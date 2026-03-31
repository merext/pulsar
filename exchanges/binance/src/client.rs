use anyhow::Result;
use bytes::Bytes;
use csv::ReaderBuilder;
use futures_util::{SinkExt, StreamExt, stream::{self, Stream}};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::borrow::Cow;
use std::io::Cursor;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};
use trade::market::{BookLevel, BookTicker, DepthLevel, DepthSnapshot, MarketEvent};
use trade::models::Trade as PulsarTrade;
use zip::read::ZipArchive;

pub struct BinanceClient;

#[derive(Debug, Clone, Default)]
pub struct CapturedMarketDataSummary {
    pub total_lines: usize,
    pub parsed_events: usize,
    pub trade_events: usize,
    pub book_ticker_events: usize,
    pub depth_events: usize,
    pub parse_errors: usize,
    pub first_event_time: Option<u64>,
    pub last_event_time: Option<u64>,
    pub event_time_regressions: usize,
    pub symbols: Vec<String>,
}

#[derive(Debug)]
pub struct ParsedMarketEventData {
    pub events: Vec<MarketEvent>,
    pub summary: CapturedMarketDataSummary,
}

#[derive(Debug, Deserialize)]
struct BinanceTradeMessage {
    #[serde(rename = "e")]
    event_type: String,
    #[serde(rename = "E")]
    event_time: u64,
    #[serde(rename = "s")]
    symbol: String,
    #[serde(rename = "t")]
    trade_id: u64,
    #[serde(rename = "p")]
    price: String,
    #[serde(rename = "q")]
    quantity: String,
    #[serde(rename = "T")]
    trade_time: u64,
    #[serde(rename = "m")]
    is_buyer_market_maker: bool,
}

#[derive(Debug, Deserialize)]
struct BinanceBookTickerMessage {
    #[serde(rename = "E")]
    event_time: Option<u64>,
    #[serde(rename = "u")]
    update_id: Option<u64>,
    #[serde(rename = "b")]
    bid_price: String,
    #[serde(rename = "B")]
    bid_quantity: String,
    #[serde(rename = "a")]
    ask_price: String,
    #[serde(rename = "A")]
    ask_quantity: String,
}

#[derive(Debug, Deserialize)]
struct BinanceDepthMessage {
    #[serde(rename = "E")]
    event_time: Option<u64>,
    #[serde(rename = "lastUpdateId")]
    last_update_id: u64,
    #[serde(rename = "bids")]
    bids: Vec<[String; 2]>,
    #[serde(rename = "asks")]
    asks: Vec<[String; 2]>,
}

#[derive(Debug, Deserialize)]
struct CapturedDepthLevel {
    price: f64,
    quantity: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
enum CapturedMarketEventRecord {
    Trade {
        event_time: u64,
        symbol: String,
        trade_id: u64,
        price: f64,
        quantity: f64,
        trade_time: u64,
        is_buyer_market_maker: bool,
    },
    BookTicker {
        event_time: u64,
        bid_price: f64,
        bid_quantity: f64,
        ask_price: f64,
        ask_quantity: f64,
    },
    Depth {
        event_time: u64,
        bids: Vec<CapturedDepthLevel>,
        asks: Vec<CapturedDepthLevel>,
    },
}

#[derive(Debug, Clone, Copy)]
enum CapturedMarketEventKind {
    Trade,
    BookTicker,
    Depth,
}

impl BinanceTradeMessage {
    fn into_trade(self) -> Option<PulsarTrade> {
        Some(PulsarTrade {
            event_type: self.event_type,
            event_time: self.event_time,
            symbol: self.symbol,
            trade_id: self.trade_id,
            price: self.price.parse().ok()?,
            quantity: self.quantity.parse().ok()?,
            buyer_order_id: None,
            seller_order_id: None,
            trade_time: self.trade_time,
            is_buyer_market_maker: self.is_buyer_market_maker,
        })
    }
}

impl BinanceBookTickerMessage {
    fn into_book_ticker(self) -> Option<BookTicker> {
        let event_time = self.event_time?;
        Some(BookTicker {
            bid: BookLevel {
                price: self.bid_price.parse().ok()?,
                quantity: self.bid_quantity.parse().ok()?,
            },
            ask: BookLevel {
                price: self.ask_price.parse().ok()?,
                quantity: self.ask_quantity.parse().ok()?,
            },
            event_time,
        })
    }
}

impl BinanceDepthMessage {
    fn into_depth_snapshot(self) -> Option<DepthSnapshot> {
        let event_time = self.event_time?;
        let bids = self
            .bids
            .into_iter()
            .map(|level| {
                Some(DepthLevel {
                    price: level[0].parse().ok()?,
                    quantity: level[1].parse().ok()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let asks = self
            .asks
            .into_iter()
            .map(|level| {
                Some(DepthLevel {
                    price: level[0].parse().ok()?,
                    quantity: level[1].parse().ok()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        Some(DepthSnapshot {
            bids,
            asks,
            event_time,
        })
    }
}

impl CapturedMarketEventRecord {
    fn kind(&self) -> CapturedMarketEventKind {
        match self {
            Self::Trade { .. } => CapturedMarketEventKind::Trade,
            Self::BookTicker { .. } => CapturedMarketEventKind::BookTicker,
            Self::Depth { .. } => CapturedMarketEventKind::Depth,
        }
    }

    fn event_time(&self) -> u64 {
        match self {
            Self::Trade { event_time, .. }
            | Self::BookTicker { event_time, .. }
            | Self::Depth { event_time, .. } => *event_time,
        }
    }

    fn symbol(&self) -> Option<&str> {
        match self {
            Self::Trade { symbol, .. } => Some(symbol),
            Self::BookTicker { .. } | Self::Depth { .. } => None,
        }
    }

    fn into_market_event(self) -> MarketEvent {
        match self {
            Self::Trade {
                event_time,
                symbol,
                trade_id,
                price,
                quantity,
                trade_time,
                is_buyer_market_maker,
            } => MarketEvent::Trade(PulsarTrade {
                event_type: "trade".to_string(),
                event_time,
                symbol,
                trade_id,
                price,
                quantity,
                buyer_order_id: None,
                seller_order_id: None,
                trade_time,
                is_buyer_market_maker,
            }),
            Self::BookTicker {
                event_time,
                bid_price,
                bid_quantity,
                ask_price,
                ask_quantity,
            } => MarketEvent::BookTicker(BookTicker {
                bid: BookLevel {
                    price: bid_price,
                    quantity: bid_quantity,
                },
                ask: BookLevel {
                    price: ask_price,
                    quantity: ask_quantity,
                },
                event_time,
            }),
            Self::Depth {
                event_time,
                bids,
                asks,
            } => MarketEvent::Depth(DepthSnapshot {
                bids: bids
                    .into_iter()
                    .map(|level| DepthLevel {
                        price: level.price,
                        quantity: level.quantity,
                    })
                    .collect(),
                asks: asks
                    .into_iter()
                    .map(|level| DepthLevel {
                        price: level.price,
                        quantity: level.quantity,
                    })
                    .collect(),
                event_time,
            }),
        }
    }
}

impl BinanceClient {
    /// # Errors
    ///
    /// Will return `Err` if the connection to Binance WebSocket Streams fails or times out.
    pub async fn new() -> Result<Self> {
        Ok(Self)
    }

    /// # Errors
    ///
    /// Will return `Err` if the subscription to the trade stream fails.
    pub async fn trade_stream(
        self,
        symbol: &str,
    ) -> Result<
        impl futures_util::Stream<Item = PulsarTrade>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let (tx, rx) = mpsc::unbounded_channel();

        let stream_symbol = symbol.to_lowercase();
        let stream_symbol_for_log = symbol.to_string();
        tokio::spawn(async move {
            let url = format!("wss://stream.binance.com:9443/ws/{}@trade", stream_symbol);
            loop {
                info!(symbol = %stream_symbol_for_log, url = %url, source = "websocket", "Connecting to live market data source");
                let connection = tokio::time::timeout(Duration::from_secs(20), connect_async(&url)).await;

                let Ok(connection) = connection else {
                    warn!(url = %url, "Public trade stream connection timed out; retrying websocket only");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                let Ok((ws_stream, _response)) = connection else {
                    warn!(url = %url, "Public trade stream connection failed; retrying websocket only");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                info!(symbol = %stream_symbol_for_log, url = %url, "Websocket market data connected");

                let (mut write, mut read) = ws_stream.split();
                let mut message_count = 0_u64;

                while let Some(message) = read.next().await {
                    match message {
                        Ok(Message::Text(text)) => match serde_json::from_str::<BinanceTradeMessage>(&text) {
                            Ok(parsed) => {
                                let Some(trade) = parsed.into_trade() else {
                                    warn!(payload = %text, "Failed to parse trade payload into numeric values");
                                    continue;
                                };

                                if tx.send(trade).is_err() {
                                    return;
                                }

                                message_count += 1;
                                if message_count <= 3 || message_count % 500 == 0 {
                                    debug!(
                                        symbol = %stream_symbol_for_log,
                                        source = "websocket",
                                        messages = message_count,
                                        "Received live market trade"
                                    );
                                }
                            }
                            Err(err) => {
                                warn!(error = ?err, payload = %text, "Failed to decode trade stream payload");
                            }
                        },
                        Ok(Message::Ping(payload)) => {
                            if let Err(err) = write.send(Message::Pong(payload)).await {
                                warn!(error = ?err, "Failed to respond to websocket ping");
                                break;
                            }
                        }
                        Ok(Message::Pong(_)) => {}
                        Ok(Message::Binary(_)) => {}
                        Ok(Message::Frame(_)) => {}
                        Ok(Message::Close(frame)) => {
                            warn!(?frame, "Trade stream closed by server; reconnecting");
                            break;
                        }
                        Err(err) => {
                            warn!(error = ?err, "Trade stream read error; reconnecting");
                            break;
                        }
                    }
                }

                warn!(url = %url, "Websocket trade stream ended; retrying websocket only");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });

        Ok(UnboundedReceiverStream::new(rx))
    }

    pub async fn market_event_stream(
        self,
        symbol: &str,
    ) -> Result<impl futures_util::Stream<Item = MarketEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let stream_symbol = symbol.to_lowercase();
        let stream_symbol_for_log = symbol.to_string();

        tokio::spawn(async move {
            let url = format!(
                "wss://stream.binance.com:9443/stream?streams={}@trade/{}@bookTicker",
                stream_symbol, stream_symbol
            );

            loop {
                info!(symbol = %stream_symbol_for_log, url = %url, source = "websocket", "Connecting to multiplexed market data source");
                let connection = tokio::time::timeout(Duration::from_secs(20), connect_async(&url)).await;

                let Ok(connection) = connection else {
                    warn!(url = %url, "Multiplexed market data connection timed out; retrying websocket only");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                let Ok((ws_stream, _response)) = connection else {
                    warn!(url = %url, "Multiplexed market data connection failed; retrying websocket only");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                info!(symbol = %stream_symbol_for_log, url = %url, "Multiplexed websocket market data connected");
                let (mut write, mut read) = ws_stream.split();
                let mut trade_count = 0_u64;
                let mut quote_count = 0_u64;

                while let Some(message) = read.next().await {
                    match message {
                        Ok(Message::Text(text)) => {
                            match parse_market_event_message(&text) {
                                Some(MarketEvent::Trade(trade)) => {
                                    if tx.send(MarketEvent::Trade(trade)).is_err() {
                                        return;
                                    }

                                    trade_count += 1;
                                    if trade_count <= 3 || trade_count % 500 == 0 {
                                        debug!(
                                            symbol = %stream_symbol_for_log,
                                            source = "websocket",
                                            trade_messages = trade_count,
                                            quote_messages = quote_count,
                                            "Received multiplexed trade event"
                                        );
                                    }
                                }
                                Some(MarketEvent::BookTicker(book_ticker)) => {
                                    if tx.send(MarketEvent::BookTicker(book_ticker)).is_err() {
                                        return;
                                    }

                                    quote_count += 1;
                                    if quote_count <= 3 || quote_count % 500 == 0 {
                                        debug!(
                                            symbol = %stream_symbol_for_log,
                                            source = "websocket",
                                            trade_messages = trade_count,
                                            quote_messages = quote_count,
                                            "Received multiplexed quote event"
                                        );
                                    }
                                }
                                Some(MarketEvent::Depth(_)) => {}
                                None => {
                                    warn!(payload = %text, "Failed to decode multiplexed market event payload");
                                }
                            }
                        }
                        Ok(Message::Ping(payload)) => {
                            if let Err(err) = write.send(Message::Pong(payload)).await {
                                warn!(error = ?err, "Failed to respond to websocket ping");
                                break;
                            }
                        }
                        Ok(Message::Pong(_)) => {}
                        Ok(Message::Binary(_)) => {}
                        Ok(Message::Frame(_)) => {}
                        Ok(Message::Close(frame)) => {
                            warn!(?frame, "Multiplexed market data stream closed by server; reconnecting");
                            break;
                        }
                        Err(err) => {
                            warn!(error = ?err, "Multiplexed market data read error; reconnecting");
                            break;
                        }
                    }
                }

                warn!(url = %url, "Multiplexed websocket market data ended; retrying websocket only");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });

        Ok(UnboundedReceiverStream::new(rx))
    }

    pub async fn market_event_stream_with_depth(
        self,
        symbol: &str,
        depth_levels: u32,
    ) -> Result<impl futures_util::Stream<Item = MarketEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let stream_symbol = symbol.to_lowercase();
        let stream_symbol_for_log = symbol.to_string();
        let depth_levels = depth_levels.max(5);

        tokio::spawn(async move {
            let url = format!(
                "wss://stream.binance.com:9443/stream?streams={}@trade/{}@bookTicker/{}@depth{}@100ms",
                stream_symbol, stream_symbol, stream_symbol, depth_levels
            );

            loop {
                info!(symbol = %stream_symbol_for_log, url = %url, source = "websocket", "Connecting to multiplexed market data source with depth");
                let connection = tokio::time::timeout(Duration::from_secs(20), connect_async(&url)).await;

                let Ok(connection) = connection else {
                    warn!(url = %url, "Multiplexed market data with depth timed out; retrying websocket only");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                let Ok((ws_stream, _response)) = connection else {
                    warn!(url = %url, "Multiplexed market data with depth failed; retrying websocket only");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                };

                info!(symbol = %stream_symbol_for_log, url = %url, "Multiplexed websocket market data with depth connected");
                let (mut write, mut read) = ws_stream.split();

                while let Some(message) = read.next().await {
                    match message {
                        Ok(Message::Text(text)) => match parse_market_event_message(&text) {
                            Some(event) => {
                                if tx.send(event).is_err() {
                                    return;
                                }
                            }
                            None => {
                                warn!(payload = %text, "Failed to decode multiplexed market+depth payload");
                            }
                        },
                        Ok(Message::Ping(payload)) => {
                            if let Err(err) = write.send(Message::Pong(payload)).await {
                                warn!(error = ?err, "Failed to respond to websocket ping");
                                break;
                            }
                        }
                        Ok(Message::Pong(_)) => {}
                        Ok(Message::Binary(_)) => {}
                        Ok(Message::Frame(_)) => {}
                        Ok(Message::Close(frame)) => {
                            warn!(?frame, "Multiplexed market+depth stream closed by server; reconnecting");
                            break;
                        }
                        Err(err) => {
                            warn!(error = ?err, "Multiplexed market+depth read error; reconnecting");
                            break;
                        }
                    }
                }

                warn!(url = %url, "Multiplexed websocket market+depth ended; retrying websocket only");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });

        Ok(UnboundedReceiverStream::new(rx))
    }

    /// # Errors
    ///
    /// Will return `Err` if the HTTP request fails or the data cannot be parsed.
    pub async fn trade_data(
        url: &str,
    ) -> Result<impl Stream<Item = PulsarTrade>, Box<dyn std::error::Error + Send + Sync>> {
        let response = reqwest::get(url).await?.bytes().await?;
        let trades = Self::parse_trade_data(response)?;
        Ok(stream::iter(trades))
    }

    /// # Errors
    ///
    /// Will return `Err` if the file cannot be read or the data cannot be parsed.
    /// This method automatically detects whether the input is a URL or local file path.
    pub async fn trade_data_from_uri(
        uri: &str,
    ) -> Result<impl Stream<Item = PulsarTrade>, Box<dyn std::error::Error + Send + Sync>> {
        if uri.starts_with("http://") || uri.starts_with("https://") {
            // It's a URL, use HTTP request
            let response = reqwest::get(uri).await?.bytes().await?;
            let trades = Self::parse_trade_data(response)?;
            Ok(stream::iter(trades))
        } else {
            // It's a local file path
            let file_content = tokio::fs::read(uri).await?;
            let trades = Self::parse_trade_data(Bytes::from(file_content))?;
            Ok(stream::iter(trades))
        }
    }

    /// # Errors
    ///
    /// Will return `Err` if the file cannot be read or the captured JSONL cannot be parsed.
    /// This method automatically detects whether the input is a URL or local file path.
    pub async fn market_event_data_from_uri(
        uri: &str,
    ) -> Result<impl Stream<Item = MarketEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let parsed = Self::load_captured_market_event_data_from_uri(uri).await?;
        Ok(stream::iter(parsed.events))
    }

    /// # Errors
    ///
    /// Will return `Err` if the file cannot be read or the captured JSONL cannot be parsed.
    pub async fn load_captured_market_event_data_from_uri(
        uri: &str,
    ) -> Result<ParsedMarketEventData, Box<dyn std::error::Error + Send + Sync>> {
        let payload = if uri.starts_with("http://") || uri.starts_with("https://") {
            reqwest::get(uri).await?.bytes().await?
        } else {
            Bytes::from(tokio::fs::read(uri).await?)
        };

        Self::parse_market_event_data(payload)
    }

    fn parse_trade_data(
        data: Bytes,
    ) -> Result<Vec<PulsarTrade>, Box<dyn std::error::Error + Send + Sync>> {
        let cursor = Cursor::new(data);

        // Try to parse as ZIP first
        if let Ok(mut archive) = ZipArchive::new(cursor.clone())
            && let Ok(file) = archive.by_index(0)
        {
            return Self::parse_csv_from_reader(file);
        }

        // If ZIP fails, try to parse as plain CSV
        Self::parse_csv_from_reader(cursor)
    }

    fn parse_market_event_data(
        data: Bytes,
    ) -> Result<ParsedMarketEventData, Box<dyn std::error::Error + Send + Sync>> {
        let text: Cow<'_, str> = String::from_utf8_lossy(&data);
        let mut events = Vec::new();
        let mut saw_non_empty_line = false;
        let mut symbols = BTreeSet::new();
        let mut summary = CapturedMarketDataSummary::default();
        let mut previous_event_time = None;

        for (line_index, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            saw_non_empty_line = true;
            summary.total_lines += 1;
            match serde_json::from_str::<CapturedMarketEventRecord>(line) {
                Ok(record) => {
                    summary.parsed_events += 1;
                    match record.kind() {
                        CapturedMarketEventKind::Trade => summary.trade_events += 1,
                        CapturedMarketEventKind::BookTicker => summary.book_ticker_events += 1,
                        CapturedMarketEventKind::Depth => summary.depth_events += 1,
                    }

                    if let Some(symbol) = record.symbol() {
                        symbols.insert(symbol.to_string());
                    }

                    let event_time = record.event_time();
                    summary.first_event_time = Some(summary.first_event_time.map_or(event_time, |current| current.min(event_time)));
                    summary.last_event_time = Some(summary.last_event_time.map_or(event_time, |current| current.max(event_time)));

                    if let Some(previous_event_time) = previous_event_time
                        && event_time < previous_event_time
                    {
                        summary.event_time_regressions += 1;
                    }
                    previous_event_time = Some(event_time);

                    events.push(record.into_market_event());
                }
                Err(err) => {
                    summary.parse_errors += 1;
                    error!(error = ?err, line = line_index + 1, payload = %line, "Failed to parse captured market event record.");
                }
            }
        }

        if saw_non_empty_line && events.is_empty() {
            return Err("failed to parse any captured market events from JSONL".into());
        }

        summary.symbols = symbols.into_iter().collect();

        Ok(ParsedMarketEventData { events, summary })
    }

    fn parse_csv_from_reader<R: std::io::Read>(
        reader: R,
    ) -> Result<Vec<PulsarTrade>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(reader);
        let mut trades = Vec::new();

        for result in rdr.records() {
            let record = match result {
                Ok(record) => record,
                Err(e) => {
                    error!(error = ?e, "Failed to parse CSV record.");
                    continue;
                }
            };

            if record.is_empty() {
                continue;
            }

            if record.get(0).is_some_and(|value| value.eq_ignore_ascii_case("timestamp")) {
                continue;
            }

            if record.len() == 3 {
                let trade_time: u64 = match record[0].parse() {
                    Ok(time) => time,
                    Err(e) => {
                        error!(error = ?e, record = ?record, "Failed to parse simple CSV timestamp.");
                        continue;
                    }
                };

                let price: f64 = match record[1].parse() {
                    Ok(price) => price,
                    Err(e) => {
                        error!(error = ?e, record = ?record, "Failed to parse simple CSV price.");
                        continue;
                    }
                };

                let quantity: f64 = match record[2].parse() {
                    Ok(quantity) => quantity,
                    Err(e) => {
                        error!(error = ?e, record = ?record, "Failed to parse simple CSV quantity.");
                        continue;
                    }
                };

                trades.push(PulsarTrade {
                    trade_id: trades.len() as u64 + 1,
                    price,
                    quantity,
                    trade_time,
                    is_buyer_market_maker: false,
                    ..Default::default()
                });
                continue;
            }

            if record.len() < 6 {
                continue; // Skip incomplete records
            }

            let trade_id: u64 = match record[0].parse() {
                Ok(id) => id,
                Err(e) => {
                    error!(error = ?e, record = ?record, "Failed to parse trade_id.");
                    continue;
                }
            };

            let price: f64 = match record[1].parse() {
                Ok(price) => price,
                Err(e) => {
                    error!(error = ?e, record = ?record, "Failed to parse price.");
                    continue;
                }
            };

            let quantity: f64 = match record[2].parse() {
                Ok(quantity) => quantity,
                Err(e) => {
                    error!(error = ?e, record = ?record, "Failed to parse quantity.");
                    continue;
                }
            };

            let trade_time: u64 = match record[4].parse() {
                Ok(time) => time,
                Err(e) => {
                    error!(error = ?e, record = ?record, "Failed to parse trade_time.");
                    continue;
                }
            };

            let is_buyer_market_maker: bool = match record[5].to_lowercase().parse() {
                Ok(is_buyer) => is_buyer,
                Err(e) => {
                    error!(error = ?e, record = ?record, "Failed to parse is_buyer_market_maker.");
                    continue;
                }
            };

            // Optional: Filter out market maker trades (commented out for testing)
            // if is_buyer_market_maker {
            //     continue; // Ignore trades from market makers
            // }

            trades.push(PulsarTrade {
                trade_id,
                price,
                quantity,
                trade_time,
                is_buyer_market_maker,
                ..Default::default()
            });
        }

        Ok(trades)
    }

    /// # Errors
    ///
    /// Will return `Err` if the file cannot be read or the data cannot be parsed.
    pub async fn trade_data_from_path(
        path: &str,
    ) -> Result<impl Stream<Item = PulsarTrade>, Box<dyn std::error::Error + Send + Sync>> {
        let file_content = tokio::fs::read(path).await?;
        let trades = Self::parse_trade_data(Bytes::from(file_content))?;
        Ok(stream::iter(trades))
    }

    #[cfg(test)]
    pub fn parse_market_event_message_for_test(payload: &str) -> Option<MarketEvent> {
        parse_market_event_message(payload)
    }
}

#[derive(Debug, Deserialize)]
struct BinanceCombinedStreamEnvelope {
    stream: String,
    data: serde_json::Value,
}

fn parse_market_event_message(payload: &str) -> Option<MarketEvent> {
    let envelope = serde_json::from_str::<BinanceCombinedStreamEnvelope>(payload).ok()?;

    if envelope.stream.ends_with("@trade") {
        let trade = serde_json::from_value::<BinanceTradeMessage>(envelope.data)
            .ok()?
            .into_trade()?;
        return Some(MarketEvent::Trade(trade));
    }

    if envelope.stream.ends_with("@bookTicker") {
        let book_ticker = serde_json::from_value::<BinanceBookTickerMessage>(envelope.data)
            .ok()?
            .into_book_ticker()?;
        return Some(MarketEvent::BookTicker(book_ticker));
    }

    if envelope.stream.contains("@depth") {
        let depth = serde_json::from_value::<BinanceDepthMessage>(envelope.data)
            .ok()?
            .into_depth_snapshot()?;
        return Some(MarketEvent::Depth(depth));
    }

    None
}
