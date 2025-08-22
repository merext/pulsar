use anyhow::{Context, Result, anyhow};
use binance_sdk::config::ConfigurationWebsocketStreams;
use binance_sdk::spot::{
    SpotWsStreams,
    websocket_streams::{TradeParams, WebsocketStreams},
};
use bytes::Bytes;
use csv::ReaderBuilder;
use futures_util::stream::{self, Stream};
use std::io::Cursor;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::error;
use trade::models::Trade as PulsarTrade;
use zip::read::ZipArchive;

pub struct BinanceClient {
    connection: WebsocketStreams,
}

impl BinanceClient {
    /// # Errors
    ///
    /// Will return `Err` if the connection to Binance WebSocket Streams fails or times out.
    pub async fn new() -> Result<Self> {
        let config = ConfigurationWebsocketStreams::builder()
            .build()
            .context("Failed to build config")?;

        let client = SpotWsStreams::production(config);

        let connection_result =
            tokio::time::timeout(Duration::from_secs(10), client.connect()).await;

        let connection = connection_result
            .map_err(|_| anyhow!("Connection timed out"))?
            .context("Failed to connect to WebSocket Streams")?;

        Ok(Self { connection })
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
        let params = TradeParams::builder(symbol.to_string()).build()?;

        let ws_stream = self
            .connection
            .trade(params)
            .await
            .context("Failed to subscribe to the trade stream")?;

        let (tx, rx) = mpsc::channel(1000);

        ws_stream.on_message(move |msg| {
            let trade = PulsarTrade {
                event_type: msg.e.clone().unwrap_or_default(),
                #[allow(clippy::cast_sign_loss)]
                event_time: msg.e_uppercase.unwrap_or_default() as u64,
                symbol: msg.s.clone().unwrap_or_default(),
                #[allow(clippy::cast_sign_loss)]
                trade_id: msg.t.unwrap_or_default() as u64,
                price: msg.p.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                quantity: msg.q.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                buyer_order_id: None,
                seller_order_id: None,
                #[allow(clippy::cast_sign_loss)]
                trade_time: msg.t_uppercase.unwrap_or_default() as u64,
                is_buyer_market_maker: msg.m.unwrap_or_default(),
            };

            if let Err(err) = tx.try_send(trade) {
                error!(error = ?err, "Failed to send trade to stream.");
            }
        });

        Ok(ReceiverStream::new(rx))
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

    fn parse_trade_data(
        data: Bytes,
    ) -> Result<Vec<PulsarTrade>, Box<dyn std::error::Error + Send + Sync>> {
        let cursor = Cursor::new(data);
        
        // Try to parse as ZIP first
        if let Ok(mut archive) = ZipArchive::new(cursor.clone()) {
            if let Ok(file) = archive.by_index(0) {
                return Self::parse_csv_from_reader(file);
            }
        }
        
        // If ZIP fails, try to parse as plain CSV
        Self::parse_csv_from_reader(cursor)
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

            if is_buyer_market_maker {
                continue; // Ignore trades from market makers
            }

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
}
