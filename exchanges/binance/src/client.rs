use crate::models::{KlineMessage, TradeMessage};
use binance_spot_connector_rust::{
    market::klines::KlineInterval, market_stream::kline::KlineStream,
    market_stream::trade::TradeStream, tokio_tungstenite::BinanceWebSocketClient,
};
use futures_util::StreamExt;
use log::info;
use std::io::BufReader;
use std::pin::Pin;
use strategies::models::{Kline as StrategyKline, TradeData as StrategyTradeData};
use tokio_stream::Stream;

pub struct BinanceClient;

impl BinanceClient {
    fn get_kline_interval(interval: &str) -> Result<KlineInterval, Box<dyn std::error::Error>> {
        match interval {
            "1m" => Ok(KlineInterval::Minutes1),
            "3m" => Ok(KlineInterval::Minutes3),
            "5m" => Ok(KlineInterval::Minutes5),
            "15m" => Ok(KlineInterval::Minutes15),
            "30m" => Ok(KlineInterval::Minutes30),
            "1h" => Ok(KlineInterval::Hours1),
            "2h" => Ok(KlineInterval::Hours2),
            "4h" => Ok(KlineInterval::Hours4),
            "6h" => Ok(KlineInterval::Hours6),
            "8h" => Ok(KlineInterval::Hours8),
            "12h" => Ok(KlineInterval::Hours12),
            "1d" => Ok(KlineInterval::Days1),
            "3d" => Ok(KlineInterval::Days3),
            "1w" => Ok(KlineInterval::Weeks1),
            "1M" => Ok(KlineInterval::Months1),
            _ => Err(format!("Invalid kline interval: {}", interval).into()),
        }
    }

    pub async fn subscribe_klines(
        symbol: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyKline> + Send>>, Box<dyn std::error::Error>> {
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe and discard the subscription ID (u64)
        _ = conn
            .subscribe(vec![
                &KlineStream::new(symbol, KlineInterval::Minutes1).into(),
            ])
            .await;

        info!("Subscribed to 1m Kline stream for {}", symbol);

        let stream = async_stream::stream! {
            while let Some(message) = conn.as_mut().next().await {
                match message {
                    Ok(msg) => {
                        let binary_data = msg.into_data();
                        if let Ok(data) = std::str::from_utf8(&binary_data) {
                            if let Ok(parsed) = serde_json::from_str::<KlineMessage>(data) {
                                yield parsed.data.kline.into(); // Convert to StrategyKline
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error receiving kline message: {:?}", e);
                        break;
                    }
                }
            }

            let _ = conn.close().await;
        };

        Ok(Box::pin(stream))
    }

    pub async fn subscribe_trades(
        symbol: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyTradeData> + Send>>, Box<dyn std::error::Error>>
    {
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe and discard the subscription ID (u64)
        _ = conn.subscribe(vec![&TradeStream::new(symbol).into()]).await;

        info!("Subscribed to trade stream for {}", symbol);

        let stream = async_stream::stream! {
            while let Some(message) = conn.as_mut().next().await {
                match message {
                    Ok(msg) => {
                        let binary_data = msg.into_data();
                        if let Ok(data) = std::str::from_utf8(&binary_data) {
                            if let Ok(parsed) = serde_json::from_str::<TradeMessage>(data) {
                                yield parsed.data.into(); // Convert to StrategyTradeData
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error receiving trade message: {:?}", e);
                        break;
                    }
                }
            }

            let _ = conn.close().await;
        };

        Ok(Box::pin(stream))
    }

    pub async fn kline_stream(
        symbol: &str,
        interval: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyKline> + Send>>, Box<dyn std::error::Error>> {
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe and discard the subscription ID (u64)
        _ = conn
            .subscribe(vec![
                &KlineStream::new(symbol, Self::get_kline_interval(interval)?).into(),
            ])
            .await;

        info!("Subscribed to {} Kline stream for {}", interval, symbol);

        let stream = async_stream::stream! {
            while let Some(message) = conn.as_mut().next().await {
                match message {
                    Ok(msg) => {
                        let binary_data = msg.into_data();
                        if let Ok(data) = std::str::from_utf8(&binary_data) {
                            if let Ok(parsed) = serde_json::from_str::<KlineMessage>(data) {
                                yield parsed.data.kline.into(); // Convert to StrategyKline
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error receiving kline message: {:?}", e);
                        break;
                    }
                }
            }

            let _ = conn.close().await;
        };

        Ok(Box::pin(stream))
    }

    pub async fn trade_stream(
        symbol: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyTradeData> + Send>>, Box<dyn std::error::Error>> {
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe and discard the subscription ID (u64)
        _ = conn.subscribe(vec![&TradeStream::new(symbol).into()]).await;

        info!("Subscribed to trade stream for {}", symbol);

        let stream = async_stream::stream! {
            while let Some(message) = conn.as_mut().next().await {
                match message {
                    Ok(msg) => {
                        let binary_data = msg.into_data();
                        if let Ok(data) = std::str::from_utf8(&binary_data) {
                            if let Ok(parsed) = serde_json::from_str::<TradeMessage>(data) {
                                yield parsed.data.into(); // Convert to StrategyTradeData
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Error receiving trade message: {:?}", e);
                        break;
                    }
                }
            }

            let _ = conn.close().await;
        };

        Ok(Box::pin(stream))
    }

    pub async fn backtest_klines(
        zip_url: &str,
        symbol: &str,
        interval: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyKline> + Send>>, Box<dyn std::error::Error>> {
        use csv::ReaderBuilder;
        use futures_util::stream;
        use reqwest::Client;
        use std::io::Cursor;
        use zip::ZipArchive;

        use strategies::models::Kline as StrategyKline;

        #[allow(dead_code)]
        #[derive(Debug, serde::Deserialize)]
        struct BinanceCsvKlineRow(
            u64,    // open time
            String, // open price
            String, // high price
            String, // low price
            String, // close price
            String, // volume
            u64,    // close time
            String, // quote asset volume (ignored)
            u64,    // number of trades (ignored)
            String, // taker buy base volume (ignored)
            String, // taker buy quote volume (ignored)
            String, // ignore
        );

        // Step 1: Download ZIP from URL
        let client = Client::new();
        let resp = client.get(zip_url).send().await?.bytes().await?;
        let cursor = Cursor::new(resp);

        // Step 2: Open ZIP archive
        let mut archive = ZipArchive::new(cursor)?;
        let mut file = archive.by_index(0)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .from_reader(BufReader::new(&mut file));

        // Step 3: Parse CSV into StrategyKline list
        let mut rows = Vec::new();
        let symbol = symbol.to_string();
        let interval = interval.to_string();

        for result in reader.deserialize::<BinanceCsvKlineRow>() {
            match result {
                Ok(row) => {
                    let kline = StrategyKline {
                        open_time: row.0,
                        close_time: row.6,
                        open_price: row.1,
                        high_price: row.2,
                        low_price: row.3,
                        close_price: row.4,
                        volume: row.5,
                        symbol: symbol.clone(),
                        interval: interval.clone(),
                    };
                    rows.push(kline);
                }
                Err(e) => {
                    eprintln!("Failed to parse row: {}", e);
                    continue;
                }
            }
        }

        // Step 4: Yield as async stream
        let stream = stream::iter(rows);
        Ok(Box::pin(stream))
    }
}
