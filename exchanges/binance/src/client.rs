use binance_spot_connector_rust::{
    market::klines::KlineInterval,
    market_stream::kline::KlineStream,
    market_stream::trade::TradeStream,
    tokio_tungstenite::BinanceWebSocketClient,
};
use crate::models::{KlineMessage, TradeMessage};
use futures_util::StreamExt;
use log::info;
use strategies::models::{Kline as StrategyKline, TradeData as StrategyTradeData};
use std::pin::Pin;
use tokio_stream::Stream;

pub struct BinanceClient;

impl BinanceClient {
    pub async fn subscribe_klines(
        symbol: &str,
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyKline> + Send>>, Box<dyn std::error::Error>> {
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe and discard the subscription ID (u64)
        _ = conn
            .subscribe(vec![&KlineStream::new(symbol, KlineInterval::Minutes1).into()])
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
    ) -> Result<Pin<Box<dyn Stream<Item = StrategyTradeData> + Send>>, Box<dyn std::error::Error>> {
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe and discard the subscription ID (u64)
        _ = conn
            .subscribe(vec![&TradeStream::new(symbol).into()])
            .await;

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
}