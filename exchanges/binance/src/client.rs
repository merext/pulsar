use binance_spot_connector_rust::{
    market::klines::KlineInterval,
    market_stream::kline::KlineStream,
    tokio_tungstenite::BinanceWebSocketClient,
};
use crate::models::{KlineMessage};
use futures_util::StreamExt;
use log::info;

pub struct BinanceClient;

impl BinanceClient {
    pub async fn subscribe_klines(symbol: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Connect to WebSocket
        let (mut conn, _) = BinanceWebSocketClient::connect_async_default().await?;

        // Subscribe to 1-minute Kline stream for the given symbol
        conn.subscribe(vec![
            &KlineStream::new(symbol, KlineInterval::Minutes1).into()
        ]).await;

        info!("Subscribed to 1m Kline stream for {}", symbol);

        while let Some(message) = conn.as_mut().next().await {
            match message {
                Ok(msg) => {
                    let binary_data = msg.into_data();
                    let data = std::str::from_utf8(&binary_data)?;
                    match serde_json::from_str::<KlineMessage>(data) {
                        Ok(parsed) => {
                            let kline = parsed.data.kline;
                            info!("{:?}", kline);
                            // You can now send this `kline` to a strategy or processor here
                        }
                        Err(e) => {
                            eprintln!("Failed to parse kline message: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error receiving message: {:?}", e);
                    break;
                }
            }
        }

        // Cleanly close connection if the loop exits
        conn.close().await?;

        Ok(())
    }
}