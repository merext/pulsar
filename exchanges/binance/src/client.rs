use anyhow::{Context, Result, anyhow};
use binance_sdk::config::ConfigurationWebsocketStreams;
use binance_sdk::spot::{
    SpotWsStreams,
    websocket_streams::{TradeParams, WebsocketStreams},
};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use trade::models::Trade as PulsarTrade;
use tracing::{error};

pub struct BinanceClient {
    connection: WebsocketStreams,
}

impl BinanceClient {
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

        Ok(BinanceClient { connection })
    }

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

        let (tx, rx) = mpsc::channel(10);

        ws_stream.on_message(move |msg| {
            let trade = PulsarTrade {
                event_type: msg.e.clone().unwrap_or_default(),
                event_time: msg.e_uppercase.unwrap_or_default() as u64,
                symbol: msg.s.clone().unwrap_or_default(),
                trade_id: msg.t.unwrap_or_default() as u64,
                price: msg.p.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                quantity: msg.q.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                buyer_order_id: None,
                seller_order_id: None,
                trade_time: msg.t_uppercase.unwrap_or_default() as u64,
                is_buyer_market_maker: msg.m.unwrap_or_default(),
            };

            if let Err(err) = tx.try_send(trade) {
                error!(error = ?err, "Failed to send trade to stream.");
            }
        });

        Ok(ReceiverStream::new(rx))
    }
}
