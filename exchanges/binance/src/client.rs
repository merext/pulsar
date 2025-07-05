use anyhow::{Context, Result, anyhow};
use binance_sdk::config::ConfigurationWebsocketStreams;
use binance_sdk::spot::{
    SpotWsStreams,
    websocket_streams::{KlineIntervalEnum, KlineParams, TradeParams, WebsocketStreams},
};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use trade::models::Kline as TradeKline;
use trade::models::Trade as PulsarTrade;

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

    pub async fn kline_stream(
        self,
        symbol: &str,
        _interval: &str,
    ) -> Result<
        impl futures_util::Stream<Item = TradeKline>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let params =
            KlineParams::builder(symbol.to_string(), KlineIntervalEnum::Interval1s).build()?;

        let ws_stream = self
            .connection
            .kline(params)
            .await
            .context("Failed to subscribe to the stream")?;

        let (tx, rx) = mpsc::channel(100);

        ws_stream.on_message(move |msg| {
            if let Some(k) = &msg.k {
                let trade_kline = TradeKline {
                    open_time: k.t.unwrap_or_default() as u64,
                    open: k.o.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    high: k.h.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    low: k.l.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    close: k.c.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    volume: k.v.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    close_time: k.t.unwrap_or_default() as u64,
                    quote_asset_volume: k.q.as_ref().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    number_of_trades: k.n.unwrap_or_default() as u64,
                    taker_buy_base_asset_volume: k
                        .v
                        .as_ref()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                    taker_buy_quote_asset_volume: k
                        .q
                        .as_ref()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0.0),
                };

                // log::debug!("{:#?}", trade_kline);

                if let Err(err) = tx.try_send(trade_kline) {
                    log::error!("Failed to send kline to stream: {:?}", err);
                }
            } else {
                log::warn!("Received message without kline data: {:?}", msg);
            }
        });

        Ok(ReceiverStream::new(rx))
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

        let (tx, rx) = mpsc::channel(100);

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
                log::error!("Failed to send trade to stream: {:?}", err);
            }
        });

        Ok(ReceiverStream::new(rx))
    }
}
