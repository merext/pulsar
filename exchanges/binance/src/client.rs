use anyhow::Context;
use binance_sdk::config::ConfigurationWebsocketStreams;
use binance_sdk::spot::{
    SpotWsStreams,
    websocket_streams::{KlineIntervalEnum, KlineParams, WebsocketStreams},
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use trade::models::Kline as TradeKline;

pub struct BinanceClient {
    connection: WebsocketStreams,
}

impl BinanceClient {
    pub async fn new() -> Self {
        let config = ConfigurationWebsocketStreams::builder()
            .build()
            .expect("Failed to build config");

        let client = SpotWsStreams::production(config);

        let connection = client
            .connect()
            .await
            .expect("Failed to connect to WebSocket Streams");

        BinanceClient {
            connection: connection,
        }
    }

    pub async fn kline_stream(
        &self,
        symbol: &str,
        interval: &str,
    ) -> Result<impl futures_util::Stream<Item = TradeKline>, Box<dyn std::error::Error>> {
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

                eprintln!("Kline: {:?}", trade_kline);

                if let Err(err) = tx.try_send(trade_kline) {
                    eprintln!("Failed to send kline to stream: {:?}", err);
                }
            } else {
                eprintln!("Received message without kline data: {:?}", msg);
            }
        });

        Ok(ReceiverStream::new(rx))
    }
}
