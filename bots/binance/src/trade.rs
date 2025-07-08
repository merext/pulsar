use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use std::time::Duration;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::{debug, error, warn};
use trade::trader::{TradeMode, Trader};

pub async fn run_trading(
    trading_symbol: &str,
    _api_key: &str,
    _api_secret: &str,
    mut strategy: impl Strategy + Send,
    binance_trader: &mut BinanceTrader,
    trade_mode: TradeMode,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let timeout = 30;

    loop {
        let mut trade_stream = loop {
            match BinanceClient::new().await {
                Ok(binance_client) => match binance_client.trade_stream(trading_symbol).await {
                    Ok(stream) => break stream,
                    Err(e) => {
                        error!(
                            "action=create_trade_stream status=failed error={:?} retry_in_seconds={}",
                            e, timeout
                        );
                        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                    }
                },
                Err(e) => {
                    error!(
                        "action=connect_to_binance status=failed error={:?} retry_in_seconds={}",
                        e, timeout
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(timeout)).await;
                }
            }
        };

        // Process trade stream
        #[allow(unreachable_code)] // This loop is intended to run indefinitely for a live bot
        loop {
            tokio::select! {
                trade_result = tokio::time::timeout(Duration::from_secs(timeout), trade_stream.next()) => {
                    match trade_result {
                        Ok(Some(trade)) => {
                            // log::info!("Received trade: {:?}", trade);
                            strategy.on_trade(trade.clone().into()).await;

                            let trade_price = trade.price;
                            let trade_time = trade.trade_time as f64;

                            let (signal, confidence) = strategy.get_signal(trade_price, trade_time, binance_trader.position());


                            let min_notional = 1.0 + 4.0 * confidence;
                            let raw_quantity = min_notional / trade_price;
                            let quantity_step = 1.0; // get this from exchangeInfo or hardcode per symbol
                            let quantity_to_trade = (raw_quantity / quantity_step).ceil() * quantity_step;
                            binance_trader.on_signal(signal, trade_price, quantity_to_trade, trade_mode).await;

                            debug!(
                                signal = %signal,
                                confidence = %confidence,
                                position = ?binance_trader.position(),
                                unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                                realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                            );
                        }
                        Ok(None) => {
                            // Stream ended, break the loop to reconnect
                            break;
                        }
                        Err(_) => {
                            warn!("action=trade_stream_timeout status=reconnecting");
                            break;
                        }
                    }
                },
            }
        }
    }
}
