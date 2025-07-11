use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::{debug, info};
use trade::trader::{TradeMode, Trader};

pub async fn run_trade(
    trading_symbol: &str,
    _api_key: &str,
    _api_secret: &str,
    mut strategy: impl Strategy + Send,
    binance_trader: &mut BinanceTrader,
    trade_mode: TradeMode,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new()
        .await
        .expect("Failed to create BinanceClient");
    let mut trade_stream = binance_client
        .trade_stream(trading_symbol)
        .await
        .expect("Failed to create trade stream");

    // Process trade stream
    loop {
        let trade = match tokio::time::timeout(
            std::time::Duration::from_secs(5),
            trade_stream.next(),
        )
        .await
        {
            Ok(Some(trade)) => trade,
            Ok(None) => {
                // Stream ended, break the loop
                break Ok(());
            }
            Err(_) => {
                // Timeout occurred, continue to the next iteration
                continue;
            }
        };

        strategy.on_trade(trade.clone().into()).await;

        let trade_price = trade.price;
        let trade_time = trade.trade_time as f64;

        let (signal, confidence) =
            strategy.get_signal(trade_price, trade_time, binance_trader.position());

        let min_notional = 1.0 + 3.0 * confidence;
        let raw_quantity = min_notional / trade_price;
        let quantity_step = 1.0; // get this from exchangeInfo or hardcode per symbol
        let quantity_to_trade = (raw_quantity / quantity_step).ceil() * quantity_step;

        debug!(
            signal = %signal,
            confidence = %format!("{:.2}", confidence),
            position = ?binance_trader.position(),
            unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
            realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
        );

        match trade_mode {
            TradeMode::Real => {
                binance_trader
                    .on_signal(signal, trade_price, quantity_to_trade)
                    .await;
            }
            TradeMode::Emulated => {
                binance_trader
                    .on_emulate(signal, trade_price, quantity_to_trade)
                    .await;
            }
        }
    }
}
