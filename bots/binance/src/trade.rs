use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::{debug, info};
use trade::trader::{TradeMode, Trader};
use trade::signal::Signal;

pub async fn run_trade(
    trading_symbol: &str,
    _api_key: &str,
    _api_secret: &str,
    mut strategy: impl Strategy + Send,
    binance_trader: &mut BinanceTrader,
    trade_mode: TradeMode,
    trading_size_min: f64,
    trading_size_max: f64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new()
        .await
        .expect("Failed to create BinanceClient");
    let mut trade_stream = binance_client
        .trade_stream(trading_symbol)
        .await
        .expect("Failed to create trade stream");

    let mut last_position_change_time = 0.0;
    let mut last_position_quantity = 0.0;
    let cooldown_period = 2.0; // 2 seconds cooldown after position change

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
        let current_position = binance_trader.position();

        // Check if position has changed recently
        let time_since_position_change = trade_time - last_position_change_time;
        let position_has_changed = current_position.quantity != last_position_quantity;
        
        if position_has_changed {
            last_position_change_time = trade_time;
            last_position_quantity = current_position.quantity;
        }

        // Skip signal generation if we're in cooldown period after position change
        if time_since_position_change < cooldown_period && position_has_changed {
            debug!(
                "Skipping signal generation due to cooldown period. Time since position change: {:.2}s",
                time_since_position_change
            );
            continue;
        }

        let (signal, confidence) =
            strategy.get_signal(trade_price, trade_time, current_position.clone());

        // Additional position-aware signal filtering
        let final_signal = match signal {
            Signal::Buy => {
                if current_position.quantity > 0.0 {
                    debug!("Ignoring BUY signal - already have position");
                    Signal::Hold
                } else {
                    Signal::Buy
                }
            }
            Signal::Sell => {
                if current_position.quantity == 0.0 {
                    debug!("Ignoring SELL signal - no position to sell");
                    Signal::Hold
                } else {
                    Signal::Sell
                }
            }
            Signal::Hold => Signal::Hold,
        };

        // Exchange calculates exact trade size based on symbol, price, confidence, min/max trade sizes, and step size
        let quantity_to_trade = binance_trader.calculate_trade_size(trading_symbol, trade_price, confidence, trading_size_min, trading_size_max, 1.0);

        // Log BUY/SELL signals at info level with structured format
        match final_signal {
            Signal::Buy | Signal::Sell => {
                info!(
                    signal = %final_signal,
                    confidence = %format!("{:.2}", confidence),
                    position = ?current_position,
                    unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                    realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                );
            }
            Signal::Hold => {
                // Only log HOLD at debug level
                debug!(
                    signal = %final_signal,
                    confidence = %format!("{:.2}", confidence),
                    position = ?current_position,
                    unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                    realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                );
            }
        }

        match trade_mode {
            TradeMode::Real => {
                binance_trader
                    .on_signal(final_signal, trade_price, quantity_to_trade)
                    .await;
            }
            TradeMode::Emulated => {
                binance_trader
                    .on_emulate(final_signal, trade_price, quantity_to_trade)
                    .await;
            }
        }
    }
}
