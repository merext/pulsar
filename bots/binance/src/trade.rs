use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::{debug, info};
use trade::{
    models::{Signal, TradeData},
    trader::{Position, Trader},
    trading_engine::TradingEngine,
    TradingConfig,
};
use std::time::Instant;

#[allow(clippy::too_many_lines)]
pub async fn run_trade(
    config: TradingConfig,
    trade_mode: TradeMode,
    _api_key: &str,
    _api_secret: &str,
    mut strategy: impl Strategy + Send,
    binance_trader: &mut BinanceTrader,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new()
        .await
        .expect("Failed to create Binance client");

    let trading_symbol = config.position_sizing.trading_symbol.clone();
    let mut trade_stream = binance_client
        .trade_stream(&trading_symbol)
        .await
        .expect("Failed to get trade stream");

    info!(
        "Starting to consume trade stream for {}",
        trading_symbol
    );
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
        #[allow(clippy::cast_precision_loss)]
        let trade_time = trade.trade_time as f64;
        let current_position = binance_trader.position();

        // Check if position has changed recently
        let time_since_position_change = trade_time - last_position_change_time;
        let position_has_changed = (current_position.quantity - last_position_quantity).abs() > f64::EPSILON;

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
        // Use proper trading config values instead of hardcoded zeros
        let quantity_to_trade = binance_trader.calculate_trade_size(
            &trading_symbol,
            trade_price,
            confidence,
            config.position_sizing.trading_size_min,
            config.position_sizing.trading_size_max,
            1.0, // trading_size_step - use 1.0 for DOGEUSDT
        );

        // Only log HOLD signals at debug level (no execution)
        if matches!(final_signal, Signal::Hold) {
            debug!(
                signal = %final_signal,
                confidence = %format!("{:.2}", confidence),
                position = %current_position
            );
        }

        match trade_mode {
            TradeMode::Real => {
                binance_trader
                    .on_signal(final_signal, trade_price, quantity_to_trade)
                    .await;
            }
            TradeMode::Emulated => {
                // Record execution start time for latency measurement
                let execution_start = Instant::now();
                
                // Simulate realistic network latency (1-5ms typical for HFT)
                let network_latency = if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    // Simulate variable latency based on market conditions
                    let base_latency = 1.0; // 1ms base
                    let volatility_factor = (trade_price - current_position.entry_price).abs() / trade_price;
                    let latency_variance = (volatility_factor * 4.0).min(4.0); // 0-4ms additional
                    std::time::Duration::from_millis((base_latency + latency_variance) as u64)
                } else {
                    std::time::Duration::from_millis(0)
                };

                // Simulate network delay
                if !network_latency.is_zero() {
                    tokio::time::sleep(network_latency).await;
                }

                // Use TradingEngine for realistic trade execution simulation
                let trade_result = if let Some(engine) = &mut trading_engine {
                    engine.on_emulate(final_signal, trade_price, quantity_to_trade).await
                } else {
                    // Fallback to BinanceTrader if TradingEngine not available
                    binance_trader.on_emulate(final_signal, trade_price, quantity_to_trade).await
                };

                // Calculate total execution time including simulated latency
                let total_execution_time = execution_start.elapsed();
                let computational_latency = total_execution_time.saturating_sub(network_latency);

                // Log executed trades with detailed execution metrics
                if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    if let Some((fill_price, fees, slippage, rebates, order_type)) = trade_result {
                        info!(
                            signal = %final_signal,
                            confidence = %format!("{:.2}", confidence),
                            price = format!("{:.8}", trade_price),
                            fill_price = format!("{:.8}", fill_price),
                            quantity = format!("{:.2}", quantity_to_trade),
                            order_type = %order_type,
                            fees = format!("{:.8}", fees),
                            slippage = format!("{:.8}", slippage),
                            rebates = format!("{:.8}", rebates),
                            computational_latency = format!("{:?}", computational_latency),
                            network_latency = format!("{:?}", network_latency),
                            total_latency = format!("{:?}", total_execution_time),
                            position = %current_position,
                            unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                            realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                        );
                    } else {
                        // Fallback logging if trade_result is None
                        info!(
                            signal = %final_signal,
                            confidence = %format!("{:.2}", confidence),
                            price = format!("{:.8}", trade_price),
                            quantity = format!("{:.2}", quantity_to_trade),
                            computational_latency = format!("{:?}", computational_latency),
                            network_latency = format!("{:?}", network_latency),
                            total_latency = format!("{:?}", total_execution_time),
                            position = %current_position,
                            unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                            realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                        );
                    }
                }
            }
        }
    }
}
