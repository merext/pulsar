use strategies::models::Signal;
use strategies::strategy::StrategyLogger;
use tracing::{debug, error, info, warn};

/// Trading logger that provides standardized logging for any exchange and strategy
pub struct TradeLogger;

impl TradeLogger {
    pub fn new() -> Self {
        Self
    }

    pub fn log_trade_received(&self, symbol: &str, price: f64, quantity: f64, timestamp: f64) {
        debug!(
            symbol = %symbol,
            action = "trade_received",
            price = %format!("{:.8}", price),
            quantity = %format!("{:.2}", quantity),
            timestamp = %timestamp
        );
    }

    pub fn log_signal_generated(&self, symbol: &str, signal: &Signal, confidence: f64, price: f64) {
        debug!(
            symbol = %symbol,
            action = "signal_generated",
            signal = %format!("{:?}", signal),
            confidence = %format!("{:.4}", confidence),
            price = %format!("{:.8}", price)
        );
    }

    pub fn log_trade_executed(&self, symbol: &str, signal: &Signal, price: f64, quantity: f64, pnl: Option<f64>, profit: Option<f64>, trade_summary: Option<(usize, usize)>) {
        let trades_info = if let Some((total_ticks, total_trades)) = trade_summary {
            format!("{}/{}", total_ticks, total_trades)
        } else {
            "unknown".to_string()
        };

        match signal {
            Signal::Buy => {
                info!(
                    trades = %trades_info,
                    symbol = %symbol,
                    action = "buy_executed",
                    price = %format!("{:.8}", price),
                    quantity = %format!("{:.2}", quantity)
                );
            }
            Signal::Sell => {
                if let Some(pnl) = pnl {
                    if let Some(profit) = profit {
                        info!(
                            trades = %trades_info,
                            symbol = %symbol,
                            action = "sell_executed",
                            price = %format!("{:.8}", price),
                            quantity = %format!("{:.2}", quantity),
                            pnl = %format!("{:.6}", pnl),
                            profit = %format!("{:.6}", profit)
                        );
                    } else {
                        info!(
                            trades = %trades_info,
                            symbol = %symbol,
                            action = "sell_executed",
                            price = %format!("{:.8}", price),
                            quantity = %format!("{:.2}", quantity),
                            pnl = %format!("{:.6}", pnl)
                        );
                    }
                } else if let Some(profit) = profit {
                    info!(
                        trades = %trades_info,
                        symbol = %symbol,
                        action = "sell_executed",
                        price = %format!("{:.8}", price),
                        quantity = %format!("{:.2}", quantity),
                        profit = %format!("{:.6}", profit)
                    );
                } else {
                    info!(
                        trades = %trades_info,
                        symbol = %symbol,
                        action = "sell_executed",
                        price = %format!("{:.8}", price),
                        quantity = %format!("{:.2}", quantity)
                    );
                }
            }
            Signal::Hold => {
                // No action needed for hold signals
            }
        }
    }

    pub fn log_error(&self, symbol: &str, action: &str, error: &str) {
        error!(
            symbol = %symbol,
            action = %action,
            error = %error
        );
    }

    pub fn log_warning(&self, symbol: &str, action: &str, warning: &str) {
        warn!(
            symbol = %symbol,
            action = %action,
            warning = %warning
        );
    }
}

impl Default for TradeLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of StrategyLogger that delegates to TradeLogger
pub struct StrategyLoggerAdapter<'a> {
    trade_logger: &'a TradeLogger,
}

impl<'a> StrategyLoggerAdapter<'a> {
    pub fn new(trade_logger: &'a TradeLogger) -> Self {
        Self { trade_logger }
    }
}

impl<'a> StrategyLogger for StrategyLoggerAdapter<'a> {
    fn log_signal_generated(&self, symbol: &str, signal: &Signal, confidence: f64, price: f64) {
        self.trade_logger
            .log_signal_generated(symbol, signal, confidence, price);
    }

    fn log_trade_executed(&self, symbol: &str, signal: &Signal, price: f64, quantity: f64, pnl: Option<f64>, profit: Option<f64>, trade_summary: Option<(usize, usize)>) {
        self.trade_logger
            .log_trade_executed(symbol, signal, price, quantity, pnl, profit, trade_summary);
    }
}
