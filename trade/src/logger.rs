use tracing::{info, debug, warn, error};
use strategies::strategy::StrategyLogger;
use strategies::models::{Signal, TradeData, Position};

/// Trading logger that provides standardized logging for any exchange and strategy
pub struct TradeLogger {
    exchange_name: String,
    strategy_name: String,
    symbol: String,
}

impl TradeLogger {
    pub fn new(exchange_name: &str, strategy_name: &str, symbol: &str) -> Self {
        Self {
            exchange_name: exchange_name.to_string(),
            strategy_name: strategy_name.to_string(),
            symbol: symbol.to_string(),
        }
    }

    pub fn log_trade_received(&self, price: f64, quantity: f64, timestamp: f64) {
        debug!(
            exchange = %self.exchange_name,
            strategy = %self.strategy_name,
            symbol = %self.symbol,
            action = "trade_received",
            price = %format!("{:.8}", price),
            quantity = %format!("{:.2}", quantity),
            timestamp = %timestamp
        );
    }

    pub fn log_signal_generated(&self, signal: &Signal, confidence: f64, price: f64) {
        debug!(
            exchange = %self.exchange_name,
            strategy = %self.strategy_name,
            symbol = %self.symbol,
            action = "signal_generated",
            signal = %format!("{:?}", signal),
            confidence = %format!("{:.4}", confidence),
            price = %format!("{:.8}", price)
        );
    }

    pub fn log_position_change(&self, old_position: &Position, new_position: &Position) {
        if old_position.quantity != new_position.quantity || 
           (old_position.entry_price - new_position.entry_price).abs() > 0.00000001 {
            debug!(
                exchange = %self.exchange_name,
                strategy = %self.strategy_name,
                symbol = %self.symbol,
                action = "position_changed",
                old_quantity = %format!("{:.2}", old_position.quantity),
                new_quantity = %format!("{:.2}", new_position.quantity),
                old_entry_price = %format!("{:.8}", old_position.entry_price),
                new_entry_price = %format!("{:.8}", new_position.entry_price)
            );
        }
    }

    pub fn log_trade_executed(&self, signal: &Signal, price: f64, quantity: f64, pnl: Option<f64>) {
        match signal {
            Signal::Buy => {
                debug!(
                    exchange = %self.exchange_name,
                    strategy = %self.strategy_name,
                    symbol = %self.symbol,
                    action = "buy_executed",
                    price = %format!("{:.8}", price),
                    quantity = %format!("{:.2}", quantity)
                );
            }
            Signal::Sell => {
                if let Some(pnl) = pnl {
                    info!(
                        exchange = %self.exchange_name,
                        strategy = %self.strategy_name,
                        symbol = %self.symbol,
                        action = "sell_executed",
                        price = %format!("{:.8}", price),
                        quantity = %format!("{:.2}", quantity),
                        pnl = %format!("{:.6}", pnl)
                    );
                } else {
                    debug!(
                        exchange = %self.exchange_name,
                        strategy = %self.strategy_name,
                        symbol = %self.symbol,
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

    pub fn log_strategy_event(&self, event_type: &str, details: &str) {
        debug!(
            exchange = %self.exchange_name,
            strategy = %self.strategy_name,
            symbol = %self.symbol,
            action = "strategy_event",
            event_type = %event_type,
            details = %details
        );
    }

    pub fn log_indicator_update(&self, indicator_name: &str, value: f64) {
        debug!(
            exchange = %self.exchange_name,
            strategy = %self.strategy_name,
            symbol = %self.symbol,
            action = "indicator_update",
            indicator = %indicator_name,
            value = %format!("{:.6}", value)
        );
    }



    pub fn log_error(&self, action: &str, error: &str) {
        error!(
            exchange = %self.exchange_name,
            strategy = %self.strategy_name,
            symbol = %self.symbol,
            action = %action,
            error = %error
        );
    }

    pub fn log_warning(&self, action: &str, warning: &str) {
        warn!(
            exchange = %self.exchange_name,
            strategy = %self.strategy_name,
            symbol = %self.symbol,
            action = %action,
            warning = %warning
        );
    }
}

/// Implementation of StrategyLogger that delegates to TradeLogger
pub struct StrategyLoggerAdapter {
    trade_logger: TradeLogger,
}

impl StrategyLoggerAdapter {
    pub fn new(trade_logger: TradeLogger) -> Self {
        Self { trade_logger }
    }
}

impl StrategyLogger for StrategyLoggerAdapter {
    fn log_strategy_event(&self, event_type: &str, details: &str) {
        self.trade_logger.log_strategy_event(event_type, details);
    }

    fn log_signal_generated(&self, signal: &Signal, confidence: f64, price: f64) {
        self.trade_logger.log_signal_generated(signal, confidence, price);
    }

    fn log_trade_processed(&self, trade: &TradeData) {
        self.trade_logger.log_trade_received(trade.price, trade.quantity, trade.timestamp);
    }

    fn log_indicator_update(&self, indicator_name: &str, value: f64) {
        self.trade_logger.log_indicator_update(indicator_name, value);
    }

    fn log_position_change(&self, old_position: &Position, new_position: &Position) {
        self.trade_logger.log_position_change(old_position, new_position);
    }
    
    fn log_trade_executed(&self, signal: &Signal, price: f64, quantity: f64, pnl: Option<f64>) {
        self.trade_logger.log_trade_executed(signal, price, quantity, pnl);
    }
}
