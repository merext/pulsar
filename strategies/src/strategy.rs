use crate::models::{Position, Signal, TradeData};
use std::path::Path;

/// Universal logging interface for strategies
pub trait StrategyLogger: Send + Sync {
    fn log_signal_generated(&self, symbol: &str, signal: &Signal, confidence: f64, price: f64);
    fn log_trade_executed(&self, symbol: &str, signal: &Signal, price: f64, quantity: f64, pnl: Option<f64>, profit: Option<f64>);
}

#[allow(async_fn_in_trait)]
#[async_trait::async_trait]
pub trait Strategy: Send + Sync {
    /// Get the strategy logger for universal logging
    fn logger(&self) -> &dyn StrategyLogger;
    /// Create a new strategy instance from a configuration file
    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    fn get_info(&self) -> String;
    async fn on_trade(&mut self, trade: TradeData);
    fn get_signal(&mut self, current_position: Position) -> (Signal, f64);
}

/// Default implementation of StrategyLogger that does nothing
pub struct NoOpStrategyLogger;

impl StrategyLogger for NoOpStrategyLogger {
    fn log_signal_generated(&self, _symbol: &str, _signal: &Signal, _confidence: f64, _price: f64) {}
    fn log_trade_executed(&self, _symbol: &str, _signal: &Signal, _price: f64, _quantity: f64, _pnl: Option<f64>, _profit: Option<f64>) {}
}
