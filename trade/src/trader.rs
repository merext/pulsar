use crate::signal::Signal;
use crate::models::Trade;
use crate::metrics::{PerformanceMetrics, PositionManager};
use futures_util::Stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    Maker,
    Taker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeMode {
    Real,
    Emulated,
    Backtest,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Position {{ symbol: \"{}\", quantity: {:.6}, entry_price: {:.8} }}",
            self.symbol, self.quantity, self.entry_price
        )
    }
}

#[async_trait::async_trait]
pub trait Trader {
    // Centralized metrics access
    fn get_metrics(&self) -> &PerformanceMetrics;
    fn get_position_manager(&self) -> &PositionManager;
    
    // Account and trading operations
    async fn account_status(&self) -> Result<(), anyhow::Error>;
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64);
    
    // Exchange calculates exact trade size based on symbol, price, confidence, min/max trade sizes, and step size
    fn calculate_trade_size(&self, symbol: &str, price: f64, confidence: f64, trading_size_min: f64, trading_size_max: f64, trading_size_step: f64) -> f64;
    
    // Universal trading loop that handles all trading modes
    async fn trade(
        &mut self,
        trading_stream: impl Stream<Item = Trade> + Unpin + Send,
        trading_strategy: &mut dyn strategies::strategy::Strategy,
        trading_symbol: &str,
        trading_mode: TradeMode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

} 