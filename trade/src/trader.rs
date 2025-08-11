use crate::signal::Signal;

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
    fn unrealized_pnl(&self, current_price: f64) -> f64;
    fn realized_pnl(&self) -> f64;
    fn position(&self) -> Position;
    async fn account_status(&self) -> Result<(), anyhow::Error>;
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64);
    async fn on_emulate(&mut self, signal: Signal, price: f64, quantity: f64) -> Option<(f64, f64, f64, f64, crate::trading_engine::OrderType)>;
    
    // Exchange calculates exact trade size based on symbol, price, confidence, min/max trade sizes, and step size
    fn calculate_trade_size(&self, symbol: &str, price: f64, confidence: f64, trading_size_min: f64, trading_size_max: f64, trading_size_step: f64) -> f64;
} 