use crate::signal::Signal;

#[derive(Debug, Clone, PartialEq)]
pub enum TradeMode {
    Real,
    Emulated,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
}

#[async_trait::async_trait]
pub trait Trader {
    fn unrealized_pnl(&self, current_price: f64) -> f64;
    fn realized_pnl(&self) -> f64;
    fn position(&self) -> Position;
    async fn account_status(&self) -> Result<(), anyhow::Error>;
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64);
    async fn on_emulate(&mut self, signal: Signal, price: f64, quantity: f64);
} 