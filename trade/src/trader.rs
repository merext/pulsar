use async_trait::async_trait;
use crate::signal::Signal;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{ symbol: {}, qty: {:.5}, entry: {:.5} }}",
            self.symbol,
            self.quantity,
            self.entry_price
        )
    }
}

#[async_trait]
pub trait Trader {
    async fn on_signal(&mut self, signal: Signal, price: f64);
    fn unrealized_pnl(&self, current_price: f64) -> f64;
    fn realized_pnl(&self) -> f64;
    fn position(&self) -> Position;
}