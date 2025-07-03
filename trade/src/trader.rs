use async_trait::async_trait;
use crate::signal::Signal;

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
}

#[async_trait]
pub trait Trader {
    async fn on_signal(&mut self, signal: Signal, price: f64);
    fn unrealized_pnl(&self, current_price: f64) -> f64;
    fn position(&self) -> Position;
}

pub struct VirtualTrader {
    pub position: Position,
}

impl VirtualTrader {
    pub fn new() -> Self {
        VirtualTrader {
            position: Position {
                symbol: "".to_string(),
                quantity: 0.0,
                entry_price: 0.0,
            },
        }
    }
}

#[async_trait]
impl Trader for VirtualTrader {
    async fn on_signal(&mut self, signal: Signal, price: f64) {
        // Placeholder for simulating orders
        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 {
                    self.position.quantity = 1.0; // Simulate buying 1 unit
                    self.position.entry_price = price;
                    println!("VirtualTrader: BUY at {}", price);
                }
            },
            Signal::Sell => {
                if self.position.quantity > 0.0 {
                    println!("VirtualTrader: SELL at {}", price);
                    self.position.quantity = 0.0;
                    self.position.entry_price = 0.0;
                }
            },
            Signal::Hold => {
                // Do nothing
            }
        }
    }

    fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.position.quantity > 0.0 {
            (current_price - self.position.entry_price) * self.position.quantity
        } else {
            0.0
        }
    }

    fn position(&self) -> Position {
        self.position.clone()
    }
}

pub struct BinanceRealTrader;

#[async_trait]
impl Trader for BinanceRealTrader {
    async fn on_signal(&mut self, _signal: Signal, _price: f64) {
        // Placeholder for executing real trades via Binance API
    }

    fn unrealized_pnl(&self, _current_price: f64) -> f64 {
        // Placeholder for real PnL calculation
        0.0
    }

    fn position(&self) -> Position {
        Position {
            symbol: "".to_string(),
            quantity: 0.0,
            entry_price: 0.0,
        }
    }
}