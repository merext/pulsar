use crate::position::Position;

pub struct VirtualTrader {
    pub position: Position,
    pub entry_price: f64,
    pub realized_pnl: f64,
    pub trades_executed: usize,
}

impl VirtualTrader {
    pub fn new() -> Self {
        Self {
            position: Position::Flat,
            entry_price: 0.0,
            realized_pnl: 0.0,
            trades_executed: 0,
        }
    }

    /// Process a trading signal at given price, update position and pnl accordingly.
    pub fn on_signal(&mut self, signal: Signal, price: f64) {
        match signal {
            Signal::Buy => {
                if self.position == Position::Flat {
                    self.position = Position::Long;
                    self.entry_price = price;
                    self.trades_executed += 1;
                    log::info!("Entered LONG at price {:.5}", price);
                }
            }
            Signal::Sell => {
                if self.position == Position::Long {
                    let profit = price - self.entry_price;
                    self.realized_pnl += profit;
                    log::info!(
                        "Exited LONG at price {:.5}, profit {:.5}, total PnL {:.5}",
                        price,
                        profit,
                        self.realized_pnl
                    );
                    self.position = Position::Flat;
                    self.entry_price = 0.0;
                    self.trades_executed += 1;
                }
            }
            Signal::Hold => {
                // Do nothing for hold
            }
        }
    }

    /// Calculate unrealized PnL for open position at current price
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.position == Position::Long {
            current_price - self.entry_price
        } else {
            0.0
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

use std::fmt;

impl fmt::Display for Signal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Signal::Buy => "BUY",
            Signal::Sell => "SELL",
            Signal::Hold => "HOLD",
        };
        write!(f, "{s}")
    }
}
