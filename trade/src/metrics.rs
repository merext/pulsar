use crate::signal::Signal;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp: f64,
    pub price: f64,
    pub quantity: f64,
    pub signal: Signal,
    pub pnl: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_time: f64,
}

#[derive(Debug, Clone)]
pub struct PositionManager {
    positions: HashMap<String, Position>,
    metrics: PerformanceMetrics,
}

impl PositionManager {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            metrics: PerformanceMetrics::new(),
        }
    }

    // Position management
    pub fn open_position(&mut self, symbol: &str, price: f64, quantity: f64, timestamp: f64) {
        let position = Position {
            symbol: symbol.to_string(),
            quantity,
            entry_price: price,
            entry_time: timestamp,
        };
        self.positions.insert(symbol.to_string(), position);
    }

    pub fn close_position(&mut self, symbol: &str, price: f64, timestamp: f64) -> f64 {
        if let Some(position) = self.positions.remove(symbol) {
            let pnl = (price - position.entry_price) * position.quantity;
            self.metrics.record_trade(TradeRecord {
                timestamp,
                price,
                quantity: position.quantity,
                signal: Signal::Sell, // Assuming closing is selling
                pnl: Some(pnl),
            });
            pnl
        } else {
            0.0
        }
    }

    pub fn update_position(&mut self, symbol: &str, new_quantity: f64, new_price: f64, timestamp: f64) {
        if let Some(position) = self.positions.get_mut(symbol) {
            if new_quantity != position.quantity {
                // If quantity changed, calculate PnL for the change
                let quantity_change = new_quantity - position.quantity;
                if quantity_change < 0.0 {
                    // Reducing position (partial close)
                    let pnl = (new_price - position.entry_price) * quantity_change.abs();
                    self.metrics.record_trade(TradeRecord {
                        timestamp,
                        price: new_price,
                        quantity: quantity_change.abs(),
                        signal: Signal::Sell,
                        pnl: Some(pnl),
                    });
                }
                
                position.quantity = new_quantity;
                if new_quantity == 0.0 {
                    // Position fully closed
                    self.positions.remove(symbol);
                }
            }
        }
    }

    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    pub fn unrealized_pnl(&self, symbol: &str, current_price: f64) -> f64 {
        if let Some(position) = self.positions.get(symbol) {
            (current_price - position.entry_price) * position.quantity
        } else {
            0.0
        }
    }

    pub fn total_unrealized_pnl(&self, current_prices: &HashMap<String, f64>) -> f64 {
        self.positions
            .iter()
            .map(|(symbol, position)| {
                if let Some(&price) = current_prices.get(symbol) {
                    (price - position.entry_price) * position.quantity
                } else {
                    0.0
                }
            })
            .sum()
    }

    pub fn realized_pnl(&self) -> f64 {
        self.metrics.total_pnl()
    }

    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    pub fn get_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }
    
    pub fn get_current_position_symbol(&self) -> String {
        // Return the first symbol if we have any positions, otherwise empty string
        self.positions.keys().next().cloned().unwrap_or_default()
    }
    
    pub fn get_current_position(&self) -> Option<&Position> {
        // Return the first position if we have any, otherwise None
        self.positions.values().next()
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    trades: Vec<TradeRecord>,
    total_pnl: f64,
    win_count: usize,
    loss_count: usize,
    total_trades: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            trades: Vec::new(),
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
            total_trades: 0,
        }
    }

    pub fn record_trade(&mut self, trade: TradeRecord) {
        if let Some(pnl) = trade.pnl {
            self.total_pnl += pnl;
            if pnl > 0.0 {
                self.win_count += 1;
            } else if pnl < 0.0 {
                self.loss_count += 1;
            }
        }
        self.total_trades += 1;
        self.trades.push(trade);
    }

    pub fn total_pnl(&self) -> f64 {
        self.total_pnl
    }

    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.win_count as f64 / self.total_trades as f64
        }
    }

    pub fn total_trades(&self) -> usize {
        self.total_trades
    }

    pub fn average_trade_pnl(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.total_pnl / self.total_trades as f64
        }
    }

    pub fn get_trades(&self) -> &[TradeRecord] {
        &self.trades
    }
}


