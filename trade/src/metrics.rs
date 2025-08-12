use crate::signal::Signal;
use crate::trader::OrderType;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_fees: f64,
    pub total_rebates: f64,
    pub total_slippage: f64,
    pub average_fill_rate: f64,
    pub consecutive_losses: usize,
    pub max_drawdown: f64,
    pub peak_equity: f64,
    pub current_equity: f64,
    pub trade_history: VecDeque<TradeRecord>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are used in backtest statistics calculation
pub struct TradeRecord {
    pub timestamp: f64,
    pub signal: Signal,
    pub price: f64,
    pub quantity: f64,
    pub fees: f64,
    pub slippage: f64,
    pub rebates: f64,
    pub pnl: f64,
    pub order_type: OrderType,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMetrics {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_fees: 0.0,
            total_rebates: 0.0,
            total_slippage: 0.0,
            average_fill_rate: 0.0,
            consecutive_losses: 0,
            max_drawdown: 0.0,
            peak_equity: 0.0,
            current_equity: 0.0,
            trade_history: VecDeque::new(),
        }
    }

    pub fn update_equity(&mut self, new_equity: f64) {
        self.current_equity = new_equity;
        if new_equity > self.peak_equity {
            self.peak_equity = new_equity;
        }

        // Prevent division by zero when peak_equity is 0
        if self.peak_equity > 0.0 {
            let drawdown = (self.peak_equity - new_equity) / self.peak_equity;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }
    }

    pub fn record_trade(&mut self, record: TradeRecord) {
        self.total_trades += 1;
        self.total_fees += record.fees;
        self.total_slippage += record.slippage;
        self.total_rebates += record.rebates;

        if record.pnl > 0.0 {
            self.winning_trades += 1;
            self.consecutive_losses = 0;
        } else {
            self.losing_trades += 1;
            self.consecutive_losses += 1;
        }

        self.trade_history.push_back(record);
        if self.trade_history.len() > 1000 {
            self.trade_history.pop_front();
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.winning_trades as f64 / self.total_trades as f64
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_trade_pnl(&self) -> f64 {
        if self.total_trades == 0 || self.trade_history.is_empty() {
            0.0
        } else {
            self.trade_history.iter().map(|t| t.pnl).sum::<f64>() / self.total_trades as f64
        }
    }

    #[must_use]
    pub fn net_pnl_after_costs(&self) -> f64 {
        if self.trade_history.is_empty() {
            0.0
        } else {
            self.trade_history.iter().map(|t| t.pnl).sum::<f64>()
        }
    }

    #[must_use]
    pub fn gross_pnl(&self) -> f64 {
        self.net_pnl_after_costs() + self.total_costs()
    }

    #[must_use]
    pub fn total_costs(&self) -> f64 {
        self.total_fees + self.total_slippage - self.total_rebates
    }

    #[must_use]
    pub fn profit_factor(&self) -> f64 {
        if self.total_costs() == 0.0 {
            if self.net_pnl_after_costs() > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            self.gross_pnl() / self.total_costs()
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_win(&self) -> f64 {
        if self.winning_trades == 0 {
            0.0
        } else {
            let winning_pnl: f64 = self.trade_history.iter()
                .filter(|t| t.pnl > 0.0)
                .map(|t| t.pnl)
                .sum();
            winning_pnl / self.winning_trades as f64
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_loss(&self) -> f64 {
        if self.losing_trades == 0 {
            0.0
        } else {
            let losing_pnl: f64 = self.trade_history.iter()
                .filter(|t| t.pnl < 0.0)
                .map(|t| t.pnl.abs())
                .sum();
            losing_pnl / self.losing_trades as f64
        }
    }

    #[must_use]
    pub fn sharpe_ratio(&self) -> f64 {
        if self.trade_history.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.trade_history.iter()
            .map(|t| t.pnl)
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;

        if variance == 0.0 {
            0.0
        } else {
            mean_return / variance.sqrt()
        }
    }
}
