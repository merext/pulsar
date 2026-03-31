use crate::execution::{ExecutionReport, ExecutionStatus};
use crate::models::{Position, SimulationAccount};
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
pub struct TradeManager {
    positions: HashMap<String, Position>,
    metrics: PerformanceMetrics,
    total_ticks: usize, // Total number of market data ticks received
    trading_fee: f64,   // Trading fee as decimal (e.g., 0.001 for 0.1%)
    account: SimulationAccount,
}

impl TradeManager {
    pub fn new(trading_fee: f64, initial_cash: f64) -> Self {
        let mut manager = Self {
            positions: HashMap::new(),
            metrics: PerformanceMetrics::new(),
            total_ticks: 0,
            trading_fee,
            account: SimulationAccount::new(initial_cash),
        };
        manager.metrics.update_account_snapshot(
            manager.account.cash,
            manager.account.initial_cash,
            manager.account.initial_cash,
            manager.account.max_drawdown,
        );
        manager
    }

    // Position management
    pub fn open_position(
        &mut self,
        symbol: &str,
        price: f64,
        quantity: f64,
        timestamp: f64,
    ) -> Result<(), &'static str> {
        let notional_value = price * quantity;
        let fee_paid = notional_value * self.trading_fee;
        let total_cost = notional_value + fee_paid;

        if total_cost > self.account.cash + 1e-9 {
            return Err("insufficient_cash");
        }

        let position = Position {
            symbol: symbol.to_string(),
            quantity,
            entry_price: price,
            entry_time: timestamp,
        };
        self.positions.insert(symbol.to_string(), position);
        self.account.record_buy(notional_value, fee_paid);
        self.metrics.set_fees_paid(self.account.fees_paid);
        let equity = self
            .account
            .update_drawdown(price, self.positions.get(symbol));
        self.metrics.update_account_snapshot(
            self.account.cash,
            equity,
            self.account.equity_peak,
            self.account.max_drawdown,
        );

        // Record the buy trade
        self.metrics.record_trade(TradeRecord {
            timestamp,
            price,
            quantity,
            signal: Signal::Buy,
            pnl: None, // No PnL for opening position
        });

        Ok(())
    }

    pub fn close_position(&mut self, symbol: &str, price: f64, timestamp: f64) -> f64 {
        if let Some(position) = self.positions.remove(symbol) {
            // Calculate gross PnL
            let gross_pnl = (price - position.entry_price) * position.quantity;

            // Calculate trading fees (entry + exit fees)
            let entry_fee = position.entry_price * position.quantity * self.trading_fee;
            let exit_fee = price * position.quantity * self.trading_fee;
            let total_fees = entry_fee + exit_fee;

            // Net PnL after fees
            let net_pnl = gross_pnl - total_fees;
            let notional_value = price * position.quantity;
            self.account.record_sell(notional_value, exit_fee, net_pnl);
            self.metrics.set_fees_paid(self.account.fees_paid);
            let equity = self.account.update_drawdown(price, None);
            self.metrics.update_account_snapshot(
                self.account.cash,
                equity,
                self.account.equity_peak,
                self.account.max_drawdown,
            );

            self.metrics.record_trade(TradeRecord {
                timestamp,
                price,
                quantity: position.quantity,
                signal: Signal::Sell, // Assuming closing is selling
                pnl: Some(net_pnl),
            });
            net_pnl
        } else {
            0.0
        }
    }

    pub fn update_position(
        &mut self,
        symbol: &str,
        new_quantity: f64,
        new_price: f64,
        timestamp: f64,
    ) {
        if let Some(position) = self.positions.get_mut(symbol)
            && new_quantity != position.quantity
        {
            // If quantity changed, calculate PnL for the change
            let quantity_change = new_quantity - position.quantity;
            if quantity_change < 0.0 {
                // Reducing position (partial close)
                let gross_pnl = (new_price - position.entry_price) * quantity_change.abs();

                // Calculate trading fees for the partial close
                let entry_fee = position.entry_price * quantity_change.abs() * self.trading_fee;
                let exit_fee = new_price * quantity_change.abs() * self.trading_fee;
                let total_fees = entry_fee + exit_fee;

                // Net PnL after fees
                let net_pnl = gross_pnl - total_fees;

                self.metrics.record_trade(TradeRecord {
                    timestamp,
                    price: new_price,
                    quantity: quantity_change.abs(),
                    signal: Signal::Sell,
                    pnl: Some(net_pnl),
                });
            }

            position.quantity = new_quantity;
            if new_quantity == 0.0 {
                // Position fully closed
                self.positions.remove(symbol);
            }
        }
    }

    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    pub fn unrealized_pnl(&self, symbol: &str, current_price: f64) -> f64 {
        if let Some(position) = self.positions.get(symbol) {
            // Calculate gross unrealized PnL
            let gross_pnl = (current_price - position.entry_price) * position.quantity;

            // Calculate trading fees (entry fee + estimated exit fee)
            let entry_fee = position.entry_price * position.quantity * self.trading_fee;
            let estimated_exit_fee = current_price * position.quantity * self.trading_fee;
            let total_fees = entry_fee + estimated_exit_fee;

            // Net unrealized PnL after fees
            gross_pnl - total_fees
        } else {
            0.0
        }
    }

    pub fn total_unrealized_pnl(&self, current_prices: &HashMap<String, f64>) -> f64 {
        self.positions
            .iter()
            .map(|(symbol, position)| {
                if let Some(&price) = current_prices.get(symbol) {
                    // Calculate gross unrealized PnL
                    let gross_pnl = (price - position.entry_price) * position.quantity;

                    // Calculate trading fees (entry fee + estimated exit fee)
                    let entry_fee = position.entry_price * position.quantity * self.trading_fee;
                    let estimated_exit_fee = price * position.quantity * self.trading_fee;
                    let total_fees = entry_fee + estimated_exit_fee;

                    // Net unrealized PnL after fees
                    gross_pnl - total_fees
                } else {
                    0.0
                }
            })
            .sum()
    }

    pub fn realized_pnl(&self) -> f64 {
        self.metrics.realized_pnl()
    }

    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    pub fn record_execution_report(&mut self, report: &ExecutionReport) {
        self.metrics.record_execution_report(report);
    }

    pub fn get_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    pub fn get_current_trade(&self) -> Option<&Position> {
        // Return the first position if we have any, otherwise None
        self.positions.values().next()
    }

    pub fn increment_ticks(&mut self) {
        self.total_ticks += 1;
    }

    pub fn get_total_ticks(&self) -> usize {
        self.total_ticks
    }

    pub fn get_trade_summary(&self) -> (usize, usize) {
        (self.total_ticks, self.metrics.total_trades())
    }

    pub fn available_cash(&self) -> f64 {
        self.account.cash
    }

    pub fn equity(&self, symbol: &str, mark_price: f64) -> f64 {
        self.account.equity(mark_price, self.positions.get(symbol))
    }

    pub fn fees_paid(&self) -> f64 {
        self.account.fees_paid
    }

    pub fn max_drawdown(&self) -> f64 {
        self.account.max_drawdown
    }

    pub fn current_drawdown(&self, symbol: &str, mark_price: f64) -> f64 {
        self.account
            .current_drawdown(mark_price, self.positions.get(symbol))
    }

    pub fn mark_to_market(&mut self, symbol: &str, mark_price: f64) -> f64 {
        let equity = self
            .account
            .update_drawdown(mark_price, self.positions.get(symbol));
        self.metrics.update_account_snapshot(
            self.account.cash,
            equity,
            self.account.equity_peak,
            self.account.max_drawdown,
        );
        equity
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    trades: Vec<TradeRecord>,
    total_pnl: f64,
    win_count: usize,
    loss_count: usize,
    total_trades: usize,
    entry_trades: usize,
    closed_trades: usize,
    filled_orders: usize,
    partially_filled_orders: usize,
    rejected_orders: usize,
    cancelled_orders: usize,
    ignored_decisions: usize,
    total_requested_quantity: f64,
    total_executed_quantity: f64,
    total_latency_seconds: f64,
    total_synthetic_half_spread_bps: f64,
    total_slippage_bps: f64,
    total_latency_impact_bps: f64,
    total_market_impact_bps: f64,
    total_expected_edge_bps: f64,
    fees_paid: f64,
    current_cash: f64,
    last_equity: f64,
    equity_peak: f64,
    max_drawdown: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            trades: Vec::new(),
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
            total_trades: 0,
            entry_trades: 0,
            closed_trades: 0,
            filled_orders: 0,
            partially_filled_orders: 0,
            rejected_orders: 0,
            cancelled_orders: 0,
            ignored_decisions: 0,
            total_requested_quantity: 0.0,
            total_executed_quantity: 0.0,
            total_latency_seconds: 0.0,
            total_synthetic_half_spread_bps: 0.0,
            total_slippage_bps: 0.0,
            total_latency_impact_bps: 0.0,
            total_market_impact_bps: 0.0,
            total_expected_edge_bps: 0.0,
            fees_paid: 0.0,
            current_cash: 0.0,
            last_equity: 0.0,
            equity_peak: 0.0,
            max_drawdown: 0.0,
        }
    }

    pub fn record_trade(&mut self, trade: TradeRecord) {
        match trade.signal {
            Signal::Buy => self.entry_trades += 1,
            Signal::Sell => self.closed_trades += 1,
            Signal::Hold => {}
        }

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

    pub fn record_execution_report(&mut self, report: &ExecutionReport) {
        match report.status {
            ExecutionStatus::Filled => self.filled_orders += 1,
            ExecutionStatus::PartiallyFilled => self.partially_filled_orders += 1,
            ExecutionStatus::Rejected => self.rejected_orders += 1,
            ExecutionStatus::Cancelled => self.cancelled_orders += 1,
            ExecutionStatus::Ignored => self.ignored_decisions += 1,
        }

        self.total_requested_quantity += report.requested_quantity;
        self.total_executed_quantity += report.executed_quantity;

        if matches!(
            report.status,
            ExecutionStatus::Filled | ExecutionStatus::PartiallyFilled
        ) {
            self.total_latency_seconds += report.latency_seconds;
            self.total_synthetic_half_spread_bps += report.synthetic_half_spread_bps;
            self.total_slippage_bps += report.slippage_bps;
            self.total_latency_impact_bps += report.latency_impact_bps;
            self.total_market_impact_bps += report.market_impact_bps;
            self.total_expected_edge_bps += report.expected_edge_bps;
        }
    }

    pub fn realized_pnl(&self) -> f64 {
        self.total_pnl
    }

    pub fn win_rate(&self) -> f64 {
        if self.closed_trades == 0 {
            0.0
        } else {
            self.win_count as f64 / self.closed_trades as f64
        }
    }

    pub fn total_trades(&self) -> usize {
        self.total_trades
    }

    pub fn entry_trades(&self) -> usize {
        self.entry_trades
    }

    pub fn closed_trades(&self) -> usize {
        self.closed_trades
    }

    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self
            .trades
            .iter()
            .filter_map(|trade| trade.pnl.filter(|pnl| *pnl > 0.0))
            .sum();
        let gross_loss: f64 = self
            .trades
            .iter()
            .filter_map(|trade| trade.pnl.filter(|pnl| *pnl < 0.0))
            .map(f64::abs)
            .sum();

        if gross_loss == 0.0 {
            if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            }
        } else {
            gross_profit / gross_loss
        }
    }

    pub fn avg_pnl_per_closed_trade(&self) -> f64 {
        if self.closed_trades == 0 {
            0.0
        } else {
            self.total_pnl / self.closed_trades as f64
        }
    }

    pub fn get_trades(&self) -> &[TradeRecord] {
        &self.trades
    }

    pub fn set_fees_paid(&mut self, fees_paid: f64) {
        self.fees_paid = fees_paid;
    }

    pub fn update_account_snapshot(
        &mut self,
        current_cash: f64,
        last_equity: f64,
        equity_peak: f64,
        max_drawdown: f64,
    ) {
        self.current_cash = current_cash;
        self.last_equity = last_equity;
        self.equity_peak = equity_peak;
        self.max_drawdown = max_drawdown;
    }

    pub fn fees_paid(&self) -> f64 {
        self.fees_paid
    }

    pub fn current_cash(&self) -> f64 {
        self.current_cash
    }

    pub fn last_equity(&self) -> f64 {
        self.last_equity
    }

    pub fn equity_peak(&self) -> f64 {
        self.equity_peak
    }

    pub fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }

    pub fn filled_orders(&self) -> usize {
        self.filled_orders
    }

    pub fn partially_filled_orders(&self) -> usize {
        self.partially_filled_orders
    }

    pub fn rejected_orders(&self) -> usize {
        self.rejected_orders
    }

    pub fn cancelled_orders(&self) -> usize {
        self.cancelled_orders
    }

    pub fn ignored_decisions(&self) -> usize {
        self.ignored_decisions
    }

    pub fn fill_ratio(&self) -> f64 {
        if self.total_requested_quantity <= f64::EPSILON {
            0.0
        } else {
            (self.total_executed_quantity / self.total_requested_quantity).clamp(0.0, 1.0)
        }
    }

    pub fn rejection_rate(&self) -> f64 {
        let total_reports = self.filled_orders
            + self.partially_filled_orders
            + self.rejected_orders
            + self.cancelled_orders
            + self.ignored_decisions;
        if total_reports == 0 {
            0.0
        } else {
            self.rejected_orders as f64 / total_reports as f64
        }
    }

    fn executed_report_count(&self) -> usize {
        self.filled_orders + self.partially_filled_orders
    }

    pub fn avg_latency_seconds(&self) -> f64 {
        let count = self.executed_report_count();
        if count == 0 {
            0.0
        } else {
            self.total_latency_seconds / count as f64
        }
    }

    pub fn avg_synthetic_half_spread_bps(&self) -> f64 {
        let count = self.executed_report_count();
        if count == 0 {
            0.0
        } else {
            self.total_synthetic_half_spread_bps / count as f64
        }
    }

    pub fn avg_slippage_bps(&self) -> f64 {
        let count = self.executed_report_count();
        if count == 0 {
            0.0
        } else {
            self.total_slippage_bps / count as f64
        }
    }

    pub fn avg_latency_impact_bps(&self) -> f64 {
        let count = self.executed_report_count();
        if count == 0 {
            0.0
        } else {
            self.total_latency_impact_bps / count as f64
        }
    }

    pub fn avg_market_impact_bps(&self) -> f64 {
        let count = self.executed_report_count();
        if count == 0 {
            0.0
        } else {
            self.total_market_impact_bps / count as f64
        }
    }

    pub fn avg_expected_edge_bps(&self) -> f64 {
        let count = self.executed_report_count();
        if count == 0 {
            0.0
        } else {
            self.total_expected_edge_bps / count as f64
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}
