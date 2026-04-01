use crate::execution::{ExecutionReport, ExecutionStatus};
use crate::models::{Position, SimulationAccount};
use crate::signal::Signal;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub trade_id: usize,
    pub symbol: String,
    pub timestamp: f64,
    pub price: f64,
    pub quantity: f64,
    pub signal: Signal,
    pub pnl: Option<f64>,
    pub gross_pnl: Option<f64>,
    pub fee_paid: f64,
    pub expected_edge_bps: f64,
    pub rationale: Option<&'static str>,
    pub decision_confidence: f64,
    pub requested_quantity: f64,
    pub executed_quantity: f64,
    pub synthetic_half_spread_bps: f64,
    pub slippage_bps: f64,
    pub latency_impact_bps: f64,
    pub market_impact_bps: f64,
    pub hold_time_millis: Option<u64>,
    pub exit_reason: Option<&'static str>,
    pub entry_price: Option<f64>,
}

#[derive(Debug, Clone, Default)]
struct PendingEntryAttribution {
    rationale: Option<&'static str>,
    decision_confidence: f64,
    expected_edge_bps: f64,
    requested_quantity: f64,
    executed_quantity: f64,
    fee_paid: f64,
    synthetic_half_spread_bps: f64,
    slippage_bps: f64,
    latency_impact_bps: f64,
    market_impact_bps: f64,
}

#[derive(Debug, Clone, Default)]
struct PendingExitAttribution {
    rationale: Option<&'static str>,
    decision_confidence: f64,
    expected_edge_bps: f64,
    requested_quantity: f64,
    executed_quantity: f64,
    fee_paid: f64,
    synthetic_half_spread_bps: f64,
    slippage_bps: f64,
    latency_impact_bps: f64,
    market_impact_bps: f64,
}

#[derive(Debug, Clone)]
pub struct TradeManager {
    positions: HashMap<String, Position>,
    metrics: PerformanceMetrics,
    pending_entry_attribution: HashMap<String, PendingEntryAttribution>,
    pending_exit_attribution: HashMap<String, PendingExitAttribution>,
    total_ticks: usize, // Total number of market data ticks received
    trading_fee: f64,   // Trading fee as decimal (e.g., 0.001 for 0.1%)
    account: SimulationAccount,
}

impl TradeManager {
    pub fn new(trading_fee: f64, initial_cash: f64) -> Self {
        let mut manager = Self {
            positions: HashMap::new(),
            metrics: PerformanceMetrics::new(),
            pending_entry_attribution: HashMap::new(),
            pending_exit_attribution: HashMap::new(),
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
        report: Option<&ExecutionReport>,
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
        let attribution = report.map_or_else(
            || {
                self.pending_entry_attribution
                    .remove(symbol)
                    .unwrap_or_default()
            },
            |report| PendingEntryAttribution {
                rationale: report.rationale,
                decision_confidence: report.decision_confidence,
                expected_edge_bps: report.expected_edge_bps,
                requested_quantity: report.requested_quantity,
                executed_quantity: report.executed_quantity,
                fee_paid: report.fee_paid,
                synthetic_half_spread_bps: report.synthetic_half_spread_bps,
                slippage_bps: report.slippage_bps,
                latency_impact_bps: report.latency_impact_bps,
                market_impact_bps: report.market_impact_bps,
            },
        );
        let trade_id = self.metrics.next_trade_id();
        self.metrics.record_trade(TradeRecord {
            trade_id,
            symbol: symbol.to_string(),
            timestamp,
            price,
            quantity,
            signal: Signal::Buy,
            pnl: None, // No PnL for opening position
            gross_pnl: None,
            fee_paid: attribution.fee_paid.max(fee_paid),
            expected_edge_bps: attribution.expected_edge_bps,
            rationale: attribution.rationale,
            decision_confidence: attribution.decision_confidence,
            requested_quantity: attribution.requested_quantity.max(quantity),
            executed_quantity: attribution.executed_quantity.max(quantity),
            synthetic_half_spread_bps: attribution.synthetic_half_spread_bps,
            slippage_bps: attribution.slippage_bps,
            latency_impact_bps: attribution.latency_impact_bps,
            market_impact_bps: attribution.market_impact_bps,
            hold_time_millis: None,
            exit_reason: None,
            entry_price: Some(price),
        });

        Ok(())
    }

    pub fn close_position(&mut self, symbol: &str, price: f64, timestamp: f64) -> f64 {
        self.close_position_with_report(symbol, price, timestamp, None)
    }

    pub fn close_position_with_report(
        &mut self,
        symbol: &str,
        price: f64,
        timestamp: f64,
        report: Option<&ExecutionReport>,
    ) -> f64 {
        if let Some(position) = self.positions.remove(symbol) {
            // Calculate gross PnL
            let gross_pnl = (price - position.entry_price) * position.quantity;

            // Calculate exit fee only — entry fee was already deducted from cash
            // at position open time in record_buy(notional, fee)
            let exit_fee = price * position.quantity * self.trading_fee;

            // Net PnL after exit fee (entry fee already reflected in cash balance)
            let net_pnl = gross_pnl - exit_fee;
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

            let attribution = report.map_or_else(
                || {
                    self.pending_exit_attribution
                        .remove(symbol)
                        .unwrap_or_default()
                },
                |report| PendingExitAttribution {
                    rationale: report.rationale,
                    decision_confidence: report.decision_confidence,
                    expected_edge_bps: report.expected_edge_bps,
                    requested_quantity: report.requested_quantity,
                    executed_quantity: report.executed_quantity,
                    fee_paid: report.fee_paid,
                    synthetic_half_spread_bps: report.synthetic_half_spread_bps,
                    slippage_bps: report.slippage_bps,
                    latency_impact_bps: report.latency_impact_bps,
                    market_impact_bps: report.market_impact_bps,
                },
            );
            let trade_id = self.metrics.next_trade_id();
            self.metrics.record_trade(TradeRecord {
                trade_id,
                symbol: symbol.to_string(),
                timestamp,
                price,
                quantity: position.quantity,
                signal: Signal::Sell, // Assuming closing is selling
                pnl: Some(net_pnl),
                gross_pnl: Some(gross_pnl),
                fee_paid: attribution.fee_paid.max(exit_fee),
                expected_edge_bps: attribution.expected_edge_bps,
                rationale: attribution.rationale,
                decision_confidence: if attribution.decision_confidence > 0.0 {
                    attribution.decision_confidence
                } else {
                    1.0
                },
                requested_quantity: attribution.requested_quantity.max(position.quantity),
                executed_quantity: attribution.executed_quantity.max(position.quantity),
                synthetic_half_spread_bps: attribution.synthetic_half_spread_bps,
                slippage_bps: attribution.slippage_bps,
                latency_impact_bps: attribution.latency_impact_bps,
                market_impact_bps: attribution.market_impact_bps,
                hold_time_millis: Some(
                    ((timestamp * 1000.0).max(0.0) as u64)
                        .saturating_sub((position.entry_time * 1000.0).max(0.0) as u64),
                ),
                exit_reason: attribution.rationale,
                entry_price: Some(position.entry_price),
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

                // Exit fee only — entry fee was already deducted from cash at open time
                let exit_fee = new_price * quantity_change.abs() * self.trading_fee;

                // Net PnL after exit fee
                let net_pnl = gross_pnl - exit_fee;

                let trade_id = self.metrics.next_trade_id();
                self.metrics.record_trade(TradeRecord {
                    trade_id,
                    symbol: symbol.to_string(),
                    timestamp,
                    price: new_price,
                    quantity: quantity_change.abs(),
                    signal: Signal::Sell,
                    pnl: Some(net_pnl),
                    gross_pnl: Some(gross_pnl),
                    fee_paid: exit_fee,
                    expected_edge_bps: 0.0,
                    rationale: None,
                    decision_confidence: 1.0,
                    requested_quantity: quantity_change.abs(),
                    executed_quantity: quantity_change.abs(),
                    synthetic_half_spread_bps: 0.0,
                    slippage_bps: 0.0,
                    latency_impact_bps: 0.0,
                    market_impact_bps: 0.0,
                    hold_time_millis: None,
                    exit_reason: None,
                    entry_price: Some(position.entry_price),
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

            // Only subtract estimated exit fee — entry fee already reflected in cash
            let estimated_exit_fee = current_price * position.quantity * self.trading_fee;

            // Net unrealized PnL after estimated exit fee
            gross_pnl - estimated_exit_fee
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

                    // Only subtract estimated exit fee — entry fee already in cash
                    let estimated_exit_fee = price * position.quantity * self.trading_fee;

                    // Net unrealized PnL after estimated exit fee
                    gross_pnl - estimated_exit_fee
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
        if matches!(report.side, Some(crate::execution::Side::Buy))
            && matches!(
                report.status,
                ExecutionStatus::Filled | ExecutionStatus::PartiallyFilled
            )
            && report.executed_quantity > 0.0
        {
            let Some(symbol) = report.symbol.as_ref() else {
                self.metrics.record_execution_report(report);
                return;
            };
            self.pending_entry_attribution.insert(
                symbol.clone(),
                PendingEntryAttribution {
                    rationale: report.rationale,
                    decision_confidence: report.decision_confidence,
                    expected_edge_bps: report.expected_edge_bps,
                    requested_quantity: report.requested_quantity,
                    executed_quantity: report.executed_quantity,
                    fee_paid: report.fee_paid,
                    synthetic_half_spread_bps: report.synthetic_half_spread_bps,
                    slippage_bps: report.slippage_bps,
                    latency_impact_bps: report.latency_impact_bps,
                    market_impact_bps: report.market_impact_bps,
                },
            );
        } else if matches!(report.side, Some(crate::execution::Side::Sell))
            && matches!(
                report.status,
                ExecutionStatus::Filled | ExecutionStatus::PartiallyFilled
            )
            && report.executed_quantity > 0.0
        {
            let Some(symbol) = report.symbol.as_ref() else {
                self.metrics.record_execution_report(report);
                return;
            };
            self.pending_exit_attribution.insert(
                symbol.clone(),
                PendingExitAttribution {
                    rationale: report.rationale,
                    decision_confidence: report.decision_confidence,
                    expected_edge_bps: report.expected_edge_bps,
                    requested_quantity: report.requested_quantity,
                    executed_quantity: report.executed_quantity,
                    fee_paid: report.fee_paid,
                    synthetic_half_spread_bps: report.synthetic_half_spread_bps,
                    slippage_bps: report.slippage_bps,
                    latency_impact_bps: report.latency_impact_bps,
                    market_impact_bps: report.market_impact_bps,
                },
            );
        }
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
    next_trade_id: usize,
    total_pnl: f64,
    win_count: usize,
    loss_count: usize,
    total_trades: usize,
    entry_trades: usize,
    closed_trades: usize,
    pending_orders: usize,
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
            next_trade_id: 1,
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
            total_trades: 0,
            entry_trades: 0,
            closed_trades: 0,
            pending_orders: 0,
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

    pub fn next_trade_id(&mut self) -> usize {
        let trade_id = self.next_trade_id;
        self.next_trade_id += 1;
        trade_id
    }

    pub fn record_execution_report(&mut self, report: &ExecutionReport) {
        match report.status {
            ExecutionStatus::Pending => self.pending_orders += 1,
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

    pub fn pending_orders(&self) -> usize {
        self.pending_orders
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
            + self.pending_orders
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
