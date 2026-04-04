use crate::execution::{ExecutionReport, ExecutionStatus, OrderIntent, Side};
use crate::strategy::StrategyLogger;
use tracing::{debug, error, info, warn};

/// Format a price/quantity value: up to 8 decimal places, trailing zeros stripped.
/// Handles IEEE 754 artifacts like 0.0013900000000000002 → "0.00139".
pub fn fmt_price(v: f64) -> String {
    let s = format!("{:.8}", v);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

/// Format an Option<f64> price value.
fn fmt_price_opt(v: Option<f64>) -> String {
    match v {
        Some(val) => fmt_price(val),
        None => "none".to_string(),
    }
}

/// Format a bps value: 2 decimal places.
fn fmt_bps(v: f64) -> String {
    format!("{:.2}", v)
}

/// Format a USDT monetary amount: up to 8 decimal places, trailing zeros stripped.
pub fn fmt_usd(v: f64) -> String {
    fmt_price(v) // same logic: 8 dp, strip trailing zeros
}

/// Format an Option<f64> USDT value.
fn fmt_usd_opt(v: Option<f64>) -> String {
    match v {
        Some(val) => fmt_usd(val),
        None => "none".to_string(),
    }
}

/// Format a ratio/percentage: 2 decimal places.
fn fmt_pct(v: f64) -> String {
    format!("{:.2}", v)
}

/// Format latency in seconds: 4 decimal places.
fn fmt_latency(v: f64) -> String {
    format!("{:.4}", v)
}

pub struct TradeLogger {
    pub exchange_name: String,
}

impl TradeLogger {
    pub fn new(exchange: String) -> Self {
        Self {
            exchange_name: exchange,
        }
    }

    pub fn log_trade_received(&self, symbol: &str, price: f64, quantity: f64, timestamp: f64) {
        debug!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = "trade_received",
            price,
            quantity,
            timestamp
        );
    }

    pub fn log_decision(
        &self,
        symbol: &str,
        intent: &OrderIntent,
        confidence: f64,
        reference_price: f64,
    ) {
        debug!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = "decision_generated",
            intent = ?intent,
            confidence,
            reference_price
        );
    }

    pub fn log_execution(
        &self,
        symbol: &str,
        report: &ExecutionReport,
        pnl: Option<f64>,
        realized_pnl: Option<f64>,
        trade_summary: Option<(usize, usize)>,
    ) {
        let trades_info = if let Some((total_ticks, total_trades)) = trade_summary {
            format!("{}/{}", total_ticks, total_trades)
        } else {
            "unknown".to_string()
        };

        let action = match (report.status, report.side) {
            (ExecutionStatus::Pending, Some(Side::Buy)) => "buy_pending",
            (ExecutionStatus::Pending, Some(Side::Sell)) => "sell_pending",
            (ExecutionStatus::Pending, _) => "order_pending",
            (ExecutionStatus::Filled, Some(Side::Buy)) => "buy_executed",
            (ExecutionStatus::Filled, Some(Side::Sell)) => "sell_executed",
            (ExecutionStatus::PartiallyFilled, Some(Side::Buy)) => "buy_partially_filled",
            (ExecutionStatus::PartiallyFilled, Some(Side::Sell)) => "sell_partially_filled",
            (ExecutionStatus::Rejected, _) => "order_rejected",
            (ExecutionStatus::Cancelled, _) => "order_cancelled",
            (ExecutionStatus::Ignored, _) => "decision_ignored",
            _ => "execution_report",
        };

        if report.status == ExecutionStatus::Ignored {
            debug!(
                exchange = %self.exchange_name,
                trades = %trades_info,
                symbol = %symbol,
                action,
                realized_pnl
            );
            return;
        }

        // Suppress repeated "limit_order_resting" spam — demote to debug.
        // The initial placement is already logged as INFO by the trader.
        if report.status == ExecutionStatus::Pending && report.reason == Some("limit_order_resting")
        {
            let requested_quantity = fmt_price(report.requested_quantity);
            let execution_price = fmt_price_opt(report.execution_price);
            let expected_edge_bps = fmt_bps(report.expected_edge_bps);
            debug!(
                exchange = %self.exchange_name,
                trades = %trades_info,
                symbol = %symbol,
                action,
                side = ?report.side,
                requested_quantity = %requested_quantity,
                execution_price = %execution_price,
                expected_edge_bps = %expected_edge_bps,
                reason = report.reason,
            );
            return;
        }

        let requested_quantity = fmt_price(report.requested_quantity);
        let executed_quantity = fmt_price(report.executed_quantity);
        let execution_price = fmt_price_opt(report.execution_price);
        let fee_paid = fmt_usd(report.fee_paid);
        let latency_seconds = fmt_latency(report.latency_seconds);
        let synthetic_half_spread_bps = fmt_bps(report.synthetic_half_spread_bps);
        let slippage_bps = fmt_bps(report.slippage_bps);
        let latency_impact_bps = fmt_bps(report.latency_impact_bps);
        let market_impact_bps = fmt_bps(report.market_impact_bps);
        let expected_edge_bps = fmt_bps(report.expected_edge_bps);
        let pnl_str = fmt_usd_opt(pnl);
        let realized_pnl_str = fmt_usd_opt(realized_pnl);

        info!(
            exchange = %self.exchange_name,
            trades = %trades_info,
            symbol = %symbol,
            action,
            side = ?report.side,
            order_type = ?report.order_type,
            requested_quantity = %requested_quantity,
            executed_quantity = %executed_quantity,
            execution_price = %execution_price,
            fee_paid = %fee_paid,
            latency_seconds = %latency_seconds,
            synthetic_half_spread_bps = %synthetic_half_spread_bps,
            slippage_bps = %slippage_bps,
            latency_impact_bps = %latency_impact_bps,
            market_impact_bps = %market_impact_bps,
            expected_edge_bps = %expected_edge_bps,
            reason = report.reason,
            pnl = %pnl_str,
            realized_pnl = %realized_pnl_str
        );
    }

    pub fn log_market_data_source(&self, symbol: &str, source: &str, status: &str) {
        info!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = "market_data_source",
            source = %source,
            status = %status
        );
    }

    pub fn log_market_state_snapshot(
        &self,
        symbol: &str,
        last_price: Option<f64>,
        mid_price: Option<f64>,
        spread_bps: Option<f64>,
        trade_flow_imbalance: f64,
        order_book_imbalance: Option<f64>,
    ) {
        debug!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = "market_state_snapshot",
            last_price,
            mid_price,
            spread_bps,
            trade_flow_imbalance,
            order_book_imbalance
        );
    }

    pub fn log_replay_event_mix(
        &self,
        symbol: &str,
        trade_events: usize,
        book_ticker_events: usize,
        depth_events: usize,
        trade_without_quote_events: usize,
        stale_quote_events: usize,
        stale_depth_events: usize,
    ) {
        info!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            trade_events,
            book_ticker_events,
            depth_events,
            trade_without_quote_events,
            stale_quote_events,
            stale_depth_events,
            action = "replay_event_mix"
        );
    }

    pub fn log_error(&self, symbol: &str, action: &str, error: &str) {
        error!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = %action,
            error = %error
        );
    }

    pub fn log_warning(&self, symbol: &str, action: &str, warning: &str) {
        warn!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = %action,
            warning = %warning
        );
    }

    pub fn log_order_error(&self, symbol: &str, action: &str, status: &str, error: &str) {
        error!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            action = %action,
            status = %status,
            error = %error
        );
    }

    pub fn log_session_summary(
        &self,
        symbol: &str,
        total_ticks: usize,
        entries: usize,
        closed_trades: usize,
        realized_pnl: f64,
        fees_paid: f64,
        ending_cash: f64,
        ending_equity: f64,
        win_rate: f64,
        profit_factor: f64,
        avg_pnl_per_trade: f64,
        max_drawdown: f64,
        fill_ratio: f64,
        rejection_rate: f64,
        avg_latency_seconds: f64,
        avg_synthetic_half_spread_bps: f64,
        avg_slippage_bps: f64,
        avg_latency_impact_bps: f64,
        avg_market_impact_bps: f64,
        avg_expected_edge_bps: f64,
    ) {
        let realized_pnl = fmt_usd(realized_pnl);
        let fees_paid = fmt_usd(fees_paid);
        let ending_cash = fmt_usd(ending_cash);
        let ending_equity = fmt_usd(ending_equity);
        let win_rate = fmt_pct(win_rate);
        let profit_factor = fmt_pct(profit_factor);
        let avg_pnl_per_trade = fmt_usd(avg_pnl_per_trade);
        let max_drawdown = fmt_pct(max_drawdown);
        let fill_ratio = fmt_pct(fill_ratio);
        let rejection_rate = fmt_pct(rejection_rate);
        let avg_latency_seconds = fmt_latency(avg_latency_seconds);
        let avg_synthetic_half_spread_bps = fmt_bps(avg_synthetic_half_spread_bps);
        let avg_slippage_bps = fmt_bps(avg_slippage_bps);
        let avg_latency_impact_bps = fmt_bps(avg_latency_impact_bps);
        let avg_market_impact_bps = fmt_bps(avg_market_impact_bps);
        let avg_expected_edge_bps = fmt_bps(avg_expected_edge_bps);

        info!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            total_ticks,
            entries,
            closed_trades,
            realized_pnl = %realized_pnl,
            fees_paid = %fees_paid,
            ending_cash = %ending_cash,
            ending_equity = %ending_equity,
            win_rate = %win_rate,
            profit_factor = %profit_factor,
            avg_pnl_per_trade = %avg_pnl_per_trade,
            max_drawdown = %max_drawdown,
            fill_ratio = %fill_ratio,
            rejection_rate = %rejection_rate,
            avg_latency_seconds = %avg_latency_seconds,
            avg_synthetic_half_spread_bps = %avg_synthetic_half_spread_bps,
            avg_slippage_bps = %avg_slippage_bps,
            avg_latency_impact_bps = %avg_latency_impact_bps,
            avg_market_impact_bps = %avg_market_impact_bps,
            avg_expected_edge_bps = %avg_expected_edge_bps,
            action = "session_summary"
        );
    }
}

impl Default for TradeLogger {
    fn default() -> Self {
        Self {
            exchange_name: "unknown".to_string(),
        }
    }
}

pub struct StrategyLoggerAdapter<'a> {
    trade_logger: &'a TradeLogger,
}

impl<'a> StrategyLoggerAdapter<'a> {
    pub fn new(trade_logger: &'a TradeLogger) -> Self {
        Self { trade_logger }
    }
}

impl<'a> StrategyLogger for StrategyLoggerAdapter<'a> {
    fn log_decision(
        &self,
        symbol: &str,
        intent: &OrderIntent,
        confidence: f64,
        reference_price: f64,
    ) {
        self.trade_logger
            .log_decision(symbol, intent, confidence, reference_price);
    }

    fn log_execution(
        &self,
        symbol: &str,
        report: &ExecutionReport,
        pnl: Option<f64>,
        realized_pnl: Option<f64>,
        trade_summary: Option<(usize, usize)>,
    ) {
        self.trade_logger
            .log_execution(symbol, report, pnl, realized_pnl, trade_summary);
    }
}
