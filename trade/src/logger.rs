use crate::execution::{ExecutionReport, ExecutionStatus, OrderIntent, Side};
use crate::strategy::StrategyLogger;
use tracing::{debug, error, info, warn};

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

        info!(
            exchange = %self.exchange_name,
            trades = %trades_info,
            symbol = %symbol,
            action,
            side = ?report.side,
            order_type = ?report.order_type,
            requested_quantity = report.requested_quantity,
            executed_quantity = report.executed_quantity,
            execution_price = report.execution_price,
            fee_paid = report.fee_paid,
            latency_seconds = report.latency_seconds,
            synthetic_half_spread_bps = report.synthetic_half_spread_bps,
            slippage_bps = report.slippage_bps,
            latency_impact_bps = report.latency_impact_bps,
            market_impact_bps = report.market_impact_bps,
            expected_edge_bps = report.expected_edge_bps,
            reason = report.reason,
            pnl,
            realized_pnl
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
        info!(
            exchange = %self.exchange_name,
            symbol = %symbol,
            total_ticks,
            entries,
            closed_trades,
            realized_pnl,
            fees_paid,
            ending_cash,
            ending_equity,
            win_rate,
            profit_factor,
            avg_pnl_per_trade,
            max_drawdown,
            fill_ratio,
            rejection_rate,
            avg_latency_seconds,
            avg_synthetic_half_spread_bps,
            avg_slippage_bps,
            avg_latency_impact_bps,
            avg_market_impact_bps,
            avg_expected_edge_bps,
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
