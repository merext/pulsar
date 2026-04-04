use crate::execution::{DecisionMetric, ExecutionReport, OrderIntent};
use crate::market::{MarketEvent, MarketState};
use crate::models::Position;
use std::collections::BTreeMap;
use std::path::Path;

pub trait StrategyLogger: Send + Sync {
    fn log_decision(&self, symbol: &str, intent: &OrderIntent, confidence: f64, reference_price: f64);
    fn log_execution(
        &self,
        symbol: &str,
        report: &ExecutionReport,
        pnl: Option<f64>,
        realized_pnl: Option<f64>,
        trade_summary: Option<(usize, usize)>,
    );
}

#[derive(Debug, Clone)]
pub struct StrategyContext {
    pub symbol: String,
    pub current_position: Position,
    pub available_cash: f64,
    pub max_position_notional: f64,
    /// Initial capital at the start of the session. Used by dynamic sizing
    /// to guard against budget exhaustion.
    pub initial_capital: f64,
    /// Minimum price increment for the exchange pair.
    pub tick_size: f64,
}

impl StrategyContext {
    pub fn capped_entry_quantity(
        &self,
        reference_price: f64,
        cash_fraction: f64,
        notional_cap: Option<f64>,
    ) -> Option<f64> {
        if reference_price <= f64::EPSILON {
            return None;
        }

        let mut target_notional = self.max_position_notional.min(self.available_cash * cash_fraction);
        if let Some(notional_cap) = notional_cap {
            target_notional = target_notional.min(notional_cap);
        }

        if target_notional <= 0.0 {
            return None;
        }

        let quantity = target_notional / reference_price;
        (quantity > 0.0).then_some(quantity)
    }
}

#[derive(Debug, Clone)]
pub struct StrategyDecision {
    pub confidence: f64,
    pub intent: OrderIntent,
    pub metrics: Vec<DecisionMetric>,
}

impl StrategyDecision {
    pub fn no_action() -> Self {
        Self {
            confidence: 0.0,
            intent: OrderIntent::NoAction,
            metrics: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StrategyDiagnostics {
    pub counters: BTreeMap<String, usize>,
    pub gauges: BTreeMap<String, f64>,
}

#[allow(async_fn_in_trait)]
#[async_trait::async_trait]
pub trait Strategy: Send + Sync {
    fn logger(&self) -> &dyn StrategyLogger;
    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    fn get_info(&self) -> String;
    fn diagnostics(&self) -> StrategyDiagnostics {
        StrategyDiagnostics::default()
    }
    fn market_state_window_millis(&self) -> u64 {
        1_000
    }
    async fn on_event(&mut self, event: &MarketEvent, market_state: &MarketState);
    fn decide(&mut self, market_state: &MarketState, context: &StrategyContext) -> StrategyDecision;
}

pub struct NoOpStrategyLogger;

impl StrategyLogger for NoOpStrategyLogger {
    fn log_decision(&self, _symbol: &str, _intent: &OrderIntent, _confidence: f64, _reference_price: f64) {}

    fn log_execution(
        &self,
        _symbol: &str,
        _report: &ExecutionReport,
        _pnl: Option<f64>,
        _realized_pnl: Option<f64>,
        _trade_summary: Option<(usize, usize)>,
    ) {
    }
}
