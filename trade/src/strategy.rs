use crate::execution::{ExecutionReport, OrderIntent};
use crate::market::{MarketEvent, MarketState};
use crate::models::Position;
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
}

#[derive(Debug, Clone)]
pub struct StrategyDecision {
    pub confidence: f64,
    pub intent: OrderIntent,
}

impl StrategyDecision {
    pub fn no_action() -> Self {
        Self {
            confidence: 0.0,
            intent: OrderIntent::NoAction,
        }
    }
}

#[allow(async_fn_in_trait)]
#[async_trait::async_trait]
pub trait Strategy: Send + Sync {
    fn logger(&self) -> &dyn StrategyLogger;
    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    fn get_info(&self) -> String;
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
