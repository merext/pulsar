use crate::cost_gate::{
    clears_taker_cost_gate, expected_edge_after_cost_bps, DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS,
    DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use trade::execution::{DecisionMetric, OrderIntent, Side, TimeInForce};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyDiagnostics, StrategyLogger};
use trade::trader::OrderType;

#[derive(Debug, Clone, Deserialize)]
pub struct TradeFlowReclaimConfig {
    pub trade_window_millis: u64,
    pub min_trades_in_window: usize,
    pub min_pullback_from_high_bps: f64,
    pub max_pullback_from_high_bps: f64,
    pub min_reclaim_from_low_bps: f64,
    pub min_trade_flow_imbalance: f64,
    pub min_recent_trade_flow_imbalance: f64,
    pub max_spread_bps: f64,
    pub stop_loss_bps: f64,
    pub take_profit_bps: f64,
    pub hold_time_millis: u64,
    pub entry_cooldown_millis: u64,
    #[serde(default = "default_assumed_round_trip_taker_cost_bps")]
    pub assumed_round_trip_taker_cost_bps: f64,
    #[serde(default = "default_min_expected_edge_after_cost_bps")]
    pub min_expected_edge_after_cost_bps: f64,
}

fn default_assumed_round_trip_taker_cost_bps() -> f64 {
    DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS
}

fn default_min_expected_edge_after_cost_bps() -> f64 {
    DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS
}

impl Default for TradeFlowReclaimConfig {
    fn default() -> Self {
        Self {
            trade_window_millis: 2_000,
            min_trades_in_window: 14,
            min_pullback_from_high_bps: 4.0,
            max_pullback_from_high_bps: 28.0,
            min_reclaim_from_low_bps: 3.0,
            min_trade_flow_imbalance: 0.10,
            min_recent_trade_flow_imbalance: 0.18,
            max_spread_bps: 12.0,
            stop_loss_bps: 14.0,
            take_profit_bps: 18.0,
            hold_time_millis: 4_000,
            entry_cooldown_millis: 2_000,
            assumed_round_trip_taker_cost_bps: DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS,
            min_expected_edge_after_cost_bps: DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
        }
    }
}

#[derive(Default)]
struct ReclaimDiagnostics {
    total_decisions: usize,
    blocked_min_trades: usize,
    blocked_spread: usize,
    blocked_pullback_band: usize,
    blocked_reclaim: usize,
    blocked_flow: usize,
    blocked_recent_flow: usize,
    blocked_cost_gate: usize,
    entries: usize,
    exits_stop_loss: usize,
    exits_take_profit: usize,
    exits_flow_reversal: usize,
    exits_max_hold: usize,
    last_pullback_bps: f64,
    last_reclaim_bps: f64,
    last_flow_imbalance: f64,
    last_recent_flow_imbalance: f64,
    last_expected_edge_bps: f64,
    last_edge_after_cost_bps: f64,
}

pub struct TradeFlowReclaimStrategy {
    config: TradeFlowReclaimConfig,
    logger: NoOpStrategyLogger,
    last_entry_time_millis: Option<u64>,
    diagnostics: ReclaimDiagnostics,
}

impl TradeFlowReclaimStrategy {
    fn load_config<P: AsRef<Path>>(config_path: P) -> Result<TradeFlowReclaimConfig, Box<dyn std::error::Error>> {
        let path = config_path.as_ref();
        if path == Path::new("/dev/null") || !path.exists() {
            return Ok(TradeFlowReclaimConfig::default());
        }

        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str::<TradeFlowReclaimConfig>(&content)?;
        Ok(config)
    }

    fn reference_price(&self, market_state: &MarketState) -> f64 {
        market_state.mid_price().or_else(|| market_state.last_price()).unwrap_or(0.0)
    }

    fn pullback_from_high_bps(&self, market_state: &MarketState) -> f64 {
        let high = market_state.trade_window_high_price().unwrap_or(0.0);
        let now = self.reference_price(market_state);
        if high <= f64::EPSILON || now <= f64::EPSILON || now >= high {
            return 0.0;
        }
        (high - now) / high * 10_000.0
    }

    fn reclaim_from_low_bps(&self, market_state: &MarketState) -> f64 {
        let low = market_state.trade_window_low_price().unwrap_or(0.0);
        let now = self.reference_price(market_state);
        if low <= f64::EPSILON || now <= f64::EPSILON || now <= low {
            return 0.0;
        }
        (now - low) / low * 10_000.0
    }

    fn recent_trade_flow_imbalance(&self, market_state: &MarketState) -> f64 {
        market_state.recent_trade_flow_imbalance((self.config.min_trades_in_window / 2).max(6))
    }

    fn expected_edge_bps(&self, market_state: &MarketState) -> f64 {
        self.reclaim_from_low_bps(market_state)
    }

    fn should_enter_long(&mut self, market_state: &MarketState) -> bool {
        self.diagnostics.total_decisions += 1;
        let stats = market_state.trade_window_stats();
        if stats.trade_count < self.config.min_trades_in_window {
            self.diagnostics.blocked_min_trades += 1;
            return false;
        }

        if market_state.spread_bps().is_some_and(|spread| spread > self.config.max_spread_bps) {
            self.diagnostics.blocked_spread += 1;
            return false;
        }

        let pullback_bps = self.pullback_from_high_bps(market_state);
        self.diagnostics.last_pullback_bps = pullback_bps;
        if pullback_bps < self.config.min_pullback_from_high_bps
            || pullback_bps > self.config.max_pullback_from_high_bps
        {
            self.diagnostics.blocked_pullback_band += 1;
            return false;
        }

        let reclaim_bps = self.reclaim_from_low_bps(market_state);
        self.diagnostics.last_reclaim_bps = reclaim_bps;
        if reclaim_bps < self.config.min_reclaim_from_low_bps {
            self.diagnostics.blocked_reclaim += 1;
            return false;
        }

        let flow = market_state.trade_flow_imbalance();
        self.diagnostics.last_flow_imbalance = flow;
        if flow < self.config.min_trade_flow_imbalance {
            self.diagnostics.blocked_flow += 1;
            return false;
        }

        let recent_flow = self.recent_trade_flow_imbalance(market_state);
        self.diagnostics.last_recent_flow_imbalance = recent_flow;
        if recent_flow < self.config.min_recent_trade_flow_imbalance {
            self.diagnostics.blocked_recent_flow += 1;
            return false;
        }

        let expected_edge_bps = self.expected_edge_bps(market_state);
        let edge_after_cost_bps = expected_edge_after_cost_bps(
            expected_edge_bps,
            self.config.assumed_round_trip_taker_cost_bps,
        );
        self.diagnostics.last_expected_edge_bps = expected_edge_bps;
        self.diagnostics.last_edge_after_cost_bps = edge_after_cost_bps;
        if !clears_taker_cost_gate(
            expected_edge_bps,
            self.config.assumed_round_trip_taker_cost_bps,
            self.config.min_expected_edge_after_cost_bps,
        ) {
            self.diagnostics.blocked_cost_gate += 1;
            return false;
        }

        true
    }

    fn should_exit_long(&mut self, market_state: &MarketState, context: &StrategyContext) -> Option<&'static str> {
        if context.current_position.quantity <= 0.0 {
            return None;
        }

        let reference_price = market_state
            .top_of_book()
            .map(|book| book.bid.price)
            .or_else(|| market_state.last_price())?;
        let pnl_bps = (reference_price - context.current_position.entry_price)
            / context.current_position.entry_price
            * 10_000.0;

        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_stop_loss += 1;
            return Some("stop_loss");
        }
        if pnl_bps >= self.config.take_profit_bps {
            self.diagnostics.exits_take_profit += 1;
            return Some("take_profit");
        }
        if self.recent_trade_flow_imbalance(market_state) < -0.05 {
            self.diagnostics.exits_flow_reversal += 1;
            return Some("flow_reversal");
        }

        let now = market_state.last_event_time_millis()?;
        let held_millis = now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
        if held_millis >= self.config.hold_time_millis {
            self.diagnostics.exits_max_hold += 1;
            return Some("max_hold_time");
        }

        None
    }

    fn in_entry_cooldown(&self, market_state: &MarketState) -> bool {
        let Some(last_entry) = self.last_entry_time_millis else {
            return false;
        };
        let Some(now) = market_state.last_event_time_millis() else {
            return false;
        };
        now.saturating_sub(last_entry) < self.config.entry_cooldown_millis
    }
}

#[async_trait::async_trait]
impl Strategy for TradeFlowReclaimStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        &self.logger
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let config = Self::load_config(config_path)?;
        Ok(Self {
            config,
            logger: NoOpStrategyLogger,
            last_entry_time_millis: None,
            diagnostics: ReclaimDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "TradeFlowReclaimStrategy - taker reclaim after pullback with buyer flow confirmation".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert("reclaim.total_decisions".to_string(), self.diagnostics.total_decisions);
        counters.insert("reclaim.blocked_min_trades".to_string(), self.diagnostics.blocked_min_trades);
        counters.insert("reclaim.blocked_spread".to_string(), self.diagnostics.blocked_spread);
        counters.insert("reclaim.blocked_pullback_band".to_string(), self.diagnostics.blocked_pullback_band);
        counters.insert("reclaim.blocked_reclaim".to_string(), self.diagnostics.blocked_reclaim);
        counters.insert("reclaim.blocked_flow".to_string(), self.diagnostics.blocked_flow);
        counters.insert("reclaim.blocked_recent_flow".to_string(), self.diagnostics.blocked_recent_flow);
        counters.insert("reclaim.blocked_cost_gate".to_string(), self.diagnostics.blocked_cost_gate);
        counters.insert("reclaim.entries".to_string(), self.diagnostics.entries);
        counters.insert("reclaim.exits_stop_loss".to_string(), self.diagnostics.exits_stop_loss);
        counters.insert("reclaim.exits_take_profit".to_string(), self.diagnostics.exits_take_profit);
        counters.insert("reclaim.exits_flow_reversal".to_string(), self.diagnostics.exits_flow_reversal);
        counters.insert("reclaim.exits_max_hold".to_string(), self.diagnostics.exits_max_hold);

        let mut gauges = BTreeMap::new();
        gauges.insert("reclaim.last_pullback_bps".to_string(), self.diagnostics.last_pullback_bps);
        gauges.insert("reclaim.last_reclaim_bps".to_string(), self.diagnostics.last_reclaim_bps);
        gauges.insert("reclaim.last_flow_imbalance".to_string(), self.diagnostics.last_flow_imbalance);
        gauges.insert(
            "reclaim.last_recent_flow_imbalance".to_string(),
            self.diagnostics.last_recent_flow_imbalance,
        );
        gauges.insert("reclaim.last_expected_edge_bps".to_string(), self.diagnostics.last_expected_edge_bps);
        gauges.insert("reclaim.last_edge_after_cost_bps".to_string(), self.diagnostics.last_edge_after_cost_bps);

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, _event: &MarketEvent, _market_state: &MarketState) {}

    fn decide(&mut self, market_state: &MarketState, context: &StrategyContext) -> StrategyDecision {
        if let Some(rationale) = self.should_exit_long(market_state, context) {
            return StrategyDecision {
                confidence: 1.0,
                intent: OrderIntent::Place {
                    side: Side::Sell,
                    order_type: OrderType::Taker,
                    price: None,
                    quantity: context.current_position.quantity,
                    time_in_force: TimeInForce::Ioc,
                    rationale,
                    expected_edge_bps: 0.0,
                },
                metrics: vec![DecisionMetric { name: "position_quantity", value: context.current_position.quantity }],
            };
        }

        if context.current_position.quantity > 0.0 || self.in_entry_cooldown(market_state) {
            return StrategyDecision::no_action();
        }

        if !self.should_enter_long(market_state) {
            return StrategyDecision::no_action();
        }

        let expected_edge_bps = self.expected_edge_bps(market_state);
        let reference_price = self.reference_price(market_state);
        if reference_price <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        let Some(quantity) = context.capped_entry_quantity(reference_price, 0.9, None) else {
            return StrategyDecision::no_action();
        };

        self.last_entry_time_millis = market_state.last_event_time_millis();
        self.diagnostics.entries += 1;

        StrategyDecision {
            confidence: 0.78,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Taker,
                price: None,
                quantity,
                time_in_force: TimeInForce::Ioc,
                rationale: "trade_flow_reclaim_entry",
                expected_edge_bps,
            },
            metrics: vec![
                DecisionMetric { name: "pullback_bps", value: self.pullback_from_high_bps(market_state) },
                DecisionMetric { name: "reclaim_bps", value: expected_edge_bps },
                DecisionMetric { name: "flow_imbalance", value: market_state.trade_flow_imbalance() },
                DecisionMetric { name: "recent_flow_imbalance", value: self.recent_trade_flow_imbalance(market_state) },
                DecisionMetric {
                    name: "edge_after_cost_bps",
                    value: expected_edge_after_cost_bps(
                        expected_edge_bps,
                        self.config.assumed_round_trip_taker_cost_bps,
                    ),
                },
            ],
        }
    }
}
