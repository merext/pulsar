use crate::cost_gate::{
    DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS, DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
    clears_taker_cost_gate, expected_edge_after_cost_bps,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use trade::execution::{DecisionMetric, OrderIntent, Side, TimeInForce};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{
    NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyDiagnostics,
    StrategyLogger,
};
use trade::trader::OrderType;

#[derive(Default)]
struct MomentumDiagnostics {
    total_decisions: usize,
    blocked_min_trades: usize,
    blocked_spread: usize,
    blocked_flow: usize,
    blocked_recent_flow: usize,
    blocked_drift_band: usize,
    blocked_vwap_stretch: usize,
    blocked_order_book: usize,
    blocked_burst: usize,
    blocked_cost_gate: usize,
    entries: usize,
    exits_stop_loss: usize,
    exits_take_profit: usize,
    exits_flow_reversal: usize,
    exits_max_hold: usize,
    last_flow_imbalance: f64,
    last_recent_flow_imbalance: f64,
    last_drift_bps: f64,
    last_drift_above_vwap_bps: f64,
    last_burst_per_second: f64,
    last_expected_edge_bps: f64,
    last_edge_after_cost_bps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TradeFlowMomentumConfig {
    pub trade_window_millis: u64,
    pub min_trades_in_window: usize,
    pub min_trade_flow_imbalance: f64,
    pub min_recent_trade_flow_imbalance: f64,
    pub min_price_drift_bps: f64,
    pub max_price_drift_bps: f64,
    pub max_drift_above_vwap_bps: f64,
    pub max_spread_bps: f64,
    pub min_burst_per_second: f64,
    pub min_order_book_imbalance: f64,
    pub position_size_confidence_floor: f64,
    pub entry_cooldown_millis: u64,
    pub hold_time_millis: u64,
    pub exit_on_flow_reversal: f64,
    pub stop_loss_bps: f64,
    pub take_profit_bps: f64,
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

impl Default for TradeFlowMomentumConfig {
    fn default() -> Self {
        Self {
            trade_window_millis: 1_500,
            min_trades_in_window: 12,
            min_trade_flow_imbalance: 0.18,
            min_recent_trade_flow_imbalance: 0.24,
            min_price_drift_bps: 6.0,
            max_price_drift_bps: 35.0,
            max_drift_above_vwap_bps: 8.0,
            max_spread_bps: 12.0,
            min_burst_per_second: 10.0,
            min_order_book_imbalance: 0.03,
            position_size_confidence_floor: 0.72,
            entry_cooldown_millis: 2_500,
            hold_time_millis: 4_000,
            exit_on_flow_reversal: -0.05,
            stop_loss_bps: 18.0,
            take_profit_bps: 24.0,
            assumed_round_trip_taker_cost_bps: DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS,
            min_expected_edge_after_cost_bps: DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
        }
    }
}

pub struct TradeFlowMomentumStrategy {
    config: TradeFlowMomentumConfig,
    logger: NoOpStrategyLogger,
    last_entry_time_millis: Option<u64>,
    diagnostics: MomentumDiagnostics,
}

impl TradeFlowMomentumStrategy {
    fn load_config<P: AsRef<Path>>(
        config_path: P,
    ) -> Result<TradeFlowMomentumConfig, Box<dyn std::error::Error>> {
        let path = config_path.as_ref();
        if path == Path::new("/dev/null") || !path.exists() {
            return Ok(TradeFlowMomentumConfig::default());
        }

        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str::<TradeFlowMomentumConfig>(&content)?;
        Ok(config)
    }

    fn trade_burst_per_second(&self, market_state: &MarketState) -> f64 {
        let stats = market_state.trade_window_stats();
        if self.config.trade_window_millis == 0 {
            return 0.0;
        }

        stats.trade_count as f64 / (self.config.trade_window_millis as f64 / 1000.0)
    }

    fn price_drift_bps(&self, market_state: &MarketState) -> f64 {
        let stats = market_state.trade_window_stats();
        let mid = market_state
            .mid_price()
            .or_else(|| market_state.last_price())
            .unwrap_or(0.0);

        if mid <= f64::EPSILON {
            0.0
        } else {
            stats.price_change / mid * 10_000.0
        }
    }

    fn recent_trade_flow_imbalance(&self, market_state: &MarketState) -> f64 {
        let sample_size = (self.config.min_trades_in_window / 2).max(6);
        market_state.recent_trade_flow_imbalance(sample_size)
    }

    fn drift_above_vwap_bps(&self, market_state: &MarketState) -> f64 {
        let reference_price = market_state
            .mid_price()
            .or_else(|| market_state.last_price())
            .unwrap_or(0.0);
        let Some(vwap) = market_state.trade_window_vwap() else {
            return 0.0;
        };

        if reference_price <= f64::EPSILON || vwap <= f64::EPSILON {
            0.0
        } else {
            (reference_price - vwap) / vwap * 10_000.0
        }
    }

    fn entry_confidence(&self, flow_imbalance: f64, drift_bps: f64, burst_per_second: f64) -> f64 {
        let flow_component = (flow_imbalance.abs()
            / self.config.min_trade_flow_imbalance.max(f64::EPSILON))
        .min(2.0);
        let drift_component =
            (drift_bps.abs() / self.config.min_price_drift_bps.max(f64::EPSILON)).min(2.0);
        let burst_component =
            (burst_per_second / self.config.min_burst_per_second.max(f64::EPSILON)).min(2.0);

        ((flow_component * 0.45) + (drift_component * 0.35) + (burst_component * 0.20)) / 2.0
    }

    fn expected_edge_bps(&self, market_state: &MarketState) -> f64 {
        self.price_drift_bps(market_state)
            .max(self.config.min_price_drift_bps)
    }

    fn should_enter_long(&mut self, market_state: &MarketState) -> bool {
        let stats = market_state.trade_window_stats();
        self.diagnostics.total_decisions += 1;
        if stats.trade_count < self.config.min_trades_in_window {
            self.diagnostics.blocked_min_trades += 1;
            return false;
        }

        if market_state
            .spread_bps()
            .is_some_and(|spread| spread > self.config.max_spread_bps)
        {
            self.diagnostics.blocked_spread += 1;
            return false;
        }

        let flow_imbalance = market_state.trade_flow_imbalance();
        self.diagnostics.last_flow_imbalance = flow_imbalance;
        if flow_imbalance < self.config.min_trade_flow_imbalance {
            self.diagnostics.blocked_flow += 1;
            return false;
        }

        let recent_flow = self.recent_trade_flow_imbalance(market_state);
        self.diagnostics.last_recent_flow_imbalance = recent_flow;
        if recent_flow < self.config.min_recent_trade_flow_imbalance {
            self.diagnostics.blocked_recent_flow += 1;
            return false;
        }

        let drift_bps = self.price_drift_bps(market_state);
        self.diagnostics.last_drift_bps = drift_bps;
        if drift_bps < self.config.min_price_drift_bps
            || drift_bps > self.config.max_price_drift_bps
        {
            self.diagnostics.blocked_drift_band += 1;
            return false;
        }

        let drift_above_vwap_bps = self.drift_above_vwap_bps(market_state);
        self.diagnostics.last_drift_above_vwap_bps = drift_above_vwap_bps;
        if drift_above_vwap_bps > self.config.max_drift_above_vwap_bps {
            self.diagnostics.blocked_vwap_stretch += 1;
            return false;
        }

        if market_state
            .order_book_imbalance()
            .is_some_and(|imbalance| imbalance < self.config.min_order_book_imbalance)
        {
            self.diagnostics.blocked_order_book += 1;
            return false;
        }

        let burst_per_second = self.trade_burst_per_second(market_state);
        self.diagnostics.last_burst_per_second = burst_per_second;
        if burst_per_second < self.config.min_burst_per_second {
            self.diagnostics.blocked_burst += 1;
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

    fn should_exit_long(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> Option<&'static str> {
        if context.current_position.quantity <= 0.0 {
            return None;
        }

        let entry_price = context.current_position.entry_price;
        let reference_price = market_state
            .top_of_book()
            .map(|book| book.bid.price)
            .or_else(|| market_state.last_price())?;

        let pnl_bps = (reference_price - entry_price) / entry_price * 10_000.0;
        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_stop_loss += 1;
            return Some("stop_loss");
        }
        if pnl_bps >= self.config.take_profit_bps {
            self.diagnostics.exits_take_profit += 1;
            return Some("take_profit");
        }

        if self.recent_trade_flow_imbalance(market_state) <= self.config.exit_on_flow_reversal {
            self.diagnostics.exits_flow_reversal += 1;
            return Some("flow_reversal");
        }

        let now = market_state.last_event_time_millis()?;
        if context.current_position.entry_time > 0.0 {
            let held_millis =
                now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
            if held_millis >= self.config.hold_time_millis {
                self.diagnostics.exits_max_hold += 1;
                return Some("max_hold_time");
            }
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
impl Strategy for TradeFlowMomentumStrategy {
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
            diagnostics: MomentumDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "TradeFlowMomentumStrategy - aggressive trade-flow taker".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert(
            "momentum.total_decisions".to_string(),
            self.diagnostics.total_decisions,
        );
        counters.insert(
            "momentum.blocked_min_trades".to_string(),
            self.diagnostics.blocked_min_trades,
        );
        counters.insert(
            "momentum.blocked_spread".to_string(),
            self.diagnostics.blocked_spread,
        );
        counters.insert(
            "momentum.blocked_flow".to_string(),
            self.diagnostics.blocked_flow,
        );
        counters.insert(
            "momentum.blocked_recent_flow".to_string(),
            self.diagnostics.blocked_recent_flow,
        );
        counters.insert(
            "momentum.blocked_drift_band".to_string(),
            self.diagnostics.blocked_drift_band,
        );
        counters.insert(
            "momentum.blocked_vwap_stretch".to_string(),
            self.diagnostics.blocked_vwap_stretch,
        );
        counters.insert(
            "momentum.blocked_order_book".to_string(),
            self.diagnostics.blocked_order_book,
        );
        counters.insert(
            "momentum.blocked_burst".to_string(),
            self.diagnostics.blocked_burst,
        );
        counters.insert(
            "momentum.blocked_cost_gate".to_string(),
            self.diagnostics.blocked_cost_gate,
        );
        counters.insert("momentum.entries".to_string(), self.diagnostics.entries);
        counters.insert(
            "momentum.exits_stop_loss".to_string(),
            self.diagnostics.exits_stop_loss,
        );
        counters.insert(
            "momentum.exits_take_profit".to_string(),
            self.diagnostics.exits_take_profit,
        );
        counters.insert(
            "momentum.exits_flow_reversal".to_string(),
            self.diagnostics.exits_flow_reversal,
        );
        counters.insert(
            "momentum.exits_max_hold".to_string(),
            self.diagnostics.exits_max_hold,
        );

        let mut gauges = BTreeMap::new();
        gauges.insert(
            "momentum.last_flow_imbalance".to_string(),
            self.diagnostics.last_flow_imbalance,
        );
        gauges.insert(
            "momentum.last_recent_flow_imbalance".to_string(),
            self.diagnostics.last_recent_flow_imbalance,
        );
        gauges.insert(
            "momentum.last_drift_bps".to_string(),
            self.diagnostics.last_drift_bps,
        );
        gauges.insert(
            "momentum.last_drift_above_vwap_bps".to_string(),
            self.diagnostics.last_drift_above_vwap_bps,
        );
        gauges.insert(
            "momentum.last_burst_per_second".to_string(),
            self.diagnostics.last_burst_per_second,
        );
        gauges.insert(
            "momentum.last_expected_edge_bps".to_string(),
            self.diagnostics.last_expected_edge_bps,
        );
        gauges.insert(
            "momentum.last_edge_after_cost_bps".to_string(),
            self.diagnostics.last_edge_after_cost_bps,
        );

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, event: &MarketEvent, _market_state: &MarketState) {
        if let MarketEvent::Trade(_) = event {}
    }

    fn decide(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> StrategyDecision {
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
                metrics: vec![DecisionMetric {
                    name: "position_quantity",
                    value: context.current_position.quantity,
                }],
            };
        }

        if context.current_position.quantity > 0.0 || self.in_entry_cooldown(market_state) {
            return StrategyDecision::no_action();
        }

        if !self.should_enter_long(market_state) {
            return StrategyDecision::no_action();
        }

        let flow_imbalance = market_state.trade_flow_imbalance();
        let drift_bps = self.price_drift_bps(market_state);
        let burst_per_second = self.trade_burst_per_second(market_state);
        let confidence = self
            .entry_confidence(flow_imbalance, drift_bps, burst_per_second)
            .max(self.config.position_size_confidence_floor)
            .min(1.0);

        let expected_edge_bps = self.expected_edge_bps(market_state);
        let reference_price = market_state
            .mid_price()
            .or_else(|| market_state.last_price())
            .unwrap_or(0.0);
        if reference_price <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        let Some(quantity) = context.capped_entry_quantity(reference_price, 0.9, None) else {
            return StrategyDecision::no_action();
        };

        self.last_entry_time_millis = market_state.last_event_time_millis();
        self.diagnostics.entries += 1;

        StrategyDecision {
            confidence,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Taker,
                price: None,
                quantity,
                time_in_force: TimeInForce::Ioc,
                rationale: "trade_flow_momentum_entry",
                expected_edge_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "flow_imbalance",
                    value: flow_imbalance,
                },
                DecisionMetric {
                    name: "recent_flow_imbalance",
                    value: self.recent_trade_flow_imbalance(market_state),
                },
                DecisionMetric {
                    name: "drift_bps",
                    value: drift_bps,
                },
                DecisionMetric {
                    name: "burst_per_second",
                    value: burst_per_second,
                },
                DecisionMetric {
                    name: "drift_above_vwap_bps",
                    value: self.drift_above_vwap_bps(market_state),
                },
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
