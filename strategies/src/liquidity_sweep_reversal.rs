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
struct SweepDiagnostics {
    total_decisions: usize,
    blocked_min_trades: usize,
    blocked_spread: usize,
    blocked_sweep_drop: usize,
    blocked_reclaim_band: usize,
    blocked_flow: usize,
    blocked_recent_flow: usize,
    blocked_vwap_stretch: usize,
    blocked_order_book: usize,
    blocked_large_trade_ratio: usize,
    blocked_cost_gate: usize,
    entries: usize,
    exits_stop_loss: usize,
    exits_take_profit: usize,
    exits_reversal_failed: usize,
    exits_max_hold: usize,
    last_sweep_drop_bps: f64,
    last_reclaim_bps: f64,
    last_recent_buyer_imbalance: f64,
    last_large_trade_ratio: f64,
    last_reclaim_above_vwap_bps: f64,
    last_expected_edge_bps: f64,
    last_edge_after_cost_bps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LiquiditySweepReversalConfig {
    pub trade_window_millis: u64,
    pub min_trades_in_window: usize,
    pub min_sweep_drop_bps: f64,
    pub min_buyer_reclaim_imbalance: f64,
    pub min_recent_buyer_imbalance: f64,
    pub min_reclaim_bps: f64,
    pub max_reclaim_bps: f64,
    pub max_reclaim_above_vwap_bps: f64,
    pub max_spread_bps: f64,
    pub min_large_trade_ratio: f64,
    pub min_order_book_imbalance: f64,
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

impl Default for LiquiditySweepReversalConfig {
    fn default() -> Self {
        Self {
            trade_window_millis: 2_000,
            min_trades_in_window: 18,
            min_sweep_drop_bps: 10.0,
            min_buyer_reclaim_imbalance: 0.12,
            min_recent_buyer_imbalance: 0.20,
            min_reclaim_bps: 2.0,
            max_reclaim_bps: 20.0,
            max_reclaim_above_vwap_bps: 6.0,
            max_spread_bps: 14.0,
            min_large_trade_ratio: 0.12,
            min_order_book_imbalance: 0.05,
            stop_loss_bps: 16.0,
            take_profit_bps: 20.0,
            hold_time_millis: 5_000,
            entry_cooldown_millis: 3_000,
            assumed_round_trip_taker_cost_bps: DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS,
            min_expected_edge_after_cost_bps: DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
        }
    }
}

pub struct LiquiditySweepReversalStrategy {
    config: LiquiditySweepReversalConfig,
    logger: NoOpStrategyLogger,
    last_entry_time_millis: Option<u64>,
    diagnostics: SweepDiagnostics,
}

impl LiquiditySweepReversalStrategy {
    fn load_config<P: AsRef<Path>>(
        config_path: P,
    ) -> Result<LiquiditySweepReversalConfig, Box<dyn std::error::Error>> {
        let path = config_path.as_ref();
        if path == Path::new("/dev/null") || !path.exists() {
            return Ok(LiquiditySweepReversalConfig::default());
        }

        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str::<LiquiditySweepReversalConfig>(&content)?;
        Ok(config)
    }

    fn current_reference_price(&self, market_state: &MarketState) -> Option<f64> {
        market_state
            .top_of_book()
            .map(|book| book.ask.price)
            .or_else(|| market_state.last_price())
    }

    fn low_to_now_reclaim_bps(&self, market_state: &MarketState) -> f64 {
        let low = market_state.trade_window_low_price().unwrap_or(0.0);
        let now = self.current_reference_price(market_state).unwrap_or(0.0);
        if low <= f64::EPSILON || now <= f64::EPSILON {
            return 0.0;
        }
        (now - low) / low * 10_000.0
    }

    fn high_to_low_drop_bps(&self, market_state: &MarketState) -> f64 {
        let high = market_state.trade_window_high_price().unwrap_or(0.0);
        let low = market_state.trade_window_low_price().unwrap_or(0.0);
        if high <= f64::EPSILON || low <= f64::EPSILON || low >= high {
            return 0.0;
        }
        (high - low) / high * 10_000.0
    }

    fn reclaim_above_vwap_bps(&self, market_state: &MarketState) -> f64 {
        let Some(current_price) = self.current_reference_price(market_state) else {
            return 0.0;
        };
        let Some(vwap) = market_state.trade_window_vwap() else {
            return 0.0;
        };
        if vwap <= f64::EPSILON {
            return 0.0;
        }

        (current_price - vwap) / vwap * 10_000.0
    }

    fn recent_buyer_imbalance(&self, market_state: &MarketState) -> f64 {
        let sample_size = (self.config.min_trades_in_window / 2).max(6);
        market_state.recent_trade_flow_imbalance(sample_size)
    }

    fn large_trade_ratio(&self, market_state: &MarketState) -> f64 {
        let stats = market_state.trade_window_stats();
        if stats.trade_count == 0 {
            return 0.0;
        }

        let avg_size = stats.volume / stats.trade_count as f64;
        if avg_size <= f64::EPSILON {
            return 0.0;
        }

        let large_count = market_state
            .recent_trades()
            .filter(|trade| trade.quantity >= avg_size * 1.5)
            .count();

        large_count as f64 / stats.trade_count as f64
    }

    fn expected_edge_bps(&self, market_state: &MarketState) -> f64 {
        self.low_to_now_reclaim_bps(market_state)
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

        let sweep_drop_bps = self.high_to_low_drop_bps(market_state);
        self.diagnostics.last_sweep_drop_bps = sweep_drop_bps;
        if sweep_drop_bps < self.config.min_sweep_drop_bps {
            self.diagnostics.blocked_sweep_drop += 1;
            return false;
        }

        let reclaim_bps = self.low_to_now_reclaim_bps(market_state);
        self.diagnostics.last_reclaim_bps = reclaim_bps;
        if reclaim_bps < self.config.min_reclaim_bps || reclaim_bps > self.config.max_reclaim_bps {
            self.diagnostics.blocked_reclaim_band += 1;
            return false;
        }

        if market_state.trade_flow_imbalance() < self.config.min_buyer_reclaim_imbalance {
            self.diagnostics.blocked_flow += 1;
            return false;
        }

        let recent_buyer_imbalance = self.recent_buyer_imbalance(market_state);
        self.diagnostics.last_recent_buyer_imbalance = recent_buyer_imbalance;
        if recent_buyer_imbalance < self.config.min_recent_buyer_imbalance {
            self.diagnostics.blocked_recent_flow += 1;
            return false;
        }

        let reclaim_above_vwap_bps = self.reclaim_above_vwap_bps(market_state);
        self.diagnostics.last_reclaim_above_vwap_bps = reclaim_above_vwap_bps;
        if reclaim_above_vwap_bps > self.config.max_reclaim_above_vwap_bps {
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

        let large_trade_ratio = self.large_trade_ratio(market_state);
        self.diagnostics.last_large_trade_ratio = large_trade_ratio;
        if large_trade_ratio < self.config.min_large_trade_ratio {
            self.diagnostics.blocked_large_trade_ratio += 1;
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

        let reference_price = market_state
            .top_of_book()
            .map(|book| book.bid.price)
            .or_else(|| market_state.last_price())?;
        let entry_price = context.current_position.entry_price;
        let pnl_bps = (reference_price - entry_price) / entry_price * 10_000.0;

        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_stop_loss += 1;
            return Some("stop_loss");
        }
        if pnl_bps >= self.config.take_profit_bps {
            self.diagnostics.exits_take_profit += 1;
            return Some("take_profit");
        }
        if self.recent_buyer_imbalance(market_state) < -0.08 {
            self.diagnostics.exits_reversal_failed += 1;
            return Some("reversal_failed");
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
impl Strategy for LiquiditySweepReversalStrategy {
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
            diagnostics: SweepDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "LiquiditySweepReversalStrategy - taker rebound after local sweep".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert(
            "sweep.total_decisions".to_string(),
            self.diagnostics.total_decisions,
        );
        counters.insert(
            "sweep.blocked_min_trades".to_string(),
            self.diagnostics.blocked_min_trades,
        );
        counters.insert(
            "sweep.blocked_spread".to_string(),
            self.diagnostics.blocked_spread,
        );
        counters.insert(
            "sweep.blocked_sweep_drop".to_string(),
            self.diagnostics.blocked_sweep_drop,
        );
        counters.insert(
            "sweep.blocked_reclaim_band".to_string(),
            self.diagnostics.blocked_reclaim_band,
        );
        counters.insert(
            "sweep.blocked_flow".to_string(),
            self.diagnostics.blocked_flow,
        );
        counters.insert(
            "sweep.blocked_recent_flow".to_string(),
            self.diagnostics.blocked_recent_flow,
        );
        counters.insert(
            "sweep.blocked_vwap_stretch".to_string(),
            self.diagnostics.blocked_vwap_stretch,
        );
        counters.insert(
            "sweep.blocked_order_book".to_string(),
            self.diagnostics.blocked_order_book,
        );
        counters.insert(
            "sweep.blocked_large_trade_ratio".to_string(),
            self.diagnostics.blocked_large_trade_ratio,
        );
        counters.insert(
            "sweep.blocked_cost_gate".to_string(),
            self.diagnostics.blocked_cost_gate,
        );
        counters.insert("sweep.entries".to_string(), self.diagnostics.entries);
        counters.insert(
            "sweep.exits_stop_loss".to_string(),
            self.diagnostics.exits_stop_loss,
        );
        counters.insert(
            "sweep.exits_take_profit".to_string(),
            self.diagnostics.exits_take_profit,
        );
        counters.insert(
            "sweep.exits_reversal_failed".to_string(),
            self.diagnostics.exits_reversal_failed,
        );
        counters.insert(
            "sweep.exits_max_hold".to_string(),
            self.diagnostics.exits_max_hold,
        );

        let mut gauges = BTreeMap::new();
        gauges.insert(
            "sweep.last_sweep_drop_bps".to_string(),
            self.diagnostics.last_sweep_drop_bps,
        );
        gauges.insert(
            "sweep.last_reclaim_bps".to_string(),
            self.diagnostics.last_reclaim_bps,
        );
        gauges.insert(
            "sweep.last_recent_buyer_imbalance".to_string(),
            self.diagnostics.last_recent_buyer_imbalance,
        );
        gauges.insert(
            "sweep.last_large_trade_ratio".to_string(),
            self.diagnostics.last_large_trade_ratio,
        );
        gauges.insert(
            "sweep.last_reclaim_above_vwap_bps".to_string(),
            self.diagnostics.last_reclaim_above_vwap_bps,
        );
        gauges.insert(
            "sweep.last_expected_edge_bps".to_string(),
            self.diagnostics.last_expected_edge_bps,
        );
        gauges.insert(
            "sweep.last_edge_after_cost_bps".to_string(),
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
            confidence: 0.8,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Taker,
                price: None,
                quantity,
                time_in_force: TimeInForce::Ioc,
                rationale: "liquidity_sweep_reversal_entry",
                expected_edge_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "sweep_drop_bps",
                    value: self.high_to_low_drop_bps(market_state),
                },
                DecisionMetric {
                    name: "reclaim_bps",
                    value: expected_edge_bps,
                },
                DecisionMetric {
                    name: "recent_buyer_imbalance",
                    value: self.recent_buyer_imbalance(market_state),
                },
                DecisionMetric {
                    name: "large_trade_ratio",
                    value: self.large_trade_ratio(market_state),
                },
                DecisionMetric {
                    name: "reclaim_above_vwap_bps",
                    value: self.reclaim_above_vwap_bps(market_state),
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
