use serde::Deserialize;
use std::path::Path;
use trade::execution::{OrderIntent, Side, TimeInForce};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyLogger};
use trade::trader::OrderType;

#[derive(Debug, Clone, Deserialize)]
pub struct TradeFlowMomentumConfig {
    pub trade_window_millis: u64,
    pub min_trades_in_window: usize,
    pub min_trade_flow_imbalance: f64,
    pub min_price_drift_bps: f64,
    pub max_price_drift_bps: f64,
    pub max_spread_bps: f64,
    pub min_burst_per_second: f64,
    pub position_size_confidence_floor: f64,
    pub entry_cooldown_millis: u64,
    pub hold_time_millis: u64,
    pub exit_on_flow_reversal: f64,
    pub stop_loss_bps: f64,
    pub take_profit_bps: f64,
}

impl Default for TradeFlowMomentumConfig {
    fn default() -> Self {
        Self {
            trade_window_millis: 1_500,
            min_trades_in_window: 12,
            min_trade_flow_imbalance: 0.18,
            min_price_drift_bps: 6.0,
            max_price_drift_bps: 35.0,
            max_spread_bps: 12.0,
            min_burst_per_second: 10.0,
            position_size_confidence_floor: 0.72,
            entry_cooldown_millis: 2_500,
            hold_time_millis: 4_000,
            exit_on_flow_reversal: -0.05,
            stop_loss_bps: 18.0,
            take_profit_bps: 24.0,
        }
    }
}

pub struct TradeFlowMomentumStrategy {
    config: TradeFlowMomentumConfig,
    logger: NoOpStrategyLogger,
    last_entry_time_millis: Option<u64>,
}

impl TradeFlowMomentumStrategy {
    fn load_config<P: AsRef<Path>>(config_path: P) -> Result<TradeFlowMomentumConfig, Box<dyn std::error::Error>> {
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

    fn entry_confidence(&self, flow_imbalance: f64, drift_bps: f64, burst_per_second: f64) -> f64 {
        let flow_component =
            (flow_imbalance.abs() / self.config.min_trade_flow_imbalance.max(f64::EPSILON)).min(2.0);
        let drift_component =
            (drift_bps.abs() / self.config.min_price_drift_bps.max(f64::EPSILON)).min(2.0);
        let burst_component =
            (burst_per_second / self.config.min_burst_per_second.max(f64::EPSILON)).min(2.0);

        ((flow_component * 0.45) + (drift_component * 0.35) + (burst_component * 0.20)) / 2.0
    }

    fn should_enter_long(&self, market_state: &MarketState) -> bool {
        let stats = market_state.trade_window_stats();
        if stats.trade_count < self.config.min_trades_in_window {
            return false;
        }

        if market_state
            .spread_bps()
            .is_some_and(|spread| spread > self.config.max_spread_bps)
        {
            return false;
        }

        let flow_imbalance = market_state.trade_flow_imbalance();
        if flow_imbalance < self.config.min_trade_flow_imbalance {
            return false;
        }

        let drift_bps = self.price_drift_bps(market_state);
        if drift_bps < self.config.min_price_drift_bps || drift_bps > self.config.max_price_drift_bps {
            return false;
        }

        self.trade_burst_per_second(market_state) >= self.config.min_burst_per_second
    }

    fn should_exit_long(&self, market_state: &MarketState, context: &StrategyContext) -> Option<&'static str> {
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
            return Some("stop_loss");
        }
        if pnl_bps >= self.config.take_profit_bps {
            return Some("take_profit");
        }

        if market_state.trade_flow_imbalance() <= self.config.exit_on_flow_reversal {
            return Some("flow_reversal");
        }

        let now = market_state.last_event_time_millis()?;
        if context.current_position.entry_time > 0.0 {
            let held_millis = now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
            if held_millis >= self.config.hold_time_millis {
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
        })
    }

    fn get_info(&self) -> String {
        "TradeFlowMomentumStrategy - aggressive trade-flow taker".to_string()
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, event: &MarketEvent, _market_state: &MarketState) {
        if let MarketEvent::Trade(_) = event {}
    }

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

        let reference_price = market_state
            .mid_price()
            .or_else(|| market_state.last_price())
            .unwrap_or(0.0);
        if reference_price <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        let target_notional = context.max_position_notional.min(context.available_cash * 0.9);
        if target_notional <= 0.0 {
            return StrategyDecision::no_action();
        }

        let quantity = target_notional / reference_price;
        if quantity <= 0.0 {
            return StrategyDecision::no_action();
        }

        self.last_entry_time_millis = market_state.last_event_time_millis();

        StrategyDecision {
            confidence,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Taker,
                price: None,
                quantity,
                time_in_force: TimeInForce::Ioc,
                rationale: "trade_flow_momentum_entry",
                expected_edge_bps: drift_bps.max(self.config.min_price_drift_bps),
            },
        }
    }
}
