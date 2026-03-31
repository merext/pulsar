use serde::Deserialize;
use std::path::Path;
use trade::execution::{OrderIntent, Side, TimeInForce};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyLogger};
use trade::trader::OrderType;

#[derive(Debug, Clone, Deserialize)]
pub struct LiquiditySweepReversalConfig {
    pub trade_window_millis: u64,
    pub min_trades_in_window: usize,
    pub min_sweep_drop_bps: f64,
    pub min_buyer_reclaim_imbalance: f64,
    pub min_reclaim_bps: f64,
    pub max_reclaim_bps: f64,
    pub max_spread_bps: f64,
    pub min_large_trade_ratio: f64,
    pub stop_loss_bps: f64,
    pub take_profit_bps: f64,
    pub hold_time_millis: u64,
    pub entry_cooldown_millis: u64,
}

impl Default for LiquiditySweepReversalConfig {
    fn default() -> Self {
        Self {
            trade_window_millis: 2_000,
            min_trades_in_window: 18,
            min_sweep_drop_bps: 10.0,
            min_buyer_reclaim_imbalance: 0.12,
            min_reclaim_bps: 2.0,
            max_reclaim_bps: 20.0,
            max_spread_bps: 14.0,
            min_large_trade_ratio: 0.12,
            stop_loss_bps: 16.0,
            take_profit_bps: 20.0,
            hold_time_millis: 5_000,
            entry_cooldown_millis: 3_000,
        }
    }
}

pub struct LiquiditySweepReversalStrategy {
    config: LiquiditySweepReversalConfig,
    logger: NoOpStrategyLogger,
    last_entry_time_millis: Option<u64>,
}

impl LiquiditySweepReversalStrategy {
    fn load_config<P: AsRef<Path>>(config_path: P) -> Result<LiquiditySweepReversalConfig, Box<dyn std::error::Error>> {
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

        if self.high_to_low_drop_bps(market_state) < self.config.min_sweep_drop_bps {
            return false;
        }

        let reclaim_bps = self.low_to_now_reclaim_bps(market_state);
        if reclaim_bps < self.config.min_reclaim_bps || reclaim_bps > self.config.max_reclaim_bps {
            return false;
        }

        if market_state.trade_flow_imbalance() < self.config.min_buyer_reclaim_imbalance {
            return false;
        }

        self.large_trade_ratio(market_state) >= self.config.min_large_trade_ratio
    }

    fn should_exit_long(&self, market_state: &MarketState, context: &StrategyContext) -> Option<&'static str> {
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
            return Some("stop_loss");
        }
        if pnl_bps >= self.config.take_profit_bps {
            return Some("take_profit");
        }
        if market_state.trade_flow_imbalance() < -0.08 {
            return Some("reversal_failed");
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
        })
    }

    fn get_info(&self) -> String {
        "LiquiditySweepReversalStrategy - taker rebound after local sweep".to_string()
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
            confidence: 0.8,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Taker,
                price: None,
                quantity,
                time_in_force: TimeInForce::Ioc,
                rationale: "liquidity_sweep_reversal_entry",
                expected_edge_bps: self.low_to_now_reclaim_bps(market_state),
            },
        }
    }
}
