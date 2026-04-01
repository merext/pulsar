use serde::Deserialize;
use std::path::Path;
use trade::execution::{DecisionMetric, OrderIntent, Side, TimeInForce};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyLogger};
use trade::trader::OrderType;

#[derive(Debug, Clone, Deserialize)]
pub struct MicropriceImbalanceMakerConfig {
    pub min_order_book_imbalance: f64,
    pub min_microprice_edge_bps: f64,
    pub max_spread_bps: f64,
    pub max_quote_size_notional: f64,
    pub hold_time_millis: u64,
    pub stop_loss_bps: f64,
    pub take_profit_bps: f64,
    pub entry_cooldown_millis: u64,
}

impl Default for MicropriceImbalanceMakerConfig {
    fn default() -> Self {
        Self {
            min_order_book_imbalance: 0.08,
            min_microprice_edge_bps: 1.5,
            max_spread_bps: 8.0,
            max_quote_size_notional: 20.0,
            hold_time_millis: 3_000,
            stop_loss_bps: 10.0,
            take_profit_bps: 12.0,
            entry_cooldown_millis: 2_000,
        }
    }
}

pub struct MicropriceImbalanceMakerStrategy {
    config: MicropriceImbalanceMakerConfig,
    logger: NoOpStrategyLogger,
    last_entry_time_millis: Option<u64>,
}

impl MicropriceImbalanceMakerStrategy {
    fn load_config<P: AsRef<Path>>(
        config_path: P,
    ) -> Result<MicropriceImbalanceMakerConfig, Box<dyn std::error::Error>> {
        let path = config_path.as_ref();
        if path == Path::new("/dev/null") || !path.exists() {
            return Ok(MicropriceImbalanceMakerConfig::default());
        }

        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str::<MicropriceImbalanceMakerConfig>(&content)?;
        Ok(config)
    }

    fn maker_edge_bps(&self, market_state: &MarketState) -> Option<f64> {
        let microprice = market_state.microprice()?;
        let mid = market_state.mid_price()?;
        if mid <= f64::EPSILON {
            return None;
        }
        Some((microprice - mid) / mid * 10_000.0)
    }

    fn should_enter_long(&self, market_state: &MarketState) -> bool {
        if market_state
            .spread_bps()
            .is_some_and(|spread| spread > self.config.max_spread_bps)
        {
            return false;
        }

        if market_state
            .order_book_imbalance()
            .is_none_or(|imbalance| imbalance < self.config.min_order_book_imbalance)
        {
            return false;
        }

        self.maker_edge_bps(market_state)
            .is_some_and(|edge| edge >= self.config.min_microprice_edge_bps)
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
        if market_state
            .order_book_imbalance()
            .is_some_and(|imbalance| imbalance < 0.0)
        {
            return Some("imbalance_reversal");
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
impl Strategy for MicropriceImbalanceMakerStrategy {
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
        "MicropriceImbalanceMakerStrategy - passive quote on positive microprice edge".to_string()
    }

    async fn on_event(&mut self, event: &MarketEvent, _market_state: &MarketState) {
        match event {
            MarketEvent::Trade(_) | MarketEvent::BookTicker(_) | MarketEvent::Depth(_) => {}
        }
    }

    fn decide(&mut self, market_state: &MarketState, context: &StrategyContext) -> StrategyDecision {
        if let Some(rationale) = self.should_exit_long(market_state, context) {
            return StrategyDecision {
                confidence: 1.0,
                intent: OrderIntent::Place {
                    side: Side::Sell,
                    order_type: OrderType::Maker,
                    price: market_state.top_of_book().map(|book| book.ask.price),
                    quantity: context.current_position.quantity,
                    time_in_force: TimeInForce::Gtc,
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

        let reference_price = market_state.mid_price().unwrap_or(0.0);
        if reference_price <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        let Some(quantity) = context.capped_entry_quantity(
            reference_price,
            1.0,
            Some(self.config.max_quote_size_notional),
        ) else {
            return StrategyDecision::no_action();
        };

        self.last_entry_time_millis = market_state.last_event_time_millis();
        StrategyDecision {
            confidence: 0.75,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Maker,
                price: market_state.top_of_book().map(|book| book.bid.price),
                quantity,
                time_in_force: TimeInForce::Gtc,
                rationale: "microprice_imbalance_maker_entry",
                expected_edge_bps: self.maker_edge_bps(market_state).unwrap_or(0.0),
            },
            metrics: vec![
                DecisionMetric {
                    name: "order_book_imbalance",
                    value: market_state.order_book_imbalance().unwrap_or(0.0),
                },
                DecisionMetric {
                    name: "maker_edge_bps",
                    value: self.maker_edge_bps(market_state).unwrap_or(0.0),
                },
                DecisionMetric {
                    name: "spread_bps",
                    value: market_state.spread_bps().unwrap_or(0.0),
                },
            ],
        }
    }
}
