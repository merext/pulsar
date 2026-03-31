use std::path::Path;
use trade::execution::OrderIntent;
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyLogger};

pub struct NullStrategy {
    logger: NoOpStrategyLogger,
}

#[async_trait::async_trait]
impl Strategy for NullStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        &self.logger
    }

    fn from_file<P: AsRef<Path>>(_config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        Ok(Self {
            logger: NoOpStrategyLogger,
        })
    }

    fn get_info(&self) -> String {
        "NullStrategy - infrastructure validation only".to_string()
    }

    async fn on_event(&mut self, _event: &MarketEvent, _market_state: &MarketState) {}

    fn decide(&mut self, market_state: &MarketState, _context: &StrategyContext) -> StrategyDecision {
        let _ = market_state;
        StrategyDecision {
            confidence: 0.0,
            intent: OrderIntent::NoAction,
        }
    }
}
