use crate::market::MarketDataProvider;
use crate::models::{MarketEvent, TradeData};
use crate::trader::Trader;
use tokio::select;

#[allow(async_fn_in_trait)]
pub trait Strategy {
    fn get_signal(
        &self,
        price: f64,
        ts: f64,
        position: crate::trader::Position,
    ) -> crate::signal::Signal;
    async fn on_trade(&mut self, trade: TradeData);
}

pub struct TradingExecutor<S, T, M>
where
    S: Strategy + Send,
    T: Trader + Send,
    M: MarketDataProvider + Send,
{
    strategy: S,
    trader: T,
    market: M,
}

impl<S, T, M> TradingExecutor<S, T, M>
where
    S: Strategy + Send,
    T: Trader + Send,
    M: MarketDataProvider + Send,
{
    pub fn new(strategy: S, trader: T, market: M) -> Self {
        TradingExecutor {
            strategy,
            trader,
            market,
        }
    }

    pub async fn run(&mut self) {
        loop {
            select! {
                Some(event) = self.market.next_event() => {
                    match event {
                        
                        MarketEvent::Trade(trade) => {
                            self.strategy.on_trade(trade).await;
                        },
                    }
                },
            }
        }
    }
}
