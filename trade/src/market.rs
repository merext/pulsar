use async_trait::async_trait;
use crate::models::MarketEvent;

#[async_trait]
pub trait MarketDataProvider {
    async fn next_event(&mut self) -> Option<MarketEvent>;
}

pub struct BinanceMarketStream;

#[async_trait]
impl MarketDataProvider for BinanceMarketStream {
    async fn next_event(&mut self) -> Option<MarketEvent> {
        // Placeholder for live WebSocket data
        None
    }
}

pub struct BacktestMarketStream;

#[async_trait]
impl MarketDataProvider for BacktestMarketStream {
    async fn next_event(&mut self) -> Option<MarketEvent> {
        // Placeholder for reading from ZIP/CSV files
        None
    }
}