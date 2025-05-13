use crate::models::{Kline, TradeData};
use crate::strategy::Strategy;

pub struct MeanReversionStrategy {
    pub moving_average: f64,
}

#[async_trait::async_trait]
impl Strategy for MeanReversionStrategy {
    async fn on_kline(&mut self, kline: Kline) {
        println!("Processing Kline: {:?}", kline);
    }

    async fn on_trade(&mut self, trade: TradeData) {
        println!("Processing Trade: {:?}", trade);
    }
}