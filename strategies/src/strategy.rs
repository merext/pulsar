use crate::models::{Kline, TradeData};

#[async_trait::async_trait]
pub trait Strategy {
    async fn on_kline(&mut self, kline: Kline);
    async fn on_trade(&mut self, trade: TradeData);
}