use crate::models::TradeData;

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


