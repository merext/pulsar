use trade::models::{Kline, TradeData};
use trade::trader::Position;
use trade::signal::Signal;

#[allow(async_fn_in_trait)]
#[async_trait::async_trait]
pub trait Strategy {
    async fn on_kline(&mut self, kline: Kline);
    async fn on_trade(&mut self, trade: TradeData);
    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> Signal {
        Signal::Hold
    }
}
