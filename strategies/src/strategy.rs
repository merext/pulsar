use trade::models::TradeData;
use trade::trader::Position;
use trade::signal::Signal;

#[allow(async_fn_in_trait)]
#[async_trait::async_trait]
pub trait Strategy {
    fn get_info(&self) -> String;
    async fn on_trade(&mut self, trade: TradeData);
    fn get_signal(
        &mut self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        (Signal::Hold, 0.0)
    }
}
