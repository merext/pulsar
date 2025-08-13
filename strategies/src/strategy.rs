use crate::models::{TradeData, Position, Signal};
use std::path::Path;

#[allow(async_fn_in_trait)]
#[async_trait::async_trait]
pub trait Strategy: Send + Sync {
    /// Create a new strategy instance from a configuration file
    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    
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
