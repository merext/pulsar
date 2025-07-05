//! # VWAP Deviation Strategy
//! 
//! This strategy aims to identify trading opportunities when the current price deviates significantly from the Volume Weighted Average Price (VWAP).
//! VWAP represents the average price an asset has traded at throughout the day, based on both volume and price.
//! 
//! **Note:** As of the current implementation, this strategy is a placeholder. It previously relied on K-line (candlestick) data,
//! which has been removed to align with a tick-data-centric HFT approach. To become functional, it requires re-implementation
//! to calculate VWAP using granular trade data (tick data) rather than aggregated K-lines.
//! 
//! Once re-implemented, the strategy would typically generate a buy signal if the price falls significantly below VWAP (indicating undervaluation)
//! and a sell signal if the price rises significantly above VWAP (indicating overvaluation).

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use async_trait::async_trait;


pub struct VwapDeviationStrategy {
    period: usize,
    deviation_threshold: f64,
}

impl VwapDeviationStrategy {
    pub fn new(period: usize, deviation_threshold: f64) -> Self {
        Self {
            period,
            deviation_threshold,
        }
    }

    
}

#[async_trait]
impl Strategy for VwapDeviationStrategy {
    async fn on_trade(&mut self, _trade: TradeData) {
        // Not used in this strategy
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> Signal {
        // This strategy requires K-line data, which is no longer available.
        // It needs to be re-implemented using TradeData or removed.
        Signal::Hold
    }
}
