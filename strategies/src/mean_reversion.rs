//! # Mean Reversion Strategy
//! 
//! This strategy is based on the principle that asset prices tend to revert to their historical average or mean price over time.
//! It identifies opportunities when the price deviates significantly from its moving average, expecting it to return to the mean.
//! 
//! The strategy generates a buy signal when the current price falls below its Simple Moving Average (SMA),
//! anticipating a bounce back towards the mean. Conversely, it generates a sell signal when the price rises above its SMA,
//! expecting a pull back.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;


use std::collections::VecDeque;

pub struct MeanReversionStrategy {
    pub window_size: usize,
    pub prices: VecDeque<f64>,
    pub last_sma: Option<f64>,
    pub recent_trades: VecDeque<f64>,
    pub max_trade_window: usize,
}

impl MeanReversionStrategy {
    pub fn new(window_size: usize, max_trade_window: usize) -> Self {
        Self {
            window_size,
            prices: VecDeque::with_capacity(window_size),
            last_sma: None,
            recent_trades: VecDeque::with_capacity(max_trade_window),
            max_trade_window,
        }
    }
}

#[async_trait::async_trait]
impl Strategy for MeanReversionStrategy {
    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        self.prices.push_back(price);
        if self.prices.len() > self.window_size {
            self.prices.pop_front();
        }

        self.recent_trades.push_back(price);
        if self.recent_trades.len() > self.max_trade_window {
            self.recent_trades.pop_front();
        }

        // Update SMA
        if self.prices.len() == self.window_size {
            let sum: f64 = self.prices.iter().sum();
            self.last_sma = Some(sum / self.window_size as f64);
        }
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if let Some(sma) = self.last_sma {
            let deviation = current_price - sma;
            let signal: Signal;
            let mut confidence: f64 = 0.0;

            // A simple confidence calculation: larger deviation means higher confidence
            // Normalize deviation to a 0-1 range. You might need to tune the divisor.
            let max_deviation_for_confidence = sma * 0.01; // Example: 1% of SMA as max deviation for 100% confidence

            if deviation < 0.0 && current_price < sma { // Price is below SMA, potential buy
                signal = Signal::Buy;
                confidence = (deviation.abs() / max_deviation_for_confidence).min(1.0);
            } else if deviation > 0.0 && current_price > sma { // Price is above SMA, potential sell
                signal = Signal::Sell;
                confidence = (deviation.abs() / max_deviation_for_confidence).min(1.0);
            } else {
                signal = Signal::Hold;
                confidence = 0.0;
            }
            (signal, confidence)
        } else {
            (Signal::Hold, 0.0)
        }
    }
}