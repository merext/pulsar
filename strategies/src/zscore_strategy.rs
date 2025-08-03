//! # Z-Score Strategy
//! 
//! This strategy identifies trading opportunities by measuring how many standard deviations the current price
//! is away from its historical mean (its Z-score). It is a statistical approach to mean reversion.
//! 
//! The strategy calculates the mean and standard deviation of prices over a defined period.
//! A buy signal is generated when the Z-score falls below a negative threshold (e.g., -2),
//! indicating that the price is significantly oversold and likely to revert upwards.
//! Conversely, a sell signal is generated when the Z-score rises above a positive threshold (e.g., +2),
//! indicating that the price is significantly overbought and likely to revert downwards.

use crate::confidence::{scale_from_threshold, scale_from_threshold_inverse};
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct ZScoreStrategy {
    period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    scale: f64,
    prices: VecDeque<f64>,
}

impl ZScoreStrategy {
    pub fn new(period: usize, buy_threshold: f64, sell_threshold: f64, scale: f64) -> Self {
        Self {
            period,
            buy_threshold,
            sell_threshold,
            scale,
            prices: VecDeque::new(),
        }
    }

    fn calculate_mean_and_std_dev(&self) -> (f64, f64) {
        if self.prices.len() < self.period {
            return (0.0, 0.0);
        }

        let relevant_prices: Vec<f64> = self.prices.iter().rev().take(self.period).cloned().collect();
        let sum: f64 = relevant_prices.iter().sum();
        let mean = sum / self.period as f64;

        let sum_of_squares: f64 = relevant_prices.iter().map(|p| (p - mean).powi(2)).sum();
        let std_dev = (sum_of_squares / self.period as f64).sqrt();

        (mean, std_dev)
    }
}

#[async_trait::async_trait]
impl Strategy for ZScoreStrategy {
    fn get_info(&self) -> String {
        format!("Z-Score Strategy (period: {}, buy_threshold: {}, sell_threshold: {}, scale: {})", self.period, self.buy_threshold, self.sell_threshold, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        self.prices.push_back(price);
        if self.prices.len() > self.period {
            self.prices.pop_front();
        }
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.prices.len() < self.period {
            return (Signal::Hold, 0.0);
        }

        let (mean, std_dev) = self.calculate_mean_and_std_dev();

        if std_dev == 0.0 {
            return (Signal::Hold, 0.0);
        }

        let z_score = (current_price - mean) / std_dev;

        let signal: Signal;
        let confidence: f64;

        if z_score > self.sell_threshold {
            signal = Signal::Sell;
            confidence = scale_from_threshold(z_score, self.sell_threshold, self.scale);
        } else if z_score < self.buy_threshold {
            signal = Signal::Buy;
            confidence = scale_from_threshold_inverse(z_score, self.buy_threshold, self.scale);
        } else {
            signal = Signal::Hold;
            confidence = 0.0; // No confidence for Hold signal
        }

        (signal, confidence)
    }
}