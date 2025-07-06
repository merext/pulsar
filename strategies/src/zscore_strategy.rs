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

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;

pub struct ZScoreStrategy {
    period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    prices: VecDeque<f64>,
}

impl ZScoreStrategy {
    pub fn new(period: usize, buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            period,
            buy_threshold,
            sell_threshold,
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
        let mut confidence: f64 = 0.0;

        if z_score > self.sell_threshold {
            signal = Signal::Sell;
            // Calculate confidence for Sell signal
            // The further z_score is above sell_threshold, the higher the confidence
            confidence = ((z_score - self.sell_threshold) / 1.0).min(1.0); // Assuming 1.0 is a reasonable scaling factor for deviation
        } else if z_score < self.buy_threshold {
            signal = Signal::Buy;
            // Calculate confidence for Buy signal
            // The further z_score is below buy_threshold, the higher the confidence
            confidence = ((self.buy_threshold - z_score) / 1.0).min(1.0); // Assuming 1.0 is a reasonable scaling factor for deviation
        } else {
            signal = Signal::Hold;
            confidence = 0.0; // No confidence for Hold signal
        }

        (signal, confidence)
    }
}