//! # Momentum Scalping Strategy
//!
//! This strategy aims to profit from very short-term price momentum by identifying rapid price changes.
//! It is designed for high-frequency trading (HFT) environments where quick entry and exit are crucial.
//!
//! The strategy tracks recent price movements within a defined window.
//! A buy signal is generated if the price change over this window exceeds a positive threshold,
//! indicating upward momentum. Conversely, a sell signal is generated if the price change falls below a negative threshold,
//! indicating downward momentum.

use crate::confidence::{scale_from_threshold, scale_from_threshold_inverse};
use crate::strategy::Strategy;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::f64;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;

#[derive(Clone)]
pub struct MomentumScalping {
    trade_window_size: usize,
    price_change_threshold: f64,
    scale: f64,
    recent_prices: VecDeque<f64>,
}

impl MomentumScalping {
    pub fn new(trade_window_size: usize, price_change_threshold: f64, scale: f64) -> Self {
        Self {
            trade_window_size,
            price_change_threshold,
            scale,
            recent_prices: VecDeque::new(),
        }
    }
}

fn std_dev(prices: &VecDeque<f64>) -> f64 {
    let mean = prices.iter().copied().sum::<f64>() / prices.len() as f64;
    let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
    variance.sqrt()
}

#[async_trait]
impl Strategy for MomentumScalping {
    fn get_info(&self) -> String {
        format!("Momentum Scalping Strategy (trade_window_size: {}, price_change_threshold: {}, scale: {})", self.trade_window_size, self.price_change_threshold, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        self.recent_prices.push_back(price);
        if self.recent_prices.len() > self.trade_window_size {
            self.recent_prices.pop_front();
        }
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.recent_prices.len() < self.trade_window_size {
            return (Signal::Hold, 0.0);
        }

        let first_price = self.recent_prices.front().unwrap_or(&0.0);
        let last_price = self.recent_prices.back().unwrap_or(&0.0);

        let price_change = last_price - first_price;

        let signal: Signal;
        let confidence: f64;

        let std = std_dev(&self.recent_prices);
        if std == 0.0 {
            return (Signal::Hold, 0.0);
        }

        let normalized_price_change = price_change / std;
        let normalized_threshold = self.price_change_threshold / std;

        if normalized_price_change > normalized_threshold {
            signal = Signal::Buy;
            confidence = scale_from_threshold(normalized_price_change, normalized_threshold, self.scale);
        } else if normalized_price_change < -normalized_threshold {
            signal = Signal::Sell;
            confidence = scale_from_threshold_inverse(normalized_price_change, -normalized_threshold, self.scale);
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }
        (signal, confidence)
    }
}
