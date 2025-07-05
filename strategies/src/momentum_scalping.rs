//! # Momentum Scalping Strategy
//! 
//! This strategy aims to profit from very short-term price momentum by identifying rapid price changes.
//! It is designed for high-frequency trading (HFT) environments where quick entry and exit are crucial.
//! 
//! The strategy tracks recent price movements within a defined window.
//! A buy signal is generated if the price change over this window exceeds a positive threshold,
//! indicating upward momentum. Conversely, a sell signal is generated if the price change falls below a negative threshold,
//! indicating downward momentum.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use async_trait::async_trait;
use std::collections::VecDeque;

pub struct MomentumScalping {
    trade_window_size: usize,
    price_change_threshold: f64,
    recent_prices: VecDeque<f64>,
}

impl MomentumScalping {
    pub fn new(trade_window_size: usize, price_change_threshold: f64) -> Self {
        Self {
            trade_window_size,
            price_change_threshold,
            recent_prices: VecDeque::new(),
        }
    }
}

#[async_trait]
impl Strategy for MomentumScalping {
    

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
    ) -> Signal {
        if self.recent_prices.len() < self.trade_window_size {
            return Signal::Hold;
        }

        let first_price = self.recent_prices.front().unwrap_or(&0.0);
        let last_price = self.recent_prices.back().unwrap_or(&0.0);

        let price_change = last_price - first_price;

        if price_change > self.price_change_threshold {
            Signal::Buy
        } else if price_change < -self.price_change_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}
