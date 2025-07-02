use crate::models::{Kline, TradeData};
use crate::position::Position;
use crate::strategy::Strategy;
use crate::trader::Signal;
use async_trait::async_trait;
use std::collections::VecDeque;

pub struct RsiStrategy {
    period: usize,
    overbought: f64,
    oversold: f64,
    prices: VecDeque<f64>,
}

impl RsiStrategy {
    pub fn new(period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            period,
            overbought,
            oversold,
            prices: VecDeque::new(),
        }
    }

    fn calculate_rsi(&self) -> f64 {
        if self.prices.len() < self.period + 1 {
            return 50.0; // Neutral RSI if not enough data
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..=self.period {
            let current_price = self.prices[self.prices.len() - i];
            let prev_price = self.prices[self.prices.len() - i - 1];
            let change = current_price - prev_price;

            if change > 0.0 {
                gains += change;
            } else {
                losses -= change; // losses are positive values
            }
        }

        let avg_gain = gains / self.period as f64;
        let avg_loss = losses / self.period as f64;

        if avg_loss == 0.0 {
            return 100.0; // Avoid division by zero, strong uptrend
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

#[async_trait]
impl Strategy for RsiStrategy {
    async fn on_kline(&mut self, kline: Kline) {
        let close_price = kline.close_price.parse::<f64>().unwrap_or_default();
        self.prices.push_back(close_price);
        if self.prices.len() > self.period + 1 {
            self.prices.pop_front();
        }
    }

    async fn on_trade(&mut self, _trade: TradeData) {
        // Not used in this strategy
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> Signal {
        let rsi = self.calculate_rsi();

        if rsi > self.overbought {
            Signal::Sell
        } else if rsi < self.oversold {
            Signal::Buy
        } else {
            Signal::Hold
        }
    }
}
