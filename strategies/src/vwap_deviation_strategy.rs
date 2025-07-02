use crate::models::{Kline, TradeData};
use crate::position::Position;
use crate::strategy::Strategy;
use crate::trader::Signal;
use async_trait::async_trait;
use std::collections::VecDeque;

pub struct VwapDeviationStrategy {
    period: usize,
    deviation_threshold: f64,
    klines: VecDeque<Kline>,
}

impl VwapDeviationStrategy {
    pub fn new(period: usize, deviation_threshold: f64) -> Self {
        Self {
            period,
            deviation_threshold,
            klines: VecDeque::new(),
        }
    }

    fn calculate_vwap(&self) -> f64 {
        if self.klines.len() < self.period {
            return 0.0;
        }

        let mut typical_price_volume_sum = 0.0;
        let mut total_volume = 0.0;

        for kline in self.klines.iter().rev().take(self.period) {
            let high = kline.high_price.parse::<f64>().unwrap_or_default();
            let low = kline.low_price.parse::<f64>().unwrap_or_default();
            let close = kline.close_price.parse::<f64>().unwrap_or_default();
            let volume = kline.volume.parse::<f64>().unwrap_or_default();

            let typical_price = (high + low + close) / 3.0;
            typical_price_volume_sum += typical_price * volume;
            total_volume += volume;
        }

        if total_volume == 0.0 {
            0.0
        } else {
            typical_price_volume_sum / total_volume
        }
    }
}

#[async_trait]
impl Strategy for VwapDeviationStrategy {
    async fn on_kline(&mut self, kline: Kline) {
        self.klines.push_back(kline);
        if self.klines.len() > self.period {
            self.klines.pop_front();
        }
    }

    async fn on_trade(&mut self, _trade: TradeData) {
        // Not used in this strategy
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> Signal {
        if self.klines.len() < self.period {
            return Signal::Hold;
        }

        let vwap = self.calculate_vwap();
        if vwap == 0.0 {
            return Signal::Hold;
        }

        let deviation = (current_price - vwap) / vwap;

        if deviation > self.deviation_threshold {
            Signal::Sell // Price is significantly above VWAP
        } else if deviation < -self.deviation_threshold {
            Signal::Buy // Price is significantly below VWAP
        } else {
            Signal::Hold
        }
    }
}
