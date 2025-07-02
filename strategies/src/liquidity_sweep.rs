use crate::models::{Kline, TradeData};
use crate::position::Position;
use crate::strategy::Strategy;
use crate::trader::Signal;
use async_trait::async_trait;
use std::collections::VecDeque;

pub struct LiquiditySweep {
    window_size: usize,
    klines: VecDeque<Kline>,
}

impl LiquiditySweep {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            klines: VecDeque::new(),
        }
    }

    fn find_swing_high_low(&self) -> (f64, f64) {
        if self.klines.is_empty() {
            return (0.0, 0.0);
        }

        let mut high = 0.0;
        let mut low = f64::MAX;

        for kline in self.klines.iter() {
            let kline_high = kline.high_price.parse::<f64>().unwrap_or_default();
            let kline_low = kline.low_price.parse::<f64>().unwrap_or_default();

            if kline_high > high {
                high = kline_high;
            }
            if kline_low < low {
                low = kline_low;
            }
        }
        (high, low)
    }
}

#[async_trait]
impl Strategy for LiquiditySweep {
    async fn on_kline(&mut self, kline: Kline) {
        self.klines.push_back(kline);
        if self.klines.len() > self.window_size {
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
        if self.klines.len() < self.window_size {
            return Signal::Hold;
        }

        let (swing_high, swing_low) = self.find_swing_high_low();

        // Check for bullish sweep (price sweeps below swing low and reverses)
        if current_price > swing_low && self.klines.back().unwrap().low_price.parse::<f64>().unwrap_or_default() < swing_low {
            // Simple reversal: current price is above the swept low
            return Signal::Buy;
        }

        // Check for bearish sweep (price sweeps above swing high and reverses)
        if current_price < swing_high && self.klines.back().unwrap().high_price.parse::<f64>().unwrap_or_default() > swing_high {
            // Simple reversal: current price is below the swept high
            return Signal::Sell;
        }

        Signal::Hold
    }
}
