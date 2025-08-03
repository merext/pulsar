//! # Fractal Approximation Strategy
//!
//! This strategy is based on the concept of fractal geometry and the Hurst Exponent,
//! which is used to measure the long-term memory of a time series. The Hurst Exponent
//! can indicate whether a market is trending, mean-reverting, or moving randomly.
//!
//! - **H > 0.5**: Indicates a trending market (persistent behavior). The strategy will
//!   follow the trend.
//! - **H < 0.5**: Indicates a mean-reverting market (anti-persistent behavior). The
//!   strategy will trade against the recent price movement.
//! - **H = 0.5**: Indicates a random walk, where no particular strategy is favored.

use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use std::collections::VecDeque;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;


#[derive(Clone)]
pub struct FractalApproximationStrategy {
    period: usize,
    prices: VecDeque<f64>,
    // Configuration parameters
    signal_threshold: f64,
    momentum_threshold: f64,
    last_price: f64,
    price_momentum: f64,
}

impl FractalApproximationStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("fractal_approximation_strategy");
        
        let period = config.as_ref().map(|c| c.get_or("period", 50)).unwrap_or(50);
        let signal_threshold = config.as_ref().map(|c| c.get_or("signal_threshold", 0.3)).unwrap_or(0.3);
        let momentum_threshold = config.as_ref().map(|c| c.get_or("momentum_threshold", 0.0001)).unwrap_or(0.0001);

        Self {
            period,
            prices: VecDeque::with_capacity(period + 1),
            signal_threshold,
            momentum_threshold,
            last_price: 0.0,
            price_momentum: 0.0,
        }
    }


}

#[async_trait::async_trait]
impl Strategy for FractalApproximationStrategy {
    fn get_info(&self) -> String {
        format!("Fractal Approximation Strategy (period: {})", self.period)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (price - self.last_price) / self.last_price;
        }
        
        self.prices.push_back(price);
        if self.prices.len() > self.period {
            self.prices.pop_front();
        }
        
        self.last_price = price;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Pure momentum approach like successful strategies (ignore Hurst exponent)
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };
        let momentum_strength = (self.price_momentum * 3000.0).min(1.0);

        let signal: Signal;
        let confidence: f64;

        if self.price_momentum > self.momentum_threshold {
            signal = Signal::Buy;
            confidence = momentum_strength * momentum_factor;
        } else if self.price_momentum < -self.momentum_threshold {
            signal = Signal::Sell;
            confidence = momentum_strength * momentum_factor;
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply signal threshold filter
        if confidence < self.signal_threshold {
            return (Signal::Hold, 0.0);
        }

        (signal, confidence.min(1.0))
    }
}
