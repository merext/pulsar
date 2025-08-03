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

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct ZScoreStrategy {
    period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    scale: f64,
    prices: VecDeque<f64>,
    // Configuration parameters
    signal_threshold: f64,
    momentum_threshold: f64,
    // Performance tracking
    last_price: f64,
    price_momentum: f64,
}

impl ZScoreStrategy {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("zscore_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let period = config.get_or("period", 30);
        let buy_threshold = config.get_or("buy_threshold", -0.000005);
        let sell_threshold = config.get_or("sell_threshold", 0.000005);
        let scale = config.get_or("scale", 1.2);
        let signal_threshold = config.get_or("signal_threshold", 0.1);

        let momentum_threshold = config.get_or("momentum_threshold", 0.00005);

        Self {
            period,
            buy_threshold,
            sell_threshold,
            scale,
            prices: VecDeque::new(),
            signal_threshold,

            momentum_threshold,
            last_price: 0.0,
            price_momentum: 0.0,
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
        if self.prices.len() < self.period {
            return (Signal::Hold, 0.0);
        }

        let (_, std_dev) = self.calculate_mean_and_std_dev();

        if std_dev == 0.0 {
            return (Signal::Hold, 0.0);
        }



        // Pure momentum approach like HFT Ultra Fast (ignore Z-score direction)
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };

        let signal: Signal;
        let confidence: f64;

        // Simple momentum-based signal generation (like successful strategies)
        if self.price_momentum > self.momentum_threshold {
            // Any positive momentum - buy (like HFT Ultra Fast)
            let momentum_strength = (self.price_momentum * 3000.0).min(1.0);
            signal = Signal::Buy;
            confidence = momentum_strength * momentum_factor * self.scale;
        } else if self.price_momentum < -self.momentum_threshold {
            // Any negative momentum - sell (like HFT Ultra Fast)
            let momentum_strength = (self.price_momentum.abs() * 3000.0).min(1.0);
            signal = Signal::Sell;
            confidence = momentum_strength * momentum_factor * self.scale;
        } else {
            // No momentum - hold
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply signal threshold filter
        if confidence < self.signal_threshold {
            return (Signal::Hold, 0.0);
        } else {
            return (signal, confidence);
        }
    }
}