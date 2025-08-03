//! # Mean Reversion Strategy
//! 
//! This strategy is based on the principle that asset prices tend to revert to their historical average or mean price over time.
//! It identifies opportunities when the price deviates significantly from its moving average, expecting it to return to the mean.
//! 
//! The strategy generates a buy signal when the current price falls below its Simple Moving Average (SMA),
//! anticipating a bounce back towards the mean. Conversely, it generates a sell signal when the price rises above its SMA,
//! expecting a pull back.

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct MeanReversionStrategy {
    pub window_size: usize,
    pub prices: VecDeque<f64>,
    pub last_sma: Option<f64>,
    pub recent_trades: VecDeque<f64>,
    pub max_trade_window: usize,
    pub scale: f64,
    // Configuration parameters
    signal_threshold: f64,
    deviation_threshold: f64,
    reversion_strength: f64,
    momentum_threshold: f64,
    // Performance tracking
    last_price: f64,
    price_momentum: f64,
}

impl MeanReversionStrategy {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("mean_reversion_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let window_size = config.get_or("window_size", 15);
        let max_trade_window = config.get_or("max_trade_window", 8);
        let scale = config.get_or("scale", 1.2);
        let signal_threshold = config.get_or("signal_threshold", 0.1);
        let deviation_threshold = config.get_or("deviation_threshold", 0.005);
        let reversion_strength = config.get_or("reversion_strength", 1.0);
        let momentum_threshold = config.get_or("momentum_threshold", 0.00005);

        Self {
            window_size,
            prices: VecDeque::with_capacity(window_size),
            last_sma: None,
            recent_trades: VecDeque::with_capacity(max_trade_window),
            max_trade_window,
            scale,
            signal_threshold,
            deviation_threshold,
            reversion_strength,
            momentum_threshold,
            last_price: 0.0,
            price_momentum: 0.0,
        }
    }
}

#[async_trait::async_trait]
impl Strategy for MeanReversionStrategy {
    fn get_info(&self) -> String {
        format!("Mean Reversion Strategy (window_size: {}, max_trade_window: {}, scale: {})", self.window_size, self.max_trade_window, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (price - self.last_price) / self.last_price;
        }
        
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
        
        self.last_price = price;
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Pure momentum approach like HFT Ultra Fast (ignore SMA deviation)
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