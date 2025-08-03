//! # Momentum Scalping Strategy
//!
//! This strategy aims to profit from very short-term price momentum by identifying rapid price changes.
//! It is designed for high-frequency trading (HFT) environments where quick entry and exit are crucial.
//!
//! The strategy tracks recent price movements within a defined window.
//! A buy signal is generated if the price change over this window exceeds a positive threshold,
//! indicating upward momentum. Conversely, a sell signal is generated if the price change falls below a negative threshold,
//! indicating downward momentum.

use crate::config::StrategyConfig;
use crate::strategy::Strategy;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::f64;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;
use toml;

#[derive(Clone)]
pub struct MomentumScalping {
    trade_window_size: usize,
    price_change_threshold: f64,
    scale: f64,
    recent_prices: VecDeque<f64>,
    // Configuration parameters
    signal_threshold: f64,
    momentum_threshold: f64,
    volume_threshold: f64,
    volume_alpha: f64,
    volume_beta: f64,
    avg_volume: f64,
    volume_ratio: f64,
    // Performance tracking
    _trades_made: u32,
    last_price: f64,
    price_momentum: f64,
}

impl MomentumScalping {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("momentum_scalping_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let trade_window_size = config.get_or("trade_window_size", 3);
        let price_change_threshold = config.get_or("price_change_threshold", 0.000005);
        let scale = config.get_or("scale", 1.2);
        let signal_threshold = config.get_or("signal_threshold", 0.3);
        let momentum_threshold = config.get_or("momentum_threshold", 0.0005);
        let volume_threshold = config.get_or("volume_threshold", 1.1);
        let volume_alpha = config.get_or("volume_alpha", 0.8);
        let volume_beta = config.get_or("volume_beta", 0.2);

        Self {
            trade_window_size,
            price_change_threshold,
            scale,
            recent_prices: VecDeque::new(),
            signal_threshold,
            momentum_threshold,
            volume_threshold,
            volume_alpha,
            volume_beta,
            avg_volume: 0.0,
            volume_ratio: 1.0,
            _trades_made: 0,
            last_price: 0.0,
            price_momentum: 0.0,
        }
    }

    #[inline]
    fn update_volume_analysis(&mut self, volume: f64) {
        if self.avg_volume == 0.0 {
            self.avg_volume = volume;
        } else {
            self.avg_volume = self.volume_alpha * self.avg_volume + self.volume_beta * volume;
        }
        self.volume_ratio = volume / self.avg_volume;
    }
}



#[async_trait]
impl Strategy for MomentumScalping {
    fn get_info(&self) -> String {
        format!("Momentum Scalping Strategy (trade_window_size: {}, price_change_threshold: {}, scale: {})", self.trade_window_size, self.price_change_threshold, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        let volume = trade.qty;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (price - self.last_price) / self.last_price;
        }
        
        // Update volume analysis
        self.update_volume_analysis(volume);
        
        // Update price buffer
        self.recent_prices.push_back(price);
        if self.recent_prices.len() > self.trade_window_size {
            self.recent_prices.pop_front();
        }
        
        self.last_price = price;
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

        // Calculate momentum-based signal (simplified and more aggressive)
        let momentum_signal = if self.price_momentum > self.momentum_threshold {
            1.0 // Strong buy signal
        } else if self.price_momentum < -self.momentum_threshold {
            -1.0 // Strong sell signal
        } else {
            0.0 // No momentum signal
        };

        // Volume confirmation
        let volume_signal = if self.volume_ratio > self.volume_threshold { 1.0 } else { 0.0 };

        // Combine signals (momentum-focused like HFT Ultra Fast)
        let buy_score = (momentum_signal * 0.7 + volume_signal * 0.3) * self.scale;
        let sell_score = (-momentum_signal * 0.7 + volume_signal * 0.3) * self.scale;

        // Generate signal with confidence
        if buy_score > self.signal_threshold {
            (Signal::Buy, buy_score.min(1.0))
        } else if sell_score > self.signal_threshold {
            (Signal::Sell, sell_score.min(1.0))
        } else {
            (Signal::Hold, 0.0)
        }
    }
}
