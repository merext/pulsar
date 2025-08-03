//! # Refined RSI Strategy
//! 
//! Minimal enhancements to the original RSI strategy:
//! 1. Volume confirmation (simple)
//! 2. Basic trend alignment
//! 3. Dynamic position sizing
//! 4. Improved risk management
//! 
//! Keeps the core momentum logic that works!

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct RefinedRsiStrategy {
    // Core RSI parameters (from original)
    period: usize,
    overbought: f64,
    oversold: f64,
    scale: f64,
    signal_threshold: f64,
    momentum_threshold: f64,
    
    // Minimal enhancements
    volume_period: usize,
    volume_threshold: f64,
    trend_period: usize,
    
    // Data storage
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    
    // State tracking
    last_price: f64,
    price_momentum: f64,
    volume_ratio: f64,
    trend_direction: i8,
}

impl RefinedRsiStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("refined_rsi_strategy")
            .unwrap_or_else(|_| {
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        // Core parameters (optimized from original)
        let period = config.get_or("period", 4);
        let overbought = config.get_or("overbought", 68.0);
        let oversold = config.get_or("oversold", 32.0);
        let scale = config.get_or("scale", 3.5);
        let signal_threshold = config.get_or("signal_threshold", 0.1);
        let momentum_threshold = config.get_or("momentum_threshold", 0.00003);
        
        // Minimal enhancements
        let volume_period = config.get_or("volume_period", 8);
        let volume_threshold = config.get_or("volume_threshold", 1.0);
        let trend_period = config.get_or("trend_period", 12);

        Self {
            period,
            overbought,
            oversold,
            scale,
            signal_threshold,
            momentum_threshold,
            volume_period,
            volume_threshold,
            trend_period,
            prices: VecDeque::new(),
            volumes: VecDeque::new(),
            last_price: 0.0,
            price_momentum: 0.0,
            volume_ratio: 1.0,
            trend_direction: 0,
        }
    }

    fn calculate_rsi(&self) -> f64 {
        if self.prices.len() < self.period + 1 {
            return 50.0;
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
                losses -= change;
            }
        }

        let avg_gain = gains / self.period as f64;
        let avg_loss = losses / self.period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_volume_ratio(&self) -> f64 {
        if self.volumes.len() < self.volume_period {
            return 1.0;
        }
        
        let current_volume = self.volumes[self.volumes.len() - 1];
        let avg_volume: f64 = self.volumes.iter().rev().take(self.volume_period).sum::<f64>() / self.volume_period as f64;
        
        current_volume / avg_volume
    }

    fn calculate_trend_direction(&self) -> i8 {
        if self.prices.len() < self.trend_period {
            return 0;
        }
        
        let recent_prices: Vec<f64> = self.prices.iter().rev().take(self.trend_period).cloned().collect();
        let first_half: Vec<f64> = recent_prices.iter().take(self.trend_period / 2).cloned().collect();
        let second_half: Vec<f64> = recent_prices.iter().skip(self.trend_period / 2).cloned().collect();
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        let trend_strength = (second_avg - first_avg) / first_avg;
        
        if trend_strength > 0.0005 { 1 }
        else if trend_strength < -0.0005 { -1 }
        else { 0 }
    }
}

#[async_trait::async_trait]
impl Strategy for RefinedRsiStrategy {
    fn get_info(&self) -> String {
        format!("Refined RSI Strategy (period: {}, overbought: {}, oversold: {}, scale: {})", 
                self.period, self.overbought, self.oversold, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (trade.price - self.last_price) / self.last_price;
        }
        
        self.prices.push_back(trade.price);
        self.volumes.push_back(trade.qty);
        
        // Keep only necessary data
        let max_len = self.period.max(self.volume_period).max(self.trend_period) + 10;
        if self.prices.len() > max_len {
            self.prices.pop_front();
            self.volumes.pop_front();
        }
        
        // Update metrics
        self.volume_ratio = self.calculate_volume_ratio();
        self.trend_direction = self.calculate_trend_direction();
        
        self.last_price = trade.price;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        let rsi = self.calculate_rsi();

        // Core momentum logic (from original strategy)
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };
        
        // Simple volume confirmation
        let volume_factor = if self.volume_ratio > self.volume_threshold { 1.2 } else { 1.0 };
        
        // Simple trend alignment
        let trend_factor = if (self.price_momentum > 0.0 && self.trend_direction == 1) || 
                              (self.price_momentum < 0.0 && self.trend_direction == -1) { 1.3 } else { 0.9 };
        
        let signal: Signal;
        let confidence: f64;

        // Pure momentum strategy with minimal enhancements
        if self.price_momentum > 0.00005 {
            signal = Signal::Buy;
            let momentum_strength = (self.price_momentum * 2000.0).min(1.0);
            let rsi_confirmation = if rsi > 50.0 { 1.2 } else { 0.8 };
            confidence = momentum_strength * rsi_confirmation * momentum_factor * volume_factor * trend_factor * self.scale;
        } else if self.price_momentum < -0.00005 {
            signal = Signal::Sell;
            let momentum_strength = (self.price_momentum.abs() * 2000.0).min(1.0);
            let rsi_confirmation = if rsi < 50.0 { 1.2 } else { 0.8 };
            confidence = momentum_strength * rsi_confirmation * momentum_factor * volume_factor * trend_factor * self.scale;
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply signal threshold
        if confidence < self.signal_threshold {
            return (Signal::Hold, 0.0);
        }

        (signal, confidence)
    }
} 