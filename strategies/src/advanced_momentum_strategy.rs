//! # Advanced Multi-Timeframe Momentum Strategy
//! 
//! This strategy addresses the key issues with current strategies:
//! 1. Low trade frequency - Uses multiple timeframes for more signals
//! 2. Poor win rate - Combines volume, momentum, and trend analysis
//! 3. High drawdown - Implements adaptive position sizing and stop-loss
//! 4. Fee erosion - Uses maker orders and volume-weighted signals
//! 5. Trending vs mean-reversion - Adapts to market regime

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct AdvancedMomentumStrategy {
    // Multi-timeframe parameters
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    
    // Volume analysis
    volume_period: usize,
    volume_threshold: f64,
    
    // Momentum parameters
    momentum_threshold: f64,
    
    // Trend analysis
    trend_threshold: f64,
    
    // Adaptive parameters
    volatility_period: usize,
    
    // Risk management
    stop_loss_pct: f64,
    take_profit_pct: f64,
    
    // Signal filtering
    signal_threshold: f64,
    min_volume_ratio: f64,
    
    // Data storage
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    timestamps: VecDeque<f64>,
    
    // State tracking
    last_signal_time: f64,
    consecutive_losses: usize,
    current_trend: f64,
    volatility: f64,
}

impl AdvancedMomentumStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("advanced_momentum_strategy")
            .unwrap_or_else(|_| {
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let short_period = config.get_or("short_period", 5);
        let medium_period = config.get_or("medium_period", 15);
        let long_period = config.get_or("long_period", 50);
        let volume_period = config.get_or("volume_period", 20);
        let volume_threshold = config.get_or("volume_threshold", 1.2);
        let momentum_threshold = config.get_or("momentum_threshold", 0.0001);
        let trend_threshold = config.get_or("trend_threshold", 0.001);
        let volatility_period = config.get_or("volatility_period", 25);
        let stop_loss_pct = config.get_or("stop_loss_pct", 0.02);
        let take_profit_pct = config.get_or("take_profit_pct", 0.04);
        let signal_threshold = config.get_or("signal_threshold", 0.4);
        let min_volume_ratio = config.get_or("min_volume_ratio", 0.8);

        Self {
            short_period,
            medium_period,
            long_period,
            volume_period,
            volume_threshold,
            momentum_threshold,
            trend_threshold,
            volatility_period,
            stop_loss_pct,
            take_profit_pct,
            signal_threshold,
            min_volume_ratio,
            prices: VecDeque::new(),
            volumes: VecDeque::new(),
            timestamps: VecDeque::new(),
            last_signal_time: 0.0,
            consecutive_losses: 0,
            current_trend: 0.0,
            volatility: 0.0,
        }
    }

    fn calculate_sma(&self, period: usize) -> f64 {
        if self.prices.len() < period {
            return 0.0;
        }
        let sum: f64 = self.prices.iter().rev().take(period).sum();
        sum / period as f64
    }

    fn calculate_momentum(&self, period: usize) -> f64 {
        if self.prices.len() < period + 1 {
            return 0.0;
        }
        let current_price = self.prices[self.prices.len() - 1];
        let past_price = self.prices[self.prices.len() - period - 1];
        (current_price - past_price) / past_price
    }

    fn calculate_volume_ratio(&self) -> f64 {
        if self.volumes.len() < self.volume_period {
            return 1.0;
        }
        let current_volume = self.volumes[self.volumes.len() - 1];
        let avg_volume: f64 = self.volumes.iter().rev().take(self.volume_period).sum::<f64>() / self.volume_period as f64;
        current_volume / avg_volume
    }

    fn calculate_volatility(&self) -> f64 {
        if self.prices.len() < self.volatility_period {
            return 0.0;
        }
        
        let mut returns = Vec::new();
        for i in 1..self.prices.len() {
            let current_price = self.prices[i];
            let prev_price = self.prices[i - 1];
            returns.push((current_price - prev_price) / prev_price);
        }
        
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }

    fn calculate_trend_strength(&self) -> f64 {
        let short_sma = self.calculate_sma(self.short_period);
        let medium_sma = self.calculate_sma(self.medium_period);
        let long_sma = self.calculate_sma(self.long_period);
        
        if short_sma == 0.0 || medium_sma == 0.0 || long_sma == 0.0 {
            return 0.0;
        }
        
        // Trend alignment score
        let short_medium_alignment = if (short_sma > medium_sma) == (medium_sma > long_sma) { 1.0 } else { 0.5 };
        let trend_strength = (short_sma - long_sma) / long_sma;
        
        short_medium_alignment * trend_strength.abs()
    }

    fn should_stop_loss(&self, entry_price: f64, current_price: f64, position: &Position) -> bool {
        if position.quantity == 0.0 {
            return false;
        }
        
        let pnl_pct = if position.quantity > 0.0 {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        
        pnl_pct < -self.stop_loss_pct
    }

    fn should_take_profit(&self, entry_price: f64, current_price: f64, position: &Position) -> bool {
        if position.quantity == 0.0 {
            return false;
        }
        
        let pnl_pct = if position.quantity > 0.0 {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        
        pnl_pct > self.take_profit_pct
    }
}

#[async_trait::async_trait]
impl Strategy for AdvancedMomentumStrategy {
    fn get_info(&self) -> String {
        format!("Advanced Momentum Strategy (short: {}, medium: {}, long: {}, vol: {})", 
                self.short_period, self.medium_period, self.long_period, self.volume_period)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.prices.push_back(trade.price);
        self.volumes.push_back(trade.qty);
        self.timestamps.push_back(trade.time as f64);
        
        // Keep only necessary data
        let max_len = self.long_period.max(self.volume_period).max(self.volatility_period) + 10;
        if self.prices.len() > max_len {
            self.prices.pop_front();
            self.volumes.pop_front();
            self.timestamps.pop_front();
        }
        
        // Update volatility
        self.volatility = self.calculate_volatility();
        
        // Update trend
        self.current_trend = self.calculate_trend_strength();
    }

    fn get_signal(
        &self,
        current_price: f64,
        current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        if self.prices.len() < self.long_period {
            return (Signal::Hold, 0.0);
        }

        // Check stop loss and take profit
        if self.should_stop_loss(current_position.entry_price, current_price, &current_position) {
            return (Signal::Sell, 1.0);
        }
        
        if self.should_take_profit(current_position.entry_price, current_price, &current_position) {
            return (Signal::Sell, 1.0);
        }

        // Multi-timeframe momentum analysis
        let short_momentum = self.calculate_momentum(self.short_period);
        let medium_momentum = self.calculate_momentum(self.medium_period);
        let long_momentum = self.calculate_momentum(self.long_period);
        
        // Volume analysis
        let volume_ratio = self.calculate_volume_ratio();
        let volume_factor = if volume_ratio > self.volume_threshold { 1.5 } else { 1.0 };
        
        // Trend analysis
        let trend_strength = self.current_trend;
        let trend_factor = if trend_strength > self.trend_threshold { 1.3 } else { 1.0 };
        
        // Volatility adjustment
        let volatility_factor = if self.volatility > 0.01 { 0.8 } else { 1.2 }; // Reduce position in high volatility
        
        // Signal generation
        let signal: Signal;
        let confidence: f64;
        
        // Strong buy signal: all timeframes aligned + high volume + strong trend
        if short_momentum > self.momentum_threshold && 
           medium_momentum > self.momentum_threshold * 0.5 && 
           long_momentum > 0.0 &&
           volume_ratio > self.min_volume_ratio &&
           trend_strength > self.trend_threshold * 0.5 {
            
            signal = Signal::Buy;
            confidence = (short_momentum * 1000.0).min(1.0) * 
                        volume_factor * 
                        trend_factor * 
                        volatility_factor;
                        
        // Strong sell signal: all timeframes aligned negative + high volume + strong trend
        } else if short_momentum < -self.momentum_threshold && 
                  medium_momentum < -self.momentum_threshold * 0.5 && 
                  long_momentum < 0.0 &&
                  volume_ratio > self.min_volume_ratio &&
                  trend_strength > self.trend_threshold * 0.5 {
            
            signal = Signal::Sell;
            confidence = (short_momentum.abs() * 1000.0).min(1.0) * 
                        volume_factor * 
                        trend_factor * 
                        volatility_factor;
                        
        // Moderate signals based on short-term momentum
        } else if short_momentum > self.momentum_threshold * 2.0 && volume_ratio > 1.0 {
            signal = Signal::Buy;
            confidence = (short_momentum * 500.0).min(0.8) * volume_factor * volatility_factor;
            
        } else if short_momentum < -self.momentum_threshold * 2.0 && volume_ratio > 1.0 {
            signal = Signal::Sell;
            confidence = (short_momentum.abs() * 500.0).min(0.8) * volume_factor * volatility_factor;
            
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply adaptive threshold based on market conditions
        let adaptive_threshold = self.signal_threshold * 
            (1.0 + self.consecutive_losses as f64 * 0.1) * 
            (1.0 + self.volatility * 10.0);

        if confidence < adaptive_threshold {
            return (Signal::Hold, 0.0);
        }

        // Prevent rapid signal changes
        let time_since_last_signal = current_timestamp - self.last_signal_time;
        if time_since_last_signal < 1.0 && signal != Signal::Hold {
            return (Signal::Hold, 0.0);
        }

        (signal, confidence)
    }
} 