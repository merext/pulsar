//! # Trend-Following Volume-Weighted Strategy
//! 
//! Specifically designed for trending markets like DOGEUSDT:
//! 1. Trend-following instead of mean-reversion
//! 2. Volume-weighted signals for better quality
//! 3. Adaptive thresholds based on market conditions
//! 4. Position scaling based on trend strength
//! 5. Reduced frequency but higher quality trades

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct TrendVolumeStrategy {
    // Trend parameters
    trend_period: usize,
    trend_threshold: f64,
    
    // Volume parameters
    volume_period: usize,
    volume_threshold: f64,
    
    // Momentum parameters
    momentum_period: usize,
    momentum_threshold: f64,
    
    // Signal parameters
    signal_threshold: f64,
    min_trend_strength: f64,
    
    // Risk management
    max_position_size: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    
    // Data storage
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    
    // State tracking
    current_trend: f64,
    trend_direction: i8, // -1: downtrend, 0: sideways, 1: uptrend
    volume_ratio: f64,
    last_signal_time: f64,
}

impl TrendVolumeStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("trend_volume_strategy")
            .unwrap_or_else(|_| {
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let trend_period = config.get_or("trend_period", 20);
        let trend_threshold = config.get_or("trend_threshold", 0.001);
        let volume_period = config.get_or("volume_period", 15);
        let volume_threshold = config.get_or("volume_threshold", 1.2);
        let momentum_period = config.get_or("momentum_period", 5);
        let momentum_threshold = config.get_or("momentum_threshold", 0.0001);
        let signal_threshold = config.get_or("signal_threshold", 0.3);
        let min_trend_strength = config.get_or("min_trend_strength", 0.5);
        let max_position_size = config.get_or("max_position_size", 1000.0);
        let stop_loss_pct = config.get_or("stop_loss_pct", 0.02);
        let take_profit_pct = config.get_or("take_profit_pct", 0.04);

        Self {
            trend_period,
            trend_threshold,
            volume_period,
            volume_threshold,
            momentum_period,
            momentum_threshold,
            signal_threshold,
            min_trend_strength,
            max_position_size,
            stop_loss_pct,
            take_profit_pct,
            prices: VecDeque::new(),
            volumes: VecDeque::new(),
            current_trend: 0.0,
            trend_direction: 0,
            volume_ratio: 1.0,
            last_signal_time: 0.0,
        }
    }

    fn calculate_trend(&self) -> (f64, i8) {
        if self.prices.len() < self.trend_period {
            return (0.0, 0);
        }
        
        let recent_prices: Vec<f64> = self.prices.iter().rev().take(self.trend_period).cloned().collect();
        let first_half: Vec<f64> = recent_prices.iter().take(self.trend_period / 2).cloned().collect();
        let second_half: Vec<f64> = recent_prices.iter().skip(self.trend_period / 2).cloned().collect();
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        let trend_strength = (second_avg - first_avg) / first_avg;
        let direction = if trend_strength > self.trend_threshold { 1 }
                       else if trend_strength < -self.trend_threshold { -1 }
                       else { 0 };
        
        (trend_strength, direction)
    }

    fn calculate_volume_ratio(&self) -> f64 {
        if self.volumes.len() < self.volume_period {
            return 1.0;
        }
        
        let current_volume = self.volumes[self.volumes.len() - 1];
        let avg_volume: f64 = self.volumes.iter().rev().take(self.volume_period).sum::<f64>() / self.volume_period as f64;
        
        current_volume / avg_volume
    }

    fn calculate_momentum(&self) -> f64 {
        if self.prices.len() < self.momentum_period + 1 {
            return 0.0;
        }
        
        let current_price = self.prices[self.prices.len() - 1];
        let past_price = self.prices[self.prices.len() - self.momentum_period - 1];
        
        (current_price - past_price) / past_price
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
impl Strategy for TrendVolumeStrategy {
    fn get_info(&self) -> String {
        format!("Trend Volume Strategy (trend: {}, volume: {}, momentum: {})", 
                self.trend_period, self.volume_period, self.momentum_period)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.prices.push_back(trade.price);
        self.volumes.push_back(trade.qty);
        
        // Keep only necessary data
        let max_len = self.trend_period.max(self.volume_period).max(self.momentum_period) + 10;
        if self.prices.len() > max_len {
            self.prices.pop_front();
            self.volumes.pop_front();
        }
        
        // Update trend and volume
        let (trend_strength, direction) = self.calculate_trend();
        self.current_trend = trend_strength;
        self.trend_direction = direction;
        self.volume_ratio = self.calculate_volume_ratio();
    }

    fn get_signal(
        &self,
        current_price: f64,
        current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        if self.prices.len() < self.trend_period {
            return (Signal::Hold, 0.0);
        }

        // Check stop loss and take profit
        if self.should_stop_loss(current_position.entry_price, current_price, &current_position) {
            return (Signal::Sell, 1.0);
        }
        
        if self.should_take_profit(current_position.entry_price, current_price, &current_position) {
            return (Signal::Sell, 1.0);
        }

        let momentum = self.calculate_momentum();
        let trend_strength = self.current_trend.abs();
        
        // Signal generation based on trend-following logic
        let signal: Signal;
        let confidence: f64;
        
        // Strong uptrend with volume confirmation
        if self.trend_direction == 1 && 
           trend_strength > self.min_trend_strength &&
           momentum > self.momentum_threshold &&
           self.volume_ratio > self.volume_threshold {
            
            signal = Signal::Buy;
            confidence = (trend_strength * 100.0).min(1.0) * 
                        (self.volume_ratio / self.volume_threshold).min(1.5) * 
                        (momentum / self.momentum_threshold).min(2.0);
                        
        // Strong downtrend with volume confirmation
        } else if self.trend_direction == -1 && 
                  trend_strength > self.min_trend_strength &&
                  momentum < -self.momentum_threshold &&
                  self.volume_ratio > self.volume_threshold {
            
            signal = Signal::Sell;
            confidence = (trend_strength * 100.0).min(1.0) * 
                        (self.volume_ratio / self.volume_threshold).min(1.5) * 
                        (momentum.abs() / self.momentum_threshold).min(2.0);
                        
        // Moderate trend signals
        } else if self.trend_direction == 1 && 
                  momentum > self.momentum_threshold * 0.5 &&
                  self.volume_ratio > 1.0 {
            
            signal = Signal::Buy;
            confidence = (trend_strength * 50.0).min(0.8) * 
                        (self.volume_ratio).min(1.2);
            
        } else if self.trend_direction == -1 && 
                  momentum < -self.momentum_threshold * 0.5 &&
                  self.volume_ratio > 1.0 {
            
            signal = Signal::Sell;
            confidence = (trend_strength * 50.0).min(0.8) * 
                        (self.volume_ratio).min(1.2);
            
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply signal threshold
        if confidence < self.signal_threshold {
            return (Signal::Hold, 0.0);
        }

        // Prevent rapid signal changes
        let time_since_last_signal = current_timestamp - self.last_signal_time;
        if time_since_last_signal < 2.0 && signal != Signal::Hold {
            return (Signal::Hold, 0.0);
        }

        (signal, confidence)
    }
} 