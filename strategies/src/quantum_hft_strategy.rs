//! # Fractal-Based HFT Strategy
//! 
//! A sophisticated trading strategy based on fractal geometry and market structure analysis

use std::collections::VecDeque;
use trade::signal::Signal;
use trade::trader::Position;
use trade::models::TradeData;
use tracing::debug;
use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use async_trait::async_trait;

/// Fractal point representing a local high or low
#[derive(Clone, Debug)]
struct FractalPoint {
    price: f64,
    timestamp: f64,
    fractal_type: FractalType,
    strength: f64,
}

/// Type of fractal point
#[derive(Clone, Debug, PartialEq)]
enum FractalType {
    High,
    Low,
}

/// Fractal-Based HFT Strategy
#[derive(Clone)]
pub struct QuantumHftStrategy {
    config: StrategyConfig,
    price_window: VecDeque<f64>,
    volume_window: VecDeque<f64>,
    timestamp_window: VecDeque<f64>,
    fractal_period: usize,
    fractal_highs: VecDeque<FractalPoint>,
    fractal_lows: VecDeque<FractalPoint>,
    fractal_dimension: f64,
    trend_direction: f64,
    pattern_strength: f64,
    win_rate: f64,
    consecutive_wins: usize,
    consecutive_losses: usize,
}

impl QuantumHftStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("quantum_hft_strategy")
            .expect("Failed to load quantum_hft_strategy configuration");
        
        Self {
            config,
            price_window: VecDeque::with_capacity(200),
            volume_window: VecDeque::with_capacity(200),
            timestamp_window: VecDeque::with_capacity(200),
            fractal_period: 5,
            fractal_highs: VecDeque::with_capacity(50),
            fractal_lows: VecDeque::with_capacity(50),
            fractal_dimension: 1.5,
            trend_direction: 0.0,
            pattern_strength: 0.0,
            win_rate: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
        }
    }

    /// Identify fractal highs and lows
    fn identify_fractals(&mut self) {
        if self.price_window.len() < self.fractal_period * 2 + 1 {
            return;
        }
        
        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        let timestamps: Vec<f64> = self.timestamp_window.iter().cloned().collect();
        
        // Check for fractal highs and lows
        for i in self.fractal_period..(prices.len() - self.fractal_period) {
            let current_price = prices[i];
            let current_timestamp = timestamps[i];
            
            // Check for fractal high
            let mut is_fractal_high = true;
            for j in (i - self.fractal_period)..i {
                if prices[j] >= current_price {
                    is_fractal_high = false;
                    break;
                }
            }
            for j in (i + 1)..=(i + self.fractal_period) {
                if prices[j] >= current_price {
                    is_fractal_high = false;
                    break;
                }
            }
            
            if is_fractal_high {
                let strength = self.calculate_fractal_strength(&prices, i, FractalType::High);
                let fractal = FractalPoint {
                    price: current_price,
                    timestamp: current_timestamp,
                    fractal_type: FractalType::High,
                    strength,
                };
                
                self.fractal_highs.push_back(fractal);
                if self.fractal_highs.len() > 50 {
                    self.fractal_highs.pop_front();
                }
            }
            
            // Check for fractal low
            let mut is_fractal_low = true;
            for j in (i - self.fractal_period)..i {
                if prices[j] <= current_price {
                    is_fractal_low = false;
                    break;
                }
            }
            for j in (i + 1)..=(i + self.fractal_period) {
                if prices[j] <= current_price {
                    is_fractal_low = false;
                    break;
                }
            }
            
            if is_fractal_low {
                let strength = self.calculate_fractal_strength(&prices, i, FractalType::Low);
                let fractal = FractalPoint {
                    price: current_price,
                    timestamp: current_timestamp,
                    fractal_type: FractalType::Low,
                    strength,
                };
                
                self.fractal_lows.push_back(fractal);
                if self.fractal_lows.len() > 50 {
                    self.fractal_lows.pop_front();
                }
            }
        }
    }

    /// Calculate fractal strength based on surrounding price action
    fn calculate_fractal_strength(&self, prices: &[f64], index: usize, fractal_type: FractalType) -> f64 {
        let current_price = prices[index];
        let mut strength = 0.0;
        
        // Calculate strength based on price differences
        for i in (index.saturating_sub(self.fractal_period))..index {
            match fractal_type {
                FractalType::High => {
                    strength += (current_price - prices[i]) / current_price;
                }
                FractalType::Low => {
                    strength += (prices[i] - current_price) / current_price;
                }
            }
        }
        
        for i in (index + 1)..=(index + self.fractal_period).min(prices.len() - 1) {
            match fractal_type {
                FractalType::High => {
                    strength += (current_price - prices[i]) / current_price;
                }
                FractalType::Low => {
                    strength += (prices[i] - current_price) / current_price;
                }
            }
        }
        
        strength / (self.fractal_period * 2) as f64
    }

    /// Calculate trend direction based on fractals
    fn calculate_trend_direction(&mut self) {
        if self.fractal_highs.len() < 2 || self.fractal_lows.len() < 2 {
            self.trend_direction = 0.0;
            return;
        }
        
        // Get recent fractals
        let recent_highs: Vec<&FractalPoint> = self.fractal_highs.iter().rev().take(3).collect();
        let recent_lows: Vec<&FractalPoint> = self.fractal_lows.iter().rev().take(3).collect();
        
        let mut trend_score: f64 = 0.0;
        
        // Analyze high trend
        if recent_highs.len() >= 2 {
            for i in 1..recent_highs.len() {
                if recent_highs[i - 1].price > recent_highs[i].price {
                    trend_score += 0.5;
                } else {
                    trend_score -= 0.5;
                }
            }
        }
        
        // Analyze low trend
        if recent_lows.len() >= 2 {
            for i in 1..recent_lows.len() {
                if recent_lows[i - 1].price > recent_lows[i].price {
                    trend_score += 0.5;
                } else {
                    trend_score -= 0.5;
                }
            }
        }
        
        self.trend_direction = trend_score.max(-1.0).min(1.0);
    }

    /// Calculate pattern strength based on fractal analysis
    fn calculate_pattern_strength(&mut self) {
        let mut strength = 0.0;
        
        // Strength from trend consistency
        strength += self.trend_direction.abs() * 0.5;
        
        // Strength from fractal count
        let fractal_count = self.fractal_highs.len() + self.fractal_lows.len();
        strength += (fractal_count as f64 * 0.02).min(0.3);
        
        // Strength from average fractal strength
        let avg_fractal_strength = if !self.fractal_highs.is_empty() && !self.fractal_lows.is_empty() {
            let high_strength: f64 = self.fractal_highs.iter().map(|f| f.strength).sum();
            let low_strength: f64 = self.fractal_lows.iter().map(|f| f.strength).sum();
            (high_strength + low_strength) / (self.fractal_highs.len() + self.fractal_lows.len()) as f64
        } else {
            0.0
        };
        strength += avg_fractal_strength * 0.2;
        
        self.pattern_strength = strength.max(0.0).min(1.0);
    }

    /// Should enter long position based on fractal analysis
    fn should_enter_long(&self, current_price: f64) -> bool {
        let pattern_threshold = self.config.get_or("pattern_threshold", 0.4);
        
        // Enter long on:
        // 1. Positive trend direction
        // 2. Good pattern strength
        // 3. Price above recent fractal low
        
        if self.trend_direction > 0.3 && self.pattern_strength > pattern_threshold {
            // Check if we're above recent fractal lows
            if let Some(recent_low) = self.fractal_lows.back() {
                if current_price > recent_low.price * 1.001 { // 0.1% above
                    return true;
                }
            }
        }
        
        false
    }

    /// Should enter short position based on fractal analysis
    fn should_enter_short(&self, current_price: f64) -> bool {
        let pattern_threshold = self.config.get_or("pattern_threshold", 0.4);
        
        // Enter short on:
        // 1. Negative trend direction
        // 2. Good pattern strength
        // 3. Price below recent fractal high
        
        if self.trend_direction < -0.3 && self.pattern_strength > pattern_threshold {
            // Check if we're below recent fractal highs
            if let Some(recent_high) = self.fractal_highs.back() {
                if current_price < recent_high.price * 0.999 { // 0.1% below
                    return true;
                }
            }
        }
        
        false
    }

    /// Should exit long position
    fn should_exit_long(&self, _current_price: f64) -> bool {
        // Exit when trend turns negative
        self.trend_direction < -0.2
    }

    /// Should exit short position
    fn should_exit_short(&self, _current_price: f64) -> bool {
        // Exit when trend turns positive
        self.trend_direction > 0.2
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, trade_result: f64) {
        if trade_result > 0.0 {
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
        } else {
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
        }
        
        let total_trades = self.consecutive_wins + self.consecutive_losses;
        if total_trades > 0 {
            self.win_rate = self.consecutive_wins as f64 / total_trades as f64;
        }
    }
}

#[async_trait]
impl Strategy for QuantumHftStrategy {
    fn get_info(&self) -> String {
        "Fractal-Based HFT Strategy - Geometric Pattern Recognition".to_string()
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price, volume, and timestamp windows
        self.price_window.push_back(trade.price);
        self.volume_window.push_back(trade.qty);
        self.timestamp_window.push_back(trade.time as f64);
        
        // Keep windows at capacity
        if self.price_window.len() > self.price_window.capacity() {
            self.price_window.pop_front();
        }
        if self.volume_window.len() > self.volume_window.capacity() {
            self.volume_window.pop_front();
        }
        if self.timestamp_window.len() > self.timestamp_window.capacity() {
            self.timestamp_window.pop_front();
        }
        
        // Perform fractal analysis when we have enough data
        if self.price_window.len() >= self.fractal_period * 2 + 10 {
            self.identify_fractals();
            self.calculate_trend_direction();
            self.calculate_pattern_strength();
            
            debug!("Fractal Analysis: trend={:.3}, pattern={:.3}, highs={}, lows={}", 
                   self.trend_direction, self.pattern_strength, 
                   self.fractal_highs.len(), self.fractal_lows.len());
        }
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        if self.price_window.len() < self.fractal_period * 2 + 10 {
            return (Signal::Hold, 0.0);
        }
        
        // Calculate signal confidence based on fractal analysis
        let confidence = self.pattern_strength.min(1.0);
        
        // Position management
        if current_position.quantity > 0.0 {
            // We have a long position
            if self.should_exit_long(current_price) {
                debug!("Fractal Long Exit: trend={:.3}", self.trend_direction);
                return (Signal::Sell, confidence);
            }
        } else if current_position.quantity < 0.0 {
            // We have a short position
            if self.should_exit_short(current_price) {
                debug!("Fractal Short Exit: trend={:.3}", self.trend_direction);
                return (Signal::Buy, confidence);
            }
        } else {
            // No position - look for entry signals
            if self.should_enter_long(current_price) {
                debug!("Fractal Long Entry: trend={:.3}, pattern={:.3}", 
                       self.trend_direction, self.pattern_strength);
                return (Signal::Buy, confidence);
            } else if self.should_enter_short(current_price) {
                debug!("Fractal Short Entry: trend={:.3}, pattern={:.3}", 
                       self.trend_direction, self.pattern_strength);
                return (Signal::Sell, confidence);
            }
            
            // Fallback: generate signals based on simple fractal patterns
            if self.pattern_strength > 0.3 && self.trend_direction > 0.2 {
                debug!("Fractal Fallback Long: pattern={:.3}, trend={:.3}", 
                       self.pattern_strength, self.trend_direction);
                return (Signal::Buy, confidence * 0.8);
            } else if self.pattern_strength > 0.3 && self.trend_direction < -0.2 {
                debug!("Fractal Fallback Short: pattern={:.3}, trend={:.3}", 
                       self.pattern_strength, self.trend_direction);
                return (Signal::Sell, confidence * 0.8);
            }
        }
        
        (Signal::Hold, 0.0)
    }
}

impl QuantumHftStrategy {
    pub fn on_trade_result(&mut self, result: f64) {
        self.update_performance_metrics(result);
    }
}
