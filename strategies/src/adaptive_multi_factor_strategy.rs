//! # Adaptive Multi-Factor Strategy
//! 
//! This is a sophisticated trading strategy that combines multiple advanced techniques:
//! - Adaptive volatility bands (Bollinger Bands with dynamic width)
//! - Volume-weighted price analysis (VWAP with momentum)
//! - Momentum convergence/divergence detection
//! - Machine learning-inspired pattern recognition
//! - Risk-adjusted position sizing
//! - Market regime detection
//! 
//! The strategy adapts to different market conditions and uses ensemble methods
//! to generate more reliable signals than single-indicator approaches.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use std::f64;

#[derive(Clone, Debug)]
pub struct AdaptiveMultiFactorStrategy {
    // Core parameters
    short_window: usize,
    long_window: usize,
    volatility_window: usize,
    volume_window: usize,
    
    // Price data
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    timestamps: VecDeque<u64>,
    
    // Technical indicators
    short_sma: Option<f64>,
    long_sma: Option<f64>,
    vwap: Option<f64>,
    volatility: Option<f64>,
    
    // Adaptive parameters
    volatility_multiplier: f64,
    momentum_threshold: f64,
    volume_threshold: f64,
    

    
    // Market regime detection
    trend_strength: f64,
    volatility_regime: VolatilityRegime,
    
    // Signal history for pattern recognition
    signal_history: VecDeque<Signal>,
    confidence_history: VecDeque<f64>,
    
    // Performance tracking
    consecutive_losses: u32,
    total_trades: u32,
    win_rate: f64,
}

#[derive(Clone, Debug, PartialEq)]
enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

impl AdaptiveMultiFactorStrategy {
    pub fn new(
        short_window: usize,
        long_window: usize,
        volatility_window: usize,
        volume_window: usize,
    ) -> Self {
        Self {
            short_window,
            long_window,
            volatility_window,
            volume_window,
            prices: VecDeque::with_capacity(long_window.max(volatility_window)),
            volumes: VecDeque::with_capacity(volume_window),
            timestamps: VecDeque::with_capacity(long_window),
            short_sma: None,
            long_sma: None,
            vwap: None,
            volatility: None,
            volatility_multiplier: 2.0,
            momentum_threshold: 0.02, // 2% price change
            volume_threshold: 1.5, // 50% above average volume

            trend_strength: 0.0,
            volatility_regime: VolatilityRegime::Medium,
            signal_history: VecDeque::with_capacity(100),
            confidence_history: VecDeque::with_capacity(100),
            consecutive_losses: 0,
            total_trades: 0,
            win_rate: 0.0,
        }
    }

    fn calculate_sma(&self, window: usize) -> Option<f64> {
        if self.prices.len() < window {
            return None;
        }
        let sum: f64 = self.prices.iter().rev().take(window).sum();
        Some(sum / window as f64)
    }

    fn calculate_vwap(&self) -> Option<f64> {
        if self.prices.len() < self.volume_window || self.volumes.len() < self.volume_window {
            return None;
        }
        
        let mut total_volume = 0.0;
        let mut volume_price_sum = 0.0;
        
        for (price, volume) in self.prices.iter().rev().take(self.volume_window)
            .zip(self.volumes.iter().rev().take(self.volume_window)) {
            total_volume += volume;
            volume_price_sum += price * volume;
        }
        
        if total_volume > 0.0 {
            Some(volume_price_sum / total_volume)
        } else {
            None
        }
    }

    fn calculate_volatility(&self) -> Option<f64> {
        if self.prices.len() < self.volatility_window {
            return None;
        }
        
        let returns: Vec<f64> = self.prices.iter()
            .rev()
            .take(self.volatility_window)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|window| (window[0] - window[1]) / window[1])
            .collect();
        
        if returns.is_empty() {
            return None;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        Some(variance.sqrt() * (252.0_f64).sqrt()) // Annualized volatility
    }

    fn detect_volatility_regime(&mut self) {
        if let Some(vol) = self.volatility {
            self.volatility_regime = if vol < 0.15 {
                VolatilityRegime::Low
            } else if vol < 0.30 {
                VolatilityRegime::Medium
            } else if vol < 0.50 {
                VolatilityRegime::High
            } else {
                VolatilityRegime::Extreme
            };
        }
    }

    fn calculate_trend_strength(&mut self) {
        if let (Some(short), Some(long)) = (self.short_sma, self.long_sma) {
            let price = self.prices.back().unwrap_or(&0.0);
            let sma_diff = (short - long) / long;
            let price_vs_short = (price - short) / short;
            
            // Trend strength combines SMA divergence and price momentum
            self.trend_strength = (sma_diff + price_vs_short) / 2.0;
        }
    }

    fn calculate_momentum_score(&self) -> f64 {
        if self.prices.len() < 10 {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.prices.iter().rev().take(10).cloned().collect();
        let mut momentum = 0.0;
        
        for i in 1..recent_prices.len() {
            let change = (recent_prices[i-1] - recent_prices[i]) / recent_prices[i];
            momentum += change * (i as f64); // Weight recent changes more heavily
        }
        
        momentum / recent_prices.len() as f64
    }

    fn calculate_volume_score(&self) -> f64 {
        if self.volumes.len() < self.volume_window {
            return 0.0;
        }
        
        let avg_volume: f64 = self.volumes.iter().sum::<f64>() / self.volumes.len() as f64;
        let current_volume = self.volumes.back().unwrap_or(&0.0);
        
        if avg_volume > 0.0 {
            (current_volume / avg_volume - 1.0).max(0.0)
        } else {
            0.0
        }
    }

    fn detect_patterns(&self) -> f64 {
        if self.signal_history.len() < 5 {
            return 0.0;
        }
        
        let recent_signals: Vec<&Signal> = self.signal_history.iter().rev().take(5).collect();
        
        // Pattern 1: Signal reversal (good for mean reversion)
        let reversals = recent_signals.windows(2)
            .filter(|window| window[0] != window[1])
            .count();
        
        // Pattern 2: Strong signal sequence (good for momentum)
        let strong_signals = recent_signals.iter()
            .zip(self.confidence_history.iter().rev().take(5))
            .filter(|&(_, &conf)| conf > 0.7)
            .count();
        
        // Pattern 3: Signal consistency
        let consistent_signals = if recent_signals.len() >= 3 {
            let last_three: Vec<&Signal> = recent_signals.iter().take(3).cloned().collect();
            if last_three.iter().all(|&s| s == last_three[0]) {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        (reversals as f64 * 0.3 + strong_signals as f64 * 0.4 + consistent_signals * 0.3) / 5.0
    }

    fn calculate_risk_adjusted_confidence(&self, base_confidence: f64) -> f64 {
        // Adjust confidence based on market conditions and risk metrics
        let mut adjusted_confidence = base_confidence;
        
        // Volatility adjustment
        match self.volatility_regime {
            VolatilityRegime::Low => adjusted_confidence *= 1.2,
            VolatilityRegime::Medium => adjusted_confidence *= 1.0,
            VolatilityRegime::High => adjusted_confidence *= 0.8,
            VolatilityRegime::Extreme => adjusted_confidence *= 0.6,
        }
        
        // Win rate adjustment
        if self.total_trades > 10 {
            let win_rate_factor = if self.win_rate > 0.6 { 1.1 } else if self.win_rate < 0.4 { 0.9 } else { 1.0 };
            adjusted_confidence *= win_rate_factor;
        }
        
        // Consecutive losses penalty
        if self.consecutive_losses > 3 {
            adjusted_confidence *= 0.8_f64.powi(self.consecutive_losses as i32 - 3);
        }
        
        adjusted_confidence.max(0.0).min(1.0)
    }

    fn generate_ensemble_signal(&self, current_price: f64) -> (Signal, f64) {
        let mut buy_score = 0.0;
        let mut sell_score = 0.0;
        let mut total_weight = 0.0;
        
        // Factor 1: SMA crossover (weight: 0.25)
        if let (Some(short), Some(long)) = (self.short_sma, self.long_sma) {
            let sma_weight = 0.25;
            total_weight += sma_weight;
            
            if short > long && current_price > short {
                buy_score += sma_weight;
            } else if short < long && current_price < short {
                sell_score += sma_weight;
            }
        }
        
        // Factor 2: VWAP deviation (weight: 0.20)
        if let Some(vwap) = self.vwap {
            let vwap_weight = 0.20;
            total_weight += vwap_weight;
            
            let vwap_deviation = (current_price - vwap) / vwap;
            if vwap_deviation < -0.01 { // Price below VWAP
                buy_score += vwap_weight * (0.01 + vwap_deviation).abs() * 100.0;
            } else if vwap_deviation > 0.01 { // Price above VWAP
                sell_score += vwap_weight * vwap_deviation * 100.0;
            }
        }
        
        // Factor 3: Momentum (weight: 0.20)
        let momentum = self.calculate_momentum_score();
        let momentum_weight = 0.20;
        total_weight += momentum_weight;
        
        if momentum > self.momentum_threshold {
            buy_score += momentum_weight;
        } else if momentum < -self.momentum_threshold {
            sell_score += momentum_weight;
        }
        
        // Factor 4: Volume confirmation (weight: 0.15)
        let volume_score = self.calculate_volume_score();
        let volume_weight = 0.15;
        total_weight += volume_weight;
        
        if volume_score > self.volume_threshold {
            if buy_score > sell_score {
                buy_score += volume_weight * volume_score;
            } else {
                sell_score += volume_weight * volume_score;
            }
        }
        
        // Factor 5: Pattern recognition (weight: 0.10)
        let pattern_score = self.detect_patterns();
        let pattern_weight = 0.10;
        total_weight += pattern_weight;
        
        if pattern_score > 0.5 {
            if buy_score > sell_score {
                buy_score += pattern_weight * pattern_score;
            } else {
                sell_score += pattern_weight * pattern_score;
            }
        }
        
        // Factor 6: Trend strength (weight: 0.10)
        let trend_weight = 0.10;
        total_weight += trend_weight;
        
        if self.trend_strength > 0.02 {
            buy_score += trend_weight;
        } else if self.trend_strength < -0.02 {
            sell_score += trend_weight;
        }
        
        // Normalize scores
        let normalized_buy = if total_weight > 0.0 { buy_score / total_weight } else { 0.0 };
        let normalized_sell = if total_weight > 0.0 { sell_score / total_weight } else { 0.0 };
        
        // Generate final signal
        let signal_threshold = 0.6; // Require 60% confidence
        
        if normalized_buy > signal_threshold && normalized_buy > normalized_sell {
            let confidence = self.calculate_risk_adjusted_confidence(normalized_buy);
            (Signal::Buy, confidence)
        } else if normalized_sell > signal_threshold && normalized_sell > normalized_buy {
            let confidence = self.calculate_risk_adjusted_confidence(normalized_sell);
            (Signal::Sell, confidence)
        } else {
            (Signal::Hold, 0.0)
        }
    }


}

#[async_trait::async_trait]
impl Strategy for AdaptiveMultiFactorStrategy {
    fn get_info(&self) -> String {
        format!(
            "Adaptive Multi-Factor Strategy (short: {}, long: {}, vol: {}, vol_window: {}, win_rate: {:.2}%)",
            self.short_window,
            self.long_window,
            self.volatility_window,
            self.volume_window,
            self.win_rate * 100.0
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        let volume = trade.qty;
        let timestamp = trade.time;
        
        // Update price data
        self.prices.push_back(price);
        if self.prices.len() > self.long_window.max(self.volatility_window) {
            self.prices.pop_front();
        }
        
        // Update volume data
        self.volumes.push_back(volume);
        if self.volumes.len() > self.volume_window {
            self.volumes.pop_front();
        }
        
        // Update timestamp data
        self.timestamps.push_back(timestamp);
        if self.timestamps.len() > self.long_window {
            self.timestamps.pop_front();
        }
        
        // Update technical indicators
        self.short_sma = self.calculate_sma(self.short_window);
        self.long_sma = self.calculate_sma(self.long_window);
        self.vwap = self.calculate_vwap();
        self.volatility = self.calculate_volatility();
        
        // Update adaptive parameters
        self.detect_volatility_regime();
        self.calculate_trend_strength();
        
        // Adjust parameters based on market conditions
        match self.volatility_regime {
            VolatilityRegime::Low => {
                self.momentum_threshold = 0.015; // Lower threshold in low volatility
                self.volatility_multiplier = 1.5;
            },
            VolatilityRegime::Medium => {
                self.momentum_threshold = 0.02;
                self.volatility_multiplier = 2.0;
            },
            VolatilityRegime::High => {
                self.momentum_threshold = 0.025; // Higher threshold in high volatility
                self.volatility_multiplier = 2.5;
            },
            VolatilityRegime::Extreme => {
                self.momentum_threshold = 0.03;
                self.volatility_multiplier = 3.0;
            },
        }
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Check if we have enough data
        if self.prices.len() < self.long_window || self.volumes.len() < self.volume_window {
            return (Signal::Hold, 0.0);
        }
        
        // Generate ensemble signal
        let (signal, confidence) = self.generate_ensemble_signal(current_price);
        
        // Additional safety checks
        if let Some(vol) = self.volatility {
            // Don't trade in extremely volatile conditions unless very confident
            if vol > 0.8 && confidence < 0.8 {
                return (Signal::Hold, 0.0);
            }
        }
        
        // Position size adjustment based on confidence
        let adjusted_confidence = if signal != Signal::Hold {
            self.calculate_risk_adjusted_confidence(confidence)
        } else {
            0.0
        };
        
        (signal, adjusted_confidence)
    }
} 