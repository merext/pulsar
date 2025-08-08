//! # Smooth Confidence Functions
//!
//! This module provides smooth, adaptive confidence functions that can be used
//! across all trading strategies to generate continuous confidence values
//! based on market conditions, strategy performance, and signal strength.

use std::collections::VecDeque;

/// Smooth confidence function parameters
#[derive(Debug, Clone)]
pub struct ConfidenceConfig {
    /// Base confidence level (0.0 to 1.0)
    pub base_confidence: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Maximum confidence threshold
    pub max_confidence: f64,
    /// Volatility scaling factor
    pub volatility_scale: f64,
    /// Performance scaling factor
    pub performance_scale: f64,
    /// Market condition scaling factor
    pub market_scale: f64,
    /// Signal strength scaling factor
    pub signal_scale: f64,
    /// Decay factor for historical performance
    pub decay_factor: f64,
}

impl Default for ConfidenceConfig {
    fn default() -> Self {
        Self {
            base_confidence: 0.5,
            min_confidence: 0.10,
            max_confidence: 0.95,
            volatility_scale: 0.15,
            performance_scale: 0.25,
            market_scale: 0.15,
            signal_scale: 0.25,
            decay_factor: 0.95,
        }
    }
}

/// Market conditions for confidence calculation
#[derive(Debug, Clone)]
pub struct MarketConditions {
    /// Current volatility (0.0 to 1.0)
    pub volatility: f64,
    /// Market trend strength (-1.0 to 1.0)
    pub trend_strength: f64,
    /// Liquidity score (0.0 to 1.0)
    pub liquidity: f64,
    /// Spread size (0.0 to 1.0)
    pub spread: f64,
    /// Volume relative to average (0.0 to 1.0)
    pub volume_ratio: f64,
}

impl Default for MarketConditions {
    fn default() -> Self {
        Self {
            volatility: 0.5,
            trend_strength: 0.0,
            liquidity: 0.5,
            spread: 0.5,
            volume_ratio: 0.5,
        }
    }
}

/// Strategy performance metrics for confidence calculation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Recent win rate (0.0 to 1.0)
    pub win_rate: f64,
    /// Recent average PnL per trade
    pub avg_pnl: f64,
    /// Number of consecutive wins
    pub consecutive_wins: usize,
    /// Number of consecutive losses
    pub consecutive_losses: usize,
    /// Recent trade count
    pub recent_trades: usize,
    /// Historical performance window
    pub performance_history: VecDeque<f64>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            win_rate: 0.5,
            avg_pnl: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
            recent_trades: 0,
            performance_history: VecDeque::with_capacity(100),
        }
    }
}

/// Signal strength metrics for confidence calculation
#[derive(Debug, Clone)]
pub struct SignalStrength {
    /// Primary signal strength (0.0 to 1.0)
    pub primary_strength: f64,
    /// Secondary signal strength (0.0 to 1.0)
    pub secondary_strength: f64,
    /// Signal consistency (0.0 to 1.0)
    pub consistency: f64,
    /// Signal duration (number of periods)
    pub duration: usize,
    /// Signal divergence from baseline
    pub divergence: f64,
}

impl Default for SignalStrength {
    fn default() -> Self {
        Self {
            primary_strength: 0.5,
            secondary_strength: 0.5,
            consistency: 0.5,
            duration: 1,
            divergence: 0.0,
        }
    }
}

/// Smooth confidence calculator
#[derive(Clone)]
pub struct ConfidenceCalculator {
    config: ConfidenceConfig,
    market_conditions: MarketConditions,
    performance_metrics: PerformanceMetrics,
    signal_strength: SignalStrength,
}

impl ConfidenceCalculator {
    /// Create a new confidence calculator with default configuration
    pub fn new() -> Self {
        Self {
            config: ConfidenceConfig::default(),
            market_conditions: MarketConditions::default(),
            performance_metrics: PerformanceMetrics::default(),
            signal_strength: SignalStrength::default(),
        }
    }

    /// Create a new confidence calculator with custom configuration
    pub fn with_config(config: ConfidenceConfig) -> Self {
        Self {
            config,
            market_conditions: MarketConditions::default(),
            performance_metrics: PerformanceMetrics::default(),
            signal_strength: SignalStrength::default(),
        }
    }

    /// Update market conditions
    pub fn update_market_conditions(&mut self, conditions: MarketConditions) {
        self.market_conditions = conditions;
    }

    /// Update performance metrics
    pub fn update_performance(&mut self, pnl: f64, is_win: bool) {
        // Update performance history
        self.performance_metrics.performance_history.push_back(pnl);
        if self.performance_metrics.performance_history.len() > 100 {
            self.performance_metrics.performance_history.pop_front();
        }

        // Update consecutive wins/losses
        if is_win {
            self.performance_metrics.consecutive_wins += 1;
            self.performance_metrics.consecutive_losses = 0;
        } else {
            self.performance_metrics.consecutive_losses += 1;
            self.performance_metrics.consecutive_wins = 0;
        }

        // Update average PnL with decay
        let history_len = self.performance_metrics.performance_history.len() as f64;
        if history_len > 0.0 {
            let recent_pnl: f64 = self.performance_metrics.performance_history
                .iter()
                .rev()
                .take(20)
                .sum::<f64>() / 20.0_f64.min(history_len);
            
            self.performance_metrics.avg_pnl = self.performance_metrics.avg_pnl * self.config.decay_factor 
                + recent_pnl * (1.0 - self.config.decay_factor);
        }

        self.performance_metrics.recent_trades += 1;
    }

    /// Update signal strength
    pub fn update_signal_strength(&mut self, strength: SignalStrength) {
        self.signal_strength = strength;
    }

    /// Calculate smooth confidence based on all factors
    pub fn calculate_confidence(&self) -> f64 {
        // Base confidence
        let mut confidence = self.config.base_confidence;

        // Market condition adjustment
        let market_adjustment = self.calculate_market_adjustment();
        confidence += market_adjustment * self.config.market_scale;

        // Performance adjustment
        let performance_adjustment = self.calculate_performance_adjustment();
        confidence += performance_adjustment * self.config.performance_scale;

        // Signal strength adjustment
        let signal_adjustment = self.calculate_signal_adjustment();
        confidence += signal_adjustment * self.config.signal_scale;

        // Volatility adjustment
        let volatility_adjustment = self.calculate_volatility_adjustment();
        confidence += volatility_adjustment * self.config.volatility_scale;

        // Apply bounds
        confidence.clamp(self.config.min_confidence, self.config.max_confidence)
    }

    /// Calculate market condition adjustment
    fn calculate_market_adjustment(&self) -> f64 {
        let conditions = &self.market_conditions;
        
        // Favorable conditions: low volatility, high liquidity, low spread, high volume
        let volatility_score = 1.0 - conditions.volatility; // Lower volatility is better
        let liquidity_score = conditions.liquidity;
        let spread_score = 1.0 - conditions.spread; // Lower spread is better
        let volume_score = conditions.volume_ratio;
        
        // Trend strength can be positive or negative
        let trend_score = conditions.trend_strength.abs();
        
        // Weighted average of market conditions
        (volatility_score * 0.25 + liquidity_score * 0.25 + spread_score * 0.2 + 
         volume_score * 0.15 + trend_score * 0.15) - 0.5
    }

    /// Calculate performance adjustment
    fn calculate_performance_adjustment(&self) -> f64 {
        let metrics = &self.performance_metrics;
        
        // Win rate adjustment (0.0 to 0.3)
        let win_rate_adjustment = (metrics.win_rate - 0.5) * 0.6;
        
        // PnL adjustment (normalized)
        let pnl_adjustment = (metrics.avg_pnl * 1000.0).clamp(-0.3, 0.3);
        
        // Consecutive wins/losses adjustment
        let streak_adjustment = if metrics.consecutive_wins > 0 {
            (metrics.consecutive_wins as f64 * 0.02).min(0.2)
        } else if metrics.consecutive_losses > 0 {
            -(metrics.consecutive_losses as f64 * 0.02).min(0.2)
        } else {
            0.0
        };
        
        win_rate_adjustment + pnl_adjustment + streak_adjustment
    }

    /// Calculate signal strength adjustment
    fn calculate_signal_adjustment(&self) -> f64 {
        let signal = &self.signal_strength;
        
        // Primary signal strength (0.0 to 0.4)
        let primary_adjustment = (signal.primary_strength - 0.5) * 0.8;
        
        // Secondary signal strength (0.0 to 0.2)
        let secondary_adjustment = (signal.secondary_strength - 0.5) * 0.4;
        
        // Consistency adjustment (0.0 to 0.2)
        let consistency_adjustment = (signal.consistency - 0.5) * 0.4;
        
        // Duration adjustment (longer signals get higher confidence)
        let duration_adjustment = (signal.duration as f64 / 10.0).min(0.1);
        
        // Divergence adjustment (higher divergence can be good or bad)
        let divergence_adjustment = signal.divergence.abs() * 0.1;
        
        primary_adjustment + secondary_adjustment + consistency_adjustment + 
        duration_adjustment + divergence_adjustment
    }

    /// Calculate volatility adjustment
    fn calculate_volatility_adjustment(&self) -> f64 {
        let volatility = self.market_conditions.volatility;
        
        // Moderate volatility is good, extreme volatility is bad
        if volatility < 0.3 {
            // Low volatility: slightly positive
            volatility * 0.2
        } else if volatility < 0.7 {
            // Moderate volatility: optimal
            0.1
        } else {
            // High volatility: negative
            -(volatility - 0.7) * 0.3
        }
    }

    /// Calculate adaptive confidence based on signal type
    pub fn calculate_signal_confidence(&self, signal_type: &str, raw_strength: f64) -> f64 {
        let base_confidence = self.calculate_confidence();
        
        // Adjust based on signal type
        let signal_multiplier = match signal_type {
            "momentum" => 1.0,
            "mean_reversion" => 0.9,
            "arbitrage" => 1.1,
            "breakout" => 0.8,
            "scalping" => 0.7,
            "swing" => 1.0,
            _ => 1.0,
        };
        
        // Combine base confidence with signal strength
        let signal_confidence = base_confidence * signal_multiplier * raw_strength;
        
        signal_confidence.clamp(self.config.min_confidence, self.config.max_confidence)
    }

    /// Get confidence breakdown for debugging
    pub fn get_confidence_breakdown(&self) -> (f64, f64, f64, f64, f64) {
        let base = self.config.base_confidence;
        let market = self.calculate_market_adjustment() * self.config.market_scale;
        let performance = self.calculate_performance_adjustment() * self.config.performance_scale;
        let signal = self.calculate_signal_adjustment() * self.config.signal_scale;
        let volatility = self.calculate_volatility_adjustment() * self.config.volatility_scale;
        
        (base, market, performance, signal, volatility)
    }
}

impl Default for ConfidenceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common confidence calculations

/// Calculate smooth confidence for momentum signals
pub fn momentum_confidence(
    momentum_strength: f64,
    trend_alignment: f64,
    volatility: f64,
    performance: &PerformanceMetrics,
) -> f64 {
    let mut calc = ConfidenceCalculator::new();
    
    calc.update_market_conditions(MarketConditions {
        volatility,
        trend_strength: trend_alignment,
        ..Default::default()
    });
    
    calc.update_performance(performance.avg_pnl, performance.win_rate > 0.5);
    
    calc.update_signal_strength(SignalStrength {
        primary_strength: momentum_strength,
        secondary_strength: trend_alignment.abs(),
        consistency: 0.7,
        duration: 1,
        divergence: 0.0,
    });
    
    calc.calculate_signal_confidence("momentum", momentum_strength)
}

/// Calculate smooth confidence for mean reversion signals
pub fn mean_reversion_confidence(
    deviation_strength: f64,
    rsi: f64,
    volatility: f64,
    performance: &PerformanceMetrics,
) -> f64 {
    let mut calc = ConfidenceCalculator::new();
    
    calc.update_market_conditions(MarketConditions {
        volatility,
        trend_strength: 0.0, // Mean reversion is neutral on trend
        ..Default::default()
    });
    
    calc.update_performance(performance.avg_pnl, performance.win_rate > 0.5);
    
    // RSI-based signal strength
    let rsi_strength = if rsi < 30.0 || rsi > 70.0 {
        ((rsi - 50.0).abs() / 50.0).min(1.0)
    } else {
        0.0
    };
    
    calc.update_signal_strength(SignalStrength {
        primary_strength: deviation_strength,
        secondary_strength: rsi_strength,
        consistency: 0.8,
        duration: 1,
        divergence: 0.0,
    });
    
    calc.calculate_signal_confidence("mean_reversion", deviation_strength.max(rsi_strength))
}

/// Calculate smooth confidence for arbitrage signals
pub fn arbitrage_confidence(
    spread_size: f64,
    liquidity: f64,
    volatility: f64,
    performance: &PerformanceMetrics,
) -> f64 {
    let mut calc = ConfidenceCalculator::new();
    
    calc.update_market_conditions(MarketConditions {
        volatility,
        liquidity,
        spread: spread_size,
        ..Default::default()
    });
    
    calc.update_performance(performance.avg_pnl, performance.win_rate > 0.5);
    
    calc.update_signal_strength(SignalStrength {
        primary_strength: spread_size,
        secondary_strength: liquidity,
        consistency: 0.9,
        duration: 1,
        divergence: 0.0,
    });
    
    calc.calculate_signal_confidence("arbitrage", spread_size)
}
