use std::collections::VecDeque;
use serde::Deserialize;

use crate::strategy::Strategy;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;

#[derive(Debug, Clone, PartialEq)]
pub enum MarketRegime {
    Trending,      // Strong directional movement
    MeanReverting, // Oscillating around a level
    Volatile,      // High volatility, choppy
    Sideways,      // Low volatility, range-bound
    Breakout,      // Breaking out of range
}

#[derive(Debug, Clone)]
pub struct MarketState {
    pub regime: MarketRegime,
    pub volatility: f64,
    pub trend_strength: f64,
    pub momentum: f64,
    pub support_level: f64,
    pub resistance_level: f64,
    pub volume_profile: f64,
}

#[derive(Deserialize)]
struct EnhancedPulsarConfig {
    regime_detection: RegimeDetectionConfig,
    signals: SignalConfig,
    risk_management: RiskConfig,
}

#[derive(Deserialize)]
struct RegimeDetectionConfig {
    volatility_window: usize,
    trend_window: usize,
    momentum_window: usize,
    volatility_threshold_high: f64,
    volatility_threshold_low: f64,
    trend_strength_threshold: f64,
}

#[derive(Deserialize)]
struct SignalConfig {
    min_confidence: f64,
    regime_weights: RegimeWeights,
    momentum_threshold: f64,
    mean_reversion_threshold: f64,
    breakout_threshold: f64,
}

#[derive(Deserialize)]
struct RegimeWeights {
    trending_weight: f64,
    mean_reverting_weight: f64,
    volatile_weight: f64,
    sideways_weight: f64,
    breakout_weight: f64,
}

#[derive(Deserialize)]
struct RiskConfig {
    max_drawdown: f64,
    max_consecutive_losses: usize,
}



pub struct EnhancedPulsarStrategy {
    config: EnhancedPulsarConfig,
    
    // Market data buffers
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    timestamp_history: VecDeque<f64>,
    
    // Market regime detection
    market_state: MarketState,
    regime_history: VecDeque<MarketRegime>,
    
    // Performance tracking
    trade_counter: usize,
    win_count: usize,
    loss_count: usize,
    consecutive_losses: usize,
    total_pnl: f64,
    current_position: f64,
    entry_price: f64,
    
    // Dynamic thresholds
    current_thresholds: DynamicThresholds,
    
    // Risk management
    max_drawdown: f64,
    peak_equity: f64,
    current_equity: f64,
}

#[derive(Debug, Clone)]
struct DynamicThresholds {
    confidence_multiplier: f64,
    position_size_multiplier: f64,
}

impl EnhancedPulsarStrategy {
    pub fn new() -> Self {
        Self::from_file("config/enhanced_pulsar_strategy.toml")
            .expect("Failed to load configuration file")
    }
    
    pub fn from_file<P: AsRef<std::path::Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(config_path)?;
        let config: EnhancedPulsarConfig = toml::from_str(&content)?;
        
        let buffer_capacity = config.regime_detection.volatility_window.max(
            config.regime_detection.trend_window.max(config.regime_detection.momentum_window)
        );
        
        Ok(Self {
            config,
            price_history: VecDeque::with_capacity(buffer_capacity),
            volume_history: VecDeque::with_capacity(buffer_capacity),
            timestamp_history: VecDeque::with_capacity(buffer_capacity),
            market_state: MarketState {
                regime: MarketRegime::Sideways,
                volatility: 0.0,
                trend_strength: 0.0,
                momentum: 0.0,
                support_level: 0.0,
                resistance_level: 0.0,
                volume_profile: 0.0,
            },
            regime_history: VecDeque::with_capacity(100),
            trade_counter: 0,
            win_count: 0,
            loss_count: 0,
            consecutive_losses: 0,
            total_pnl: 0.0,
            current_position: 0.0,
            entry_price: 0.0,
            current_thresholds: DynamicThresholds {
                confidence_multiplier: 1.0,
                position_size_multiplier: 1.0,
            },
            max_drawdown: 0.0,
            peak_equity: 0.0,
            current_equity: 0.0,
        })
    }
    
    fn detect_market_regime(&mut self) -> MarketRegime {
        if self.price_history.len() < self.config.regime_detection.volatility_window {
            return MarketRegime::Sideways;
        }
        
        // Calculate volatility
        let volatility = self.calculate_volatility();
        
        // Calculate trend strength
        let trend_strength = self.calculate_trend_strength();
        
        // Calculate momentum
        let momentum = self.calculate_momentum();
        
        // Calculate volume profile
        let volume_profile = self.calculate_volume_profile();
        
        // Update market state
        self.market_state.volatility = volatility;
        self.market_state.trend_strength = trend_strength;
        self.market_state.momentum = momentum;
        self.market_state.volume_profile = volume_profile;
        
        // Determine regime based on indicators
        let regime = if volatility > self.config.regime_detection.volatility_threshold_high {
            if trend_strength > self.config.regime_detection.trend_strength_threshold {
                MarketRegime::Breakout
            } else {
                MarketRegime::Volatile
            }
        } else if volatility < self.config.regime_detection.volatility_threshold_low {
            if trend_strength > self.config.regime_detection.trend_strength_threshold {
                MarketRegime::Trending
            } else {
                MarketRegime::Sideways
            }
        } else {
            if momentum.abs() > self.config.signals.momentum_threshold {
                MarketRegime::MeanReverting
            } else {
                MarketRegime::Sideways
            }
        };
        
        // Update regime history
        self.regime_history.push_back(regime.clone());
        if self.regime_history.len() > 50 {
            self.regime_history.pop_front();
        }
        
        // Update support/resistance levels
        self.update_support_resistance();
        
        regime
    }
    
    fn calculate_volatility(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }
        
        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        let returns: Vec<f64> = prices.windows(2)
            .map(|window| (window[1] - window[0]) / window[0])
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }
    
    fn calculate_trend_strength(&self) -> f64 {
        if self.price_history.len() < self.config.regime_detection.trend_window {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(self.config.regime_detection.trend_window)
            .cloned()
            .collect();
        
        let first_price = recent_prices.first().unwrap();
        let last_price = recent_prices.last().unwrap();
        let _total_return = (last_price - first_price) / first_price;
        
        // Calculate linear regression R-squared
        let n = recent_prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = recent_prices.iter().sum::<f64>() / n;
        
        let numerator: f64 = recent_prices.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator_x: f64 = (0..recent_prices.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();
        
        let denominator_y: f64 = recent_prices.iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();
        
        if denominator_x > 0.0 && denominator_y > 0.0 {
            (numerator.powi(2) / (denominator_x * denominator_y)).abs()
        } else {
            0.0
        }
    }
    
    fn calculate_momentum(&self) -> f64 {
        if self.price_history.len() < self.config.regime_detection.momentum_window {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(self.config.regime_detection.momentum_window)
            .cloned()
            .collect();
        
        let first_price = recent_prices.first().unwrap();
        let last_price = recent_prices.last().unwrap();
        
        (last_price - first_price) / first_price
    }
    
    fn calculate_volume_profile(&self) -> f64 {
        if self.volume_history.len() < 10 {
            return 0.0;
        }
        
        let recent_volume: Vec<f64> = self.volume_history.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let avg_volume = recent_volume.iter().sum::<f64>() / recent_volume.len() as f64;
        let current_volume = recent_volume.first().unwrap();
        
        current_volume / avg_volume
    }
    
    fn update_support_resistance(&mut self) {
        if self.price_history.len() < 20 {
            return;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(20)
            .cloned()
            .collect();
        
        self.market_state.support_level = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        self.market_state.resistance_level = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    }
    
    fn generate_regime_optimized_signal(&self, current_price: f64) -> (Signal, f64) {
        let regime = &self.market_state.regime;
        
        match regime {
            MarketRegime::Trending => self.generate_trending_signal(current_price),
            MarketRegime::MeanReverting => self.generate_mean_reversion_signal(current_price),
            MarketRegime::Volatile => self.generate_volatile_signal(current_price),
            MarketRegime::Sideways => self.generate_sideways_signal(current_price),
            MarketRegime::Breakout => self.generate_breakout_signal(current_price),
        }
    }
    
    fn generate_trending_signal(&self, _current_price: f64) -> (Signal, f64) {
        let momentum = self.market_state.momentum;
        let trend_strength = self.market_state.trend_strength;
        
        let confidence = (trend_strength * self.config.signals.regime_weights.trending_weight).min(1.0);
        
        if momentum > self.config.signals.momentum_threshold && trend_strength > 0.7 {
            (Signal::Buy, confidence)
        } else if momentum < -self.config.signals.momentum_threshold && trend_strength > 0.7 {
            (Signal::Sell, confidence)
        } else {
            (Signal::Hold, 0.0)
        }
    }
    
    fn generate_mean_reversion_signal(&self, _current_price: f64) -> (Signal, f64) {
        let momentum = self.market_state.momentum;
        let volatility = self.market_state.volatility;
        
        let confidence = (volatility * self.config.signals.regime_weights.mean_reverting_weight).min(1.0);
        
        if momentum > self.config.signals.mean_reversion_threshold {
            (Signal::Sell, confidence)
        } else if momentum < -self.config.signals.mean_reversion_threshold {
            (Signal::Buy, confidence)
        } else {
            (Signal::Hold, 0.0)
        }
    }
    
    fn generate_volatile_signal(&self, _current_price: f64) -> (Signal, f64) {
        let momentum = self.market_state.momentum;
        let volume_profile = self.market_state.volume_profile;
        
        let confidence = (volume_profile * self.config.signals.regime_weights.volatile_weight).min(1.0);
        
        // In volatile markets, be more conservative
        if momentum.abs() > self.config.signals.momentum_threshold * 1.5 {
            if momentum > 0.0 {
                (Signal::Buy, confidence * 0.7)
            } else {
                (Signal::Sell, confidence * 0.7)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }
    
    fn generate_sideways_signal(&self, current_price: f64) -> (Signal, f64) {
        let momentum = self.market_state.momentum;
        let support = self.market_state.support_level;
        let resistance = self.market_state.resistance_level;
        
        let confidence = self.config.signals.regime_weights.sideways_weight;
        
        // Range trading logic
        let range = resistance - support;
        let position_in_range = (current_price - support) / range;
        
        if position_in_range < 0.2 && momentum > 0.0 {
            (Signal::Buy, confidence)
        } else if position_in_range > 0.8 && momentum < 0.0 {
            (Signal::Sell, confidence)
        } else {
            (Signal::Hold, 0.0)
        }
    }
    
    fn generate_breakout_signal(&self, current_price: f64) -> (Signal, f64) {
        let momentum = self.market_state.momentum;
        let volume_profile = self.market_state.volume_profile;
        let resistance = self.market_state.resistance_level;
        let support = self.market_state.support_level;
        
        let confidence = (volume_profile * self.config.signals.regime_weights.breakout_weight).min(1.0);
        
        // Breakout logic
        if current_price > resistance && momentum > self.config.signals.breakout_threshold {
            (Signal::Buy, confidence)
        } else if current_price < support && momentum < -self.config.signals.breakout_threshold {
            (Signal::Sell, confidence)
        } else {
            (Signal::Hold, 0.0)
        }
    }
    
    fn update_dynamic_thresholds(&mut self) {
        let regime = &self.market_state.regime;
        
        // Adjust thresholds based on regime
        match regime {
            MarketRegime::Trending => {
                self.current_thresholds.confidence_multiplier = 1.2;
                self.current_thresholds.position_size_multiplier = 1.1;
            }
            MarketRegime::MeanReverting => {
                self.current_thresholds.confidence_multiplier = 1.0;
                self.current_thresholds.position_size_multiplier = 0.9;
            }
            MarketRegime::Volatile => {
                self.current_thresholds.confidence_multiplier = 0.8;
                self.current_thresholds.position_size_multiplier = 0.7;
            }
            MarketRegime::Sideways => {
                self.current_thresholds.confidence_multiplier = 0.9;
                self.current_thresholds.position_size_multiplier = 0.8;
            }
            MarketRegime::Breakout => {
                self.current_thresholds.confidence_multiplier = 1.3;
                self.current_thresholds.position_size_multiplier = 1.2;
            }
        }
        
        // Adjust based on performance
        if self.consecutive_losses > 2 {
            self.current_thresholds.confidence_multiplier *= 0.8;
            self.current_thresholds.position_size_multiplier *= 0.7;
        }
        
        if self.win_count > self.loss_count && self.trade_counter > 10 {
            self.current_thresholds.confidence_multiplier *= 1.1;
            self.current_thresholds.position_size_multiplier *= 1.05;
        }
    }
    

}

#[async_trait::async_trait]
impl Strategy for EnhancedPulsarStrategy {
    fn get_info(&self) -> String {
        format!(
            "EnhancedPulsarStrategy - Regime: {:?}, Trades: {}, Win Rate: {:.1}%, PnL: {:.6}, Vol: {:.4}, Trend: {:.3}",
            self.market_state.regime,
            self.trade_counter,
            if self.trade_counter > 0 {
                (self.win_count as f64 / self.trade_counter as f64) * 100.0
            } else {
                0.0
            },
            self.total_pnl,
            self.market_state.volatility,
            self.market_state.trend_strength
        )
    }
    
    async fn on_trade(&mut self, trade: TradeData) {
        // Update price and volume history
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.qty);
        self.timestamp_history.push_back(trade.time as f64);
        
        // Maintain buffer size
        let max_size = self.config.regime_detection.volatility_window.max(
            self.config.regime_detection.trend_window.max(self.config.regime_detection.momentum_window)
        );
        
        if self.price_history.len() > max_size {
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.timestamp_history.pop_front();
        }
        
        // Detect market regime
        self.market_state.regime = self.detect_market_regime();
        
        // Update dynamic thresholds
        self.update_dynamic_thresholds();
        
        // Update equity tracking
        if self.current_position != 0.0 {
            let unrealized_pnl = (trade.price - self.entry_price) * self.current_position;
            self.current_equity = self.total_pnl + unrealized_pnl;
            
            if self.current_equity > self.peak_equity {
                self.peak_equity = self.current_equity;
            }
            
            let drawdown = if self.peak_equity > 0.0 {
                (self.peak_equity - self.current_equity) / self.peak_equity
            } else {
                0.0
            };
            
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }
    }
    
    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // Check if we have enough data
        if self.price_history.len() < 10 {
            return (Signal::Hold, 0.0);
        }
        
        // Check risk management
        if self.consecutive_losses >= self.config.risk_management.max_consecutive_losses {
            return (Signal::Hold, 0.0);
        }
        
        if self.max_drawdown > self.config.risk_management.max_drawdown {
            return (Signal::Hold, 0.0);
        }
        
        // Generate regime-optimized signal
        let (signal, base_confidence) = self.generate_regime_optimized_signal(current_price);
        
        // Apply dynamic confidence multiplier
        let adjusted_confidence = base_confidence * self.current_thresholds.confidence_multiplier;
        
        // Apply minimum confidence threshold
        if adjusted_confidence < self.config.signals.min_confidence {
            return (Signal::Hold, 0.0);
        }
        
        // Position management
        match signal {
            Signal::Buy => {
                if current_position.quantity == 0.0 {
                    (Signal::Buy, adjusted_confidence)
                } else {
                    (Signal::Hold, 0.0)
                }
            }
            Signal::Sell => {
                if current_position.quantity > 0.0 {
                    (Signal::Sell, adjusted_confidence)
                } else {
                    (Signal::Hold, 0.0)
                }
            }
            Signal::Hold => (Signal::Hold, 0.0),
        }
    }
}
