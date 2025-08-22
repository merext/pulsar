use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use crate::models::{Position, Signal, TradeData};

#[derive(serde::Deserialize)]
struct StochasticHftConfig {
    general: GeneralConfig,
    stochastic: StochasticConfig,
    signals: SignalsConfig,
    risk: RiskConfig,
}

#[derive(serde::Deserialize)]
struct GeneralConfig {
    strategy_name: String,
}

#[derive(serde::Deserialize)]
struct StochasticConfig {
    k_period: usize,
    d_period: usize,
    oversold: f64,
    overbought: f64,
    buffer_capacity: usize,
}

#[derive(serde::Deserialize)]
struct SignalsConfig {
    min: f64,
    oversold: f64,
    overbought: f64,
    crossover: f64,
}

#[derive(serde::Deserialize)]
struct RiskConfig {
    max_consecutive_losses: usize,
    signal_cooldown_ms: u64,
}





pub struct StochasticHftStrategy {
    // Configuration
    config: StochasticHftConfig,
    
    // Price data
    price_history: VecDeque<f64>,
    
    // Stochastic Oscillator values
    stochastic_k: VecDeque<f64>,
    stochastic_d: VecDeque<f64>,
    
    // Enhanced indicators
    volatility_history: VecDeque<f64>,
    trend_strength: VecDeque<f64>,
    
    // Strategy state
    trade_counter: usize,
    last_signal_time: f64,
    consecutive_losses: usize,
    
    // Position tracking for backtesting
    current_position: f64,
    entry_price: f64,
    entry_time: f64,
    
    // Strategy logger
    logger: Box<dyn StrategyLogger>,
}



impl StochasticHftStrategy {
    /// # Panics
    ///
    /// Panics if the configuration file cannot be loaded.
    #[must_use]
    pub fn new() -> Self {
        Self::from_file("config/stochastic_hft_strategy.toml").expect("Failed to load configuration file")
    }
}

impl Default for StochasticHftStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticHftStrategy {
    /// # Errors
    ///
    /// Will return `Err` if the config file cannot be read or parsed.
    pub fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(config_path)?;
        let config: StochasticHftConfig = toml::from_str(&content)?;
        let buffer_capacity = config.stochastic.buffer_capacity;
        
        Ok(Self {
            config,
            price_history: VecDeque::with_capacity(buffer_capacity),
            stochastic_k: VecDeque::with_capacity(buffer_capacity),
            stochastic_d: VecDeque::with_capacity(buffer_capacity),
            volatility_history: VecDeque::with_capacity(100),
            trend_strength: VecDeque::with_capacity(100),
            trade_counter: 0,
            last_signal_time: -1000.0, // Initialize to allow immediate signals
            consecutive_losses: 0,
            current_position: 0.0,
            entry_price: 0.0,
            entry_time: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        })
    }
    
    /// Set the strategy logger for logging
    pub fn with_logger(mut self, logger: Box<dyn StrategyLogger>) -> Self {
        self.logger = logger;
        self
    }
    
    fn calculate_stochastic(&self) -> Option<(f64, f64)> {
        if self.price_history.len() < self.config.stochastic.k_period {
            return None;
        }
        
        let prices: Vec<f64> = self.price_history.iter().rev().take(self.config.stochastic.k_period).copied().collect();
        let current_price = *self.price_history.back().unwrap();
        
        let low_min = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let high_max = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (high_max - low_min).abs() < f64::EPSILON {
            return Some((50.0, 50.0));
        }
        
        let k = 100.0 * ((current_price - low_min) / (high_max - low_min));
        
        Some((k, 0.0)) // %D will be calculated in update_stochastic
    }
    
    #[allow(clippy::cast_precision_loss)]
    fn update_stochastic(&mut self) {
        if let Some((k, _)) = self.calculate_stochastic() {
            // Add new %K value first
            self.stochastic_k.push_back(k);
            
            // Calculate %D as a moving average of %K values
            let d = if self.stochastic_k.len() >= self.config.stochastic.d_period {
                // Use the most recent d_period %K values for %D calculation
                let k_values: Vec<f64> = self.stochastic_k.iter().rev().take(self.config.stochastic.d_period).copied().collect();
                k_values.iter().sum::<f64>() / k_values.len() as f64
            } else {
                k
            };
            
            self.stochastic_d.push_back(d);
            
            // Maintain buffer size
            if self.stochastic_k.len() > self.config.stochastic.buffer_capacity {
                self.stochastic_k.pop_front();
                self.stochastic_d.pop_front();
            }
        }
    }
    
    #[allow(clippy::cast_precision_loss)]
    fn calculate_volatility(&mut self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.0;
        }
        
        let returns: Vec<f64> = self.price_history.iter()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[0] - w[1]) / w[1])
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        let volatility = variance.sqrt();
        self.volatility_history.push_back(volatility);
        if self.volatility_history.len() > 50 {
            self.volatility_history.pop_front();
        }
        
        volatility
    }
    
    #[allow(clippy::cast_precision_loss)]
    fn calculate_trend_strength(&mut self) -> f64 {
        if self.price_history.len() < 50 {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(50)
            .copied()
            .collect();
        
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
        
        let trend_strength = if denominator_x > 0.0 && denominator_y > 0.0 {
            (numerator.powi(2) / (denominator_x * denominator_y)).abs()
        } else {
            0.0
        };
        
        self.trend_strength.push_back(trend_strength);
        if self.trend_strength.len() > 20 {
            self.trend_strength.pop_front();
        }
        
        trend_strength
    }
    
    fn update_indicators(&mut self) {
        self.update_stochastic();
        self.calculate_volatility();
        self.calculate_trend_strength();
    }
    
    fn generate_signal(&self) -> (Signal, f64) {
        if self.stochastic_k.len() < 2 || self.stochastic_d.len() < 2 {
            return (Signal::Hold, 0.0);
        }
        
        let current_k = *self.stochastic_k.back().unwrap();
        let current_d = *self.stochastic_d.back().unwrap();
        
        // Get previous values for crossover detection
        let prev_k = self.stochastic_k.get(self.stochastic_k.len() - 2).unwrap();
        let prev_d = self.stochastic_d.get(self.stochastic_d.len() - 2).unwrap();
        

        
        let mut buy_signals = Vec::new();
        let mut sell_signals = Vec::new();
        
        // Stochastic oversold/overbought signals - more sensitive for historical data
        if current_k < self.config.stochastic.oversold {
            buy_signals.push(self.config.signals.oversold);
        } else if current_k > self.config.stochastic.overbought {
            sell_signals.push(self.config.signals.overbought);
        }
        
        // Alternative signals for low-volatility historical data
        // When stochastic is at extremes (0 or 100), use momentum-based signals
        if current_k <= 5.0 && self.price_history.len() >= 20 {
            // Very oversold - check for price momentum reversal
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(20).copied().collect();
            let price_momentum = (recent_prices[0] - recent_prices[19]) / recent_prices[19];
            if price_momentum > 0.001 { // Slight upward momentum
                buy_signals.push(self.config.signals.oversold * 0.8); // Lower confidence
            }
        } else if current_k >= 95.0 && self.price_history.len() >= 20 {
            // Very overbought - check for price momentum reversal
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(20).copied().collect();
            let price_momentum = (recent_prices[0] - recent_prices[19]) / recent_prices[19];
            if price_momentum < -0.001 { // Slight downward momentum
                sell_signals.push(self.config.signals.overbought * 0.8); // Lower confidence
            }
        }
        
        // K/D crossover signals
        let k_d_cross_up = current_k > current_d && *prev_k <= *prev_d;
        let k_d_cross_down = current_k < current_d && *prev_k >= *prev_d;
        
        if k_d_cross_up {
            buy_signals.push(self.config.signals.crossover);
        } else if k_d_cross_down {
            sell_signals.push(self.config.signals.crossover);
        }
        
        // Calculate signal direction and confidence
        #[allow(clippy::cast_precision_loss)]
        let buy_confidence = if buy_signals.is_empty() {
            0.0
        } else {
            buy_signals.iter().sum::<f64>() / buy_signals.len() as f64
        };
        
        #[allow(clippy::cast_precision_loss)]
        let sell_confidence = if sell_signals.is_empty() {
            0.0
        } else {
            sell_signals.iter().sum::<f64>() / sell_signals.len() as f64
        };
        
        // Determine final signal and confidence
        let (signal_direction, base_confidence) = if buy_confidence > sell_confidence {
            (1.0, buy_confidence)
        } else if sell_confidence > buy_confidence {
            (-1.0, sell_confidence)
        } else {
            (0.0, 0.0)
        };
        
        // Apply trend and volatility adjustments to confidence
        let mut adjusted_confidence = base_confidence;
        
        // Trend confirmation - only boost if trend is strong
        #[allow(clippy::cast_precision_loss)]
        let avg_trend = self.trend_strength.iter().sum::<f64>() / self.trend_strength.len().max(1) as f64;
        if avg_trend > 0.5 {
            adjusted_confidence *= 1.1; // Moderate boost for strong trends
        } else if avg_trend < 0.2 {
            adjusted_confidence *= 0.7; // Reduce confidence in weak trends
        }
        
        // Volatility adjustment - balanced
        #[allow(clippy::cast_precision_loss)]
        let avg_volatility = self.volatility_history.iter().sum::<f64>() / self.volatility_history.len().max(1) as f64;
        if avg_volatility > 0.008 {
            adjusted_confidence *= 0.7; // Moderate reduction in very high volatility
        } else if avg_volatility < 0.002 {
            adjusted_confidence *= 1.05; // Slight boost in low volatility
        }
        
        // Additional signal quality checks
        let signal_strength = (current_k - current_d).abs();
        if signal_strength < 5.0 {
            adjusted_confidence *= 0.5; // Weak signal strength
        }
        
        // Stochastic range check - avoid signals near middle
        if current_k > 45.0 && current_k < 55.0 {
            adjusted_confidence *= 0.5; // Lower confidence in middle range
        }
        
        let final_confidence = adjusted_confidence.min(1.0);
        
        if final_confidence >= self.config.signals.min {
            if signal_direction > 0.0 {
                (Signal::Buy, final_confidence)
            } else if signal_direction < 0.0 {
                (Signal::Sell, final_confidence)
            } else {
                (Signal::Hold, 0.0)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }
}

#[async_trait::async_trait]
impl Strategy for StochasticHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }
    
    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_file(config_path)
    }
    
    fn get_info(&self) -> String {
        let current_k = self.stochastic_k.back().unwrap_or(&0.0);
        let current_d = self.stochastic_d.back().unwrap_or(&0.0);
        
        format!(
            "{} - K: {:.1}, D: {:.1}",
            self.config.general.strategy_name,
            current_k,
            current_d
        )
    }
    
    async fn on_trade(&mut self, trade: TradeData) {
        // Update price data
        self.price_history.push_back(trade.price);
        
        // Keep only recent data
        if self.price_history.len() > self.config.stochastic.buffer_capacity {
            self.price_history.pop_front();
        }
        
        // Update all indicators
        self.update_indicators();
        

    }
    
    fn get_signal(
        &mut self,
        _current_price: f64,
        current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // Check cooldown
        #[allow(clippy::cast_precision_loss)]
        let cooldown = self.config.risk.signal_cooldown_ms as f64 / 1000.0;
        if current_timestamp - self.last_signal_time < cooldown {
            return (Signal::Hold, 0.0);
        }
        
        // Check if we have enough data
        if self.price_history.len() < self.config.stochastic.k_period {
            return (Signal::Hold, 0.0);
        }
        
        // Check risk management
        if self.consecutive_losses >= self.config.risk.max_consecutive_losses {
            return (Signal::Hold, 0.0);
        }
        
        // Generate base signal based on Stochastic Oscillator
        let (base_signal, confidence) = self.generate_signal();
        
                
        
        // Position-aware signal filtering with minimum holding time
        match base_signal {
            Signal::Buy => {
                // Only generate BUY if we don't have a long position
                if current_position.quantity <= 0.0 {
                    // Update entry time when opening a new position
                    self.entry_time = current_timestamp;
                    (Signal::Buy, confidence)
                } else {
                    (Signal::Hold, 0.0)
                }
            }
            Signal::Sell => {
                // Only generate SELL if we have a long position and have held it for minimum time
                if current_position.quantity > 0.0 {
                    // Minimum holding time of 5 seconds to allow for more frequent trading
                    let min_hold_time = 5.0; // seconds
                    let holding_time = current_timestamp - self.entry_time;
                    if holding_time >= min_hold_time {
                        (Signal::Sell, confidence)
                    } else {
                        (Signal::Hold, 0.0)
                    }
                } else {
                    (Signal::Hold, 0.0)
                }
            }
            Signal::Hold => (Signal::Hold, 0.0),
        }
    }
}

impl StochasticHftStrategy {

    
    #[allow(dead_code)]
    fn execute_trade(&mut self, signal: Signal, current_price: f64, current_time: f64, confidence: f64) {
        // Use trading_config position sizing: min + (confidence * (max - min))
        // These should ideally be read from trading_config.toml, but for now using reasonable defaults
        let trading_size_min = 20.0; // From trading_config.toml
        let trading_size_max = 30.0; // From trading_config.toml
        let position_size = confidence.mul_add(trading_size_max - trading_size_min, trading_size_min);
        
        match signal {
            Signal::Buy => {
                if self.current_position <= 0.0 {
                    // Open long position
                    self.current_position = position_size;
                    self.entry_price = current_price;
                    self.entry_time = current_time;
                    self.trade_counter += 1;
                }
            }
            Signal::Sell => {
                if self.current_position >= 0.0 {
                    // Open short position
                    self.current_position = -position_size;
                    self.entry_price = current_price;
                    self.entry_time = current_time;
                    self.trade_counter += 1;
                }
            }
            Signal::Hold => {
                // No action needed for hold signals
            }
        }
    }
    

    

}
