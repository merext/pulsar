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
    buffer_capacity: usize,
}

#[derive(serde::Deserialize)]
struct SignalsConfig {
    min: f64,
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
    

    
    // Strategy state (no additional state needed)
    
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

            last_signal_time: -1000.0, // Initialize to allow immediate signals
            consecutive_losses: 0,
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
            
            // Simple %D calculation as moving average of last 3 %K values
            let d = if self.stochastic_k.len() >= 3 {
                let k_values: Vec<f64> = self.stochastic_k.iter().rev().take(3).copied().collect();
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
    
    fn update_indicators(&mut self) {
        self.update_stochastic();
    }
    
    fn generate_signal(&self) -> (Signal, f64) {
        if self.stochastic_k.len() < 2 || self.stochastic_d.len() < 2 {
            return (Signal::Hold, 0.0);
        }
        
        let current_k = *self.stochastic_k.back().unwrap();
        let current_d = *self.stochastic_d.back().unwrap();
        

        

        
        // Simple stochastic-based signal generation
        let mut signal_direction = 0.0;
        let mut base_confidence = 0.0;
        
        // Basic stochastic signals
        if current_k <= 20.0 {
            signal_direction = 1.0; // Buy signal
            base_confidence = 0.8;
        } else if current_k >= 80.0 {
            signal_direction = -1.0; // Sell signal
            base_confidence = 0.8;
        }
        
        // Simple confidence adjustment based on signal strength
        let signal_strength = (current_k - current_d).abs();
        let mut adjusted_confidence: f64 = base_confidence;
        
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
                    (Signal::Buy, confidence)
                } else {
                    (Signal::Hold, 0.0)
                }
            }
            Signal::Sell => {
                // Only generate SELL if we have a long position
                if current_position.quantity > 0.0 {
                    (Signal::Sell, confidence)
                } else {
                    (Signal::Hold, 0.0)
                }
            }
            Signal::Hold => (Signal::Hold, 0.0),
        }
    }
}


