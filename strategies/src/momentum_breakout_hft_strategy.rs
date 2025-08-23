use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct MomentumBreakoutHftConfig {
    pub general: GeneralConfig,
    pub momentum: MomentumConfig,
    pub volatility: VolatilityConfig,
    pub volume: VolumeConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct MomentumConfig {
    pub short_period: usize,      // Short-term momentum window
    pub long_period: usize,       // Long-term momentum window
    pub momentum_threshold: f64,  // Minimum momentum for signal
    pub acceleration_threshold: f64, // Rate of momentum change
}

#[derive(Debug, Deserialize)]
pub struct VolatilityConfig {
    pub volatility_window: usize, // Window for volatility calculation
    pub volatility_threshold: f64, // Minimum volatility for trading
    pub breakout_multiplier: f64,  // Multiplier for breakout detection
}

#[derive(Debug, Deserialize)]
pub struct VolumeConfig {
    pub volume_window: usize,     // Window for volume analysis
    pub volume_threshold: f64,    // Minimum volume increase
    pub volume_momentum: f64,     // Volume momentum threshold
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,     // Minimum confidence for trade
    pub max_position_time: f64,  // Maximum time to hold position (seconds)
    pub profit_target: f64,      // Profit target percentage
    pub stop_loss: f64,          // Stop loss percentage
}

pub struct MomentumBreakoutHftStrategy {
    config: MomentumBreakoutHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    momentum_history: VecDeque<f64>,
    volatility_history: VecDeque<f64>,
    last_signal_time: f64,
    logger: Box<dyn StrategyLogger>,
}

impl MomentumBreakoutHftStrategy {
    pub fn new(config: MomentumBreakoutHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            momentum_history: VecDeque::new(),
            volatility_history: VecDeque::new(),
            last_signal_time: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_momentum(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }
        
        let recent = prices.iter().rev().take(period).sum::<f64>() / period as f64;
        let older = prices.iter().rev().skip(period).take(period).sum::<f64>() / period as f64;
        
        if older > 0.0 {
            (recent - older) / older * 100.0
        } else {
            0.0
        }
    }

    fn calculate_volatility(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
        let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        
        let variance = recent_prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
        
        variance.sqrt() / mean * 100.0
    }

    fn calculate_volume_momentum(&self, volumes: &VecDeque<f64>, period: usize) -> f64 {
        if volumes.len() < period {
            return 0.0;
        }
        
        let recent_avg = volumes.iter().rev().take(period).sum::<f64>() / period as f64;
        let older_avg = volumes.iter().rev().skip(period).take(period).sum::<f64>() / period as f64;
        
        if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg * 100.0
        } else {
            0.0
        }
    }

    fn detect_breakout(&self, current_price: f64) -> bool {
        if self.price_history.len() < self.config.volatility.volatility_window {
            return false;
        }
        
        let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
        let avg_price = self.price_history.iter().rev().take(self.config.volatility.volatility_window).sum::<f64>() / self.config.volatility.volatility_window as f64;
        
        let upper_band = avg_price * (1.0 + volatility / 100.0 * self.config.volatility.breakout_multiplier);
        let lower_band = avg_price * (1.0 - volatility / 100.0 * self.config.volatility.breakout_multiplier);
        
        current_price > upper_band || current_price < lower_band
    }

    fn should_exit_position(&self, current_price: f64, entry_price: f64, entry_time: f64, current_time: f64) -> bool {
        // Time-based exit
        if current_time - entry_time > self.config.signals.max_position_time {
            return true;
        }
        
        // Profit target or stop loss
        let price_change = (current_price - entry_price) / entry_price * 100.0;
        
        price_change >= self.config.signals.profit_target || price_change <= -self.config.signals.stop_loss
    }
}

#[async_trait]
impl Strategy for MomentumBreakoutHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: MomentumBreakoutHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "MomentumBreakoutHftStrategy - Price: {:.8}, Data Points: {}, Momentum: {:.4}, Volatility: {:.4}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.momentum_history.back().unwrap_or(&0.0),
            self.volatility_history.back().unwrap_or(&0.0)
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // CRITICAL DEBUG: Log EVERY trade to see if data is reaching the strategy
        println!("DEBUG: on_trade called! Price: {:.8}, Quantity: {:.2}, Symbol: {}", 
            trade.price, trade.quantity, trade.symbol);

        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.momentum.long_period * 2 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > self.config.volume.volume_window * 2 {
            self.volume_history.pop_front();
        }

        // Calculate and store momentum
        if self.price_history.len() >= self.config.momentum.short_period {
            let momentum = self.calculate_momentum(&self.price_history, self.config.momentum.short_period);
            self.momentum_history.push_back(momentum);
            if self.momentum_history.len() > self.config.momentum.long_period {
                self.momentum_history.pop_front();
            }
        }

        // Calculate and store volatility
        if self.price_history.len() >= self.config.volatility.volatility_window {
            let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
            self.volatility_history.push_back(volatility);
            if self.volatility_history.len() > self.config.volatility.volatility_window {
                self.volatility_history.pop_front();
            }
        }

        // Debug logging every 100 trades (more frequent)
        if self.price_history.len() % 100 == 0 {
            println!("DEBUG: Processed {} trades, current price: {:.8}, momentum: {:.4}, volatility: {:.4}", 
                self.price_history.len(), 
                trade.price,
                self.momentum_history.back().unwrap_or(&0.0),
                self.volatility_history.back().unwrap_or(&0.0));
        }
    }

    fn get_signal(&mut self, current_position: Position) -> (Signal, f64) {
        // CRITICAL DEBUG: Log EVERY signal check
        println!("DEBUG: get_signal called! Price history length: {}, Current price: {:.8}", 
            self.price_history.len(), 
            self.price_history.back().unwrap_or(&0.0));

        if self.price_history.len() < self.config.momentum.long_period {
            println!("DEBUG: Not enough data yet. Need {} but have {}", 
                self.config.momentum.long_period, self.price_history.len());
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Calculate momentum indicators
        let short_momentum = self.calculate_momentum(&self.price_history, self.config.momentum.short_period);
        let long_momentum = self.calculate_momentum(&self.price_history, self.config.momentum.long_period);
        
        // Calculate momentum acceleration
        let momentum_acceleration = if self.momentum_history.len() >= 2 {
            self.momentum_history.back().unwrap() - self.momentum_history.iter().rev().nth(1).unwrap()
        } else {
            0.0
        };

        // Calculate volatility
        let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
        
        // Calculate volume momentum
        let volume_momentum = self.calculate_volume_momentum(&self.volume_history, self.config.volume.volume_window);

        // Check for breakout
        let breakout_detected = self.detect_breakout(current_price);

        // Generate signal based on multiple factors
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Momentum analysis
        if short_momentum > self.config.momentum.momentum_threshold && 
           long_momentum > self.config.momentum.momentum_threshold {
            signal_strength += 0.3;
            signal_direction = 1.0;
        } else if short_momentum < -self.config.momentum.momentum_threshold && 
                  long_momentum < -self.config.momentum.momentum_threshold {
            signal_strength += 0.3;
            signal_direction = -1.0;
        }

        // Acceleration analysis
        if momentum_acceleration.abs() > self.config.momentum.acceleration_threshold {
            signal_strength += 0.2;
        }

        // Volatility analysis
        if volatility > self.config.volatility.volatility_threshold {
            signal_strength += 0.2;
        }

        // Volume analysis
        if volume_momentum > self.config.volume.volume_threshold {
            signal_strength += 0.2;
        }

        // Breakout analysis
        if breakout_detected {
            signal_strength += 0.1;
        }

        // Time-based signal filtering
        if current_time - self.last_signal_time < 1.0 {
            signal_strength *= 0.5; // Reduce signal strength for rapid signals
        }

        // Generate final signal
        if signal_strength >= self.config.signals.min_confidence {
            self.last_signal_time = current_time;
            
            if signal_direction > 0.0 && current_position.quantity == 0.0 {
                (Signal::Buy, signal_strength)
            } else if signal_direction < 0.0 && current_position.quantity > 0.0 {
                (Signal::Sell, signal_strength)
            } else {
                (Signal::Hold, 0.0)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }
}
