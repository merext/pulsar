use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct MeanReversionHftConfig {
    pub general: GeneralConfig,
    pub mean_reversion: MeanReversionConfig,
    pub volatility: VolatilityConfig,
    pub volume: VolumeConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct MeanReversionConfig {
    pub mean_window: usize,        // Window for calculating moving average
    pub deviation_threshold: f64,  // Price deviation threshold from mean
    pub reversion_strength: f64,   // Strength of mean reversion signal
    pub max_deviation: f64,        // Maximum deviation before extreme signal
}

#[derive(Debug, Deserialize)]
pub struct VolatilityConfig {
    pub volatility_window: usize,  // Window for volatility calculation
    pub volatility_threshold: f64, // Minimum volatility for trading
    pub adaptive_threshold: bool,  // Whether to adapt thresholds to market
}

#[derive(Debug, Deserialize)]
pub struct VolumeConfig {
    pub volume_window: usize,      // Window for volume analysis
    pub volume_threshold: f64,     // Minimum volume increase
    pub volume_confirmation: bool, // Whether to require volume confirmation
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,      // Minimum confidence for trade
    pub max_position_time: f64,   // Maximum time to hold position
    pub profit_target: f64,       // Profit target percentage
    pub stop_loss: f64,           // Stop loss percentage
    pub max_trades_per_hour: usize, // Maximum trades per hour
}

pub struct MeanReversionHftStrategy {
    config: MeanReversionHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    mean_history: VecDeque<f64>,
    volatility_history: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_hour: usize,
    last_hour_reset: f64,
    logger: Box<dyn StrategyLogger>,
}

impl MeanReversionHftStrategy {
    pub fn new(config: MeanReversionHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            mean_history: VecDeque::new(),
            volatility_history: VecDeque::new(),
            last_signal_time: 0.0,
            trades_this_hour: 0,
            last_hour_reset: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_moving_average(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
        recent_prices.iter().sum::<f64>() / recent_prices.len() as f64
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

    fn detect_mean_reversion(&self, current_price: f64, mean_price: f64) -> (f64, f64) {
        let deviation = (current_price - mean_price) / mean_price * 100.0;
        let deviation_abs = deviation.abs();
        
        // Calculate reversion strength based on deviation
        let reversion_strength = if deviation_abs > self.config.mean_reversion.max_deviation {
            1.0 // Extreme deviation - strong signal
        } else if deviation_abs > self.config.mean_reversion.deviation_threshold {
            (deviation_abs - self.config.mean_reversion.deviation_threshold) / 
            (self.config.mean_reversion.max_deviation - self.config.mean_reversion.deviation_threshold)
        } else {
            0.0 // No significant deviation
        };
        
        (deviation, reversion_strength)
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

    fn reset_hourly_trades(&mut self, current_time: f64) {
        if current_time - self.last_hour_reset >= 3600.0 { // 1 hour
            self.trades_this_hour = 0;
            self.last_hour_reset = current_time;
        }
    }
}

#[async_trait]
impl Strategy for MeanReversionHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: MeanReversionHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "MeanReversionHftStrategy - Price: {:.8}, Data Points: {}, Mean: {:.8}, Volatility: {:.4}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.mean_history.back().unwrap_or(&0.0),
            self.volatility_history.back().unwrap_or(&0.0)
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.mean_reversion.mean_window * 3 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > self.config.volume.volume_window * 3 {
            self.volume_history.pop_front();
        }

        // Calculate and store moving average
        if self.price_history.len() >= self.config.mean_reversion.mean_window {
            let mean = self.calculate_moving_average(&self.price_history, self.config.mean_reversion.mean_window);
            self.mean_history.push_back(mean);
            if self.mean_history.len() > 100 {
                self.mean_history.pop_front();
            }
        }

        // Calculate and store volatility
        if self.price_history.len() >= self.config.volatility.volatility_window {
            let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
            self.volatility_history.push_back(volatility);
            if self.volatility_history.len() > 100 {
                self.volatility_history.pop_front();
            }
        }

        // Debug logging every 100 trades
        if self.price_history.len() % 100 == 0 {
            println!("DEBUG: Processed {} trades, current price: {:.8}, mean: {:.8}, volatility: {:.4}", 
                self.price_history.len(), 
                trade.price,
                self.mean_history.back().unwrap_or(&0.0),
                self.volatility_history.back().unwrap_or(&0.0));
        }
    }

    fn get_signal(&mut self, current_position: Position) -> (Signal, f64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Reset hourly trade counter
        self.reset_hourly_trades(current_time);

        // Check if we've exceeded hourly trade limit
        if self.trades_this_hour >= self.config.signals.max_trades_per_hour {
            return (Signal::Hold, 0.0);
        }

        if self.price_history.len() < self.config.mean_reversion.mean_window {
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();
        let mean_price = *self.mean_history.back().unwrap();

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Detect mean reversion opportunity
        let (deviation, reversion_strength) = self.detect_mean_reversion(current_price, mean_price);
        
        // Calculate volatility
        let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
        
        // Calculate volume momentum
        let volume_momentum = self.calculate_volume_momentum(&self.volume_history, self.config.volume.volume_window);

        // Generate signal based on mean reversion
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Mean reversion analysis
        if reversion_strength > 0.0 {
            if deviation > 0.0 {
                // Price above mean - expect reversion down (sell)
                signal_strength = reversion_strength * self.config.mean_reversion.reversion_strength;
                signal_direction = -1.0;
            } else {
                // Price below mean - expect reversion up (buy)
                signal_strength = reversion_strength * self.config.mean_reversion.reversion_strength;
                signal_direction = 1.0;
            }
        }

        // Volatility analysis
        if volatility > self.config.volatility.volatility_threshold {
            signal_strength += 0.2;
        }

        // Volume analysis
        if self.config.volume.volume_confirmation {
            if volume_momentum > self.config.volume.volume_threshold {
                signal_strength += 0.2;
            }
        } else {
            // Volume not required for confirmation
            signal_strength += 0.1;
        }

        // Time-based signal filtering
        if current_time - self.last_signal_time < 5.0 {
            signal_strength *= 0.5; // Reduce signal strength for rapid signals
        }

        // Generate final signal
        if signal_strength >= self.config.signals.min_confidence {
            self.last_signal_time = current_time;
            self.trades_this_hour += 1;
            
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
