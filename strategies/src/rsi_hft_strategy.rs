use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct RsiHftConfig {
    pub general: GeneralConfig,
    pub rsi: RsiConfig,
    pub volatility: VolatilityConfig,
    pub volume: VolumeConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct RsiConfig {
    pub rsi_period: usize,           // RSI calculation period
    pub oversold_threshold: f64,     // RSI oversold threshold
    pub overbought_threshold: f64,   // RSI overbought threshold
    pub rsi_smoothing: f64,          // RSI smoothing factor
    pub divergence_lookback: usize,  // Lookback for divergence detection
}

#[derive(Debug, Deserialize)]
pub struct VolatilityConfig {
    pub volatility_window: usize,    // Window for volatility calculation
    pub volatility_threshold: f64,   // Minimum volatility for trading
    pub adaptive_threshold: bool,    // Whether to adapt thresholds to market
}

#[derive(Debug, Deserialize)]
pub struct VolumeConfig {
    pub volume_window: usize,        // Window for volume analysis
    pub volume_threshold: f64,       // Minimum volume increase
    pub volume_confirmation: bool,   // Whether to require volume confirmation
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,        // Minimum confidence for trade
    pub max_position_time: f64,     // Maximum time to hold position
    pub profit_target: f64,         // Profit target percentage
    pub stop_loss: f64,             // Stop loss percentage
    pub max_trades_per_hour: usize, // Maximum trades per hour
    pub signal_cooldown: f64,       // Minimum time between signals
}

pub struct RsiHftStrategy {
    config: RsiHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    rsi_history: VecDeque<f64>,
    volatility_history: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_hour: usize,
    last_hour_reset: f64,
    logger: Box<dyn StrategyLogger>,
}

impl RsiHftStrategy {
    pub fn new(config: RsiHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            rsi_history: VecDeque::new(),
            volatility_history: VecDeque::new(),
            last_signal_time: 0.0,
            trades_this_hour: 0,
            last_hour_reset: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_rsi(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI when insufficient data
        }

        let recent_prices: Vec<f64> = prices.iter().rev().take(period + 1).cloned().collect();
        let mut gains = Vec::new();
        let mut losses = Vec::new();

        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i-1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(change.abs());
            }
        }

        let avg_gain: f64 = gains.iter().sum::<f64>() / gains.len() as f64;
        let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
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
        if volumes.len() < period * 2 {
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

    fn detect_rsi_signal(&self, current_rsi: f64, previous_rsi: f64) -> (f64, f64) {
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Oversold condition (RSI below threshold)
        if current_rsi <= self.config.rsi.oversold_threshold {
            let oversold_strength = (self.config.rsi.oversold_threshold - current_rsi) / self.config.rsi.oversold_threshold;
            signal_strength = oversold_strength.min(1.0);
            signal_direction = 1.0; // Buy signal
        }
        // Overbought condition (RSI above threshold)
        else if current_rsi >= self.config.rsi.overbought_threshold {
            let overbought_strength = (current_rsi - self.config.rsi.overbought_threshold) / (100.0 - self.config.rsi.overbought_threshold);
            signal_strength = overbought_strength.min(1.0);
            signal_direction = -1.0; // Sell signal
        }

        // RSI momentum (direction change)
        if previous_rsi > 0.0 {
            let rsi_change = current_rsi - previous_rsi;
            if signal_direction > 0.0 && rsi_change > 0.0 {
                signal_strength += 0.2; // RSI rising from oversold
            } else if signal_direction < 0.0 && rsi_change < 0.0 {
                signal_strength += 0.2; // RSI falling from overbought
            }
        }

        (signal_strength, signal_direction)
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
impl Strategy for RsiHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: RsiHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "RsiHftStrategy - Price: {:.8}, Data Points: {}, RSI: {:.2}, Volatility: {:.4}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.rsi_history.back().unwrap_or(&50.0),
            self.volatility_history.back().unwrap_or(&0.0)
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.rsi.rsi_period * 5 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > self.config.volume.volume_window * 3 {
            self.volume_history.pop_front();
        }

        // Calculate and store RSI
        if self.price_history.len() >= self.config.rsi.rsi_period + 1 {
            let rsi = self.calculate_rsi(&self.price_history, self.config.rsi.rsi_period);
            self.rsi_history.push_back(rsi);
            if self.rsi_history.len() > 100 {
                self.rsi_history.pop_front();
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

        // Debug logging every 100 trades with confidence information
        if self.price_history.len() % 100 == 0 {
            let rsi_value = self.rsi_history.back().unwrap_or(&50.0);
            
            // Log RSI and confidence information
            self.logger.log_signal_generated(
                &trade.symbol,
                &Signal::Hold,
                *rsi_value / 100.0, // Use RSI as confidence (0.0-1.0)
                trade.price,
            );
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

        // Check signal cooldown
        if current_time - self.last_signal_time < self.config.signals.signal_cooldown {
            return (Signal::Hold, 0.0);
        }

        if self.price_history.len() < self.config.rsi.rsi_period + 1 {
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();
        let current_rsi = *self.rsi_history.back().unwrap();
        let previous_rsi = self.rsi_history.iter().rev().nth(1).unwrap_or(&50.0);

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Detect RSI signal
        let (rsi_strength, rsi_direction) = self.detect_rsi_signal(current_rsi, *previous_rsi);
        
        // Calculate volatility
        let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
        
        // Calculate volume momentum
        let volume_momentum = self.calculate_volume_momentum(&self.volume_history, self.config.volume.volume_window);

        // Generate final signal
        let mut signal_strength = rsi_strength;

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

        // Generate final signal
        if signal_strength >= self.config.signals.min_confidence {
            self.last_signal_time = current_time;
            self.trades_this_hour += 1;
            
            if rsi_direction > 0.0 && current_position.quantity == 0.0 {
                (Signal::Buy, signal_strength)
            } else if rsi_direction < 0.0 && current_position.quantity > 0.0 {
                (Signal::Sell, signal_strength)
            } else {
                (Signal::Hold, 0.0)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }
}
