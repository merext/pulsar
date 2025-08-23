use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct StatisticalArbitrageHftConfig {
    pub general: GeneralConfig,
    pub statistical: StatisticalConfig,
    pub mean_reversion: MeanReversionConfig,
    pub volatility: VolatilityConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct StatisticalConfig {
    pub lookback_period: usize,   // Period for statistical calculations
    pub z_score_threshold: f64,   // Z-score threshold for signals
    pub mean_window: usize,       // Window for mean calculation
    pub std_window: usize,        // Window for standard deviation
}

#[derive(Debug, Deserialize)]
pub struct MeanReversionConfig {
    pub reversion_strength: f64,  // Strength of mean reversion
    pub max_deviation: f64,       // Maximum deviation before extreme signal
    pub reversion_speed: f64,     // Speed of mean reversion
}

#[derive(Debug, Deserialize)]
pub struct VolatilityConfig {
    pub volatility_window: usize, // Window for volatility calculation
    pub volatility_threshold: f64, // Volatility threshold for trading
    pub adaptive_threshold: bool, // Whether to adapt thresholds
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,      // Minimum confidence for trade
    pub max_position_time: f64,   // Maximum time to hold position
    pub profit_target: f64,       // Profit target percentage
    pub stop_loss: f64,           // Stop loss percentage
    pub max_trades_per_minute: usize, // Maximum trades per minute
}

pub struct StatisticalArbitrageHftStrategy {
    config: StatisticalArbitrageHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    mean_history: VecDeque<f64>,
    std_history: VecDeque<f64>,
    z_score_history: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_minute: usize,
    last_minute_reset: f64,
    logger: Box<dyn StrategyLogger>,
}

impl StatisticalArbitrageHftStrategy {
    pub fn new(config: StatisticalArbitrageHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            mean_history: VecDeque::new(),
            std_history: VecDeque::new(),
            z_score_history: VecDeque::new(),
            last_signal_time: 0.0,
            trades_this_minute: 0,
            last_minute_reset: 0.0,
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

    fn calculate_standard_deviation(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
        let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        
        let variance = recent_prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
        
        variance.sqrt()
    }

    fn calculate_z_score(&self, current_price: f64, mean: f64, std: f64) -> f64 {
        if std <= 0.0 {
            return 0.0;
        }
        
        (current_price - mean) / std
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

    fn detect_statistical_anomaly(&self, current_price: f64, mean: f64, std: f64) -> (bool, f64, f64) {
        let z_score = self.calculate_z_score(current_price, mean, std);
        let z_score_abs = z_score.abs();
        
        let is_anomaly = z_score_abs > self.config.statistical.z_score_threshold;
        let anomaly_strength = if is_anomaly {
            (z_score_abs - self.config.statistical.z_score_threshold) / 
            (self.config.statistical.z_score_threshold * 2.0)
        } else {
            0.0
        };
        
        (is_anomaly, z_score, anomaly_strength.min(1.0))
    }

    fn calculate_mean_reversion_probability(&self, z_score: f64, volatility: f64) -> f64 {
        // Higher z-score and lower volatility increase reversion probability
        let z_score_factor = z_score.abs() / self.config.statistical.z_score_threshold;
        let volatility_factor = 1.0 - (volatility / 100.0).min(1.0);
        
        (z_score_factor + volatility_factor) / 2.0
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

    fn reset_minute_trades(&mut self, current_time: f64) {
        if current_time - self.last_minute_reset >= 60.0 {
            self.trades_this_minute = 0;
            self.last_minute_reset = current_time;
        }
    }
}

#[async_trait]
impl Strategy for StatisticalArbitrageHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: StatisticalArbitrageHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "StatisticalArbitrageHftStrategy - Price: {:.8}, Data Points: {}, Mean: {:.8}, Std: {:.8}, Z-Score: {:.4}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.mean_history.back().unwrap_or(&0.0),
            self.std_history.back().unwrap_or(&0.0),
            self.z_score_history.back().unwrap_or(&0.0)
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.statistical.lookback_period * 3 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }

        // Calculate and store moving average
        if self.price_history.len() >= self.config.statistical.mean_window {
            let mean = self.calculate_moving_average(&self.price_history, self.config.statistical.mean_window);
            self.mean_history.push_back(mean);
            if self.mean_history.len() > 100 {
                self.mean_history.pop_front();
            }
        }

        // Calculate and store standard deviation
        if self.price_history.len() >= self.config.statistical.std_window {
            let std = self.calculate_standard_deviation(&self.price_history, self.config.statistical.std_window);
            self.std_history.push_back(std);
            if self.std_history.len() > 100 {
                self.std_history.pop_front();
            }
        }

        // Calculate and store z-score
        if self.mean_history.len() > 0 && self.std_history.len() > 0 {
            let mean = *self.mean_history.back().unwrap();
            let std = *self.std_history.back().unwrap();
            let z_score = self.calculate_z_score(trade.price, mean, std);
            self.z_score_history.push_back(z_score);
            if self.z_score_history.len() > 100 {
                self.z_score_history.pop_front();
            }
        }

        // Debug logging every 100 trades
        if self.price_history.len() % 100 == 0 {
            println!("DEBUG: Processed {} trades, price: {:.8}, mean: {:.8}, std: {:.8}, z_score: {:.4}", 
                self.price_history.len(), 
                trade.price,
                self.mean_history.back().unwrap_or(&0.0),
                self.std_history.back().unwrap_or(&0.0),
                self.z_score_history.back().unwrap_or(&0.0));
        }
    }

    fn get_signal(&mut self, current_position: Position) -> (Signal, f64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Reset minute trade counter
        self.reset_minute_trades(current_time);

        // Check if we've exceeded minute trade limit
        if self.trades_this_minute >= self.config.signals.max_trades_per_minute {
            return (Signal::Hold, 0.0);
        }

        if self.price_history.len() < self.config.statistical.lookback_period {
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();
        let mean = *self.mean_history.back().unwrap();
        let std = *self.std_history.back().unwrap();

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Detect statistical anomaly
        let (is_anomaly, z_score, anomaly_strength) = self.detect_statistical_anomaly(current_price, mean, std);
        
        // Calculate volatility
        let volatility = self.calculate_volatility(&self.price_history, self.config.volatility.volatility_window);
        
        // Calculate mean reversion probability
        let reversion_probability = self.calculate_mean_reversion_probability(z_score, volatility);

        // Generate signal based on statistical arbitrage
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Statistical anomaly analysis
        if is_anomaly {
            signal_strength += anomaly_strength * 0.4;
            
            // Determine direction based on z-score
            if z_score > 0.0 {
                // Price above mean - expect reversion down (sell)
                signal_direction = -1.0;
            } else {
                // Price below mean - expect reversion up (buy)
                signal_direction = 1.0;
            }
        }

        // Mean reversion analysis
        if reversion_probability > 0.5 {
            signal_strength += reversion_probability * 0.3;
        }

        // Volatility analysis
        if volatility > self.config.volatility.volatility_threshold {
            signal_strength += 0.2;
        }

        // Time-based signal filtering
        if current_time - self.last_signal_time < 3.0 {
            signal_strength *= 0.6; // Reduce signal strength for rapid signals
        }

        // Generate final signal
        if signal_strength >= self.config.signals.min_confidence {
            self.last_signal_time = current_time;
            self.trades_this_minute += 1;
            
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
