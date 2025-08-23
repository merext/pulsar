use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct VolumeBreakoutHftConfig {
    pub general: GeneralConfig,
    pub volume: VolumeConfig,
    pub price: PriceConfig,
    pub breakout: BreakoutConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct VolumeConfig {
    pub volume_window: usize,      // Window for volume analysis
    pub volume_threshold: f64,     // Volume spike threshold
    pub volume_momentum: f64,      // Volume momentum threshold
    pub adaptive_volume: bool,     // Whether to adapt volume thresholds
}

#[derive(Debug, Deserialize)]
pub struct PriceConfig {
    pub price_window: usize,       // Window for price analysis
    pub price_threshold: f64,      // Price movement threshold
    pub price_acceleration: f64,   // Price acceleration threshold
}

#[derive(Debug, Deserialize)]
pub struct BreakoutConfig {
    pub breakout_window: usize,    // Window for breakout detection
    pub breakout_threshold: f64,   // Breakout confirmation threshold
    pub false_breakout_filter: f64, // Filter for false breakouts
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,      // Minimum confidence for trade
    pub max_position_time: f64,   // Maximum time to hold position
    pub profit_target: f64,       // Profit target percentage
    pub stop_loss: f64,           // Stop loss percentage
    pub max_trades_per_minute: usize, // Maximum trades per minute
}

pub struct VolumeBreakoutHftStrategy {
    config: VolumeBreakoutHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    volume_ma_history: VecDeque<f64>,
    price_momentum_history: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_minute: usize,
    last_minute_reset: f64,
    logger: Box<dyn StrategyLogger>,
}

impl VolumeBreakoutHftStrategy {
    pub fn new(config: VolumeBreakoutHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            volume_ma_history: VecDeque::new(),
            price_momentum_history: VecDeque::new(),
            last_signal_time: 0.0,
            trades_this_minute: 0,
            last_minute_reset: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_volume_moving_average(&self, volumes: &VecDeque<f64>, period: usize) -> f64 {
        if volumes.len() < period {
            return 0.0;
        }
        
        let recent_volumes: Vec<f64> = volumes.iter().rev().take(period).cloned().collect();
        recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64
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

    fn calculate_price_momentum(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period * 2 {
            return 0.0;
        }
        
        let recent_avg = prices.iter().rev().take(period).sum::<f64>() / period as f64;
        let older_avg = prices.iter().rev().skip(period).take(period).sum::<f64>() / period as f64;
        
        if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg * 100.0
        } else {
            0.0
        }
    }

    fn calculate_price_acceleration(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period * 3 {
            return 0.0;
        }
        
        let recent_momentum = self.calculate_price_momentum(prices, period);
        let older_momentum = self.calculate_price_momentum(prices, period * 2);
        
        recent_momentum - older_momentum
    }

    fn detect_volume_breakout(&self, current_volume: f64, volume_ma: f64) -> (bool, f64) {
        if volume_ma <= 0.0 {
            return (false, 0.0);
        }
        
        let volume_ratio = current_volume / volume_ma;
        let breakout_strength = if volume_ratio > self.config.volume.volume_threshold {
            (volume_ratio - self.config.volume.volume_threshold) / 
            (self.config.volume.volume_threshold * 2.0) // Normalize to 0-1 range
        } else {
            0.0
        };
        
        (volume_ratio > self.config.volume.volume_threshold, breakout_strength.min(1.0))
    }

    fn detect_price_breakout(&self, current_price: f64) -> (bool, f64) {
        if self.price_history.len() < self.config.breakout.breakout_window {
            return (false, 0.0);
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter().rev().take(self.config.breakout.breakout_window).cloned().collect();
        let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        
        let price_deviation = (current_price - avg_price) / avg_price * 100.0;
        let breakout_strength = if price_deviation.abs() > self.config.breakout.breakout_threshold {
            (price_deviation.abs() - self.config.breakout.breakout_threshold) / 
            (self.config.breakout.breakout_threshold * 2.0)
        } else {
            0.0
        };
        
        (price_deviation.abs() > self.config.breakout.breakout_threshold, breakout_strength.min(1.0))
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
        if current_time - self.last_minute_reset >= 60.0 { // 1 minute
            self.trades_this_minute = 0;
            self.last_minute_reset = current_time;
        }
    }
}

#[async_trait]
impl Strategy for VolumeBreakoutHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: VolumeBreakoutHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "VolumeBreakoutHftStrategy - Price: {:.8}, Data Points: {}, Volume MA: {:.2}, Price Momentum: {:.4}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.volume_ma_history.back().unwrap_or(&0.0),
            self.price_momentum_history.back().unwrap_or(&0.0)
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.price.price_window * 3 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > self.config.volume.volume_window * 3 {
            self.volume_history.pop_front();
        }

        // Calculate and store volume moving average
        if self.volume_history.len() >= self.config.volume.volume_window {
            let volume_ma = self.calculate_volume_moving_average(&self.volume_history, self.config.volume.volume_window);
            self.volume_ma_history.push_back(volume_ma);
            if self.volume_ma_history.len() > 100 {
                self.volume_ma_history.pop_front();
            }
        }

        // Calculate and store price momentum
        if self.price_history.len() >= self.config.price.price_window {
            let price_momentum = self.calculate_price_momentum(&self.price_history, self.config.price.price_window);
            self.price_momentum_history.push_back(price_momentum);
            if self.price_momentum_history.len() > 100 {
                self.price_momentum_history.pop_front();
            }
        }

        // Debug logging every 100 trades
        if self.price_history.len() % 100 == 0 {
            println!("DEBUG: Processed {} trades, price: {:.8}, volume: {:.2}, volume_ma: {:.2}", 
                self.price_history.len(), 
                trade.price,
                trade.quantity,
                self.volume_ma_history.back().unwrap_or(&0.0));
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

        if self.price_history.len() < self.config.price.price_window || 
           self.volume_history.len() < self.config.volume.volume_window {
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();
        let current_volume = *self.volume_history.back().unwrap();
        let volume_ma = *self.volume_ma_history.back().unwrap();

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Detect volume breakout
        let (volume_breakout, volume_strength) = self.detect_volume_breakout(current_volume, volume_ma);
        
        // Detect price breakout
        let (price_breakout, price_strength) = self.detect_price_breakout(current_price);
        
        // Calculate volume momentum
        let volume_momentum = self.calculate_volume_momentum(&self.volume_history, self.config.volume.volume_window);
        
        // Calculate price acceleration
        let price_acceleration = self.calculate_price_acceleration(&self.price_history, self.config.price.price_window);

        // Generate signal based on volume and price breakouts
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Volume breakout analysis
        if volume_breakout {
            signal_strength += volume_strength * 0.4;
            
            // Determine direction based on price movement
            if price_acceleration > self.config.price.price_acceleration {
                signal_strength += 0.3;
                signal_direction = 1.0; // Buy on volume + upward price acceleration
            } else if price_acceleration < -self.config.price.price_acceleration {
                signal_strength += 0.3;
                signal_direction = -1.0; // Sell on volume + downward price acceleration
            }
        }

        // Price breakout analysis
        if price_breakout {
            signal_strength += price_strength * 0.3;
            
            // Confirm with volume
            if volume_momentum > self.config.volume.volume_momentum {
                signal_strength += 0.2;
            }
        }

        // Time-based signal filtering
        if current_time - self.last_signal_time < 3.0 {
            signal_strength *= 0.5; // Reduce signal strength for rapid signals
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
