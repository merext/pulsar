use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct TrendFollowingHftConfig {
    pub general: GeneralConfig,
    pub trend: TrendConfig,
    pub momentum: MomentumConfig,
    pub confirmation: ConfirmationConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct TrendConfig {
    pub short_window: usize,      // Short-term trend window
    pub long_window: usize,       // Long-term trend window
    pub trend_strength: f64,      // Minimum trend strength
    pub trend_confirmation: f64,  // Trend confirmation threshold
}

#[derive(Debug, Deserialize)]
pub struct MomentumConfig {
    pub momentum_window: usize,   // Window for momentum calculation
    pub momentum_threshold: f64,  // Momentum threshold for entry
    pub acceleration_threshold: f64, // Acceleration threshold
    pub momentum_decay: f64,      // Momentum decay factor
}

#[derive(Debug, Deserialize)]
pub struct ConfirmationConfig {
    pub volume_confirmation: bool, // Whether to require volume confirmation
    pub volume_threshold: f64,    // Volume confirmation threshold
    pub price_confirmation: bool, // Whether to require price confirmation
    pub confirmation_window: usize, // Window for confirmation
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,      // Minimum confidence for trade
    pub max_position_time: f64,   // Maximum time to hold position
    pub profit_target: f64,       // Profit target percentage
    pub stop_loss: f64,           // Stop loss percentage
    pub max_trades_per_minute: usize, // Maximum trades per minute
}

pub struct TrendFollowingHftStrategy {
    config: TrendFollowingHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    short_ma_history: VecDeque<f64>,
    long_ma_history: VecDeque<f64>,
    momentum_history: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_minute: usize,
    last_minute_reset: f64,
    current_trend: f64, // 1.0 for uptrend, -1.0 for downtrend, 0.0 for sideways
    logger: Box<dyn StrategyLogger>,
}

impl TrendFollowingHftStrategy {
    pub fn new(config: TrendFollowingHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            short_ma_history: VecDeque::new(),
            long_ma_history: VecDeque::new(),
            momentum_history: VecDeque::new(),
            last_signal_time: 0.0,
            trades_this_minute: 0,
            last_minute_reset: 0.0,
            current_trend: 0.0,
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

    fn calculate_momentum(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
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

    fn calculate_trend_strength(&self, short_ma: f64, long_ma: f64) -> f64 {
        if long_ma <= 0.0 {
            return 0.0;
        }
        
        let trend_diff = (short_ma - long_ma) / long_ma * 100.0;
        trend_diff.abs()
    }

    fn detect_trend(&self, short_ma: f64, long_ma: f64) -> (f64, f64) {
        if long_ma <= 0.0 {
            return (0.0, 0.0);
        }
        
        let trend_diff = (short_ma - long_ma) / long_ma * 100.0;
        let trend_strength = self.calculate_trend_strength(short_ma, long_ma);
        
        let trend_direction = if trend_strength > self.config.trend.trend_strength {
            if trend_diff > 0.0 {
                1.0 // Uptrend
            } else {
                -1.0 // Downtrend
            }
        } else {
            0.0 // Sideways
        };
        
        (trend_direction, trend_strength)
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
impl Strategy for TrendFollowingHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: TrendFollowingHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "TrendFollowingHftStrategy - Price: {:.8}, Data Points: {}, Trend: {:.2}, Short MA: {:.8}, Long MA: {:.8}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.current_trend,
            self.short_ma_history.back().unwrap_or(&0.0),
            self.long_ma_history.back().unwrap_or(&0.0)
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.trend.long_window * 3 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }

        // Calculate and store short-term moving average
        if self.price_history.len() >= self.config.trend.short_window {
            let short_ma = self.calculate_moving_average(&self.price_history, self.config.trend.short_window);
            self.short_ma_history.push_back(short_ma);
            if self.short_ma_history.len() > 100 {
                self.short_ma_history.pop_front();
            }
        }

        // Calculate and store long-term moving average
        if self.price_history.len() >= self.config.trend.long_window {
            let long_ma = self.calculate_moving_average(&self.price_history, self.config.trend.long_window);
            self.long_ma_history.push_back(long_ma);
            if self.long_ma_history.len() > 100 {
                self.long_ma_history.pop_front();
            }
        }

        // Calculate and store momentum
        if self.price_history.len() >= self.config.momentum.momentum_window {
            let momentum = self.calculate_momentum(&self.price_history, self.config.momentum.momentum_window);
            self.momentum_history.push_back(momentum);
            if self.momentum_history.len() > 100 {
                self.momentum_history.pop_front();
            }
        }

        // Debug logging every 100 trades
        if self.price_history.len() % 100 == 0 {
            println!("DEBUG: Processed {} trades, price: {:.8}, trend: {:.2}, short_ma: {:.8}, long_ma: {:.8}", 
                self.price_history.len(), 
                trade.price,
                self.current_trend,
                self.short_ma_history.back().unwrap_or(&0.0),
                self.long_ma_history.back().unwrap_or(&0.0));
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

        if self.price_history.len() < self.config.trend.long_window {
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();
        let short_ma = *self.short_ma_history.back().unwrap();
        let long_ma = *self.long_ma_history.back().unwrap();

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Detect trend
        let (trend_direction, trend_strength) = self.detect_trend(short_ma, long_ma);
        self.current_trend = trend_direction;
        
        // Calculate momentum
        let momentum = self.calculate_momentum(&self.price_history, self.config.momentum.momentum_window);
        
        // Calculate volume momentum
        let volume_momentum = self.calculate_volume_momentum(&self.volume_history, self.config.confirmation.confirmation_window);

        // Generate signal based on trend following
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Trend analysis
        if trend_strength > self.config.trend.trend_strength {
            signal_strength += 0.4;
            
            if trend_direction > 0.0 {
                signal_direction = 1.0; // Buy on uptrend
            } else if trend_direction < 0.0 {
                signal_direction = -1.0; // Sell on downtrend
            }
        }

        // Momentum analysis
        if momentum.abs() > self.config.momentum.momentum_threshold {
            signal_strength += 0.3;
            
            // Confirm trend direction with momentum
            if (trend_direction > 0.0 && momentum > 0.0) || (trend_direction < 0.0 && momentum < 0.0) {
                signal_strength += 0.2; // Strong confirmation
            }
        }

        // Volume confirmation
        if self.config.confirmation.volume_confirmation {
            if volume_momentum > self.config.confirmation.volume_threshold {
                signal_strength += 0.2;
            }
        } else {
            signal_strength += 0.1; // Volume not required
        }

        // Price confirmation
        if self.config.confirmation.price_confirmation {
            if trend_direction > 0.0 && current_price > short_ma {
                signal_strength += 0.1; // Price above short MA confirms uptrend
            } else if trend_direction < 0.0 && current_price < short_ma {
                signal_strength += 0.1; // Price below short MA confirms downtrend
            }
        }

        // Time-based signal filtering
        if current_time - self.last_signal_time < 5.0 {
            signal_strength *= 0.7; // Reduce signal strength for rapid signals
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
