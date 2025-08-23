use crate::models::{Position, Signal, TradeData};
use crate::strategy::{Strategy, StrategyLogger, NoOpStrategyLogger};
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::Path;
use async_trait::async_trait;

#[derive(Debug, Deserialize)]
pub struct ScalpingHftConfig {
    pub general: GeneralConfig,
    pub scalping: ScalpingConfig,
    pub micro_movements: MicroMovementsConfig,
    pub risk_management: RiskManagementConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Deserialize)]
pub struct ScalpingConfig {
    pub entry_threshold: f64,     // Minimum price movement for entry
    pub exit_threshold: f64,      // Price movement for exit
    pub max_hold_time: f64,       // Maximum time to hold position
    pub min_trade_interval: f64,  // Minimum time between trades
}

#[derive(Debug, Deserialize)]
pub struct MicroMovementsConfig {
    pub price_window: usize,      // Window for price analysis
    pub momentum_threshold: f64,  // Momentum threshold for entry
    pub reversal_threshold: f64,  // Reversal detection threshold
    pub noise_filter: f64,        // Filter out market noise
}

#[derive(Debug, Deserialize)]
pub struct RiskManagementConfig {
    pub max_position_size: f64,   // Maximum position size
    pub max_daily_loss: f64,      // Maximum daily loss
    pub max_consecutive_losses: usize, // Maximum consecutive losses
    pub profit_lock_threshold: f64, // Lock profits threshold
}

#[derive(Debug, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,      // Minimum confidence for trade
    pub profit_target: f64,       // Profit target percentage
    pub stop_loss: f64,           // Stop loss percentage
    pub max_trades_per_minute: usize, // Maximum trades per minute
}

pub struct ScalpingHftStrategy {
    config: ScalpingHftConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    momentum_history: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_minute: usize,
    last_minute_reset: f64,
    consecutive_losses: usize,
    daily_pnl: f64,
    last_day_reset: f64,
    logger: Box<dyn StrategyLogger>,
}

impl ScalpingHftStrategy {
    pub fn new(config: ScalpingHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            momentum_history: VecDeque::<f64>::new(),
            last_signal_time: 0.0,
            trades_this_minute: 0,
            last_minute_reset: 0.0,
            consecutive_losses: 0,
            daily_pnl: 0.0,
            last_day_reset: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_micro_momentum(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period * 2 {
            return 0.0;
        }
        
        let recent_avg = prices.iter().rev().take(period).sum::<f64>() / period as f64;
        let older_avg = prices.iter().rev().skip(period).take(period).sum::<f64>() / period as f64;
        
        if older_avg > 0.0 {
            (recent_avg - older_avg) / older_avg * 1000000.0 // Micro percentage
        } else {
            0.0
        }
    }

    fn detect_micro_reversal(&self, current_price: f64) -> (bool, f64) {
        if self.price_history.len() < self.config.micro_movements.price_window {
            return (false, 0.0);
        }
        
        // Calculate short-term trend using iterators directly
        let first_price = self.price_history.back().unwrap_or(&0.0);
        let fifth_price = self.price_history.iter().rev().nth(4).unwrap_or(&0.0);
        
        if *fifth_price <= 0.0 {
            return (false, 0.0);
        }
        
        let trend = (*first_price - *fifth_price) / *fifth_price * 1000000.0;
        
        // Detect reversal if trend changes direction
        let reversal_strength = if trend.abs() > self.config.micro_movements.reversal_threshold {
            trend.abs() / self.config.micro_movements.reversal_threshold
        } else {
            0.0
        };
        
        (reversal_strength > 0.0, reversal_strength.min(1.0))
    }

    fn should_exit_position(&self, current_price: f64, entry_price: f64, entry_time: f64, current_time: f64) -> bool {
        // Time-based exit
        if current_time - entry_time > self.config.scalping.max_hold_time {
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

    fn reset_daily_stats(&mut self, current_time: f64) {
        if current_time - self.last_day_reset >= 86400.0 { // 24 hours
            self.daily_pnl = 0.0;
            self.consecutive_losses = 0;
            self.last_day_reset = current_time;
        }
    }

    fn can_trade(&self, current_time: f64) -> bool {
        // Check daily loss limit
        if self.daily_pnl <= -self.config.risk_management.max_daily_loss {
            return false;
        }
        
        // Check consecutive losses
        if self.consecutive_losses >= self.config.risk_management.max_consecutive_losses {
            return false;
        }
        
        // Check trade interval
        if current_time - self.last_signal_time < self.config.scalping.min_trade_interval {
            return false;
        }
        
        true
    }
}

#[async_trait]
impl Strategy for ScalpingHftStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: ScalpingHftConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        format!(
            "ScalpingHftStrategy - Price: {:.8}, Data Points: {}, Momentum: {:.2}, Daily PnL: {:.6}",
            self.price_history.back().unwrap_or(&0.0),
            self.price_history.len(),
            self.momentum_history.back().unwrap_or(&0.0),
            self.daily_pnl
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > self.config.micro_movements.price_window * 3 {
            self.price_history.pop_front();
        }

        // Update volume history
        self.volume_history.push_back(trade.quantity);
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }

        // Calculate and store micro momentum
        if self.price_history.len() >= self.config.micro_movements.price_window {
            let momentum = self.calculate_micro_momentum(&self.price_history, self.config.micro_movements.price_window);
            self.momentum_history.push_back(momentum);
            if self.momentum_history.len() > 100 {
                self.momentum_history.pop_front();
            }
        }

        // Debug logging every 50 trades
        if self.price_history.len() % 50 == 0 {
            println!("DEBUG: Processed {} trades, price: {:.8}, momentum: {:.2}, daily_pnl: {:.6}", 
                self.price_history.len(), 
                trade.price,
                self.momentum_history.back().unwrap_or(&0.0),
                self.daily_pnl);
        }
    }

    fn get_signal(&mut self, current_position: Position) -> (Signal, f64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Reset counters
        self.reset_minute_trades(current_time);
        self.reset_daily_stats(current_time);

        // Check if we can trade
        if !self.can_trade(current_time) {
            return (Signal::Hold, 0.0);
        }

        // Check if we've exceeded minute trade limit
        if self.trades_this_minute >= self.config.signals.max_trades_per_minute {
            return (Signal::Hold, 0.0);
        }

        if self.price_history.len() < self.config.micro_movements.price_window {
            return (Signal::Hold, 0.0);
        }

        let current_price = *self.price_history.back().unwrap();

        // Check if we should exit existing position
        if current_position.quantity > 0.0 {
            if self.should_exit_position(current_price, current_position.entry_price, current_time, current_time) {
                return (Signal::Sell, 0.9);
            }
        }

        // Calculate micro momentum
        let momentum = self.calculate_micro_momentum(&self.price_history, self.config.micro_movements.price_window);
        
        // Detect micro reversal
        let (reversal_detected, reversal_strength) = self.detect_micro_reversal(current_price);

        // Generate signal based on micro movements
        let mut signal_strength = 0.0;
        let mut signal_direction = 0.0;

        // Momentum analysis
        if momentum.abs() > self.config.micro_movements.momentum_threshold {
            signal_strength += 0.4;
            
            if momentum > 0.0 {
                signal_direction = 1.0; // Buy on positive momentum
            } else {
                signal_direction = -1.0; // Sell on negative momentum
            }
        }

        // Reversal analysis
        if reversal_detected {
            signal_strength += reversal_strength * 0.4;
            
            // Reverse the signal direction for reversal trades
            signal_direction = -signal_direction;
        }

        // Noise filtering
        if momentum.abs() < self.config.micro_movements.noise_filter {
            signal_strength *= 0.5; // Reduce signal strength for noise
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
