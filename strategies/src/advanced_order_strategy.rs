use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use crate::{models::*, strategy::{Strategy, StrategyLogger, NoOpStrategyLogger}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedOrderConfig {
    pub general: GeneralConfig,
    pub twap: TwapConfig,
    pub vwap: VwapConfig,
    pub order_types: OrderTypeConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    pub time_slice_duration: f64,  // Duration of each time slice in seconds
    pub max_slices: usize,          // Maximum number of time slices
    pub min_slice_size: f64,        // Minimum order size per slice
    pub price_deviation_limit: f64, // Maximum price deviation from target
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
    pub calculation_period: usize,  // Period for VWAP calculation
    pub deviation_threshold: f64,   // Price deviation threshold from VWAP
    pub volume_threshold: f64,      // Volume threshold for signal generation
    pub reversion_strength: f64,    // Strength of mean reversion signal
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderTypeConfig {
    pub ioc_timeout: f64,           // IOC order timeout in seconds
    pub fok_min_fill_ratio: f64,    // Minimum fill ratio for FOK orders
    pub market_impact_threshold: f64, // Market impact threshold
    pub slippage_tolerance: f64,    // Slippage tolerance percentage
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub strategy_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    pub min_confidence: f64,
    pub max_position_time: f64,
    pub profit_target: f64,
    pub stop_loss: f64,
    pub max_trades_per_hour: usize,
    pub signal_cooldown: f64,
}

#[derive(Debug, Clone)]
pub struct TwapExecution {
    pub target_price: f64,
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_slices: usize,
    pub start_time: f64,
    pub last_execution_time: f64,
}

#[derive(Debug, Clone)]
pub struct VwapData {
    pub vwap_value: f64,
    pub volume_profile: VecDeque<f64>,
    pub price_profile: VecDeque<f64>,
    pub last_update: f64,
}

pub struct AdvancedOrderStrategy {
    config: AdvancedOrderConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    twap_executions: VecDeque<TwapExecution>,
    vwap_data: VwapData,
    last_signal_time: f64,
    trades_this_hour: usize,
    last_hour_reset: f64,
    current_timestamp: f64,
    logger: Box<dyn StrategyLogger>,
}

impl AdvancedOrderStrategy {
    pub fn new(config: AdvancedOrderConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            twap_executions: VecDeque::new(),
            vwap_data: VwapData {
                vwap_value: 0.0,
                volume_profile: VecDeque::new(),
                price_profile: VecDeque::new(),
                last_update: 0.0,
            },
            last_signal_time: 0.0,
            trades_this_hour: 0,
            last_hour_reset: 0.0,
            current_timestamp: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_vwap(&mut self) -> f64 {
        if self.price_history.len() < self.config.vwap.calculation_period {
            return 0.0;
        }

        let period = self.config.vwap.calculation_period;
        let recent_prices: Vec<f64> = self.price_history.iter().rev().take(period).cloned().collect();
        let recent_volumes: Vec<f64> = self.volume_history.iter().rev().take(period).cloned().collect();

        if recent_prices.len() != recent_volumes.len() || recent_prices.is_empty() {
            return 0.0;
        }

        let total_volume: f64 = recent_volumes.iter().sum();
        if total_volume == 0.0 {
            return 0.0;
        }

        let volume_weighted_sum: f64 = recent_prices.iter()
            .zip(recent_volumes.iter())
            .map(|(price, volume)| price * volume)
            .sum();

        volume_weighted_sum / total_volume
    }

    fn should_exit_position(&self, current_position: &Position, current_price: f64) -> bool {
        if current_position.quantity == 0.0 {
            return false;
        }
        
        let entry_price = current_position.entry_price;
        let profit_pct = if current_position.quantity > 0.0 {
            (current_price - entry_price) / entry_price * 100.0
        } else {
            (entry_price - current_price) / entry_price * 100.0
        };
        
        profit_pct >= self.config.signals.profit_target || 
        profit_pct <= -self.config.signals.stop_loss
    }

    fn reset_hourly_trades(&mut self, current_time: f64) {
        if current_time - self.last_hour_reset >= 3600.0 {
            self.trades_this_hour = 0;
            self.last_hour_reset = current_time;
        }
    }

    fn should_generate_signal(&self, current_time: f64) -> bool {
        let time_since_last = current_time - self.last_signal_time;
        time_since_last >= self.config.signals.signal_cooldown
    }

    fn generate_twap_signal(&self, current_price: f64, _current_volume: f64) -> Option<(Signal, f64)> {
        if self.twap_executions.is_empty() {
            return None;
        }

        let current_execution = self.twap_executions.front()?;
        let time_since_start = self.current_timestamp - current_execution.start_time;
        let time_slice_duration = self.config.twap.time_slice_duration;
        
        // Check if it's time for next slice
        if time_since_start >= time_slice_duration {
            let slice_number = (time_since_start / time_slice_duration) as usize;
            
            if slice_number < current_execution.remaining_slices {
                let price_deviation = (current_price - current_execution.target_price).abs() / current_execution.target_price;
                
                if price_deviation <= self.config.twap.price_deviation_limit {
                    let confidence = (1.0 - price_deviation / self.config.twap.price_deviation_limit).max(0.6);
                    return Some((Signal::Buy, confidence));
                }
            }
        }

        None
    }

    fn generate_vwap_signal(&self, current_price: f64, current_volume: f64) -> Option<(Signal, f64)> {
        if self.vwap_data.vwap_value == 0.0 {
            return None;
        }

        let price_deviation = (current_price - self.vwap_data.vwap_value) / self.vwap_data.vwap_value;
        let volume_ratio = if !self.volume_history.is_empty() {
            let avg_volume: f64 = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
            current_volume / avg_volume
        } else {
            1.0
        };

        // VWAP mean reversion signals with improved logic
        if price_deviation.abs() > self.config.vwap.deviation_threshold {
            if price_deviation < -self.config.vwap.deviation_threshold && volume_ratio > self.config.vwap.volume_threshold {
                // Price below VWAP with high volume - buy signal
                let confidence = (price_deviation.abs() / 0.1).min(0.9) * self.config.vwap.reversion_strength;
                return Some((Signal::Buy, confidence.max(0.7)));
            } else if price_deviation > self.config.vwap.deviation_threshold && volume_ratio > self.config.vwap.volume_threshold {
                // Price above VWAP with high volume - sell signal
                let confidence = (price_deviation / 0.1).min(0.9) * self.config.vwap.reversion_strength;
                return Some((Signal::Sell, confidence.max(0.7)));
            }
        }

        // Additional momentum signals for better profitability
        if self.price_history.len() >= 10 {
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(10).cloned().collect();
            let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
            let momentum = (current_price - avg_price) / avg_price;
            
            // Strong momentum with volume confirmation
            if momentum.abs() > 0.002 && volume_ratio > 1.8 {
                if momentum > 0.002 {
                    return Some((Signal::Buy, 0.8));
                } else {
                    return Some((Signal::Sell, 0.8));
                }
            }
        }

        None
    }

    fn generate_ioc_fok_signal(&self, current_price: f64, current_volume: f64) -> Option<(Signal, f64)> {
        if self.price_history.len() < 20 {
            return None;
        }

        // Calculate market impact and volatility
        let recent_prices: Vec<f64> = self.price_history.iter().rev().take(20).cloned().collect();
        let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let volatility = recent_prices.iter()
            .map(|&p| (p - avg_price).abs())
            .sum::<f64>() / recent_prices.len() as f64;

        let price_change = (current_price - avg_price) / avg_price;
        let volume_spike = if !self.volume_history.is_empty() {
            let avg_volume: f64 = self.volume_history.iter().sum::<f64>() / self.volume_history.len() as f64;
            current_volume / avg_volume
        } else {
            1.0
        };

        // IOC signals for high volatility periods with improved thresholds
        if volatility > self.config.order_types.market_impact_threshold {
            if price_change > 0.0015 && volume_spike > 1.8 {
                // Strong upward move with volume - IOC buy
                let confidence = (price_change / 0.01).min(0.85);
                return Some((Signal::Buy, confidence.max(0.7)));
            } else if price_change < -0.0015 && volume_spike > 1.8 {
                // Strong downward move with volume - IOC sell
                let confidence = (price_change.abs() / 0.01).min(0.85);
                return Some((Signal::Sell, confidence.max(0.7)));
            }
        }

        // FOK signals for low volatility, high volume periods
        if volatility < self.config.order_types.market_impact_threshold * 0.6 && volume_spike > 2.2 {
            if price_change.abs() < self.config.order_types.slippage_tolerance {
                // Stable price with high volume - FOK order
                let confidence = (volume_spike / 3.0).min(0.9);
                return Some((Signal::Buy, confidence.max(0.7)));
            }
        }

        None
    }

    fn generate_advanced_signal(&self, current_price: f64, current_volume: f64) -> Option<(Signal, f64)> {
        // Try VWAP signal first (most profitable)
        if let Some(signal) = self.generate_vwap_signal(current_price, current_volume) {
            return Some(signal);
        }

        // Try IOC/FOK signal (good for momentum)
        if let Some(signal) = self.generate_ioc_fok_signal(current_price, current_volume) {
            return Some(signal);
        }

        // Try TWAP signal last (most conservative)
        if let Some(signal) = self.generate_twap_signal(current_price, current_volume) {
            return Some(signal);
        }

        None
    }

    fn update_twap_executions(&mut self) {
        let current_time = self.current_timestamp;
        
        // Remove completed executions
        self.twap_executions.retain(|exec| {
            let time_since_start = current_time - exec.start_time;
            let max_duration = exec.remaining_slices as f64 * self.config.twap.time_slice_duration;
            time_since_start < max_duration
        });

        // Update remaining slices for active executions
        for exec in &mut self.twap_executions {
            let time_since_start = current_time - exec.start_time;
            let slice_number = (time_since_start / self.config.twap.time_slice_duration) as usize;
            exec.remaining_slices = exec.remaining_slices.saturating_sub(slice_number);
        }
    }
}

#[async_trait]
impl Strategy for AdvancedOrderStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        self.logger.as_ref()
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: AdvancedOrderConfig = toml::from_str(&config_content)?;
        Ok(Self::new(config))
    }

    fn get_info(&self) -> String {
        let current_price = self.price_history.back().unwrap_or(&0.0);
        let current_volume = self.volume_history.back().unwrap_or(&0.0);
        let vwap = self.vwap_data.vwap_value;
        let active_twap = self.twap_executions.len();
        
        let time_since_last = if self.last_signal_time > 0.0 {
            self.current_timestamp - self.last_signal_time
        } else {
            0.0
        };
        
        format!(
            "Advanced Order Strategy - Price: {:.8}, Volume: {:.2}, VWAP: {:.8}, Active TWAP: {}, Trades this hour: {}, Last signal: {:.1}s ago",
            current_price, current_volume, vwap, active_twap, self.trades_this_hour, time_since_last
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.current_timestamp = trade.timestamp;

        // Update price and volume history
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.quantity);
        
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > 1000 {
            self.volume_history.pop_front();
        }

        // Update VWAP
        let vwap = self.calculate_vwap();
        self.vwap_data.vwap_value = vwap;
        self.vwap_data.price_profile.push_back(trade.price);
        self.vwap_data.volume_profile.push_back(trade.quantity);
        
        if self.vwap_data.price_profile.len() > 1000 {
            self.vwap_data.price_profile.pop_front();
        }
        if self.vwap_data.volume_profile.len() > 1000 {
            self.vwap_data.volume_profile.pop_front();
        }

        // Update TWAP executions
        self.update_twap_executions();

        // Log every 100 trades
        if self.price_history.len() % 100 == 0 {
            self.logger.log_signal_generated(
                &trade.symbol,
                &Signal::Hold,
                0.5,
                trade.price,
            );
        }
    }

    fn get_signal(&mut self, current_position: Position) -> (Signal, f64) {
        let current_time = self.current_timestamp;

        // Reset hourly trade counter
        self.reset_hourly_trades(current_time);

        // Check if we've exceeded hourly trade limit
        if self.trades_this_hour >= self.config.signals.max_trades_per_hour {
            return (Signal::Hold, 0.0);
        }

        // Check if we should exit current position
        if let Some(current_price) = self.price_history.back() {
            if self.should_exit_position(&current_position, *current_price) {
                // Return exit signal instead of hold
                if current_position.quantity > 0.0 {
                    return (Signal::Sell, 0.9);
                } else if current_position.quantity < 0.0 {
                    return (Signal::Buy, 0.9);
                }
            }
        }

        // Check if it's time to generate a signal
        if self.should_generate_signal(current_time) {
            let current_price = *self.price_history.back().unwrap_or(&0.0);
            let current_volume = *self.volume_history.back().unwrap_or(&0.0);

            // Generate advanced signal
            if let Some((signal, confidence)) = self.generate_advanced_signal(current_price, current_volume) {
                if confidence >= self.config.signals.min_confidence {
                    self.last_signal_time = current_time;
                    self.trades_this_hour += 1;
                    return (signal, confidence);
                }
            }
        }

        (Signal::Hold, 0.0)
    }
}
