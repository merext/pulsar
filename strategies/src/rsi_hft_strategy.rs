use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use crate::{models::*, strategy::{Strategy, StrategyLogger, NoOpStrategyLogger}};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsiHftConfig {
    pub general: GeneralConfig,
    pub rsi: RsiConfig,
    pub trend: TrendConfig,
    pub signals: SignalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsiConfig {
    pub period: usize,
    pub oversold_threshold: f64,
    pub overbought_threshold: f64,
    pub signal_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendConfig {
    pub ma_short_period: usize,
    pub ma_long_period: usize,
    pub trend_strength: f64,
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
    pub exit_signal_confidence: f64,
}

pub struct RsiHftStrategy {
    config: RsiHftConfig,
    price_history: VecDeque<f64>,
    rsi_values: VecDeque<f64>,
    ma_short_values: VecDeque<f64>,
    ma_long_values: VecDeque<f64>,
    last_signal_time: f64,
    trades_this_hour: usize,
    last_hour_reset: f64,
    current_timestamp: f64,
    logger: Box<dyn StrategyLogger>,
}

impl RsiHftStrategy {
    pub fn new(config: RsiHftConfig) -> Self {
        Self {
            config,
            price_history: VecDeque::new(),
            rsi_values: VecDeque::new(),
            ma_short_values: VecDeque::new(),
            ma_long_values: VecDeque::new(),
            last_signal_time: 0.0,
            trades_this_hour: 0,
            last_hour_reset: 0.0,
            current_timestamp: 0.0,
            logger: Box::new(NoOpStrategyLogger),
        }
    }

    fn calculate_rsi(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0; // Neutral RSI if not enough data
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (prices.len() - period)..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return 100.0;
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    fn calculate_moving_average(&self, prices: &VecDeque<f64>, period: usize) -> f64 {
        if prices.len() < period {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(period).cloned().collect();
        recent_prices.iter().sum::<f64>() / recent_prices.len() as f64
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

    fn generate_rsi_signal(&self) -> Option<(Signal, f64)> {
        if self.rsi_values.len() < 2 || self.ma_short_values.len() < 1 || self.ma_long_values.len() < 1 {
            return None;
        }

        let current_rsi = *self.rsi_values.back().unwrap();
        let previous_rsi = self.rsi_values[self.rsi_values.len() - 2];
        let short_ma = *self.ma_short_values.back().unwrap();
        let long_ma = *self.ma_long_values.back().unwrap();

        // Determine trend direction
        let trend_bullish = short_ma > long_ma;
        let trend_bearish = short_ma < long_ma;

        // RSI signal conditions
        let rsi_oversold = current_rsi < self.config.rsi.oversold_threshold;
        let rsi_overbought = current_rsi > self.config.rsi.overbought_threshold;
        let rsi_rising = current_rsi > previous_rsi;
        let rsi_falling = current_rsi < previous_rsi;

        // Generate buy signal: oversold RSI + bullish trend + rising RSI
        if rsi_oversold && trend_bullish && rsi_rising {
            let confidence = (self.config.rsi.oversold_threshold - current_rsi) / 
                           self.config.rsi.oversold_threshold * self.config.rsi.signal_strength;
            let trend_confidence = (short_ma - long_ma) / long_ma * self.config.trend.trend_strength;
            let final_confidence = (confidence + trend_confidence).min(0.95);
            return Some((Signal::Buy, final_confidence.max(0.6)));
        }

        // Generate sell signal: overbought RSI + bearish trend + falling RSI
        if rsi_overbought && trend_bearish && rsi_falling {
            let confidence = (current_rsi - self.config.rsi.overbought_threshold) / 
                           (100.0 - self.config.rsi.overbought_threshold) * self.config.rsi.signal_strength;
            let trend_confidence = (long_ma - short_ma) / short_ma * self.config.trend.trend_strength;
            let final_confidence = (confidence + trend_confidence).min(0.95);
            return Some((Signal::Sell, final_confidence.max(0.6)));
        }

        // Additional momentum-based signals for HFT (more aggressive)
        if self.price_history.len() >= 7 {
            let current_price = *self.price_history.back().unwrap();
            let price_3_periods_ago = self.price_history[self.price_history.len() - 4];
            let price_6_periods_ago = self.price_history[self.price_history.len() - 7];
            
            // Momentum breakout signals (more sensitive)
            let short_momentum = (current_price - price_3_periods_ago) / price_3_periods_ago * 100.0;
            let long_momentum = (current_price - price_6_periods_ago) / price_6_periods_ago * 100.0;
            
            // Strong upward momentum (even with high RSI)
            if short_momentum > 0.1 && long_momentum > 0.05 && trend_bullish {
                let momentum_confidence = (short_momentum / 0.5).min(0.9);
                let rsi_confidence = if current_rsi > 50.0 { 0.8 } else { 0.6 };
                let final_confidence = (momentum_confidence + rsi_confidence) / 2.0;
                return Some((Signal::Buy, final_confidence.max(0.5)));
            }
            
            // Strong downward momentum (even with low RSI)
            if short_momentum < -0.1 && long_momentum < -0.05 && trend_bearish {
                let momentum_confidence = (short_momentum.abs() / 0.5).min(0.9);
                let rsi_confidence = if current_rsi < 50.0 { 0.8 } else { 0.6 };
                let final_confidence = (momentum_confidence + rsi_confidence) / 2.0;
                return Some((Signal::Sell, final_confidence.max(0.5)));
            }
        }

        // Mean reversion signals for sideways markets
        if self.price_history.len() >= 21 {
            let current_price = *self.price_history.back().unwrap();
            let price_20_periods_ago = self.price_history[self.price_history.len() - 21];
            let mean_reversion = (current_price - price_20_periods_ago) / price_20_periods_ago * 100.0;
            
            // Buy when price is significantly below 20-period average
            if mean_reversion < -0.5 && current_rsi < 50.0 && !trend_bearish {
                let reversion_confidence = (mean_reversion.abs() / 2.0).min(0.7);
                return Some((Signal::Buy, reversion_confidence.max(0.6)));
            }
            
            // Sell when price is significantly above 20-period average
            if mean_reversion > 0.5 && current_rsi > 50.0 && !trend_bullish {
                let reversion_confidence = (mean_reversion / 2.0).min(0.7);
                return Some((Signal::Sell, reversion_confidence.max(0.6)));
            }
        }

        // Breakout signals for trending markets
        if self.price_history.len() >= 15 {
            let current_price = *self.price_history.back().unwrap();
            
            // Calculate recent high and low levels
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(15).cloned().collect();
            let recent_high = recent_prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let recent_low = recent_prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            // Breakout above recent high with volume confirmation (RSI not overbought)
            if current_price > recent_high * 0.9995 && current_rsi < 70.0 && trend_bullish {
                let breakout_strength = (current_price - recent_high) / recent_high * 1000.0;
                let breakout_confidence = (breakout_strength / 2.0).min(0.8);
                return Some((Signal::Buy, breakout_confidence.max(0.6)));
            }
            
            // Breakdown below recent low with volume confirmation (RSI not oversold)
            if current_price < recent_low * 1.0005 && current_rsi > 30.0 && trend_bearish {
                let breakdown_strength = (recent_low - current_price) / recent_low * 1000.0;
                let breakdown_confidence = (breakdown_strength / 2.0).min(0.8);
                return Some((Signal::Sell, breakdown_confidence.max(0.6)));
            }
        }

        None
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
        let current_price = self.price_history.back().unwrap_or(&0.0);
        let current_rsi = self.rsi_values.back().unwrap_or(&50.0);
        let short_ma = self.ma_short_values.back().unwrap_or(&0.0);
        let long_ma = self.ma_long_values.back().unwrap_or(&0.0);
        
        let time_since_last = if self.last_signal_time > 0.0 {
            self.current_timestamp - self.last_signal_time
        } else {
            0.0
        };
        
        format!(
            "RSI HFT Strategy - Price: {:.8}, RSI: {:.2}, Short MA: {:.8}, Long MA: {:.8}, Trades this hour: {}, Last signal: {:.1}s ago",
            current_price, current_rsi, short_ma, long_ma, self.trades_this_hour, time_since_last
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.current_timestamp = trade.timestamp;

        // Update price history
        self.price_history.push_back(trade.price);
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
        }

        // Calculate RSI
        let rsi = self.calculate_rsi(&self.price_history, self.config.rsi.period);
        self.rsi_values.push_back(rsi);
        if self.rsi_values.len() > 1000 {
            self.rsi_values.pop_front();
        }

        // Calculate moving averages
        let short_ma = self.calculate_moving_average(&self.price_history, self.config.trend.ma_short_period);
        let long_ma = self.calculate_moving_average(&self.price_history, self.config.trend.ma_long_period);
        
        self.ma_short_values.push_back(short_ma);
        self.ma_long_values.push_back(long_ma);
        
        if self.ma_short_values.len() > 1000 {
            self.ma_short_values.pop_front();
        }
        if self.ma_long_values.len() > 1000 {
            self.ma_long_values.pop_front();
        }

        // Log every 10 trades for debugging
        if self.price_history.len() % 10 == 0 {
            println!("DEBUG: Processed {} trades, current price: {:.8}, RSI: {:.2}", 
                self.price_history.len(), 
                trade.price,
                self.rsi_values.back().unwrap_or(&50.0));
        }
    }

    fn get_signal(&mut self, current_position: Position) -> (Signal, f64) {
        let current_time = self.current_timestamp;
        
        // Debug current position
        if current_position.quantity != 0.0 {
            println!("DEBUG: Current position - quantity: {:.2}, entry_price: {:.8}", 
                current_position.quantity, current_position.entry_price);
        }

        // Reset hourly trade counter
        self.reset_hourly_trades(current_time);

        // Check if we've exceeded hourly trade limit
        if self.trades_this_hour >= self.config.signals.max_trades_per_hour {
            return (Signal::Hold, 0.0);
        }

        // Check if we should exit current position
        if let Some(current_price) = self.price_history.back() {
            if self.should_exit_position(&current_position, *current_price) {
                println!("DEBUG: Should exit position - quantity: {:.2}, entry_price: {:.8}, current_price: {:.8}", 
                    current_position.quantity, current_position.entry_price, *current_price);
                if current_position.quantity > 0.0 {
                    println!("DEBUG: Returning Sell signal for exit");
                    return (Signal::Sell, self.config.signals.exit_signal_confidence);
                } else if current_position.quantity < 0.0 {
                    println!("DEBUG: Returning Buy signal for exit");
                    return (Signal::Buy, self.config.signals.exit_signal_confidence);
                }
            }
        }

        // Check if it's time to generate a signal
        if self.should_generate_signal(current_time) {
            // Generate RSI-based signal
            if let Some((signal, confidence)) = self.generate_rsi_signal() {
                println!("DEBUG: Generated signal {:?} with confidence {:.3}", signal, confidence);
                if confidence >= self.config.signals.min_confidence {
                    self.last_signal_time = current_time;
                    self.trades_this_hour += 1;
                    println!("DEBUG: Executing signal {:?} with confidence {:.3}", signal, confidence);
                    return (signal, confidence);
                } else {
                    println!("DEBUG: Signal confidence {:.3} below minimum {:.3}", confidence, self.config.signals.min_confidence);
                }
            } else {
                println!("DEBUG: No signal generated");
            }
        } else {
            println!("DEBUG: Signal cooldown active");
        }

        (Signal::Hold, 0.0)
    }
}
