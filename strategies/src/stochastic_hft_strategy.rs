use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use crate::strategy::Strategy;
use trade::{Position, Signal};
use trade::models::TradeData;

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
    base_size: f64,
}

#[derive(serde::Deserialize)]
struct StochasticConfig {
    k_period: usize,
    d_period: usize,
    oversold: f64,
    overbought: f64,
    buffer_capacity: usize,
}

#[derive(serde::Deserialize)]
struct SignalsConfig {
    min_confidence: f64,
    oversold_confidence: f64,
    overbought_confidence: f64,
    crossover_confidence: f64,
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
    
    // Enhanced indicators
    volatility_history: VecDeque<f64>,
    trend_strength: VecDeque<f64>,
    
    // Strategy state
    trade_counter: usize,
    last_signal_time: f64,
    consecutive_losses: usize,
    
    // Position and PnL tracking for backtesting
    current_position: f64,
    entry_price: f64,
    entry_time: f64,
    unrealized_pnl: f64,
    realized_pnl: f64,
    total_pnl: f64,
    
    // Trade history for backtesting
    trades: Vec<TradeRecord>,
    
    // Performance tracking
    win_count: usize,
    loss_count: usize,
    max_drawdown: f64,
    peak_equity: f64,
    current_equity: f64,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields are used in backtest statistics calculation
struct TradeRecord {
    entry_time: f64,
    exit_time: f64,
    entry_price: f64,
    exit_price: f64,
    position_size: f64,
    pnl: f64,
    signal: Signal,
}

impl StochasticHftStrategy {
    pub fn new() -> Self {
        Self::from_file("config/stochastic_hft_strategy.toml").expect("Failed to load configuration file")
    }
    
    pub fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(config_path)?;
        let config: StochasticHftConfig = toml::from_str(&content)?;
        let buffer_capacity = config.stochastic.buffer_capacity;
        
        Ok(Self {
            config,
            price_history: VecDeque::with_capacity(buffer_capacity),
            stochastic_k: VecDeque::with_capacity(buffer_capacity),
            stochastic_d: VecDeque::with_capacity(buffer_capacity),
            volatility_history: VecDeque::with_capacity(100),
            trend_strength: VecDeque::with_capacity(100),
            trade_counter: 0,
            last_signal_time: 0.0,
            consecutive_losses: 0,
            current_position: 0.0,
            entry_price: 0.0,
            entry_time: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            total_pnl: 0.0,
            trades: Vec::new(),
            win_count: 0,
            loss_count: 0,
            max_drawdown: 0.0,
            peak_equity: 0.0,
            current_equity: 0.0,
        })
    }
    

    
    fn calculate_stochastic(&mut self) -> Option<(f64, f64)> {
        if self.price_history.len() < self.config.stochastic.k_period {
            return None;
        }
        
        let prices: Vec<f64> = self.price_history.iter().rev().take(self.config.stochastic.k_period).cloned().collect();
        let current_price = *self.price_history.back().unwrap();
        
        let low_min = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let high_max = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if high_max == low_min {
            return Some((50.0, 50.0));
        }
        
        let k = 100.0 * ((current_price - low_min) / (high_max - low_min));
        
        Some((k, 0.0)) // %D will be calculated in update_stochastic
    }
    
    fn update_stochastic(&mut self) {
        if let Some((k, _)) = self.calculate_stochastic() {
            // Add new %K value first
            self.stochastic_k.push_back(k);
            
            // Calculate %D from the updated %K buffer
            let d = if self.stochastic_k.len() >= self.config.stochastic.d_period {
                let k_values: Vec<f64> = self.stochastic_k.iter().rev().take(self.config.stochastic.d_period).cloned().collect();
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
    
    fn calculate_volatility(&mut self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.0;
        }
        
        let returns: Vec<f64> = self.price_history.iter()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| (w[0] - w[1]) / w[1])
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        let volatility = variance.sqrt();
        self.volatility_history.push_back(volatility);
        if self.volatility_history.len() > 50 {
            self.volatility_history.pop_front();
        }
        
        volatility
    }
    
    fn calculate_trend_strength(&mut self) -> f64 {
        if self.price_history.len() < 50 {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.price_history.iter()
            .rev()
            .take(50)
            .cloned()
            .collect();
        
        // Calculate linear regression R-squared
        let n = recent_prices.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = recent_prices.iter().sum::<f64>() / n;
        
        let numerator: f64 = recent_prices.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator_x: f64 = (0..recent_prices.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();
        
        let denominator_y: f64 = recent_prices.iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();
        
        let trend_strength = if denominator_x > 0.0 && denominator_y > 0.0 {
            (numerator.powi(2) / (denominator_x * denominator_y)).abs()
        } else {
            0.0
        };
        
        self.trend_strength.push_back(trend_strength);
        if self.trend_strength.len() > 20 {
            self.trend_strength.pop_front();
        }
        
        trend_strength
    }
    
    fn update_indicators(&mut self) {
        self.update_stochastic();
        self.calculate_volatility();
        self.calculate_trend_strength();
    }
    
    fn generate_signal(&self) -> (Signal, f64) {
        if self.stochastic_k.len() < 2 || self.stochastic_d.len() < 2 {
            return (Signal::Hold, 0.0);
        }
        
        let current_k = *self.stochastic_k.back().unwrap();
        let current_d = *self.stochastic_d.back().unwrap();
        
        // Get previous values for crossover detection
        let prev_k = self.stochastic_k.get(self.stochastic_k.len() - 2).unwrap();
        let prev_d = self.stochastic_d.get(self.stochastic_d.len() - 2).unwrap();
        
        let mut buy_signals = Vec::new();
        let mut sell_signals = Vec::new();
        
        // Stochastic oversold/overbought signals
        if current_k < self.config.stochastic.oversold && current_d < self.config.stochastic.oversold {
            buy_signals.push(self.config.signals.oversold_confidence);
        } else if current_k > self.config.stochastic.overbought && current_d > self.config.stochastic.overbought {
            sell_signals.push(self.config.signals.overbought_confidence);
        }
        
        // K/D crossover signals
        let k_d_cross_up = current_k > current_d && *prev_k <= *prev_d;
        let k_d_cross_down = current_k < current_d && *prev_k >= *prev_d;
        
        if k_d_cross_up {
            buy_signals.push(self.config.signals.crossover_confidence);
        } else if k_d_cross_down {
            sell_signals.push(self.config.signals.crossover_confidence);
        }
        
        // Calculate signal direction and confidence
        let buy_confidence = if !buy_signals.is_empty() {
            buy_signals.iter().sum::<f64>() / buy_signals.len() as f64
        } else {
            0.0
        };
        
        let sell_confidence = if !sell_signals.is_empty() {
            sell_signals.iter().sum::<f64>() / sell_signals.len() as f64
        } else {
            0.0
        };
        
        // Determine final signal and confidence
        let (signal_direction, base_confidence) = if buy_confidence > sell_confidence {
            (1.0, buy_confidence)
        } else if sell_confidence > buy_confidence {
            (-1.0, sell_confidence)
        } else {
            (0.0, 0.0)
        };
        
        // Apply trend and volatility adjustments to confidence
        let mut adjusted_confidence = base_confidence;
        
        // Trend confirmation
        let avg_trend = self.trend_strength.iter().sum::<f64>() / self.trend_strength.len().max(1) as f64;
        if avg_trend > 0.3 {
            adjusted_confidence *= 1.2; // Boost confidence in trending markets
        }
        
        // Volatility adjustment
        let avg_volatility = self.volatility_history.iter().sum::<f64>() / self.volatility_history.len().max(1) as f64;
        if avg_volatility > 0.01 {
            adjusted_confidence *= 0.8; // Reduce confidence in high volatility
        }
        
        let final_confidence = adjusted_confidence.min(1.0);
        
        if final_confidence >= self.config.signals.min_confidence {
            if signal_strength > 0.0 {
                (Signal::Buy, final_confidence)
            } else if signal_strength < 0.0 {
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
    fn get_info(&self) -> String {
        let stats = self.get_backtest_stats();
        let current_k = self.stochastic_k.back().unwrap_or(&0.0);
        let current_d = self.stochastic_d.back().unwrap_or(&0.0);
        
        format!(
            "{} - Trades: {}, PnL: {:.4}, Win Rate: {:.1}%, MaxDD: {:.2}%, PF: {:.2}, K: {:.1}, D: {:.1}",
            self.config.general.strategy_name,
            stats.total_trades,
            stats.total_pnl,
            stats.win_rate,
            stats.max_drawdown * 100.0,
            stats.profit_factor,
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
        
        // Update PnL for current position
        self.update_pnl(trade.price, trade.time as f64);
        
        // Generate and execute signal
        let (signal, confidence) = self.generate_signal();
        self.execute_trade(signal, trade.price, trade.time as f64, confidence);
    }
    
    fn get_signal(
        &self,
        _current_price: f64,
        current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Check cooldown
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
        
        // Generate signal based on Stochastic Oscillator
        self.generate_signal()
    }
}

impl StochasticHftStrategy {
    // PnL calculation methods for backtesting
    fn update_pnl(&mut self, current_price: f64, _current_time: f64) {
        if self.current_position != 0.0 {
            // Calculate unrealized PnL
            let price_change = if self.current_position > 0.0 {
                current_price - self.entry_price
            } else {
                self.entry_price - current_price
            };
            
            self.unrealized_pnl = self.current_position.abs() * price_change;
            self.total_pnl = self.realized_pnl + self.unrealized_pnl;
            self.current_equity = self.total_pnl;
            
            // Update max drawdown
            if self.current_equity > self.peak_equity {
                self.peak_equity = self.current_equity;
            } else {
                let drawdown = (self.peak_equity - self.current_equity) / self.peak_equity.max(1.0);
                if drawdown > self.max_drawdown {
                    self.max_drawdown = drawdown;
                }
            }
        }
    }
    
    fn execute_trade(&mut self, signal: Signal, current_price: f64, current_time: f64, confidence: f64) {
        let position_size = self.config.general.base_size * confidence;
        
        match signal {
            Signal::Buy => {
                if self.current_position <= 0.0 {
                    // Close existing short position if any
                    if self.current_position < 0.0 {
                        self.close_position(current_price, current_time);
                    }
                    
                    // Open long position
                    self.current_position = position_size;
                    self.entry_price = current_price;
                    self.entry_time = current_time;
                    self.trade_counter += 1;
                }
            }
            Signal::Sell => {
                if self.current_position >= 0.0 {
                    // Close existing long position if any
                    if self.current_position > 0.0 {
                        self.close_position(current_price, current_time);
                    }
                    
                    // Open short position
                    self.current_position = -position_size;
                    self.entry_price = current_price;
                    self.entry_time = current_time;
                    self.trade_counter += 1;
                }
            }
            Signal::Hold => {
                // Update unrealized PnL for current position
                self.update_pnl(current_price, current_time);
            }
        }
    }
    
    fn close_position(&mut self, exit_price: f64, exit_time: f64) {
        if self.current_position != 0.0 {
            let pnl = if self.current_position > 0.0 {
                (exit_price - self.entry_price) * self.current_position.abs()
            } else {
                (self.entry_price - exit_price) * self.current_position.abs()
            };
            
            // Record the trade
            let trade_record = TradeRecord {
                entry_time: self.entry_time,
                exit_time,
                entry_price: self.entry_price,
                exit_price,
                position_size: self.current_position.abs(),
                pnl,
                signal: if self.current_position > 0.0 { Signal::Buy } else { Signal::Sell },
            };
            
            self.trades.push(trade_record);
            
            // Update performance metrics
            self.realized_pnl += pnl;
            self.total_pnl = self.realized_pnl;
            
            if pnl > 0.0 {
                self.win_count += 1;
                self.consecutive_losses = 0;
            } else {
                self.loss_count += 1;
                self.consecutive_losses += 1;
            }
            
            // Reset position
            self.current_position = 0.0;
            self.entry_price = 0.0;
            self.entry_time = 0.0;
            self.unrealized_pnl = 0.0;
        }
    }
    
    // Get backtesting statistics
    fn get_backtest_stats(&self) -> BacktestStats {
        let total_trades = self.trades.len();
        let win_rate = if total_trades > 0 {
            self.win_count as f64 / total_trades as f64 * 100.0
        } else {
            0.0
        };
        
        let avg_win = if self.win_count > 0 {
            self.trades.iter()
                .filter(|t| t.pnl > 0.0)
                .map(|t| t.pnl)
                .sum::<f64>() / self.win_count as f64
        } else {
            0.0
        };
        
        let avg_loss = if self.loss_count > 0 {
            self.trades.iter()
                .filter(|t| t.pnl < 0.0)
                .map(|t| t.pnl.abs())
                .sum::<f64>() / self.loss_count as f64
        } else {
            0.0
        };
        
        let profit_factor = if avg_loss > 0.0 {
            (avg_win * self.win_count as f64) / (avg_loss * self.loss_count as f64)
        } else {
            f64::INFINITY
        };
        
        BacktestStats {
            total_trades,
            win_rate,
            total_pnl: self.total_pnl,
            realized_pnl: self.realized_pnl,
            max_drawdown: self.max_drawdown,
            profit_factor,
            avg_win,
            avg_loss,
        }
    }
    
    // Public method to get backtest statistics
    pub fn get_stats(&self) -> BacktestStats {
        self.get_backtest_stats()
    }
}

#[derive(Debug, Clone)]
pub struct BacktestStats {
    pub total_trades: usize,
    pub win_rate: f64,
    pub total_pnl: f64,
    pub realized_pnl: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
}
