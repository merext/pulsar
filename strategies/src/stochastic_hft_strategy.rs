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
        
        // Calculate %D (SMA of %K)
        let k_values: Vec<f64> = self.stochastic_k.iter().rev().take(self.config.stochastic.d_period).cloned().collect();
        let d = if k_values.len() >= self.config.stochastic.d_period {
            k_values.iter().sum::<f64>() / k_values.len() as f64
        } else {
            k
        };
        
        Some((k, d))
    }
    
    fn update_stochastic(&mut self) {
        if let Some((k, d)) = self.calculate_stochastic() {
            self.stochastic_k.push_back(k);
            self.stochastic_d.push_back(d);
            
            if self.stochastic_k.len() > self.config.stochastic.buffer_capacity {
                self.stochastic_k.pop_front();
                self.stochastic_d.pop_front();
            }
        }
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
        
        let mut signals = Vec::new();
        let mut confidences = Vec::new();
        
        // Oversold condition (potential buy signal)
        if current_k < self.config.stochastic.oversold && current_d < self.config.stochastic.oversold {
            signals.push(Signal::Buy);
            confidences.push(self.config.signals.oversold_confidence);
        }
        
        // Overbought condition (potential sell signal)
        if current_k > self.config.stochastic.overbought && current_d > self.config.stochastic.overbought {
            signals.push(Signal::Sell);
            confidences.push(self.config.signals.overbought_confidence);
        }
        
        // K/D crossover signals
        let k_d_cross_up = current_k > current_d && *prev_k <= *prev_d;
        let k_d_cross_down = current_k < current_d && *prev_k >= *prev_d;
        
        if k_d_cross_up {
            signals.push(Signal::Buy);
            confidences.push(self.config.signals.crossover_confidence);
        } else if k_d_cross_down {
            signals.push(Signal::Sell);
            confidences.push(self.config.signals.crossover_confidence);
        }
        
        // Combine signals
        if signals.is_empty() {
            return (Signal::Hold, 0.0);
        }
        
        let buy_signals = signals.iter().filter(|&&s| s == Signal::Buy).count();
        let sell_signals = signals.iter().filter(|&&s| s == Signal::Sell).count();
        
        let avg_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
        
        if buy_signals > sell_signals && avg_confidence >= self.config.signals.min_confidence {
            (Signal::Buy, avg_confidence)
        } else if sell_signals > buy_signals && avg_confidence >= self.config.signals.min_confidence {
            (Signal::Sell, avg_confidence)
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
        
        // Update Stochastic Oscillator
        self.update_stochastic();
        
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
