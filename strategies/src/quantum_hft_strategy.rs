//! # Fractal-Based HFT Strategy
//! 
//! A sophisticated trading strategy based on fractal geometry and market structure analysis

use std::collections::VecDeque;
use trade::signal::Signal;
use trade::trader::Position;
use crate::config::StrategyConfig;
use crate::strategy::Strategy;
use tracing::debug;

/// Enhanced Profitable Strategy - Technical Analysis + Smart Entry/Exit
pub struct QuantumHftStrategy {
    config: StrategyConfig,
    
    // Data windows
    price_window: VecDeque<f64>,
    volume_window: VecDeque<f64>,
    
    // Performance tracking
    win_rate: f64,
    consecutive_wins: usize,
    consecutive_losses: usize,
    trade_counter: usize,
    total_pnl: f64,
    
    // Sub-strategy tracking
    volume_spike_threshold: f64,
    avg_volume: f64,
    support_resistance_levels: VecDeque<f64>,
    
    // Profit optimization tracking
    recent_volatility: f64,
    price_velocity: f64,
    micro_trend_direction: i8, // -1, 0, 1
    profitable_exit_count: usize,
    last_profitable_threshold: f64,
    
    // Risk management
    max_position_size: f64,
    stop_loss_threshold: f64,
    take_profit_threshold: f64,
}

impl QuantumHftStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("quantum_hft_strategy")
            .expect("Failed to load quantum_hft_strategy configuration");
        
        let max_position_size = config.get_or("max_position_size", 1000.0);
        let stop_loss_threshold = config.get_or("stop_loss_threshold", 0.008);
        let take_profit_threshold = config.get_or("take_profit_threshold", 0.012);
        
        Self {
            config,
            price_window: VecDeque::with_capacity(100),
            volume_window: VecDeque::with_capacity(100),
            win_rate: 0.5,
            consecutive_wins: 0,
            consecutive_losses: 0,
            trade_counter: 0,
            total_pnl: 0.0,
            
            // Initialize sub-strategy parameters
            volume_spike_threshold: 3.0, // Volume spike if 3x average (more selective)
            avg_volume: 1000.0, // Initial average volume
            support_resistance_levels: VecDeque::with_capacity(10),
            
            // Initialize profit optimization parameters
            recent_volatility: 0.001,
            price_velocity: 0.0,
            micro_trend_direction: 0,
            profitable_exit_count: 0,
            last_profitable_threshold: 0.001,
            
            max_position_size,
            stop_loss_threshold,
            take_profit_threshold,
        }
    }
    
    /// Calculate simple moving average
    fn calculate_sma(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices[prices.len() - 1];
        }
        prices[prices.len() - period..].iter().sum::<f64>() / period as f64
    }

    /// Calculate RSI
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in prices.len() - period..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }
        
        if losses == 0.0 {
            return 100.0;
        }
        
        let rs = gains / losses;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate Bollinger Bands
    fn calculate_bollinger_bands(&self, prices: &[f64], period: usize, std_dev: f64) -> (f64, f64, f64) {
        if prices.len() < period {
            return (0.0, 0.0, 0.0);
        }
        
        let recent_prices = &prices[prices.len() - period..];
        let mean = recent_prices.iter().sum::<f64>() / period as f64;
        
        let variance = recent_prices.iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / period as f64;
        let std = variance.sqrt();
        
        let upper = mean + (std * std_dev);
        let lower = mean - (std * std_dev);
        
        (upper, mean, lower)
    }

    /// Calculate momentum
    fn calculate_momentum(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.0;
        }
        let current = prices[prices.len() - 1];
        let past = prices[prices.len() - 1 - period];
        (current - past) / past
    }

    /// Calculate Volume-Weighted Average Price (VWAP)
    fn calculate_vwap(&self, prices: &[f64], volumes: &[f64]) -> f64 {
        if prices.len() != volumes.len() || prices.is_empty() {
            return 0.0;
        }
        
        let total_volume: f64 = volumes.iter().sum();
        if total_volume == 0.0 {
            return prices[prices.len() - 1];
        }
        
        prices.iter().zip(volumes.iter())
            .map(|(&p, &v)| p * v)
            .sum::<f64>() / total_volume
    }

    /// Detect volume spike
    fn is_volume_spike(&self, current_volume: f64) -> bool {
        current_volume > self.avg_volume * self.volume_spike_threshold
    }

    /// Update support and resistance levels
    fn update_support_resistance(&mut self, price: f64) {
        // Add new price level
        self.support_resistance_levels.push_back(price);
        
        // Keep only recent levels
        if self.support_resistance_levels.len() > 10 {
            self.support_resistance_levels.pop_front();
        }
    }

    /// Check if price is near support/resistance
    fn is_near_support_resistance(&self, current_price: f64, threshold: f64) -> (bool, bool) {
        let mut near_support = false;
        let mut near_resistance = false;
        
        for &level in &self.support_resistance_levels {
            let distance = (current_price - level).abs() / level;
            if distance < threshold {
                if current_price > level {
                    near_resistance = true;
                } else {
                    near_support = true;
                }
            }
        }
        
        (near_support, near_resistance)
    }

    /// Calculate market volatility
    fn calculate_volatility(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 0.001;
        }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let recent_returns = &returns[returns.len().saturating_sub(period)..];
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / recent_returns.len() as f64;
        
        variance.sqrt().max(0.0001)
    }

    /// Calculate price velocity (rate of change acceleration)
    fn calculate_price_velocity(&self, prices: &[f64]) -> f64 {
        if prices.len() < 5 {
            return 0.0;
        }
        
        let recent = &prices[prices.len() - 5..];
        let momentum1 = (recent[2] - recent[0]) / recent[0];
        let momentum2 = (recent[4] - recent[2]) / recent[2];
        
        momentum2 - momentum1
    }

    /// Detect micro trend direction
    fn detect_micro_trend(&self, prices: &[f64]) -> i8 {
        if prices.len() < 10 {
            return 0;
        }
        
        let recent = &prices[prices.len() - 10..];
        let first_half_avg = recent[0..5].iter().sum::<f64>() / 5.0;
        let second_half_avg = recent[5..10].iter().sum::<f64>() / 5.0;
        
        let trend_strength = (second_half_avg - first_half_avg) / first_half_avg;
        
        if trend_strength > 0.0005 {
            1  // Uptrend
        } else if trend_strength < -0.0005 {
            -1 // Downtrend
        } else {
            0  // Sideways
        }
    }

    /// Generate micro-scalping signals for small profits
    fn generate_micro_scalping_signal(&self, current_price: f64) -> (Signal, f64) {
        if self.price_window.len() < 20 {
            return (Signal::Hold, 0.0);
        }
        
        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        let volatility = self.recent_volatility;
        let velocity = self.price_velocity;
        
        // Adaptive thresholds based on volatility
        let entry_threshold = (volatility * 2.0).max(0.0003).min(0.002);
        let momentum_threshold = (volatility * 1.5).max(0.0002).min(0.001);
        
        // Micro scalping: quick in and out on small price movements
        let short_momentum = self.calculate_momentum(&prices, 3);
        let micro_trend = self.micro_trend_direction;
        
        // Long scalping signal - MORE AGGRESSIVE
        if short_momentum > momentum_threshold && 
           velocity >= 0.0 && 
           volatility > 0.0003 {
            debug!("Micro Scalp Long: momentum={:.4}, velocity={:.4}, volatility={:.4}", 
                   short_momentum, velocity, volatility);
            return (Signal::Buy, 0.8);
        }
        
        // Short scalping signal - MORE AGGRESSIVE
        if short_momentum < -momentum_threshold && 
           velocity <= 0.0 && 
           volatility > 0.0003 {
            debug!("Micro Scalp Short: momentum={:.4}, velocity={:.4}, volatility={:.4}", 
                   short_momentum, velocity, volatility);
            return (Signal::Sell, 0.8);
        }
        
        // Trend continuation scalping
        if micro_trend == 1 && short_momentum > 0.0001 && velocity > -0.0001 {
            debug!("Trend Scalp Long: trend={}, momentum={:.4}", micro_trend, short_momentum);
            return (Signal::Buy, 0.7);
        }
        
        if micro_trend == -1 && short_momentum < -0.0001 && velocity < 0.0001 {
            debug!("Trend Scalp Short: trend={}, momentum={:.4}", micro_trend, short_momentum);
            return (Signal::Sell, 0.7);
        }
        
        (Signal::Hold, 0.0)
    }

    /// Generate contrarian strategy for market overreactions
    fn generate_contrarian_signal(&self, current_price: f64) -> (Signal, f64) {
        if self.price_window.len() < 20 {
            return (Signal::Hold, 0.0);
        }
        
        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        let volumes: Vec<f64> = self.volume_window.iter().cloned().collect();
        
        let current_volume = volumes[volumes.len() - 1];
        let rsi = self.calculate_rsi(&prices, 14);
        let momentum = self.calculate_momentum(&prices, 5);
        let volatility = self.recent_volatility;
        
        // Contrarian: buy when market is oversold with high volume
        if rsi < 25.0 && current_volume > self.avg_volume * 1.5 && momentum < -0.001 {
            debug!("Contrarian Long: RSI={:.1}, volume={:.1}x, momentum={:.4}", 
                   rsi, current_volume / self.avg_volume, momentum);
            return (Signal::Buy, 0.9);
        }
        
        // Contrarian: sell when market is overbought with high volume
        if rsi > 75.0 && current_volume > self.avg_volume * 1.5 && momentum > 0.001 {
            debug!("Contrarian Short: RSI={:.1}, volume={:.1}x, momentum={:.4}", 
                   rsi, current_volume / self.avg_volume, momentum);
            return (Signal::Sell, 0.9);
        }
        
        // Quick reversal trades on extreme moves
        if momentum > 0.003 && rsi > 60.0 {
            debug!("Quick Reversal Short: momentum={:.4}, RSI={:.1}", momentum, rsi);
            return (Signal::Sell, 0.8);
        }
        
        if momentum < -0.003 && rsi < 40.0 {
            debug!("Quick Reversal Long: momentum={:.4}, RSI={:.1}", momentum, rsi);
            return (Signal::Buy, 0.8);
        }
        
        (Signal::Hold, 0.0)
    }

    /// Generate Volume-Weighted Price Action sub-strategy signal
    fn generate_volume_substrategy_signal(&self, current_price: f64) -> (Signal, f64) {
        if self.price_window.len() < 20 || self.volume_window.len() < 20 {
            return (Signal::Hold, 0.0);
        }
        
        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        let volumes: Vec<f64> = self.volume_window.iter().cloned().collect();
        
        let current_volume = volumes[volumes.len() - 1];
        let vwap = self.calculate_vwap(&prices, &volumes);
        let is_volume_spike = self.is_volume_spike(current_volume);
        let (near_support, near_resistance) = self.is_near_support_resistance(current_price, 0.005);
        let momentum = self.calculate_momentum(&prices, 5);
        
        // Volume breakout strategy - WITH MOMENTUM CONFIRMATION
        if is_volume_spike && current_volume > self.avg_volume * 2.5 {
            if current_price > vwap * 1.005 && !near_resistance && momentum > 0.001 {
                debug!("Volume Breakout Long: price={:.4}, VWAP={:.4}, volume={:.1}x, momentum={:.4}", 
                       current_price, vwap, current_volume / self.avg_volume, momentum);
                return (Signal::Buy, 0.9);
            }
            
            if current_price < vwap * 0.995 && !near_support && momentum < -0.001 {
                debug!("Volume Breakout Short: price={:.4}, VWAP={:.4}, volume={:.1}x, momentum={:.4}", 
                       current_price, vwap, current_volume / self.avg_volume, momentum);
                return (Signal::Sell, 0.9);
            }
        }
        
        // Support/Resistance bounce strategy - WITH MOMENTUM CONFIRMATION
        if near_support && current_price < vwap * 0.998 && current_volume > self.avg_volume * 1.8 && momentum > 0.0005 {
            debug!("Support Bounce: price={:.4}, VWAP={:.4}, momentum={:.4}", current_price, vwap, momentum);
            return (Signal::Buy, 0.8);
        }
        
        if near_resistance && current_price > vwap * 1.002 && current_volume > self.avg_volume * 1.8 && momentum < -0.0005 {
            debug!("Resistance Bounce: price={:.4}, VWAP={:.4}, momentum={:.4}", current_price, vwap, momentum);
            return (Signal::Sell, 0.8);
        }
        
        (Signal::Hold, 0.0)
    }

    /// Generate enhanced signal with technical analysis
    fn generate_signal(&self, current_price: f64) -> (Signal, f64) {
        if self.price_window.len() < 20 {
            return (Signal::Hold, 0.0);
        }

        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        
        // Calculate technical indicators
        let rsi = self.calculate_rsi(&prices, 14);
        let (bb_upper, bb_middle, bb_lower) = self.calculate_bollinger_bands(&prices, 20, 2.0);
        let momentum = self.calculate_momentum(&prices, 5);
        let sma_short = self.calculate_sma(&prices, 10);
        let sma_long = self.calculate_sma(&prices, 20);
        
        // Enhanced entry conditions
        let bb_position = (current_price - bb_lower) / (bb_upper - bb_lower);
        let price_deviation = (current_price - sma_long) / sma_long;

        // MEAN REVERSION WITH TIGHT STOPS - buy dips, sell rallies
        let short_momentum = self.calculate_momentum(&prices, 3);
        
        if rsi < 0.1 && short_momentum > -0.001 {
            debug!("Mean Rev Buy: RSI={:.1}, momentum={:.4}", rsi, short_momentum);
            return (Signal::Buy, 0.9);
        }
        
        if rsi > 99.9 && short_momentum < 0.001 {
            debug!("Mean Rev Sell: RSI={:.1}, momentum={:.4}", rsi, short_momentum);
            return (Signal::Sell, 0.9);
        }
        
        // Simple alternating every 1000000 trades as backup
        if self.trade_counter % 1000000 == 0 {
            if (self.trade_counter / 1000000) % 2 == 0 {
                debug!("Backup Buy: counter={}", self.trade_counter);
                return (Signal::Buy, 0.00001);
            } else {
                debug!("Backup Sell: counter={}", self.trade_counter);
                return (Signal::Sell, 0.00001);
            }
        }
        
        // Simple profitable strategy - RSI + MOMENTUM + TIMING
        // Long signal: oversold RSI with positive momentum
        if rsi < 35.0 && momentum > 0.0005 {
            let confidence = (35.0 - rsi) / 35.0 * 0.8;
            debug!("RSI Long: RSI={:.1}, Momentum={:.4}", rsi, momentum);
            return (Signal::Buy, confidence);
        }
        
        // Short signal: overbought RSI with negative momentum
        if rsi > 65.0 && momentum < -0.0005 {
            let confidence = (rsi - 65.0) / 35.0 * 0.8;
            debug!("RSI Short: RSI={:.1}, Momentum={:.4}", rsi, momentum);
            return (Signal::Sell, confidence);
        }
        
        // Momentum breakout: strong momentum with RSI confirmation
        if momentum > 0.002 && rsi < 70.0 && current_price > sma_short {
            debug!("Momentum Long: RSI={:.1}, Momentum={:.4}", rsi, momentum);
            return (Signal::Buy, 0.7);
        }
        
        if momentum < -0.002 && rsi > 30.0 && current_price < sma_short {
            debug!("Momentum Short: RSI={:.1}, Momentum={:.4}", rsi, momentum);
            return (Signal::Sell, 0.7);
        }
        
        // Mean reversion: buy when price is below average, sell when above
        if price_deviation < -0.002 && rsi < 45.0 {
            debug!("Mean Reversion Buy: RSI={:.1}, Dev={:.4}", rsi, price_deviation);
            return (Signal::Buy, 0.6);
        }
        
        if price_deviation > 0.002 && rsi > 55.0 {
            debug!("Mean Reversion Sell: RSI={:.1}, Dev={:.4}", rsi, price_deviation);
            return (Signal::Sell, 0.6);
        }
        
        // Quick arbitrage-like trades on price inefficiencies
        if self.trade_counter % 100 == 0 {
            let recent_high = prices.iter().rev().take(20).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let recent_low = prices.iter().rev().take(20).fold(f64::INFINITY, |a, &b| a.min(b));
            let price_range = recent_high - recent_low;
            
            if price_range > 0.0 {
                let position_in_range = (current_price - recent_low) / price_range;
                
                // Buy near the low of recent range
                if position_in_range < 0.2 && momentum > -0.001 {
                    debug!("Range Bottom Buy: position={:.2}, range={:.4}", position_in_range, price_range);
                    return (Signal::Buy, 0.6);
                }
                
                // Sell near the high of recent range
                if position_in_range > 0.8 && momentum < 0.001 {
                    debug!("Range Top Sell: position={:.2}, range={:.4}", position_in_range, price_range);
                    return (Signal::Sell, 0.6);
                }
            }
        }
        
        // MEAN REVERSION WITH TIGHT STOPS - buy dips, sell rallies
        let short_momentum = self.calculate_momentum(&prices, 3);
        
        if rsi < 30.0 && short_momentum > -0.001 {
            debug!("Mean Rev Buy: RSI={:.1}, momentum={:.4}", rsi, short_momentum);
            return (Signal::Buy, 0.9);
        }
        
        if rsi > 70.0 && short_momentum < 0.001 {
            debug!("Mean Rev Sell: RSI={:.1}, momentum={:.4}", rsi, short_momentum);
            return (Signal::Sell, 0.9);
        }
        
        // Simple alternating every 100 trades as backup
        if self.trade_counter % 100 == 0 {
            if (self.trade_counter / 100) % 2 == 0 {
                debug!("Backup Buy: counter={}", self.trade_counter);
                return (Signal::Buy, 0.6);
            } else {
                debug!("Backup Sell: counter={}", self.trade_counter);
                return (Signal::Sell, 0.6);
            }
        }
        
        (Signal::Hold, 0.0)
    }

    /// Update performance metrics
    fn update_performance(&mut self, trade_result: f64) {
        self.total_pnl += trade_result;
        
        if trade_result > 0.0 {
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
        } else {
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
        }
        
        // Update win rate
        let total_trades = self.consecutive_wins + self.consecutive_losses;
        if total_trades > 0 {
            self.win_rate = self.consecutive_wins as f64 / total_trades as f64;
        }
    }
}

#[async_trait::async_trait]
impl Strategy for QuantumHftStrategy {
    fn get_info(&self) -> String {
        format!("Profit-Optimized Multi-Strategy: Micro-Scalping + Volume + RSI+Momentum (Win Rate: {:.1}%, PnL: {:.4})", 
                self.win_rate * 100.0, self.total_pnl)
    }

    async fn on_trade(&mut self, trade: trade::models::TradeData) {
        // Update data windows
        self.price_window.push_back(trade.price);
        self.volume_window.push_back(trade.qty);
        
        // Update volume tracking for sub-strategy
        if self.volume_window.len() > 1 {
            // Update rolling average volume
            let total_volume: f64 = self.volume_window.iter().sum();
            self.avg_volume = total_volume / self.volume_window.len() as f64;
        }
        
        // Update profit optimization metrics
        if self.price_window.len() >= 20 {
            let prices: Vec<f64> = self.price_window.iter().cloned().collect();
            self.recent_volatility = self.calculate_volatility(&prices, 10);
            self.price_velocity = self.calculate_price_velocity(&prices);
            self.micro_trend_direction = self.detect_micro_trend(&prices);
        }
        
        // Update support/resistance levels periodically
        if self.trade_counter % 50 == 0 {
            self.update_support_resistance(trade.price);
        }
        
        // Keep window size manageable
        if self.price_window.len() > 100 {
            self.price_window.pop_front();
            self.volume_window.pop_front();
        }
        
        self.trade_counter += 1;
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // Enhanced position management
        if current_position.quantity > 0.0 {
            // Long position - smart exit
            let (exit_signal, _) = self.generate_signal(current_price);
            if exit_signal == Signal::Sell || self.consecutive_losses >= 1 {
                debug!("Smart Long Exit: consecutive_losses={}", self.consecutive_losses);
                return (Signal::Sell, 0.9);
            }
            
            // Smart profit-taking with adaptive thresholds
            let prices: Vec<f64> = self.price_window.iter().cloned().collect();
            let rsi = self.calculate_rsi(&prices, 14);
            let momentum = self.calculate_momentum(&prices, 5);
            let volatility = self.recent_volatility;
            
            // Adaptive profit threshold based on volatility
            let profit_threshold = (volatility * 3.0).max(0.0005).min(0.003);
            
            // Quick scalping exit for small profits - MORE AGGRESSIVE
            if momentum > profit_threshold * 0.5 || rsi > 52.0 + (volatility * 500.0) {
                debug!("Smart Long Profit: RSI={:.1}, Momentum={:.4}, Volatility={:.4}", 
                       rsi, momentum, volatility);
                return (Signal::Sell, 0.95);
            }
            
            // CONSERVATIVE PROFIT TAKING - exit on small positive movement
            if momentum > 0.0002 || rsi > 52.0 {
                debug!("Conservative Long Profit: momentum={:.4}, RSI={:.1}", momentum, rsi);
                return (Signal::Sell, 0.95);
            }
        } else if current_position.quantity < 0.0 {
            // Short position - smart exit
            let (exit_signal, _) = self.generate_signal(current_price);
            if exit_signal == Signal::Buy || self.consecutive_losses >= 1 {
                debug!("Smart Short Exit: consecutive_losses={}", self.consecutive_losses);
                return (Signal::Buy, 0.9);
            }
            
            // Smart profit-taking with adaptive thresholds
            let prices: Vec<f64> = self.price_window.iter().cloned().collect();
            let rsi = self.calculate_rsi(&prices, 14);
            let momentum = self.calculate_momentum(&prices, 5);
            let volatility = self.recent_volatility;
            
            // Adaptive profit threshold based on volatility
            let profit_threshold = (volatility * 3.0).max(0.0005).min(0.003);
            
            // Quick scalping exit for small profits - MORE AGGRESSIVE
            if momentum < -profit_threshold * 0.5 || rsi < 48.0 - (volatility * 500.0) {
                debug!("Smart Short Profit: RSI={:.1}, Momentum={:.4}, Volatility={:.4}", 
                       rsi, momentum, volatility);
                return (Signal::Buy, 0.95);
            }
            
            // CONSERVATIVE PROFIT TAKING - exit on small negative movement
            if momentum < -0.0002 || rsi < 48.0 {
                debug!("Conservative Short Profit: momentum={:.4}, RSI={:.1}", momentum, rsi);
                return (Signal::Buy, 0.95);
            }
        } else {
            // No position - look for enhanced entry
            let (signal, confidence) = self.generate_signal(current_price);
            if signal != Signal::Hold {
                return (signal, confidence);
            }
        }
        
        (Signal::Hold, 0.0)
    }
}

impl QuantumHftStrategy {
    pub fn on_trade_result(&mut self, result: f64) {
        self.update_performance(result);
        debug!("Trade Result: {:.4}, Win Rate: {:.1}%, Total PnL: {:.4}", 
               result, self.win_rate * 100.0, self.total_pnl);
    }
}
