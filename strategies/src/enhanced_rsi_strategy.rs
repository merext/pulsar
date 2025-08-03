//! # Enhanced RSI Strategy with Advanced Features
//! 
//! Advanced features added to the base RSI strategy:
//! 1. Volume confirmation for signal quality
//! 2. Multi-timeframe RSI analysis
//! 3. Adaptive thresholds based on market volatility
//! 4. Trend detection and alignment
//! 5. Dynamic position sizing
//! 6. Advanced risk management

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct EnhancedRsiStrategy {
    // Multi-timeframe RSI parameters
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    
    // Volume analysis
    volume_period: usize,
    volume_threshold: f64,
    
    // RSI thresholds
    overbought_short: f64,
    oversold_short: f64,
    overbought_medium: f64,
    oversold_medium: f64,
    overbought_long: f64,
    oversold_long: f64,
    
    // Signal parameters
    signal_threshold: f64,
    momentum_threshold: f64,
    trend_threshold: f64,
    
    // Adaptive parameters
    volatility_period: usize,
    adaptive_factor: f64,
    
    // Risk management
    max_position_size: f64,
    stop_loss_pct: f64,
    take_profit_pct: f64,
    max_drawdown_pct: f64,
    
    // Data storage
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    
    // RSI values for different timeframes
    rsi_short: f64,
    rsi_medium: f64,
    rsi_long: f64,
    
    // State tracking
    last_price: f64,
    price_momentum: f64,
    volume_ratio: f64,
    volatility: f64,
    trend_direction: i8, // -1: downtrend, 0: sideways, 1: uptrend
    consecutive_losses: usize,
    last_signal_time: f64,
}

impl EnhancedRsiStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("enhanced_rsi_strategy")
            .unwrap_or_else(|_| {
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let short_period = config.get_or("short_period", 4);
        let medium_period = config.get_or("medium_period", 8);
        let long_period = config.get_or("long_period", 16);
        let volume_period = config.get_or("volume_period", 10);
        let volume_threshold = config.get_or("volume_threshold", 1.1);
        let overbought_short = config.get_or("overbought_short", 68.0);
        let oversold_short = config.get_or("oversold_short", 32.0);
        let overbought_medium = config.get_or("overbought_medium", 70.0);
        let oversold_medium = config.get_or("oversold_medium", 30.0);
        let overbought_long = config.get_or("overbought_long", 72.0);
        let oversold_long = config.get_or("oversold_long", 28.0);
        let signal_threshold = config.get_or("signal_threshold", 0.1);
        let momentum_threshold = config.get_or("momentum_threshold", 0.00003);
        let trend_threshold = config.get_or("trend_threshold", 0.0005);
        let volatility_period = config.get_or("volatility_period", 15);
        let adaptive_factor = config.get_or("adaptive_factor", 0.5);
        let max_position_size = config.get_or("max_position_size", 1000.0);
        let stop_loss_pct = config.get_or("stop_loss_pct", 0.015);
        let take_profit_pct = config.get_or("take_profit_pct", 0.03);
        let max_drawdown_pct = config.get_or("max_drawdown_pct", 0.05);

        Self {
            short_period,
            medium_period,
            long_period,
            volume_period,
            volume_threshold,
            overbought_short,
            oversold_short,
            overbought_medium,
            oversold_medium,
            overbought_long,
            oversold_long,
            signal_threshold,
            momentum_threshold,
            trend_threshold,
            volatility_period,
            adaptive_factor,
            max_position_size,
            stop_loss_pct,
            take_profit_pct,
            max_drawdown_pct,
            prices: VecDeque::new(),
            volumes: VecDeque::new(),
            rsi_short: 50.0,
            rsi_medium: 50.0,
            rsi_long: 50.0,
            last_price: 0.0,
            price_momentum: 0.0,
            volume_ratio: 1.0,
            volatility: 0.0,
            trend_direction: 0,
            consecutive_losses: 0,
            last_signal_time: 0.0,
        }
    }

    fn calculate_rsi(&self, period: usize) -> f64 {
        if self.prices.len() < period + 1 {
            return 50.0;
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..=period {
            let current_price = self.prices[self.prices.len() - i];
            let prev_price = self.prices[self.prices.len() - i - 1];
            let change = current_price - prev_price;

            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
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

    fn calculate_volume_ratio(&self) -> f64 {
        if self.volumes.len() < self.volume_period {
            return 1.0;
        }
        
        let current_volume = self.volumes[self.volumes.len() - 1];
        let avg_volume: f64 = self.volumes.iter().rev().take(self.volume_period).sum::<f64>() / self.volume_period as f64;
        
        current_volume / avg_volume
    }

    fn calculate_volatility(&self) -> f64 {
        if self.prices.len() < self.volatility_period {
            return 0.0;
        }
        
        let mut returns = Vec::new();
        for i in 1..self.prices.len() {
            let current_price = self.prices[i];
            let prev_price = self.prices[i - 1];
            returns.push((current_price - prev_price) / prev_price);
        }
        
        if returns.is_empty() {
            return 0.0;
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }

    fn calculate_trend_direction(&self) -> i8 {
        if self.prices.len() < self.long_period {
            return 0;
        }
        
        let recent_prices: Vec<f64> = self.prices.iter().rev().take(self.long_period).cloned().collect();
        let first_half: Vec<f64> = recent_prices.iter().take(self.long_period / 2).cloned().collect();
        let second_half: Vec<f64> = recent_prices.iter().skip(self.long_period / 2).cloned().collect();
        
        let first_avg = first_half.iter().sum::<f64>() / first_half.len() as f64;
        let second_avg = second_half.iter().sum::<f64>() / second_half.len() as f64;
        
        let trend_strength = (second_avg - first_avg) / first_avg;
        
        if trend_strength > self.trend_threshold { 1 }
        else if trend_strength < -self.trend_threshold { -1 }
        else { 0 }
    }

    fn should_stop_loss(&self, entry_price: f64, current_price: f64, position: &Position) -> bool {
        if position.quantity == 0.0 {
            return false;
        }
        
        let pnl_pct = if position.quantity > 0.0 {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        
        pnl_pct < -self.stop_loss_pct
    }

    fn should_take_profit(&self, entry_price: f64, current_price: f64, position: &Position) -> bool {
        if position.quantity == 0.0 {
            return false;
        }
        
        let pnl_pct = if position.quantity > 0.0 {
            (current_price - entry_price) / entry_price
        } else {
            (entry_price - current_price) / entry_price
        };
        
        pnl_pct > self.take_profit_pct
    }

    fn get_adaptive_threshold(&self) -> f64 {
        let volatility_factor = 1.0 + (self.volatility * 10.0);
        let loss_factor = 1.0 + (self.consecutive_losses as f64 * 0.1);
        let volume_factor = if self.volume_ratio > 1.5 { 0.8 } else { 1.2 };
        
        self.signal_threshold * volatility_factor * loss_factor * volume_factor
    }

    fn calculate_signal_strength(&self, rsi_value: f64, is_buy: bool) -> f64 {
        let base_strength = if is_buy {
            (50.0 - rsi_value) / 50.0 // RSI below 50 = stronger buy signal
        } else {
            (rsi_value - 50.0) / 50.0 // RSI above 50 = stronger sell signal
        };
        
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 1.5 } else { 1.0 };
        let volume_factor = if self.volume_ratio > self.volume_threshold { 1.3 } else { 1.0 };
        let trend_factor = if (is_buy && self.trend_direction == 1) || (!is_buy && self.trend_direction == -1) { 1.4 } else { 0.8 };
        
        base_strength * momentum_factor * volume_factor * trend_factor
    }
}

#[async_trait::async_trait]
impl Strategy for EnhancedRsiStrategy {
    fn get_info(&self) -> String {
        format!("Enhanced RSI Strategy (short: {}, medium: {}, long: {}, vol: {})", 
                self.short_period, self.medium_period, self.long_period, self.volume_period)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (trade.price - self.last_price) / self.last_price;
        }
        
        self.prices.push_back(trade.price);
        self.volumes.push_back(trade.qty);
        
        // Keep only necessary data
        let max_len = self.long_period.max(self.volume_period).max(self.volatility_period) + 10;
        if self.prices.len() > max_len {
            self.prices.pop_front();
            self.volumes.pop_front();
        }
        
        // Update RSI values for all timeframes
        self.rsi_short = self.calculate_rsi(self.short_period);
        self.rsi_medium = self.calculate_rsi(self.medium_period);
        self.rsi_long = self.calculate_rsi(self.long_period);
        
        // Update other metrics
        self.volume_ratio = self.calculate_volume_ratio();
        self.volatility = self.calculate_volatility();
        self.trend_direction = self.calculate_trend_direction();
        
        self.last_price = trade.price;
    }

    fn get_signal(
        &self,
        current_price: f64,
        current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        if self.prices.len() < self.long_period {
            return (Signal::Hold, 0.0);
        }

        // Check stop loss and take profit
        if self.should_stop_loss(current_position.entry_price, current_price, &current_position) {
            return (Signal::Sell, 1.0);
        }
        
        if self.should_take_profit(current_position.entry_price, current_price, &current_position) {
            return (Signal::Sell, 1.0);
        }

        let adaptive_threshold = self.get_adaptive_threshold();
        let signal: Signal;
        let confidence: f64;

        // Multi-timeframe RSI analysis with volume confirmation
        let short_buy = self.rsi_short < self.oversold_short;
        let medium_buy = self.rsi_medium < self.oversold_medium;
        let long_buy = self.rsi_long < self.oversold_long;
        
        let short_sell = self.rsi_short > self.overbought_short;
        let medium_sell = self.rsi_medium > self.overbought_medium;
        let long_sell = self.rsi_long > self.overbought_long;

        // Strong buy signal: all timeframes oversold + volume confirmation + momentum
        if short_buy && medium_buy && long_buy && 
           self.volume_ratio > self.volume_threshold &&
           self.price_momentum > self.momentum_threshold {
            
            signal = Signal::Buy;
            confidence = self.calculate_signal_strength(self.rsi_short, true);
            
        // Strong sell signal: all timeframes overbought + volume confirmation + momentum
        } else if short_sell && medium_sell && long_sell && 
                  self.volume_ratio > self.volume_threshold &&
                  self.price_momentum < -self.momentum_threshold {
            
            signal = Signal::Sell;
            confidence = self.calculate_signal_strength(self.rsi_short, false);
            
        // Moderate buy signal: short and medium oversold
        } else if short_buy && medium_buy && 
                  self.volume_ratio > 1.0 &&
                  self.price_momentum > self.momentum_threshold * 0.5 {
            
            signal = Signal::Buy;
            confidence = self.calculate_signal_strength(self.rsi_short, true) * 0.8;
            
        // Moderate sell signal: short and medium overbought
        } else if short_sell && medium_sell && 
                  self.volume_ratio > 1.0 &&
                  self.price_momentum < -self.momentum_threshold * 0.5 {
            
            signal = Signal::Sell;
            confidence = self.calculate_signal_strength(self.rsi_short, false) * 0.8;
            
        // Weak signals based on short-term RSI only
        } else if short_buy && self.price_momentum > 0.0 {
            signal = Signal::Buy;
            confidence = self.calculate_signal_strength(self.rsi_short, true) * 0.6;
            
        } else if short_sell && self.price_momentum < 0.0 {
            signal = Signal::Sell;
            confidence = self.calculate_signal_strength(self.rsi_short, false) * 0.6;
            
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply adaptive threshold
        if confidence < adaptive_threshold {
            return (Signal::Hold, 0.0);
        }

        // Prevent rapid signal changes
        let time_since_last_signal = current_timestamp - self.last_signal_time;
        if time_since_last_signal < 1.0 && signal != Signal::Hold {
            return (Signal::Hold, 0.0);
        }

        (signal, confidence)
    }
} 