use std::collections::VecDeque;
use async_trait::async_trait;
use crate::strategy::Strategy;
use trade::{Position, Signal};
use trade::models::TradeData;

pub struct MeanReversionStrategy {
    // Price data
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    
    // Mean reversion indicators
    sma_20: VecDeque<f64>,
    sma_50: VecDeque<f64>,
    bollinger_upper: VecDeque<f64>,
    bollinger_lower: VecDeque<f64>,
    rsi_values: VecDeque<f64>,
    
    // Strategy parameters
    sma_short: usize,
    sma_long: usize,
    bollinger_window: usize,
    bollinger_std: f64,
    rsi_window: usize,
    
    // Risk management
    trade_counter: usize,
    last_signal_time: f64,
    signal_cooldown: f64,
    consecutive_losses: usize,
    max_consecutive_losses: usize,
    
    // Performance tracking
    total_pnl: f64,
    win_count: usize,
    loss_count: usize,
}

impl MeanReversionStrategy {
    pub fn new() -> Self {
        Self {
            price_history: VecDeque::with_capacity(1000),
            volume_history: VecDeque::with_capacity(1000),
            sma_20: VecDeque::with_capacity(1000),
            sma_50: VecDeque::with_capacity(1000),
            bollinger_upper: VecDeque::with_capacity(1000),
            bollinger_lower: VecDeque::with_capacity(1000),
            rsi_values: VecDeque::with_capacity(1000),
            
            // Mean reversion parameters
            sma_short: 20,
            sma_long: 50,
            bollinger_window: 20,
            bollinger_std: 2.0,
            rsi_window: 14,
            
            trade_counter: 0,
            last_signal_time: 0.0,
            signal_cooldown: 15.0, // 15 seconds cooldown for mean reversion
            consecutive_losses: 0,
            max_consecutive_losses: 2,
            
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
        }
    }
    
    fn calculate_sma(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window {
            return 0.0;
        }
        let recent_prices: Vec<f64> = prices.iter().rev().take(window).cloned().collect();
        recent_prices.iter().sum::<f64>() / recent_prices.len() as f64
    }
    
    fn calculate_bollinger_bands(&self, prices: &[f64], window: usize, std_dev: f64) -> (f64, f64) {
        if prices.len() < window {
            return (0.0, 0.0);
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(window).cloned().collect();
        let sma = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        
        let variance = recent_prices.iter()
            .map(|p| (p - sma).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
        let std = variance.sqrt();
        
        let upper = sma + (std_dev * std);
        let lower = sma - (std_dev * std);
        
        (upper, lower)
    }
    
    fn calculate_rsi(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window + 1 {
            return 50.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(window + 1).cloned().collect();
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..recent_prices.len() {
            let change = recent_prices[i] - recent_prices[i-1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }
        
        if losses == 0.0 {
            return 100.0;
        }
        
        let avg_gain = gains / window as f64;
        let avg_loss = losses / window as f64;
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    fn update_indicators(&mut self) {
        if self.price_history.len() < self.sma_long {
            return;
        }
        
        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        
        // Update moving averages
        let sma_20 = self.calculate_sma(&prices, self.sma_short);
        let sma_50 = self.calculate_sma(&prices, self.sma_long);
        self.sma_20.push_back(sma_20);
        self.sma_50.push_back(sma_50);
        
        // Update Bollinger Bands
        let (upper, lower) = self.calculate_bollinger_bands(&prices, self.bollinger_window, self.bollinger_std);
        self.bollinger_upper.push_back(upper);
        self.bollinger_lower.push_back(lower);
        
        // Update RSI
        let rsi = self.calculate_rsi(&prices, self.rsi_window);
        self.rsi_values.push_back(rsi);
        
        // Keep only recent data
        if self.sma_20.len() > 1000 {
            self.sma_20.pop_front();
            self.sma_50.pop_front();
            self.bollinger_upper.pop_front();
            self.bollinger_lower.pop_front();
            self.rsi_values.pop_front();
        }
    }
    
    fn generate_mean_reversion_signal(&self, current_timestamp: f64) -> (Signal, f64) {
        // Check cooldown and consecutive losses
        if current_timestamp - self.last_signal_time < self.signal_cooldown {
            return (Signal::Hold, 0.0);
        }
        
        if self.consecutive_losses >= self.max_consecutive_losses {
            return (Signal::Hold, 0.0);
        }
        
        if self.price_history.len() < self.sma_long {
            return (Signal::Hold, 0.0);
        }
        
        let current_price = *self.price_history.back().unwrap();
        let sma_20 = *self.sma_20.back().unwrap();
        let sma_50 = *self.sma_50.back().unwrap();
        let upper_band = *self.bollinger_upper.back().unwrap();
        let lower_band = *self.bollinger_lower.back().unwrap();
        let rsi = *self.rsi_values.back().unwrap();
        
        // Mean Reversion Strategy 1: RSI oversold/overbought
        if rsi < 25.0 {
            return (Signal::Buy, 0.9);
        }
        
        if rsi > 75.0 {
            return (Signal::Sell, 0.9);
        }
        
        // Mean Reversion Strategy 2: Bollinger Band extremes
        if current_price < lower_band && rsi < 40.0 {
            return (Signal::Buy, 0.8);
        }
        
        if current_price > upper_band && rsi > 60.0 {
            return (Signal::Sell, 0.8);
        }
        
        // Mean Reversion Strategy 3: Price vs long-term SMA
        let price_vs_sma50 = (current_price - sma_50) / sma_50;
        if price_vs_sma50 < -0.02 && rsi < 35.0 {
            return (Signal::Buy, 0.7);
        }
        
        if price_vs_sma50 > 0.02 && rsi > 65.0 {
            return (Signal::Sell, 0.7);
        }
        
        // Mean Reversion Strategy 4: Moving average convergence
        let ma_diff = (sma_20 - sma_50) / sma_50;
        if ma_diff < -0.01 && current_price < sma_20 {
            return (Signal::Buy, 0.6);
        }
        
        if ma_diff > 0.01 && current_price > sma_20 {
            return (Signal::Sell, 0.6);
        }
        
        (Signal::Hold, 0.0)
    }
}

#[async_trait]
impl Strategy for MeanReversionStrategy {
    fn get_info(&self) -> String {
        let win_rate = if self.win_count + self.loss_count > 0 {
            (self.win_count as f64 / (self.win_count + self.loss_count) as f64) * 100.0
        } else {
            0.0
        };
        format!("Mean Reversion Strategy - Win Rate: {:.1}%, Total PnL: {:.6}", win_rate, self.total_pnl)
    }
    
    async fn on_trade(&mut self, trade: TradeData) {
        self.trade_counter += 1;
        
        // Update price and volume data
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.qty);
        
        // Keep only recent data
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }
        
        // Update technical indicators
        self.update_indicators();
    }
    
    fn get_signal(
        &self,
        _current_price: f64,
        current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.price_history.len() < self.sma_short {
            return (Signal::Hold, 0.0);
        }
        
        self.generate_mean_reversion_signal(current_timestamp)
    }
}
