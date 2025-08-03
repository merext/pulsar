//! # HFT Ultra-Fast Strategy
//! 
//! This is a high-frequency trading strategy optimized for:
//! - Ultra-low latency (< 1 microsecond signal generation)
//! - Minimal memory allocations
//! - Simple, branch-predictable calculations
//! - Tick-by-tick processing
//! - Immediate position management
//! 
//! Designed for HFT environments where speed is critical.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;

#[derive(Clone, Debug)]
pub struct HftUltraFastStrategy {
    // Pre-allocated buffers for minimal allocations
    price_buffer: [f64; 64],  // Fixed-size array, no VecDeque
    volume_buffer: [f64; 64],
    
    // Current buffer position
    buffer_pos: usize,
    buffer_size: usize,
    
    // Simple indicators (no complex calculations)
    last_price: f64,
    last_volume: f64,
    price_change: f64,
    volume_change: f64,
    
    // Fast moving averages (exponential, not simple)
    ema_fast: f64,
    ema_slow: f64,
    
    // Volatility (simple rolling standard deviation)
    volatility: f64,
    volatility_sum: f64,
    volatility_sum_sq: f64,
    
    // Volume analysis
    avg_volume: f64,
    volume_ratio: f64,
    
    // Signal thresholds (pre-calculated)
    buy_threshold: f64,
    volume_threshold: f64,
    
    // Performance tracking (minimal)
    trades_won: u32,
    trades_total: u32,
}

impl HftUltraFastStrategy {
    pub fn new() -> Self {
        Self {
            price_buffer: [0.0; 64],
            volume_buffer: [0.0; 64],
            buffer_pos: 0,
            buffer_size: 0,
            last_price: 0.0,
            last_volume: 0.0,
            price_change: 0.0,
            volume_change: 0.0,
            ema_fast: 0.0,
            ema_slow: 0.0,
            volatility: 0.0,
            volatility_sum: 0.0,
            volatility_sum_sq: 0.0,
            avg_volume: 0.0,
            volume_ratio: 1.0,
            buy_threshold: 0.0005,  // 0.05% price change
            volume_threshold: 1.2,   // 20% above average volume
            trades_won: 0,
            trades_total: 0,
        }
    }

    #[inline]
    fn update_price_buffer(&mut self, price: f64) {
        // Fast circular buffer update
        self.price_buffer[self.buffer_pos] = price;
        self.buffer_pos = (self.buffer_pos + 1) % 64;
        if self.buffer_size < 64 {
            self.buffer_size += 1;
        }
    }

    #[inline]
    fn update_volume_buffer(&mut self, volume: f64) {
        self.volume_buffer[self.buffer_pos] = volume;
    }

    #[inline]
    fn calculate_fast_ema(&mut self, price: f64, alpha: f64) -> f64 {
        if self.ema_fast == 0.0 {
            self.ema_fast = price;
        } else {
            self.ema_fast = alpha * price + (1.0 - alpha) * self.ema_fast;
        }
        self.ema_fast
    }

    #[inline]
    fn calculate_slow_ema(&mut self, price: f64, alpha: f64) -> f64 {
        if self.ema_slow == 0.0 {
            self.ema_slow = price;
        } else {
            self.ema_slow = alpha * price + (1.0 - alpha) * self.ema_slow;
        }
        self.ema_slow
    }

    #[inline]
    fn update_volatility(&mut self, price_change: f64) {
        // Simple rolling volatility calculation
        self.volatility_sum += price_change;
        self.volatility_sum_sq += price_change * price_change;
        
        if self.buffer_size > 10 {
            let mean = self.volatility_sum / self.buffer_size as f64;
            let variance = (self.volatility_sum_sq / self.buffer_size as f64) - (mean * mean);
            self.volatility = variance.sqrt();
        }
    }

    #[inline]
    fn update_volume_analysis(&mut self, volume: f64) {
        // Simple volume ratio calculation
        if self.avg_volume == 0.0 {
            self.avg_volume = volume;
        } else {
            self.avg_volume = 0.9 * self.avg_volume + 0.1 * volume;
        }
        self.volume_ratio = volume / self.avg_volume;
    }

    #[inline]
    fn generate_signal(&self, current_price: f64) -> (Signal, f64) {
        // Ultra-fast signal generation with minimal branching
        
        // Check if we have enough data
        if self.buffer_size < 10 {
            return (Signal::Hold, 0.0);
        }

        // Calculate price momentum
        let price_momentum = (current_price - self.last_price) / self.last_price;
        
        // Calculate EMA crossover
        let ema_crossover = self.ema_fast - self.ema_slow;
        let ema_signal = if ema_crossover > 0.0 { 1.0 } else { -1.0 };
        
        // Volume confirmation
        let volume_signal = if self.volume_ratio > self.volume_threshold { 1.0 } else { 0.0 };
        
        // Volatility adjustment
        let volatility_factor = if self.volatility > 0.01 { 0.5 } else { 1.0 };
        
        // Combine signals (simple weighted sum)
        let buy_score = (price_momentum * 0.4 + ema_signal * 0.4 + volume_signal * 0.2) * volatility_factor;
        let sell_score = (-price_momentum * 0.4 - ema_signal * 0.4 + volume_signal * 0.2) * volatility_factor;
        
        // Generate signal with confidence
        if buy_score > self.buy_threshold {
            (Signal::Buy, buy_score.min(1.0))
        } else if sell_score > self.buy_threshold {
            (Signal::Sell, sell_score.min(1.0))
        } else {
            (Signal::Hold, 0.0)
        }
    }



    #[inline]
    fn get_win_rate(&self) -> f64 {
        if self.trades_total == 0 {
            0.0
        } else {
            self.trades_won as f64 / self.trades_total as f64
        }
    }
}

#[async_trait::async_trait]
impl Strategy for HftUltraFastStrategy {
    fn get_info(&self) -> String {
        format!(
            "HFT Ultra-Fast Strategy (win_rate: {:.1}%, trades: {})",
            self.get_win_rate() * 100.0,
            self.trades_total
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        let volume = trade.qty;
        
        // Update price change
        if self.last_price > 0.0 {
            self.price_change = (price - self.last_price) / self.last_price;
        }
        
        // Update volume change
        if self.last_volume > 0.0 {
            self.volume_change = (volume - self.last_volume) / self.last_volume;
        }
        
        // Update buffers
        self.update_price_buffer(price);
        self.update_volume_buffer(volume);
        
        // Update indicators (fast calculations)
        self.calculate_fast_ema(price, 0.1);  // Fast EMA (10-period equivalent)
        self.calculate_slow_ema(price, 0.05); // Slow EMA (20-period equivalent)
        self.update_volatility(self.price_change);
        self.update_volume_analysis(volume);
        
        // Update last values
        self.last_price = price;
        self.last_volume = volume;
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Ultra-fast signal generation
        self.generate_signal(current_price)
    }
} 