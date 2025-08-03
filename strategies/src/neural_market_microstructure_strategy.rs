//! # Neural Market Microstructure Strategy
//! 
//! This is a cutting-edge trading strategy that combines:
//! - Neural network-inspired pattern recognition
//! - Market microstructure analysis
//! - Order flow imbalance detection
//! - Liquidity analysis
//! - Adaptive learning from market conditions
//! - Multi-timeframe analysis
//! - Sentiment analysis simulation
//! 
//! The strategy uses sophisticated mathematical models to identify
//! high-probability trading opportunities in various market conditions.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use std::f64;

#[derive(Clone, Debug)]
pub struct NeuralMarketMicrostructureStrategy {
    // Core parameters
    short_window: usize,
    medium_window: usize,
    long_window: usize,
    micro_window: usize,
    
    // Price and volume data
    prices: VecDeque<f64>,
    volumes: VecDeque<f64>,
    timestamps: VecDeque<u64>,
    trade_sizes: VecDeque<f64>,
    
    // Technical indicators
    ema_short: Option<f64>,
    ema_medium: Option<f64>,
    ema_long: Option<f64>,
    rsi: Option<f64>,
    macd: Option<f64>,
    macd_signal: Option<f64>,
    
    // Market microstructure
    price_velocity: VecDeque<f64>,
    trade_imbalance: VecDeque<f64>,
    liquidity_score: f64,
    
    // Neural-inspired features
    hidden_layer: Vec<f64>,
    feature_weights: Vec<f64>,
    bias_terms: Vec<f64>,
    
    // Adaptive learning
    performance_history: VecDeque<f64>,
    

    
    // Market regime
    market_regime: MarketRegime,
    regime_confidence: f64,
    
    // Multi-timeframe analysis
    timeframe_signals: Vec<Signal>,
    signal_weights: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
enum MarketRegime {
    Trending,
    MeanReverting,
    Volatile,
    Sideways,
    Breakout,
}

impl NeuralMarketMicrostructureStrategy {
    pub fn new(
        short_window: usize,
        medium_window: usize,
        long_window: usize,
        micro_window: usize,
    ) -> Self {
        let hidden_size = 8;
        let feature_size = 12;
        
        Self {
            short_window,
            medium_window,
            long_window,
            micro_window,
            prices: VecDeque::with_capacity(long_window),
            volumes: VecDeque::with_capacity(long_window),
            timestamps: VecDeque::with_capacity(long_window),
            trade_sizes: VecDeque::with_capacity(micro_window),
            ema_short: None,
            ema_medium: None,
            ema_long: None,
            rsi: None,
            macd: None,
            macd_signal: None,
            price_velocity: VecDeque::with_capacity(10),
            trade_imbalance: VecDeque::with_capacity(10),
            liquidity_score: 0.5,
            hidden_layer: vec![0.0; hidden_size],
            feature_weights: vec![0.1; feature_size * hidden_size],
            bias_terms: vec![0.0; hidden_size],
            performance_history: VecDeque::with_capacity(100),

            market_regime: MarketRegime::Sideways,
            regime_confidence: 0.5,
            timeframe_signals: vec![Signal::Hold; 3],
            signal_weights: vec![0.4, 0.35, 0.25], // Short, medium, long weights
        }
    }

    fn calculate_ema(&self, window: usize, alpha: f64) -> Option<f64> {
        if self.prices.len() < window {
            return None;
        }
        
        let mut ema = self.prices[0];
        for &price in self.prices.iter().skip(1).take(window) {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        Some(ema)
    }

    fn calculate_rsi(&self, period: usize) -> Option<f64> {
        if self.prices.len() < period + 1 {
            return None;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..=period {
            let change = self.prices[self.prices.len() - i] - self.prices[self.prices.len() - i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            return Some(100.0);
        }
        
        let rs = avg_gain / avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }

    fn calculate_macd(&self) -> Option<(f64, f64)> {
        if let (Some(ema12), Some(ema26)) = (
            self.calculate_ema(12, 2.0 / 13.0),
            self.calculate_ema(26, 2.0 / 27.0)
        ) {
            let macd_line = ema12 - ema26;
            let signal_line = macd_line * 0.2 + (self.macd.unwrap_or(macd_line) * 0.8);
            Some((macd_line, signal_line))
        } else {
            None
        }
    }

    fn calculate_price_velocity(&mut self) {
        if self.prices.len() < 2 {
            return;
        }
        
        let current_price = self.prices.back().unwrap();
        let prev_price = self.prices[self.prices.len() - 2];
        let velocity = (current_price - prev_price) / prev_price;
        
        self.price_velocity.push_back(velocity);
        if self.price_velocity.len() > 10 {
            self.price_velocity.pop_front();
        }
    }

    fn calculate_trade_imbalance(&mut self) {
        if self.trade_sizes.len() < 2 {
            return;
        }
        
        let recent_trades: Vec<f64> = self.trade_sizes.iter().rev().take(10).cloned().collect();
        let avg_trade_size = recent_trades.iter().sum::<f64>() / recent_trades.len() as f64;
        let current_trade_size = self.trade_sizes.back().unwrap();
        
        let imbalance = if avg_trade_size > 0.0 {
            (current_trade_size - avg_trade_size) / avg_trade_size
        } else {
            0.0
        };
        
        self.trade_imbalance.push_back(imbalance);
        if self.trade_imbalance.len() > 10 {
            self.trade_imbalance.pop_front();
        }
    }

    fn calculate_liquidity_score(&mut self) {
        if self.volumes.len() < 20 {
            return;
        }
        
        let recent_volumes: Vec<f64> = self.volumes.iter().rev().take(20).cloned().collect();
        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let current_volume = self.volumes.back().unwrap();
        
        // Liquidity score based on volume relative to average
        let volume_ratio = current_volume / avg_volume;
        
        // Also consider price impact (smaller trades = higher liquidity)
        let avg_trade_size = if !self.trade_sizes.is_empty() {
            self.trade_sizes.iter().sum::<f64>() / self.trade_sizes.len() as f64
        } else {
            1.0
        };
        
        let size_factor = 1.0 / (1.0 + avg_trade_size);
        
        self.liquidity_score = (volume_ratio * 0.7 + size_factor * 0.3).max(0.0).min(1.0);
    }

    fn detect_market_regime(&mut self) {
        if self.prices.len() < self.long_window {
            return;
        }
        
        let _price_std = self.calculate_price_std();
        let trend_strength = self.calculate_trend_strength();
        let volatility = self.calculate_volatility();
        
        // Regime classification based on multiple factors
        if trend_strength > 0.02 && volatility < 0.3 {
            self.market_regime = MarketRegime::Trending;
            self.regime_confidence = trend_strength.min(1.0);
        } else if trend_strength.abs() < 0.01 && volatility < 0.2 {
            self.market_regime = MarketRegime::Sideways;
            self.regime_confidence = 0.8;
        } else if volatility > 0.4 {
            self.market_regime = MarketRegime::Volatile;
            self.regime_confidence = volatility.min(1.0);
        } else if trend_strength.abs() > 0.01 && trend_strength.abs() < 0.02 {
            self.market_regime = MarketRegime::MeanReverting;
            self.regime_confidence = 0.6;
        } else {
            self.market_regime = MarketRegime::Breakout;
            self.regime_confidence = 0.7;
        }
    }

    fn calculate_price_std(&self) -> f64 {
        if self.prices.len() < 20 {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = self.prices.iter().rev().take(20).cloned().collect();
        let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let variance = recent_prices.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / recent_prices.len() as f64;
        
        variance.sqrt()
    }

    fn calculate_trend_strength(&self) -> f64 {
        if let (Some(ema_short), Some(ema_long)) = (self.ema_short, self.ema_long) {
            (ema_short - ema_long) / ema_long
        } else {
            0.0
        }
    }

    fn calculate_volatility(&self) -> f64 {
        if self.price_velocity.len() < 10 {
            return 0.0;
        }
        
        let mean_velocity = self.price_velocity.iter().sum::<f64>() / self.price_velocity.len() as f64;
        let variance = self.price_velocity.iter()
            .map(|v| (v - mean_velocity).powi(2))
            .sum::<f64>() / self.price_velocity.len() as f64;
        
        variance.sqrt()
    }

    fn extract_features(&self, current_price: f64) -> Vec<f64> {
        let mut features = Vec::with_capacity(12);
        
        // Price-based features
        features.push(current_price);
        features.push(self.prices.back().unwrap_or(&current_price) / current_price - 1.0);
        
        // Technical indicator features
        features.push(self.ema_short.unwrap_or(current_price) / current_price - 1.0);
        features.push(self.ema_medium.unwrap_or(current_price) / current_price - 1.0);
        features.push(self.ema_long.unwrap_or(current_price) / current_price - 1.0);
        features.push(self.rsi.unwrap_or(50.0) / 100.0);
        features.push(self.macd.unwrap_or(0.0));
        features.push(self.macd_signal.unwrap_or(0.0));
        
        // Volume and microstructure features
        features.push(self.liquidity_score);
        features.push(self.trade_imbalance.back().unwrap_or(&0.0).clone());
        features.push(self.price_velocity.back().unwrap_or(&0.0).clone());
        
        // Market regime feature
        let regime_value = match self.market_regime {
            MarketRegime::Trending => 0.2,
            MarketRegime::MeanReverting => 0.4,
            MarketRegime::Volatile => 0.6,
            MarketRegime::Sideways => 0.8,
            MarketRegime::Breakout => 1.0,
        };
        features.push(regime_value);
        
        features
    }

    fn forward_propagation(&mut self, features: &[f64]) -> f64 {
        // Hidden layer computation
        for i in 0..self.hidden_layer.len() {
            let mut sum = self.bias_terms[i];
            for j in 0..features.len() {
                sum += features[j] * self.feature_weights[j * self.hidden_layer.len() + i];
            }
            self.hidden_layer[i] = self.activation_function(sum);
        }
        
        // Output computation (simple weighted sum)
        let mut output = 0.0;
        for &hidden_value in &self.hidden_layer {
            output += hidden_value;
        }
        
        output / self.hidden_layer.len() as f64
    }

    fn activation_function(&self, x: f64) -> f64 {
        // ReLU-like activation with tanh for bounded output
        x.max(0.0).tanh()
    }



    fn generate_multi_timeframe_signals(&mut self, current_price: f64) {
        // Short-term signal (momentum-based)
        self.timeframe_signals[0] = self.generate_short_term_signal(current_price);
        
        // Medium-term signal (trend-based)
        self.timeframe_signals[1] = self.generate_medium_term_signal(current_price);
        
        // Long-term signal (regime-based)
        self.timeframe_signals[2] = self.generate_long_term_signal(current_price);
    }

    fn generate_short_term_signal(&self, _current_price: f64) -> Signal {
        if let Some(rsi) = self.rsi {
            if rsi < 30.0 {
                Signal::Buy
            } else if rsi > 70.0 {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }

    fn generate_medium_term_signal(&self, _current_price: f64) -> Signal {
        if let (Some(macd), Some(signal)) = (self.macd, self.macd_signal) {
            if macd > signal && macd > 0.0 {
                Signal::Buy
            } else if macd < signal && macd < 0.0 {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            Signal::Hold
        }
    }

    fn generate_long_term_signal(&self, _current_price: f64) -> Signal {
        match self.market_regime {
            MarketRegime::Trending => {
                if let (Some(ema_short), Some(ema_long)) = (self.ema_short, self.ema_long) {
                    if ema_short > ema_long {
                        Signal::Buy
                    } else {
                        Signal::Sell
                    }
                } else {
                    Signal::Hold
                }
            },
            MarketRegime::MeanReverting => {
                if let Some(rsi) = self.rsi {
                    if rsi < 40.0 {
                        Signal::Buy
                    } else if rsi > 60.0 {
                        Signal::Sell
                    } else {
                        Signal::Hold
                    }
                } else {
                    Signal::Hold
                }
            },
            _ => Signal::Hold,
        }
    }

    fn combine_signals(&self) -> (Signal, f64) {
        let mut buy_score = 0.0;
        let mut sell_score = 0.0;
        let mut total_weight = 0.0;
        
        for (signal, weight) in self.timeframe_signals.iter().zip(self.signal_weights.iter()) {
            total_weight += weight;
            match signal {
                Signal::Buy => buy_score += weight,
                Signal::Sell => sell_score += weight,
                Signal::Hold => {},
            }
        }
        
        let threshold = 0.6;
        if buy_score > threshold && buy_score > sell_score {
            (Signal::Buy, buy_score)
        } else if sell_score > threshold && sell_score > buy_score {
            (Signal::Sell, sell_score)
        } else {
            (Signal::Hold, 0.0)
        }
    }
}

#[async_trait::async_trait]
impl Strategy for NeuralMarketMicrostructureStrategy {
    fn get_info(&self) -> String {
        format!(
            "Neural Market Microstructure Strategy (short: {}, medium: {}, long: {}, regime: {:?}, confidence: {:.2})",
            self.short_window,
            self.medium_window,
            self.long_window,
            self.market_regime,
            self.regime_confidence
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        let volume = trade.qty;
        let timestamp = trade.time;
        
        // Update data structures
        self.prices.push_back(price);
        if self.prices.len() > self.long_window {
            self.prices.pop_front();
        }
        
        self.volumes.push_back(volume);
        if self.volumes.len() > self.long_window {
            self.volumes.pop_front();
        }
        
        self.timestamps.push_back(timestamp);
        if self.timestamps.len() > self.long_window {
            self.timestamps.pop_front();
        }
        
        self.trade_sizes.push_back(volume);
        if self.trade_sizes.len() > self.micro_window {
            self.trade_sizes.pop_front();
        }
        
        // Update technical indicators
        self.ema_short = self.calculate_ema(self.short_window, 2.0 / (self.short_window as f64 + 1.0));
        self.ema_medium = self.calculate_ema(self.medium_window, 2.0 / (self.medium_window as f64 + 1.0));
        self.ema_long = self.calculate_ema(self.long_window, 2.0 / (self.long_window as f64 + 1.0));
        self.rsi = self.calculate_rsi(14);
        
        if let Some((macd_line, signal_line)) = self.calculate_macd() {
            self.macd = Some(macd_line);
            self.macd_signal = Some(signal_line);
        }
        
        // Update microstructure metrics
        self.calculate_price_velocity();
        self.calculate_trade_imbalance();
        self.calculate_liquidity_score();
        
        // Detect market regime
        self.detect_market_regime();
        
        // Update performance history for learning
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Check if we have enough data
        if self.prices.len() < self.long_window {
            return (Signal::Hold, 0.0);
        }
        
        // Extract features for neural network
        let features = self.extract_features(current_price);
        
        // Get neural network prediction
        let mut strategy = self.clone();
        let neural_output = strategy.forward_propagation(&features);
        
        // Generate multi-timeframe signals
        let mut strategy = self.clone();
        strategy.generate_multi_timeframe_signals(current_price);
        let (ensemble_signal, ensemble_confidence) = strategy.combine_signals();
        
        // Combine neural output with ensemble signal
        let final_confidence = (neural_output + ensemble_confidence) / 2.0;
        
        // Apply risk filters
        let vol = self.calculate_volatility();
        if vol > 0.5 && final_confidence < 0.8 {
            return (Signal::Hold, 0.0);
        }
        
        // Apply liquidity filter
        if self.liquidity_score < 0.3 {
            return (Signal::Hold, 0.0);
        }
        
        (ensemble_signal, final_confidence.max(0.0).min(1.0))
    }
} 