//! # Relative Strength Index (RSI) Strategy
//! 
//! This strategy utilizes the Relative Strength Index (RSI), a momentum oscillator that measures the speed and change of price movements.
//! RSI oscillates between zero and 100 and is typically used to identify overbought or oversold conditions in an asset.
//! 
//! The strategy generates a sell signal when the RSI crosses above a defined overbought threshold (e.g., 70),
//! indicating that the asset may be overvalued and due for a price correction.
//! Conversely, a buy signal is generated when the RSI falls below a defined oversold threshold (e.g., 30),
//! suggesting the asset may be undervalued and due for a price rebound.

use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;
use toml;

#[derive(Clone)]
pub struct RsiStrategy {
    period: usize,
    overbought: f64,
    oversold: f64,
    scale: f64,
    prices: VecDeque<f64>,
    // Configuration parameters
    signal_threshold: f64,
    momentum_threshold: f64,
    // Performance tracking
    last_price: f64,
    price_momentum: f64,
    last_rsi: f64,
    rsi_momentum: f64,
}

impl RsiStrategy {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("rsi_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let period = config.get_or("period", 10);
        let overbought = config.get_or("overbought", 75.0);
        let oversold = config.get_or("oversold", 25.0);
        let scale = config.get_or("scale", 1.2);
        let signal_threshold = config.get_or("signal_threshold", 0.3);
        let momentum_threshold = config.get_or("momentum_threshold", 0.001);

        Self {
            period,
            overbought,
            oversold,
            scale,
            prices: VecDeque::new(),
            signal_threshold,
            momentum_threshold,
            last_price: 0.0,
            price_momentum: 0.0,
            last_rsi: 50.0,
            rsi_momentum: 0.0,
        }
    }

    fn calculate_rsi(&self) -> f64 {
        if self.prices.len() < self.period + 1 {
            return 50.0; // Neutral RSI if not enough data
        }

        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in 1..=self.period {
            let current_price = self.prices[self.prices.len() - i];
            let prev_price = self.prices[self.prices.len() - i - 1];
            let change = current_price - prev_price;

            if change > 0.0 {
                gains += change;
            } else {
                losses -= change; // losses are positive values
            }
        }

        let avg_gain = gains / self.period as f64;
        let avg_loss = losses / self.period as f64;

        if avg_loss == 0.0 {
            return 100.0; // Avoid division by zero, strong uptrend
        }

        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
}

#[async_trait::async_trait]
impl Strategy for RsiStrategy {
    fn get_info(&self) -> String {
        format!("RSI Strategy (period: {}, overbought: {}, oversold: {}, scale: {})", self.period, self.overbought, self.oversold, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let trade_price = trade.price;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (trade_price - self.last_price) / self.last_price;
        }
        
        self.prices.push_back(trade_price);
        if self.prices.len() > self.period + 1 {
            self.prices.pop_front();
        }
        
        // Update RSI momentum
        let current_rsi = self.calculate_rsi();
        if self.last_rsi > 0.0 {
            self.rsi_momentum = current_rsi - self.last_rsi;
        }
        self.last_rsi = current_rsi;
        
        self.last_price = trade_price;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        let rsi = self.calculate_rsi();

        // Pure momentum approach - ignore traditional RSI levels
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };
        let rsi_momentum_factor = if self.rsi_momentum.abs() > 1.0 { 1.8 } else { 1.0 };
        
        let signal: Signal;
        let confidence: f64;

        // Pure momentum strategy - RSI just confirms momentum direction
        if self.price_momentum > 0.00005 {
            // Any positive momentum - buy (like HFT Ultra Fast)
            signal = Signal::Buy;
            let momentum_strength = (self.price_momentum * 2000.0).min(1.0);
            let rsi_confirmation = if rsi > 50.0 { 1.2 } else { 0.8 }; // RSI confirms trend
            confidence = momentum_strength * rsi_confirmation * momentum_factor * rsi_momentum_factor * self.scale;
        } else if self.price_momentum < -0.00005 {
            // Any negative momentum - sell (like HFT Ultra Fast)
            signal = Signal::Sell;
            let momentum_strength = (self.price_momentum.abs() * 2000.0).min(1.0);
            let rsi_confirmation = if rsi < 50.0 { 1.2 } else { 0.8 }; // RSI confirms trend
            confidence = momentum_strength * rsi_confirmation * momentum_factor * rsi_momentum_factor * self.scale;
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply signal threshold filter
        if confidence < self.signal_threshold {
            return (Signal::Hold, 0.0);
        } else {
            return (signal, confidence);
        }
    }
}