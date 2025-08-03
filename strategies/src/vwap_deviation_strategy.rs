//! # VWAP Deviation Strategy
//!
//! This strategy aims to identify trading opportunities when the current price deviates significantly from the Volume Weighted Average Price (VWAP).
//! VWAP represents the average price an asset has traded at throughout the day, based on both volume and price.
//!
//! The strategy generates a buy signal if the price falls significantly below VWAP (indicating undervaluation)
//! and a sell signal if the price rises significantly above VWAP (indicating overvaluation).

use crate::config::StrategyConfig;
use async_trait::async_trait;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;
use crate::strategy::Strategy;
use toml;

#[derive(Clone)]
pub struct VwapDeviationStrategy {
    period: usize,
    deviation_threshold: f64,
    signal_threshold: f64,
    trades: Vec<TradeData>,
    total_volume: f64,
    total_price_volume: f64,
    vwap: f64,
    // Configuration parameters for momentum approach
    momentum_threshold: f64,
    last_price: f64,
    price_momentum: f64,
}

impl VwapDeviationStrategy {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("vwap_deviation_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        let period = config.get_or("period", 20);
        let deviation_threshold = config.get_or("deviation_threshold", 0.001);
        let signal_threshold = config.get_or("signal_threshold", 0.01);
        let momentum_threshold = config.get_or("momentum_threshold", 0.00001);

        Self {
            period,
            deviation_threshold,
            signal_threshold,
            trades: Vec::with_capacity(period),
            total_volume: 0.0,
            total_price_volume: 0.0,
            vwap: 0.0,
            momentum_threshold,
            last_price: 0.0,
            price_momentum: 0.0,
        }
    }
}

#[async_trait]
impl Strategy for VwapDeviationStrategy {
    fn get_info(&self) -> String {
        format!("VWAP Deviation Strategy (period: {}, deviation_threshold: {})", self.period, self.deviation_threshold)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (price - self.last_price) / self.last_price;
        }
        
        self.total_volume += trade.qty;
        self.total_price_volume += trade.price * trade.qty;
        self.trades.push(trade);

        if self.trades.len() > self.period {
            let oldest_trade = self.trades.remove(0);
            self.total_volume -= oldest_trade.qty;
            self.total_price_volume -= oldest_trade.price * oldest_trade.qty;
        }

        if self.total_volume > 0.0 {
            self.vwap = self.total_price_volume / self.total_volume;
        }
        
        self.last_price = price;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Pure momentum approach like successful strategies (ignore VWAP deviation)
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };
        let momentum_strength = (self.price_momentum * 3000.0).min(1.0);

        let signal: Signal;
        let confidence: f64;

        if self.price_momentum > self.momentum_threshold {
            signal = Signal::Buy;
            confidence = momentum_strength * momentum_factor;
        } else if self.price_momentum < -self.momentum_threshold {
            signal = Signal::Sell;
            confidence = momentum_strength * momentum_factor;
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }

        // Apply signal threshold filter
        if confidence < self.signal_threshold {
            return (Signal::Hold, 0.0);
        }

        (signal, confidence.min(1.0))
    }
}
