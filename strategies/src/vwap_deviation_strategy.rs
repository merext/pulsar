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
        let signal_threshold = config.get_or("signal_threshold", 0.3);

        Self {
            period,
            deviation_threshold,
            signal_threshold,
            trades: Vec::with_capacity(period),
            total_volume: 0.0,
            total_price_volume: 0.0,
            vwap: 0.0,
        }
    }
}

#[async_trait]
impl Strategy for VwapDeviationStrategy {
    fn get_info(&self) -> String {
        format!("VWAP Deviation Strategy (period: {}, deviation_threshold: {})", self.period, self.deviation_threshold)
    }

    async fn on_trade(&mut self, trade: TradeData) {
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
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.vwap == 0.0 {
            return (Signal::Hold, 0.0);
        }

        let deviation = (current_price - self.vwap) / self.vwap;

        if deviation < -self.deviation_threshold {
            let confidence = deviation.abs().min(1.0);
            if confidence < self.signal_threshold {
                (Signal::Hold, 0.0)
            } else {
                (Signal::Buy, confidence)
            }
        } else if deviation > self.deviation_threshold {
            let confidence = deviation.abs().min(1.0);
            if confidence < self.signal_threshold {
                (Signal::Hold, 0.0)
            } else {
                (Signal::Sell, confidence)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }
}
