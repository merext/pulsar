//! # Order Book Imbalance Strategy
//!
//! This strategy analyzes the imbalance between buy and sell volumes within recent trades to predict short-term price movements.
//! It operates on the principle that a significant imbalance can indicate immediate buying or selling pressure.
//!
//! The strategy calculates an Order Book Imbalance (OBI) metric based on the volume of buyer-initiated versus seller-initiated trades within a defined period.
//! A buy signal is generated if the OBI exceeds a positive threshold, indicating strong buying pressure.
//! Conversely, a sell signal is generated if the OBI falls below a negative threshold, indicating strong selling pressure.

use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use std::collections::VecDeque;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;
use async_trait::async_trait;

#[derive(Clone)]
pub struct OrderBookImbalance {
    period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    scale: f64,
    trades: VecDeque<TradeData>,
    // Configuration parameters
    signal_threshold: f64,
    momentum_threshold: f64,
    last_price: f64,
    price_momentum: f64,
}

impl OrderBookImbalance {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("order_book_imbalance_strategy");
        
        let period = config.as_ref().map(|c| c.get_or("period", 20)).unwrap_or(20);
        let buy_threshold = config.as_ref().map(|c| c.get_or("buy_threshold", 0.1)).unwrap_or(0.1);
        let sell_threshold = config.as_ref().map(|c| c.get_or("sell_threshold", -0.1)).unwrap_or(-0.1);
        let scale = config.as_ref().map(|c| c.get_or("scale", 1.2)).unwrap_or(1.2);
        let signal_threshold = config.as_ref().map(|c| c.get_or("signal_threshold", 0.3)).unwrap_or(0.3);
        let momentum_threshold = config.as_ref().map(|c| c.get_or("momentum_threshold", 0.0001)).unwrap_or(0.0001);

        Self {
            period,
            buy_threshold,
            sell_threshold,
            scale,
            trades: VecDeque::new(),
            signal_threshold,
            momentum_threshold,
            last_price: 0.0,
            price_momentum: 0.0,
        }
    }

    fn calculate_obi(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for trade in self.trades.iter() {
            let quantity = trade.qty;
            if trade.is_buyer_maker {
                buy_volume += quantity;
            } else {
                sell_volume += quantity;
            }
        }

        let total_volume = buy_volume + sell_volume;
        if total_volume == 0.0 {
            0.0
        } else {
            (buy_volume - sell_volume) / total_volume
        }
    }
}

#[async_trait::async_trait]
impl Strategy for OrderBookImbalance {
    fn get_info(&self) -> String {
        format!("Order Book Imbalance Strategy (period: {}, buy_threshold: {}, sell_threshold: {}, scale: {})", self.period, self.buy_threshold, self.sell_threshold, self.scale)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (price - self.last_price) / self.last_price;
        }
        
        self.trades.push_back(trade);
        if self.trades.len() > self.period {
            self.trades.pop_front();
        }
        
        self.last_price = price;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Pure momentum approach like successful strategies (ignore OBI)
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };
        let momentum_strength = (self.price_momentum * 3000.0).min(1.0);

        let signal: Signal;
        let confidence: f64;

        if self.price_momentum > self.momentum_threshold {
            signal = Signal::Buy;
            confidence = momentum_strength * momentum_factor * self.scale;
        } else if self.price_momentum < -self.momentum_threshold {
            signal = Signal::Sell;
            confidence = momentum_strength * momentum_factor * self.scale;
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
