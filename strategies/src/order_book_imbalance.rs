//! # Order Book Imbalance Strategy
//! 
//! This strategy analyzes the imbalance between buy and sell volumes within recent trades to predict short-term price movements.
//! It operates on the principle that a significant imbalance can indicate immediate buying or selling pressure.
//! 
//! The strategy calculates an Order Book Imbalance (OBI) metric based on the volume of buyer-initiated versus seller-initiated trades within a defined period.
//! A buy signal is generated if the OBI exceeds a positive threshold, indicating strong buying pressure.
//! Conversely, a sell signal is generated if the OBI falls below a negative threshold, indicating strong selling pressure.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct OrderBookImbalance {
    period: usize,
    buy_threshold: f64,
    sell_threshold: f64,
    trades: VecDeque<TradeData>,
}

impl OrderBookImbalance {
    pub fn new(period: usize, buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            period,
            buy_threshold,
            sell_threshold,
            trades: VecDeque::new(),
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
    

    async fn on_trade(&mut self, trade: TradeData) {
        self.trades.push_back(trade);
        if self.trades.len() > self.period {
            self.trades.pop_front();
        }
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        let obi = self.calculate_obi();

        let signal: Signal;
        let confidence: f64;

        if obi > self.buy_threshold {
            signal = Signal::Buy;
            // Confidence increases as OBI goes further above buy_threshold
            confidence = ((obi - self.buy_threshold) / (1.0 - self.buy_threshold)).min(1.0);
        } else if obi < self.sell_threshold {
            signal = Signal::Sell;
            // Confidence increases as OBI goes further below sell_threshold
            confidence = ((self.sell_threshold - obi) / (self.sell_threshold - (-1.0))).min(1.0);
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }
        (signal, confidence)
    }
}