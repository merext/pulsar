//! # Relative Strength Index (RSI) Strategy
//! 
//! This strategy utilizes the Relative Strength Index (RSI), a momentum oscillator that measures the speed and change of price movements.
//! RSI oscillates between zero and 100 and is typically used to identify overbought or oversold conditions in an asset.
//! 
//! The strategy generates a sell signal when the RSI crosses above a defined overbought threshold (e.g., 70),
//! indicating that the asset may be overvalued and due for a price correction.
//! Conversely, a buy signal is generated when the RSI falls below a defined oversold threshold (e.g., 30),
//! suggesting the asset may be undervalued and due for a price rebound.

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct RsiStrategy {
    period: usize,
    overbought: f64,
    oversold: f64,
    prices: VecDeque<f64>,
}

impl RsiStrategy {
    pub fn new(period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            period,
            overbought,
            oversold,
            prices: VecDeque::new(),
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
    

    async fn on_trade(&mut self, trade: TradeData) {
        let trade_price = trade.price;
        self.prices.push_back(trade_price);
        if self.prices.len() > self.period + 1 {
            self.prices.pop_front();
        }
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        let rsi = self.calculate_rsi();

        let signal: Signal;
        let confidence: f64;

        if rsi > self.overbought {
            signal = Signal::Sell;
            // Confidence increases as RSI goes further above overbought
            confidence = ((rsi - self.overbought) / (100.0 - self.overbought)).min(1.0);
        } else if rsi < self.oversold {
            signal = Signal::Buy;
            // Confidence increases as RSI goes further below oversold
            confidence = ((self.oversold - rsi) / self.oversold).min(1.0);
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }
        (signal, confidence)
    }
}