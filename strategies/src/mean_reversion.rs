use crate::models::{Kline, TradeData};
use crate::position::Position;
use crate::strategy::Strategy;
use crate::trader::Signal;
use log::{debug, info};

pub struct MeanReversionStrategy {
    pub window_size: usize,
    pub prices: Vec<f64>,
    pub last_sma: Option<f64>,
    pub recent_trades: Vec<f64>,
    pub max_trade_window: usize,
    // Removed: pub position: Position,
}

#[async_trait::async_trait]
impl Strategy for MeanReversionStrategy {
    async fn on_kline(&mut self, kline: Kline) {
        let close: f64 = match kline.close_price.parse() {
            Ok(v) => v,
            Err(_) => return,
        };

        self.prices.push(close);
        if self.prices.len() > self.window_size {
            self.prices.remove(0);
        }

        if self.prices.len() < self.window_size {
            info!("waiting... {}", self.window_size - self.prices.len());
            return;
        }

        let sma: f64 = self.prices.iter().sum::<f64>() / self.prices.len() as f64;
        self.last_sma = Some(sma);

        // current_position must be passed from outside (strategy does not track position)
        // Here we temporarily log HOLD, actual position is managed by VirtualTrader
        debug!("🤝 HOLD @ {:.5} (SMA {:.5})", close, sma);
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price: f64 = match trade.price.parse() {
            Ok(v) => v,
            Err(_) => return,
        };

        self.recent_trades.push(price);
        if self.recent_trades.len() > self.max_trade_window {
            self.recent_trades.remove(0);
        }

        let Some(sma) = self.last_sma else {
            return;
        };

        let avg_trade_price =
            self.recent_trades.iter().copied().sum::<f64>() / self.recent_trades.len() as f64;

        // Just log trade info here, signal to be handled outside
        info!(
            "Recent trade avg price {:.5} near SMA {:.5}",
            avg_trade_price, sma
        );
    }
}

impl MeanReversionStrategy {
    /// Determines trading signal based on price, SMA, and current position.
    pub fn get_signal(&self, close: f64, sma: f64, current_position: Position) -> Signal {
        let threshold = 0.004;
        let deviation = (close - sma) / sma;

        if current_position == Position::Flat && deviation < -threshold {
            Signal::Buy
        } else if current_position == Position::Long && deviation > threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}
