use trade::models::{Kline, TradeData};
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use log::{debug, info};

pub struct MeanReversionStrategy {
    pub window_size: usize,
    pub prices: Vec<f64>,
    pub last_sma: Option<f64>,
    pub recent_trades: Vec<f64>,
    pub max_trade_window: usize,
}

#[async_trait::async_trait]
impl Strategy for MeanReversionStrategy {
    async fn on_kline(&mut self, kline: Kline) {
        let close: f64 = kline.close;

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

        debug!("🤝 HOLD @ {:.5} (SMA {:.5})", close, sma);
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price: f64 = trade.price;

        self.recent_trades.push(price);
        if self.recent_trades.len() > self.max_trade_window {
            self.recent_trades.remove(0);
        }

        let Some(sma) = self.last_sma else {
            return;
        };

        let avg_trade_price =
            self.recent_trades.iter().copied().sum::<f64>() / self.recent_trades.len() as f64;

        info!(
            "Recent trade avg price {:.5} near SMA {:.5}",
            avg_trade_price, sma
        );
    }

    /// Determines trading signal based on price, SMA, and current position.
    fn get_signal(&self, close: f64, _ts: f64, current_position: Position) -> Signal {
        let Some(sma) = self.last_sma else {
            return Signal::Hold;
        };

        let threshold = 0.004;
        let deviation = (close - sma) / sma;

        if current_position.quantity == 0.0 && deviation < -threshold {
            Signal::Buy
        } else if current_position.quantity > 0.0 && deviation > threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}