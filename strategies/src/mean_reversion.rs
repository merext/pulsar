use crate::models::{Kline, TradeData};
use crate::strategy::Strategy;
use log::info;

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

        let threshold = 0.0002;
        let deviation = (close - sma) / sma;

        if deviation < -threshold {
            info!(
                "📈 BUY signal - price {:.2} below SMA {:.2} (deviation {:.5}, threshold {:.5})",
                close, sma, deviation, -threshold
            );
        } else if deviation > threshold {
            info!(
                "📉 SELL signal - price {:.2} above SMA {:.2} (deviation {:.5}, threshold {:.5})",
                close, sma, deviation, threshold
            );
        } else {
            info!(
                "🤝 HOLD - price {:.2} near SMA {:.2} (deviation {:.5}, threshold ±{:.5})",
                close, sma, deviation, threshold
            );
        }
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
        let deviation = (avg_trade_price - sma) / sma;

        let confirmation_threshold = 0.0025;

        if deviation < -confirmation_threshold {
            info!(
                "✅ CONFIRMED BUY - trade avg {:.2} well below SMA {:.2}",
                avg_trade_price, sma
            );
        } else if deviation > confirmation_threshold {
            info!(
                "✅ CONFIRMED SELL - trade avg {:.2} well above SMA {:.2}",
                avg_trade_price, sma
            );
        }
    }
}
