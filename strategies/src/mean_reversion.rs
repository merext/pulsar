use crate::models::{Kline, TradeData};
use crate::position::Position;
use crate::strategy::Strategy;
use log::info;

pub struct MeanReversionStrategy {
    pub window_size: usize,
    pub prices: Vec<f64>,
    pub last_sma: Option<f64>,
    pub recent_trades: Vec<f64>,
    pub max_trade_window: usize,
    pub position: Position,
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

        match self.position {
            Position::Flat => {
                if deviation < -threshold {
                    self.position = Position::Long;
                    info!(
                        "📥 ENTER LONG @ {:.5} (SMA {:.5}, deviation {:.5})",
                        close, sma, deviation
                    );
                } else {
                    info!(
                        "🤝 HOLD (Flat) @ {:.5} (SMA {:.5}, deviation {:.5})",
                        close, sma, deviation
                    );
                }
            }
            Position::Long => {
                if deviation > threshold {
                    self.position = Position::Flat;
                    info!(
                        "📤 EXIT LONG @ {:.5} (SMA {:.5}, deviation {:.5})",
                        close, sma, deviation
                    );
                } else {
                    info!(
                        "📈 STILL LONG @ {:.5} (SMA {:.5}, deviation {:.5})",
                        close, sma, deviation
                    );
                }
            }
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
