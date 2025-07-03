use trade::models::{Kline, TradeData};
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use log::{debug, info};
use splines::{Interpolation, Key, Spline};

pub struct SplineStrategy {
    pub window_size: usize,
    pub prices: Vec<f64>,
    pub timestamps: Vec<f64>,
    pub last_spline: Option<Spline<f64, f64>>,
    pub interpolation: Interpolation<f64, f64>,
}

impl SplineStrategy {
    pub fn new(window_size: usize, interpolation: Interpolation<f64, f64>) -> Self {
        Self {
            window_size,
            prices: Vec::with_capacity(window_size),
            timestamps: Vec::with_capacity(window_size),
            last_spline: None,
            interpolation,
        }
    }

    // Build spline only if we have enough data
    pub fn update_spline(&mut self) {
        if self.prices.len() == self.window_size {
            let points: Vec<_> = self
                .timestamps
                .iter()
                .zip(self.prices.iter())
                .map(|(&t, &p)| Key::new(t, p, self.interpolation))
                .collect();
            self.last_spline = Some(Spline::from_vec(points));
            debug!("Spline updated with {} points.", self.window_size);
        } else {
            self.last_spline = None;
            info!(
                "Waiting for enough data to build spline: need {}, have {}",
                self.window_size,
                self.prices.len()
            );
        }
    }
}

#[async_trait::async_trait]
impl Strategy for SplineStrategy {
    async fn on_kline(&mut self, kline: Kline) {
        let close: f64 = kline.close;
        let timestamp = kline.close_time as f64;

        if self.prices.len() == self.window_size {
            // Remove oldest
            self.prices.remove(0);
            self.timestamps.remove(0);
        }

        self.prices.push(close);
        self.timestamps.push(timestamp);

        self.update_spline();
    }

    async fn on_trade(&mut self, _trade: TradeData) {
        // Not used
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64, // Not used directly, we use last_ts
        _current_position: Position,
    ) -> Signal {
        debug!("Entering get_signal");

        if self.last_spline.is_none() {
            debug!("No spline available, returning Hold");
            return Signal::Hold;
        }

        let price_difference_threshold = 0.000000000000001; // Extremely small threshold for HFT
        let epsilon = 60_000_000_000.0; // 1 minute in nanoseconds

        let spline = self.last_spline.as_ref().unwrap(); // We already checked for None above

        let last_ts = *self.timestamps.last().unwrap();

        let price_difference = {
            let p_now_ts = last_ts;
            let p_prev_ts = last_ts - epsilon;

            // Ensure p_prev_ts is not before the first timestamp in the spline
            if p_prev_ts < *self.timestamps.first().unwrap() {
                debug!("p_prev_ts ({:.0}) is before the first spline timestamp ({:.0}). Returning Hold.", p_prev_ts, *self.timestamps.first().unwrap());
                return Signal::Hold;
            }

            debug!("Sampling spline at p_now_ts: {:.0}", p_now_ts);
            let p_now = match spline.sample(p_now_ts) {
                Some(p) => p,
                None => {
                    debug!("Spline sample at p_now_ts ({:.0}) returned None. Returning Hold.", p_now_ts);
                    return Signal::Hold;
                }
            };

            debug!("Sampling spline at p_prev_ts: {:.0}", p_prev_ts);
            let p_prev = match spline.sample(p_prev_ts) {
                Some(p) => p,
                None => {
                    debug!("Spline sample at p_prev_ts ({:.0}) returned None. Returning Hold.", p_prev_ts);
                    return Signal::Hold;
                }
            };
            p_now - p_prev
        };

        info!("Price difference calc => ts: {:.0}, difference: {:.18}", last_ts, price_difference); // Increased precision

        if price_difference > price_difference_threshold {
            debug!(
                "Signal: Buy (difference {:.18} > threshold {:.18})", // Increased precision
                price_difference, price_difference_threshold
            );
            Signal::Buy
        } else if price_difference < -price_difference_threshold {
            debug!(
                "Signal: Sell (difference {:.18} < -threshold {:.18})", // Increased precision
                price_difference, price_difference_threshold
            );
            Signal::Sell
        } else {
            debug!(
                "Signal: Hold (difference {:.18} within ±{:.18})", // Increased precision
                price_difference, price_difference_threshold
            );
            Signal::Hold
        }
    }
}
