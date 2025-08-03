//! # Fractal Approximation Strategy
//!
//! This strategy is based on the concept of fractal geometry and the Hurst Exponent,
//! which is used to measure the long-term memory of a time series. The Hurst Exponent
//! can indicate whether a market is trending, mean-reverting, or moving randomly.
//!
//! - **H > 0.5**: Indicates a trending market (persistent behavior). The strategy will
//!   follow the trend.
//! - **H < 0.5**: Indicates a mean-reverting market (anti-persistent behavior). The
//!   strategy will trade against the recent price movement.
//! - **H = 0.5**: Indicates a random walk, where no particular strategy is favored.

use crate::strategy::Strategy;
use std::collections::VecDeque;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;

#[derive(Clone)]
pub struct FractalApproximationStrategy {
    period: usize,
    prices: VecDeque<f64>,
}

impl FractalApproximationStrategy {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            prices: VecDeque::with_capacity(period + 1),
        }
    }

    // Hurst Exponent calculation using R/S analysis
    fn calculate_hurst(&self) -> f64 {
        let n = self.prices.len();
        if n < 20 {
            // Need a reasonable amount of data
            return 0.5;
        }

        let mut log_rs_values = Vec::new();
        let mut log_n_values = Vec::new();

        // We will test a few sub-period lengths
        let min_lag = 10;
        let num_lags = 5;
        let step = (n - min_lag) / num_lags;

        if step == 0 {
            return 0.5; // Not enough data for multiple lags
        }

        for i in 0..num_lags {
            let lag = min_lag + i * step;
            if lag > n {
                continue;
            }

            let num_chunks = n / lag;
            let mut rs_values = Vec::new();

            for j in 0..num_chunks {
                let chunk_start = j * lag;
                let chunk_end = (j + 1) * lag;
                if chunk_end > self.prices.len() {
                    continue;
                }
                let chunk = self
                    .prices
                    .range(chunk_start..chunk_end)
                    .cloned()
                    .collect::<Vec<f64>>();
                let mean = chunk.iter().sum::<f64>() / lag as f64;
                let y: Vec<f64> = chunk.iter().map(|&p| p - mean).collect();
                let z: Vec<f64> = y
                    .iter()
                    .scan(0.0, |acc, &x| {
                        *acc += x;
                        Some(*acc)
                    })
                    .collect();

                if let (Some(max_z), Some(min_z)) = (
                    z.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
                    z.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                ) {
                    let r = max_z - min_z;
                    let s = (y.iter().map(|&val| val.powi(2)).sum::<f64>() / lag as f64).sqrt();
                    if s > 0.0 {
                        rs_values.push(r / s);
                    }
                }
            }

            if !rs_values.is_empty() {
                let avg_rs = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
                log_rs_values.push(avg_rs.ln());
                log_n_values.push((lag as f64).ln());
            }
        }

        if log_rs_values.len() < 2 {
            return 0.5; // Not enough data points to fit a line
        }

        // Linear regression to find the slope (Hurst Exponent)
        let n_points = log_n_values.len() as f64;
        let sum_xy = log_n_values
            .iter()
            .zip(log_rs_values.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x = log_n_values.iter().sum::<f64>();
        let sum_y = log_rs_values.iter().sum::<f64>();
        let sum_x2 = log_n_values.iter().map(|x| x.powi(2)).sum::<f64>();

        let numerator = n_points * sum_xy - sum_x * sum_y;
        let denominator = n_points * sum_x2 - sum_x.powi(2);

        if denominator == 0.0 {
            return 0.5;
        }

        numerator / denominator // This is the Hurst Exponent
    }
}

#[async_trait::async_trait]
impl Strategy for FractalApproximationStrategy {
    fn get_info(&self) -> String {
        format!("Fractal Approximation Strategy (period: {})", self.period)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.prices.push_back(trade.price);
        if self.prices.len() > self.period {
            self.prices.pop_front();
        }
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        let hurst = self.calculate_hurst();
        let n = self.prices.len();

        if n < 2 {
            return (Signal::Hold, 0.0);
        }

        let start_price = self.prices[0];
        let end_price = self.prices[n - 1];
        let trend = end_price - start_price;

        let signal: Signal;
        let confidence = (hurst - 0.5).abs() * 2.0; // Confidence scales with deviation from 0.5

        if hurst > 0.55 {
            // Trending
            if trend > 0.0 {
                signal = Signal::Buy;
            } else {
                signal = Signal::Sell;
            }
        } else if hurst < 0.45 {
            // Mean-reverting
            if trend > 0.0 {
                signal = Signal::Sell; // Expect a reversal
            } else {
                signal = Signal::Buy; // Expect a rebound
            }
        } else {
            // Random walk
            signal = Signal::Hold;
        }

        (signal, confidence.min(1.0))
    }
}
