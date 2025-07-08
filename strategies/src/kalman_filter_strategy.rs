//! # Kalman Filter Strategy
//! 
//! This strategy utilizes a Kalman Filter to estimate the true price and its velocity from noisy trade data.
//! It aims to reduce lag and noise in price tracking, providing a smoother signal for trading decisions.
//! 
//! The strategy generates buy or sell signals based on the deviation of the current price from the Kalman Filter's estimated price.
//! A positive deviation exceeding a defined threshold suggests a potential sell opportunity (price is overvalued relative to its estimated true value),
//! while a negative deviation below a threshold suggests a potential buy opportunity (price is undervalued).

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use async_trait::async_trait;
use nalgebra::{Matrix2, Vector2};

#[derive(Clone)]
pub struct KalmanFilterStrategy {
    // State vector: [price, velocity]
    state_estimate: Vector2<f64>,
    // Covariance matrix
    covariance: Matrix2<f64>,
    // State transition matrix
    transition_matrix: Matrix2<f64>,
    // Observation matrix
    observation_matrix: Matrix2<f64>,
    // Process noise covariance
    process_noise_covariance: Matrix2<f64>,
    // Measurement noise covariance
    measurement_noise_covariance: Matrix2<f64>,
    // Threshold for generating signals
    signal_threshold: f64,
}

impl KalmanFilterStrategy {
    pub fn new(signal_threshold: f64) -> Self {
        let dt = 1.0; // Time step (e.g., 1 for each new kline/trade)

        // Initial state (price, velocity)
        let state_estimate = Vector2::new(0.0, 0.0);
        // Initial covariance (high uncertainty)
        let covariance = Matrix2::new(
            1000.0, 0.0,
            0.0, 1000.0,
        );
        // State transition matrix (predicts next state from current)
        // x_k = x_{k-1} + v_{k-1}*dt
        // v_k = v_{k-1}
        let transition_matrix = Matrix2::new(
            1.0, dt,
            0.0, 1.0,
        );
        // Observation matrix (relates state to measurement)
        // We only observe price
        let observation_matrix = Matrix2::new(
            1.0, 0.0,
            0.0, 0.0,
        );
        // Process noise covariance (uncertainty in our model)
        let process_noise_covariance = Matrix2::new(
            0.01, 0.0,
            0.0, 0.01,
        );
        // Measurement noise covariance (uncertainty in our measurements)
        let measurement_noise_covariance = Matrix2::new(
            0.1, 0.0,
            0.0, 0.1,
        );

        Self {
            state_estimate,
            covariance,
            transition_matrix,
            observation_matrix,
            process_noise_covariance,
            measurement_noise_covariance,
            signal_threshold,
        }
    }

    // Predict step
    fn predict(&mut self) {
        self.state_estimate = self.transition_matrix * self.state_estimate;
        self.covariance = self.transition_matrix * self.covariance * self.transition_matrix.transpose() + self.process_noise_covariance;
    }

    // Update step
    fn update(&mut self, measurement: f64) {
        let measurement_vector = Vector2::new(measurement, 0.0);
        let innovation = measurement_vector - self.observation_matrix * self.state_estimate;
        let innovation_covariance = self.observation_matrix * self.covariance * self.observation_matrix.transpose() + self.measurement_noise_covariance;
        let kalman_gain = self.covariance * self.observation_matrix.transpose() * innovation_covariance.try_inverse().unwrap_or_else(Matrix2::zeros);

        self.state_estimate = self.state_estimate + kalman_gain * innovation;
        self.covariance = (Matrix2::identity() - kalman_gain * self.observation_matrix) * self.covariance;
    }
}

#[async_trait]
impl Strategy for KalmanFilterStrategy {
    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        self.predict();
        self.update(price);
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        let estimated_price = self.state_estimate[0];
        let deviation = current_price - estimated_price;

        let signal: Signal;
        let confidence: f64;

        if deviation > self.signal_threshold {
            signal = Signal::Buy; // Current price is significantly above estimated price
            confidence = (deviation / (self.signal_threshold * 2.0)).min(1.0); // Example confidence calculation
        } else if deviation < -self.signal_threshold {
            signal = Signal::Sell; // Current price is significantly below estimated price
            confidence = (deviation.abs() / (self.signal_threshold * 2.0)).min(1.0); // Example confidence calculation
        } else {
            signal = Signal::Hold;
            confidence = 0.0;
        }
        (signal, confidence)
    }
}
