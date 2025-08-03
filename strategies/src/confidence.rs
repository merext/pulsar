//! # Confidence Scaling
//!
//! This module provides standardized functions for scaling confidence values in trading strategies.
//! These functions ensure that confidence is always represented as a value between 0.0 and 1.0,
//! making it easier to compare and combine signals from different strategies.

/// Scales a value to a confidence score based on its distance from a threshold.
///
/// This function is useful for strategies where confidence increases as a value moves
/// further away from a threshold. The scaling factor allows you to control how quickly
/// the confidence score approaches 1.0.
///
/// # Arguments
///
/// * `value` - The value to be scaled (e.g., Z-score, RSI).
/// * `threshold` - The threshold that the value is expected to cross.
/// * `scale` - A scaling factor that controls the sensitivity of the confidence score.
///
/// # Returns
///
/// A confidence score between 0.0 and 1.0.
pub fn scale_from_threshold(value: f64, threshold: f64, scale: f64) -> f64 {
    if scale <= 0.0 {
        return 0.0;
    }
    ((value - threshold) / scale).max(0.0).min(1.0)
}

/// Scales a value to a confidence score based on its distance from a threshold, for inverse relationships.
///
/// This function is similar to `scale_from_threshold`, but it's designed for situations where
/// confidence increases as a value moves *below* a threshold.
///
/// # Arguments
///
/// * `value` - The value to be scaled (e.g., Z-score, RSI).
/// * `threshold` - The threshold that the value is expected to fall below.
/// * `scale` - A scaling factor that controls the sensitivity of the confidence score.
///
/// # Returns
///
/// A confidence score between 0.0 and 1.0.
pub fn scale_from_threshold_inverse(value: f64, threshold: f64, scale: f64) -> f64 {
    if scale <= 0.0 {
        return 0.0;
    }
    ((threshold - value) / scale).max(0.0).min(1.0)
}
