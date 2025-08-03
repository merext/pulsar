use std::fs;
use std::path::Path;
use toml::Value;

/// Configuration loader for trading strategies
pub struct StrategyConfig {
    pub config: Value,
}

impl StrategyConfig {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let config = content.parse::<Value>()?;
        Ok(StrategyConfig { config })
    }

    /// Load strategy-specific configuration from the config directory
    pub fn load_strategy_config(strategy_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = format!("../../config/{}.toml", strategy_name);
        Self::from_file(config_path)
    }

    /// Get a value from the configuration
    pub fn get<T: ConfigValue>(&self, key: &str) -> Option<T> {
        T::from_config_value(&self.config, key)
    }

    /// Get a value with a default fallback
    pub fn get_or<T: ConfigValue>(&self, key: &str, default: T) -> T {
        self.get(key).unwrap_or(default)
    }

    /// Get a nested section
    pub fn section(&self, section: &str) -> Option<StrategyConfig> {
        self.config.get(section).map(|value| StrategyConfig {
            config: value.clone(),
        })
    }
}

/// Trait for converting TOML values to specific types
pub trait ConfigValue: Sized {
    fn from_config_value(config: &Value, key: &str) -> Option<Self>;
}

impl ConfigValue for f64 {
    fn from_config_value(config: &Value, key: &str) -> Option<Self> {
        config.get(key)?.as_float()
    }
}

impl ConfigValue for usize {
    fn from_config_value(config: &Value, key: &str) -> Option<Self> {
        config.get(key)?.as_integer().map(|v| v as usize)
    }
}

impl ConfigValue for bool {
    fn from_config_value(config: &Value, key: &str) -> Option<Self> {
        config.get(key)?.as_bool()
    }
}

impl ConfigValue for String {
    fn from_config_value(config: &Value, key: &str) -> Option<Self> {
        config.get(key)?.as_str().map(|s| s.to_string())
    }
}

/// Default configuration values for strategies
pub struct DefaultConfig;

impl DefaultConfig {
    // RSI Strategy defaults
    pub fn rsi_period() -> usize { 14 }
    pub fn rsi_overbought() -> f64 { 70.0 }
    pub fn rsi_oversold() -> f64 { 30.0 }
    pub fn rsi_scale() -> f64 { 1.0 }
    pub fn rsi_signal_threshold() -> f64 { 0.6 }

    // Mean Reversion Strategy defaults
    pub fn mean_reversion_window_size() -> usize { 20 }
    pub fn mean_reversion_max_trade_window() -> usize { 10 }
    pub fn mean_reversion_scale() -> f64 { 1.0 }
    pub fn mean_reversion_signal_threshold() -> f64 { 0.6 }

    // Momentum Scalping Strategy defaults
    pub fn momentum_trade_window_size() -> usize { 5 }
    pub fn momentum_price_change_threshold() -> f64 { 0.00001 }
    pub fn momentum_scale() -> f64 { 1.0 }
    pub fn momentum_signal_threshold() -> f64 { 0.6 }

    // Kalman Filter Strategy defaults
    pub fn kalman_signal_threshold() -> f64 { 0.00001 }
    pub fn kalman_scale() -> f64 { 1.0 }

    // Order Book Imbalance Strategy defaults
    pub fn obi_period() -> usize { 10 }
    pub fn obi_buy_threshold() -> f64 { 0.00001 }
    pub fn obi_sell_threshold() -> f64 { -0.00001 }
    pub fn obi_scale() -> f64 { 1.0 }
    pub fn obi_signal_threshold() -> f64 { 0.6 }

    // Spline Strategy defaults
    pub fn spline_window_size() -> usize { 5 }
    pub fn spline_derivative_buy_threshold() -> f64 { 0.000001 }
    pub fn spline_derivative_sell_threshold() -> f64 { -0.000001 }

    // VWAP Deviation Strategy defaults
    pub fn vwap_period() -> usize { 10 }
    pub fn vwap_deviation_threshold() -> f64 { 0.00001 }
    pub fn vwap_signal_threshold() -> f64 { 0.6 }

    // Z-Score Strategy defaults
    pub fn zscore_period() -> usize { 50 }
    pub fn zscore_buy_threshold() -> f64 { -0.00001 }
    pub fn zscore_sell_threshold() -> f64 { 0.00001 }
    pub fn zscore_scale() -> f64 { 1.0 }
    pub fn zscore_signal_threshold() -> f64 { 0.6 }

    // Fractal Approximation Strategy defaults
    pub fn fractal_period() -> usize { 20 }
    pub fn fractal_signal_threshold() -> f64 { 0.6 }

    // HFT Ultra-Fast Strategy defaults
    pub fn hft_signal_threshold() -> f64 { 0.6 }

    // HFT Market Maker Strategy defaults
    pub fn hft_mm_signal_threshold() -> f64 { 0.6 }

    // Adaptive Multi-Factor Strategy defaults
    pub fn adaptive_short_window() -> usize { 10 }
    pub fn adaptive_long_window() -> usize { 50 }
    pub fn adaptive_volatility_window() -> usize { 20 }
    pub fn adaptive_volume_window() -> usize { 30 }
    pub fn adaptive_signal_threshold() -> f64 { 0.6 }

    // Neural Market Microstructure Strategy defaults
    pub fn neural_short_window() -> usize { 5 }
    pub fn neural_medium_window() -> usize { 20 }
    pub fn neural_long_window() -> usize { 100 }
    pub fn neural_micro_window() -> usize { 10 }
    pub fn neural_signal_threshold() -> f64 { 0.6 }
} 