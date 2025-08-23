use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct BinanceTraderConfig {
    pub general: GeneralConfig,
    pub trading_behavior: TradingBehaviorConfig,
    pub websocket: WebSocketConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GeneralConfig {
    pub name: String,
    pub trading_fee: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TradingBehaviorConfig {
    pub volatility_high_threshold: f64,
    pub volatility_medium_threshold: f64,
    pub volatility_low_threshold: f64,
    pub volatility_high_factor: f64,
    pub volatility_medium_factor: f64,
    pub volatility_low_factor: f64,
    pub confidence_high_threshold: f64,
    pub confidence_low_threshold: f64,
    pub confidence_high_factor: f64,
    pub confidence_low_factor: f64,
    pub kelly_factor: f64,
    pub default_volatility: f64,
    pub step_size_fallback: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WebSocketConfig {
    pub connection_timeout: u64,
}

impl BinanceTraderConfig {
    pub fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: BinanceTraderConfig = toml::from_str(&config_content)?;
        Ok(config)
    }
}
