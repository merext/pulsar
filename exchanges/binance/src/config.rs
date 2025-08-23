use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct BinanceTraderConfig {
    pub general: GeneralConfig,
    pub execution: ExecutionConfig,
    pub websocket: WebSocketConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Deserialize)]
pub struct GeneralConfig {
    pub trader_name: String,
    pub exchange: String,
}



#[derive(Debug, Deserialize)]
pub struct ExecutionConfig {
    pub order_timeout: u64,
    pub max_retries: usize,
    pub retry_delay: u64,
    pub use_market_orders: bool,
    pub slippage_tolerance: f64,
}

#[derive(Debug, Deserialize)]
pub struct WebSocketConfig {
    pub connection_timeout: u64,
    pub heartbeat_interval: u64,
    pub max_reconnections: usize,
    pub reconnection_delay: u64,
}

#[derive(Debug, Deserialize)]
pub struct LoggingConfig {
    pub log_level: String,
    pub log_all_trades: bool,
    pub log_order_details: bool,
    pub log_balance_updates: bool,
}

impl BinanceTraderConfig {
    pub fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_content = std::fs::read_to_string(config_path)?;
        let config: BinanceTraderConfig = toml::from_str(&config_content)?;
        Ok(config)
    }
}
