use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct BinanceTraderConfig {
    pub websocket: WebSocketConfig,
}

#[derive(Debug, Deserialize)]
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
