use std::fs;
use std::path::Path;
use toml::Value;

/// Configuration loader for trading strategies
#[derive(Clone)]
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
        let config_path = format!("config/{}.toml", strategy_name);
        Self::from_file(config_path)
    }

    /// Load trading configuration from the config directory
    pub fn load_trading_config() -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = "config/trading_config.toml";
        match Self::from_file(config_path) {
            Ok(config) => Ok(config),
            Err(e) => {
                eprintln!("Failed to load trading config from {}: {}", config_path, e);
                eprintln!("Current working directory: {:?}", std::env::current_dir()?);
                Err(e)
            }
        }
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