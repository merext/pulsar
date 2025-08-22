pub mod config;
pub mod metrics;
pub mod executor;
pub mod market;
pub mod trader;
pub mod signal;
pub mod models;
pub mod logger;

// Re-export commonly used types
pub use models::{Trade, TradeData};
pub use trader::{Position, Trader, TradeMode, OrderType};
pub use config::TradingConfig;
pub use metrics::{PerformanceMetrics, TradeRecord};
pub use signal::Signal;

