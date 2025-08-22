pub mod config;
pub mod logger;
pub mod metrics;
pub mod models;
pub mod signal;
pub mod trader;

pub use config::TradingConfig;
pub use logger::{StrategyLoggerAdapter, TradeLogger};
pub use metrics::{PerformanceMetrics, Position, PositionManager, TradeRecord};

pub use signal::Signal;
pub use trader::{OrderType, Position as TraderPosition, TradeMode, Trader};

