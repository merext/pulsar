pub mod executor;
pub mod market;
pub mod trader;
pub mod trading_engine;
pub mod signal;
pub mod models;

// Re-export commonly used types
pub use models::Trade;
pub use trader::{Position, Trader, TradeMode};
pub use trading_engine::{TradingEngine, PerformanceMetrics, TradingConfig};
pub use signal::Signal;
