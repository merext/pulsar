pub mod trading_engine;
pub use trading_engine::TradingConfig;
pub mod executor;
pub mod market;
pub mod trader;
// (trading_engine already declared above)
pub mod signal;
pub mod models;


// Re-export commonly used types
pub use models::Trade;
pub use trader::{Position, Trader, TradeMode};
pub use trading_engine::{TradingEngine, PerformanceMetrics};
pub use signal::Signal;

