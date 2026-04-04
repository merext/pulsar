use crate::execution::{ExecutionReport, OrderIntent};
use crate::market::MarketEvent;
use crate::metrics::{PerformanceMetrics, TradeManager};
use crate::strategy::Strategy;
use futures_util::Stream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    Maker,
    Taker,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeMode {
    Real,
    Emulated,
    Backtest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketDataSourceKind {
    HistoricalTradeReplay,
    HistoricalCapturedReplay,
    WebsocketTradeBookTicker,
}

impl MarketDataSourceKind {
    pub fn status(self) -> &'static str {
        match self {
            Self::HistoricalTradeReplay => "historical_trade_replay",
            Self::HistoricalCapturedReplay => "historical_captured_replay",
            Self::WebsocketTradeBookTicker => "websocket_trade_bookticker",
        }
    }

    pub fn source(self) -> &'static str {
        match self {
            Self::HistoricalTradeReplay | Self::HistoricalCapturedReplay => {
                "binance_market_event_stream"
            }
            Self::WebsocketTradeBookTicker => "binance_market_event_stream",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExchangeInfo {
    pub name: String,
    pub trading_fee: f64,
}

#[async_trait::async_trait]
pub trait Trader {
    // Exchange information
    fn get_info(&self) -> &ExchangeInfo;

    // Centralized metrics access
    fn get_metrics(&self) -> &PerformanceMetrics;
    fn get_trade_manager(&self) -> &TradeManager;

    // Account and trading operations
    // Returns the USDT free balance if available (live mode), None otherwise.
    async fn account_status(&self) -> Result<Option<f64>, anyhow::Error>;
    async fn on_order_intent(
        &mut self,
        symbol: &str,
        reference_price: f64,
        intent: OrderIntent,
    ) -> ExecutionReport;

    // Exchange calculates exact trade size based on symbol, price, confidence, min/max trade sizes, and step size
    fn calculate_trade_size(
        &self,
        symbol: &str,
        price: f64,
        confidence: f64,
        trading_size_min: f64,
        trading_size_max: f64,
        trading_size_step: f64,
    ) -> f64;

    // Universal trading loop that handles all trading modes
    async fn trade(
        &mut self,
        trading_stream: impl Stream<Item = MarketEvent> + Send,
        trading_strategy: &mut dyn Strategy,
        trading_symbol: &str,
        trading_mode: TradeMode,
        market_data_source: MarketDataSourceKind,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
