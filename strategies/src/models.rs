// strategies/src/models.rs

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Kline {
    pub open_time: u64,
    pub close_time: u64,
    pub symbol: String,
    pub interval: String,
    pub open_price: String,
    pub close_price: String,
    pub high_price: String,
    pub low_price: String,
    pub volume: String,
}

#[derive(Debug, Deserialize)]
pub struct TradeData {
    pub event_type: String,
    pub event_time: u64,
    pub symbol: String,
    pub trade_id: u64,
    pub price: String,
    pub quantity: String,
    pub buyer_order_id: u64,
    pub seller_order_id: u64,
    pub trade_time: u64,
    pub is_buyer_market_maker: bool,
}