pub struct TradeData {
    pub id: u64,
    pub price: f64,
    pub qty: f64,
    pub quote_qty: f64,
    pub time: u64,
    pub is_buyer_maker: bool,
    pub is_best_match: bool,
}

pub enum MarketEvent {
    Trade(TradeData),
}

#[derive(Debug, Clone, Default)]
pub struct Trade {
    pub event_type: String,
    pub event_time: u64,
    pub symbol: String,
    pub trade_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub buyer_order_id: Option<u64>,
    pub seller_order_id: Option<u64>,
    pub trade_time: u64,
    pub is_buyer_market_maker: bool,
}

impl From<Trade> for TradeData {
    fn from(trade: Trade) -> Self {
        TradeData {
            id: trade.trade_id,
            price: trade.price,
            qty: trade.quantity,
            quote_qty: trade.price * trade.quantity,
            time: trade.trade_time,
            is_buyer_maker: trade.is_buyer_market_maker,
            is_best_match: false,
        }
    }
}