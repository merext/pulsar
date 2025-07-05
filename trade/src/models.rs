#[derive(Debug)]
pub struct Kline {
    pub open_time: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub close_time: u64,
    pub quote_asset_volume: f64,
    pub number_of_trades: u64,
    pub taker_buy_base_asset_volume: f64,
    pub taker_buy_quote_asset_volume: f64,
}

impl Kline {
    pub fn price(&self) -> f64 {
        self.close
    }

    pub fn ts(&self) -> f64 {
        self.close_time as f64
    }
}

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
    Kline(Kline),
    Trade(TradeData),
}

#[derive(Debug)]
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
