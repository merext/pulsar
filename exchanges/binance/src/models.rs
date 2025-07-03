use serde::Deserialize;
use trade::models::{Kline as TradeKline, TradeData as TradeTradeData};

#[derive(Debug, Deserialize)]
pub struct KlineMessage {
    pub stream: String,
    pub data: KlineData,
}

#[derive(Debug, Deserialize)]
pub struct KlineData {
    #[serde(rename = "e")]
    pub event_type: String,

    #[serde(rename = "E")]
    pub event_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "k")]
    pub kline: Kline,
}


#[derive(Debug, Deserialize, Clone)]
pub struct Kline {
    #[serde(rename = "t")]
    pub start_time: u64, // Kline start time

    #[serde(rename = "T")]
    pub close_time: u64, // Kline close time

    #[serde(rename = "s")]
    pub symbol: String, // Symbol

    #[serde(rename = "i")]
    pub interval: String, // Interval

    #[serde(rename = "o")]
    pub open_price: String, // Open price

    #[serde(rename = "c")]
    pub close_price: String, // Close price

    #[serde(rename = "h")]
    pub high_price: String, // High price

    #[serde(rename = "l")]
    pub low_price: String, // Low price

    #[serde(rename = "v")]
    pub base_volume: String, // Base asset volume

    #[serde(rename = "n")]
    pub num_trades: u64, // Number of trades

    #[serde(rename = "x")]
    pub is_closed: bool, // Is this kline closed?

    #[serde(rename = "q")]
    pub quote_volume: String, // Quote asset volume

    #[serde(rename = "V")]
    pub taker_buy_base_volume: String, // Taker buy base asset volume

    #[serde(rename = "Q")]
    pub taker_buy_quote_volume: String, // Taker buy quote asset volume

    #[serde(rename = "B")]
    pub ignore: String, // Ignore
}

impl From<Kline> for TradeKline {
    fn from(k: Kline) -> Self {
        TradeKline {
            open_time: k.start_time,
            open: k.open_price.parse().unwrap_or_default(),
            high: k.high_price.parse().unwrap_or_default(),
            low: k.low_price.parse().unwrap_or_default(),
            close: k.close_price.parse().unwrap_or_default(),
            volume: k.base_volume.parse().unwrap_or_default(),
            close_time: k.close_time,
            quote_asset_volume: k.quote_volume.parse().unwrap_or_default(),
            number_of_trades: k.num_trades,
            taker_buy_base_asset_volume: k.taker_buy_base_volume.parse().unwrap_or_default(),
            taker_buy_quote_asset_volume: k.taker_buy_quote_volume.parse().unwrap_or_default(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TradeMessage {
    pub stream: String,
    pub data: TradeData,
}

#[derive(Debug, Deserialize)]
pub struct TradeData {
    #[serde(rename = "e")]
    pub event_type: String,

    #[serde(rename = "E")]
    pub event_time: u64,

    #[serde(rename = "s")]
    pub symbol: String,

    #[serde(rename = "t")]
    pub trade_id: u64,

    #[serde(rename = "p")]
    pub price: String,

    #[serde(rename = "q")]
    pub quantity: String,

    #[serde(rename = "b")]
    pub buyer_order_id: Option<u64>,

    #[serde(rename = "a")]
    pub seller_order_id: Option<u64>,

    #[serde(rename = "T")]
    pub trade_time: u64,

    #[serde(rename = "m")]
    pub is_buyer_market_maker: bool,

    #[serde(rename = "M")]
    pub ignore: bool,
}

impl From<TradeData> for TradeTradeData {
    fn from(t: TradeData) -> Self {
        TradeTradeData {
            id: t.trade_id,
            price: t.price.parse().unwrap_or_default(),
            qty: t.quantity.parse().unwrap_or_default(),
            quote_qty: (t.price.parse::<f64>().unwrap_or_default() * t.quantity.parse::<f64>().unwrap_or_default()),
            time: t.trade_time,
            is_buyer_maker: t.is_buyer_market_maker,
            is_best_match: false, // This field is not available in Binance TradeData, setting to false
        }
    }
}
