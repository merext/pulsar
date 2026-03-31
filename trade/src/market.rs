use crate::models::Trade;
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct BookLevel {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct BookTicker {
    pub bid: BookLevel,
    pub ask: BookLevel,
    pub event_time: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DepthLevel {
    pub price: f64,
    pub quantity: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct DepthSnapshot {
    pub bids: Vec<DepthLevel>,
    pub asks: Vec<DepthLevel>,
    pub event_time: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MarketEvent {
    Trade(Trade),
    BookTicker(BookTicker),
    Depth(DepthSnapshot),
}

#[derive(Debug, Clone, Default)]
pub struct TradeWindowStats {
    pub trade_count: usize,
    pub volume: f64,
    pub notional: f64,
    pub buyer_initiated_volume: f64,
    pub seller_initiated_volume: f64,
    pub price_change: f64,
    pub last_price: f64,
}

#[derive(Debug, Clone)]
pub struct MarketState {
    symbol: String,
    last_trade: Option<Trade>,
    top_of_book: Option<BookTicker>,
    depth: Option<DepthSnapshot>,
    trades: VecDeque<Trade>,
    trade_window_millis: u64,
}

impl MarketState {
    pub fn new(symbol: impl Into<String>, trade_window_millis: u64) -> Self {
        Self {
            symbol: symbol.into(),
            last_trade: None,
            top_of_book: None,
            depth: None,
            trades: VecDeque::new(),
            trade_window_millis,
        }
    }

    pub fn apply(&mut self, event: &MarketEvent) {
        match event {
            MarketEvent::Trade(trade) => {
                self.last_trade = Some(trade.clone());
                self.trades.push_back(trade.clone());
                self.trim_trade_window(trade.trade_time);
            }
            MarketEvent::BookTicker(book_ticker) => {
                self.top_of_book = Some(*book_ticker);
            }
            MarketEvent::Depth(depth) => {
                self.depth = Some(depth.clone());
            }
        }
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn last_trade(&self) -> Option<&Trade> {
        self.last_trade.as_ref()
    }

    pub fn top_of_book(&self) -> Option<&BookTicker> {
        self.top_of_book.as_ref()
    }

    pub fn depth(&self) -> Option<&DepthSnapshot> {
        self.depth.as_ref()
    }

    pub fn last_price(&self) -> Option<f64> {
        self.last_trade.as_ref().map(|trade| trade.price)
    }

    pub fn last_event_time_millis(&self) -> Option<u64> {
        [
            self.last_trade.as_ref().map(|trade| trade.trade_time),
            self.top_of_book.as_ref().map(|book| book.event_time),
            self.depth.as_ref().map(|depth| depth.event_time),
        ]
        .into_iter()
        .flatten()
        .max()
    }

    pub fn last_event_time_secs(&self) -> Option<f64> {
        self.last_event_time_millis()
            .map(|time| time as f64 / 1000.0)
    }

    pub fn mid_price(&self) -> Option<f64> {
        self.top_of_book
            .as_ref()
            .map(|book| (book.bid.price + book.ask.price) / 2.0)
    }

    pub fn spread_bps(&self) -> Option<f64> {
        let book = self.top_of_book.as_ref()?;
        let mid = (book.bid.price + book.ask.price) / 2.0;
        if mid <= f64::EPSILON {
            return None;
        }
        Some((book.ask.price - book.bid.price) / mid * 10_000.0)
    }

    pub fn microprice(&self) -> Option<f64> {
        let book = self.top_of_book.as_ref()?;
        let total_qty = book.bid.quantity + book.ask.quantity;
        if total_qty <= f64::EPSILON {
            return None;
        }

        Some((book.ask.price * book.bid.quantity + book.bid.price * book.ask.quantity) / total_qty)
    }

    pub fn order_book_imbalance(&self) -> Option<f64> {
        let book = self.top_of_book.as_ref()?;
        let total_qty = book.bid.quantity + book.ask.quantity;
        if total_qty <= f64::EPSILON {
            return None;
        }

        Some((book.bid.quantity - book.ask.quantity) / total_qty)
    }

    pub fn trade_window_stats(&self) -> TradeWindowStats {
        Self::build_trade_window_stats(self.trades.iter())
    }

    pub fn recent_trade_window_stats(&self, trade_count: usize) -> TradeWindowStats {
        if trade_count == 0 {
            return TradeWindowStats::default();
        }

        Self::build_trade_window_stats(self.trades.iter().rev().take(trade_count).rev())
    }

    fn build_trade_window_stats<'a>(
        trades: impl IntoIterator<Item = &'a Trade>,
    ) -> TradeWindowStats {
        let mut stats = TradeWindowStats::default();
        let mut first_price = None;

        for trade in trades {
            stats.trade_count += 1;
            stats.volume += trade.quantity;
            stats.notional += trade.quantity * trade.price;
            if trade.is_buyer_market_maker {
                stats.seller_initiated_volume += trade.quantity;
            } else {
                stats.buyer_initiated_volume += trade.quantity;
            }

            if first_price.is_none() {
                first_price = Some(trade.price);
            }
            stats.last_price = trade.price;
        }

        if let Some(first_price) = first_price {
            stats.price_change = stats.last_price - first_price;
        }

        stats
    }

    pub fn trade_window_vwap(&self) -> Option<f64> {
        Self::vwap_from_stats(&self.trade_window_stats())
    }

    pub fn recent_trade_window_vwap(&self, trade_count: usize) -> Option<f64> {
        Self::vwap_from_stats(&self.recent_trade_window_stats(trade_count))
    }

    pub fn recent_trades(
        &self,
    ) -> impl DoubleEndedIterator<Item = &Trade> + ExactSizeIterator + '_ {
        self.trades.iter()
    }

    pub fn trade_window_low_price(&self) -> Option<f64> {
        self.trades
            .iter()
            .map(|trade| trade.price)
            .min_by(f64::total_cmp)
    }

    pub fn trade_window_high_price(&self) -> Option<f64> {
        self.trades
            .iter()
            .map(|trade| trade.price)
            .max_by(f64::total_cmp)
    }

    pub fn trade_flow_imbalance(&self) -> f64 {
        Self::trade_flow_imbalance_from_stats(&self.trade_window_stats())
    }

    pub fn recent_trade_flow_imbalance(&self, trade_count: usize) -> f64 {
        Self::trade_flow_imbalance_from_stats(&self.recent_trade_window_stats(trade_count))
    }

    fn trim_trade_window(&mut self, current_time: u64) {
        while let Some(front) = self.trades.front() {
            if current_time.saturating_sub(front.trade_time) > self.trade_window_millis {
                self.trades.pop_front();
            } else {
                break;
            }
        }
    }

    fn trade_flow_imbalance_from_stats(stats: &TradeWindowStats) -> f64 {
        let total = stats.buyer_initiated_volume + stats.seller_initiated_volume;
        if total <= f64::EPSILON {
            0.0
        } else {
            (stats.buyer_initiated_volume - stats.seller_initiated_volume) / total
        }
    }

    fn vwap_from_stats(stats: &TradeWindowStats) -> Option<f64> {
        if stats.volume <= f64::EPSILON {
            None
        } else {
            Some(stats.notional / stats.volume)
        }
    }
}
