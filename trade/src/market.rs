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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EventMixDiagnostics {
    pub trade_events: usize,
    pub book_ticker_events: usize,
    pub depth_events: usize,
    pub stale_quote_events: usize,
    pub stale_depth_events: usize,
    pub trade_without_quote_events: usize,
}

/// Incrementally computed microstructure features, updated on each event.
/// All EMA fields use the convention: `ema = alpha * new_value + (1 - alpha) * ema`.
#[derive(Debug, Clone)]
pub struct MicrostructureState {
    /// EMA of mid prices (from BookTicker events)
    pub ema_mid_price: f64,
    /// EMA of quoted spread in basis points
    pub ema_spread_bps: f64,
    /// EMA of squared trade returns (for realized vol estimation)
    pub ema_squared_return: f64,
    /// Realized volatility in bps (sqrt of ema_squared_return scaled to per-second)
    pub realized_vol_bps: f64,
    /// Smoothed trade arrival rate (trades per second)
    pub trade_rate_per_second: f64,
    /// Multi-level depth imbalance [-1, 1], positive = bid-heavy
    pub depth_imbalance: f64,
    /// Previous trade price for return computation
    prev_trade_price: Option<f64>,
    /// Previous trade timestamp for rate computation
    prev_trade_time_millis: Option<u64>,
    /// Whether at least one BookTicker has been received (EMA initialized)
    mid_initialized: bool,
    /// Whether at least two trades have been received (vol initialized)
    vol_initialized: bool,
    /// EMA alpha for mid price
    alpha_mid: f64,
    /// EMA alpha for spread
    alpha_spread: f64,
    /// EMA alpha for volatility (squared returns)
    alpha_vol: f64,
    /// EMA alpha for trade arrival rate
    alpha_trade_rate: f64,
}

impl MicrostructureState {
    fn new(alpha_mid: f64, alpha_spread: f64, alpha_vol: f64, alpha_trade_rate: f64) -> Self {
        Self {
            ema_mid_price: 0.0,
            ema_spread_bps: 0.0,
            ema_squared_return: 0.0,
            realized_vol_bps: 0.0,
            trade_rate_per_second: 0.0,
            depth_imbalance: 0.0,
            prev_trade_price: None,
            prev_trade_time_millis: None,
            mid_initialized: false,
            vol_initialized: false,
            alpha_mid,
            alpha_spread,
            alpha_vol,
            alpha_trade_rate,
        }
    }

    fn on_book_ticker(&mut self, book: &BookTicker) {
        let mid = (book.bid.price + book.ask.price) / 2.0;
        if mid <= f64::EPSILON {
            return;
        }
        if self.mid_initialized {
            self.ema_mid_price = self.alpha_mid * mid + (1.0 - self.alpha_mid) * self.ema_mid_price;
        } else {
            self.ema_mid_price = mid;
            self.mid_initialized = true;
        }

        let spread_bps = (book.ask.price - book.bid.price) / mid * 10_000.0;
        if self.ema_spread_bps == 0.0 && spread_bps > 0.0 {
            self.ema_spread_bps = spread_bps;
        } else {
            self.ema_spread_bps =
                self.alpha_spread * spread_bps + (1.0 - self.alpha_spread) * self.ema_spread_bps;
        }
    }

    fn on_trade(&mut self, trade: &Trade) {
        // Realized volatility: EMA of squared returns
        if let Some(prev_price) = self.prev_trade_price {
            if prev_price > f64::EPSILON {
                let ret = (trade.price - prev_price) / prev_price;
                let sq_return = ret * ret;
                if self.vol_initialized {
                    self.ema_squared_return = self.alpha_vol * sq_return
                        + (1.0 - self.alpha_vol) * self.ema_squared_return;
                } else {
                    self.ema_squared_return = sq_return;
                    self.vol_initialized = true;
                }
                // Convert to bps: sqrt(ema_sq_return) * 10_000
                // Scale by sqrt(trade_rate) to get per-second vol
                let per_trade_vol_bps = self.ema_squared_return.sqrt() * 10_000.0;
                self.realized_vol_bps =
                    per_trade_vol_bps * self.trade_rate_per_second.max(1.0).sqrt();
            }
        }

        // Trade arrival rate
        if let Some(prev_time) = self.prev_trade_time_millis {
            let dt_millis = trade.trade_time.saturating_sub(prev_time);
            if dt_millis > 0 {
                let instant_rate = 1000.0 / dt_millis as f64;
                // Clamp to avoid crazy spikes from sub-ms batched trades
                let clamped_rate = instant_rate.min(500.0);
                if self.trade_rate_per_second == 0.0 {
                    self.trade_rate_per_second = clamped_rate;
                } else {
                    self.trade_rate_per_second = self.alpha_trade_rate * clamped_rate
                        + (1.0 - self.alpha_trade_rate) * self.trade_rate_per_second;
                }
            }
        }

        self.prev_trade_price = Some(trade.price);
        self.prev_trade_time_millis = Some(trade.trade_time);
    }

    fn on_depth(&mut self, depth: &DepthSnapshot) {
        // Multi-level depth imbalance with distance-weighted levels
        let mut bid_weighted = 0.0;
        let mut ask_weighted = 0.0;
        for (i, level) in depth.bids.iter().enumerate() {
            let weight = 1.0 / (1.0 + i as f64);
            bid_weighted += level.quantity * weight;
        }
        for (i, level) in depth.asks.iter().enumerate() {
            let weight = 1.0 / (1.0 + i as f64);
            ask_weighted += level.quantity * weight;
        }
        let total = bid_weighted + ask_weighted;
        if total > f64::EPSILON {
            self.depth_imbalance = (bid_weighted - ask_weighted) / total;
        }
    }

    /// Whether EMA mid price has been initialized (at least one BookTicker received).
    pub fn mid_initialized(&self) -> bool {
        self.mid_initialized
    }
}

#[derive(Debug, Clone)]
pub struct MarketState {
    symbol: String,
    last_trade: Option<Trade>,
    top_of_book: Option<BookTicker>,
    depth: Option<DepthSnapshot>,
    trades: VecDeque<Trade>,
    trade_window_millis: u64,
    event_mix: EventMixDiagnostics,
    micro: MicrostructureState,
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
            event_mix: EventMixDiagnostics::default(),
            micro: MicrostructureState::new(0.02, 0.01, 0.05, 0.01),
        }
    }

    pub fn apply(&mut self, event: &MarketEvent) {
        match event {
            MarketEvent::Trade(trade) => {
                self.event_mix.trade_events += 1;
                if self.top_of_book.is_none() {
                    self.event_mix.trade_without_quote_events += 1;
                } else if self
                    .top_of_book
                    .as_ref()
                    .is_some_and(|book| trade.trade_time < book.event_time)
                {
                    self.event_mix.stale_quote_events += 1;
                }
                if self
                    .depth
                    .as_ref()
                    .is_some_and(|depth| trade.trade_time < depth.event_time)
                {
                    self.event_mix.stale_depth_events += 1;
                }
                self.micro.on_trade(trade);
                self.last_trade = Some(trade.clone());
                self.trades.push_back(trade.clone());
                self.trim_trade_window(trade.trade_time);
            }
            MarketEvent::BookTicker(book_ticker) => {
                self.event_mix.book_ticker_events += 1;
                self.micro.on_book_ticker(book_ticker);
                self.top_of_book = Some(*book_ticker);
            }
            MarketEvent::Depth(depth) => {
                self.event_mix.depth_events += 1;
                self.micro.on_depth(depth);
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

    pub fn event_mix_diagnostics(&self) -> &EventMixDiagnostics {
        &self.event_mix
    }

    pub fn micro(&self) -> &MicrostructureState {
        &self.micro
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
