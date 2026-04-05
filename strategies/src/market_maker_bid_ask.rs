use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use trade::execution::{DecisionMetric, OrderIntent, Side, TimeInForce};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{
    NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyDiagnostics,
    StrategyLogger,
};
use trade::trader::OrderType;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct MarketMakerBidAskConfig {
    // --- Base spreads ---
    #[serde(default = "default_bid_half_spread_bps")]
    pub bid_half_spread_bps: f64,
    #[serde(default = "default_ask_half_spread_bps")]
    pub ask_half_spread_bps: f64,

    // --- Spread bounds ---
    #[serde(default = "default_min_half_spread_bps")]
    pub min_half_spread_bps: f64,
    #[serde(default = "default_max_half_spread_bps")]
    pub max_half_spread_bps: f64,

    // --- Volatility adjustment ---
    #[serde(default = "default_vol_reference_bps")]
    pub vol_reference_bps: f64,
    #[serde(default = "default_vol_multiplier_min")]
    pub vol_multiplier_min: f64,
    #[serde(default = "default_vol_multiplier_max")]
    pub vol_multiplier_max: f64,

    // --- Trade flow filter ---
    #[serde(default = "default_flow_window_millis")]
    pub flow_window_millis: u64,
    #[serde(default = "default_flow_imbalance_threshold")]
    pub flow_imbalance_threshold: f64,
    #[serde(default = "default_flow_skew_factor")]
    pub flow_skew_factor: f64,

    // --- Microprice edge ---
    #[serde(default = "default_microprice_edge_threshold_bps")]
    pub microprice_edge_threshold_bps: f64,
    #[serde(default = "default_use_microprice_fair")]
    pub use_microprice_fair: bool,

    // --- Inventory management ---
    #[serde(default = "default_inventory_skew_factor")]
    pub inventory_skew_factor: f64,
    #[serde(default = "default_max_inventory_fraction")]
    pub max_inventory_fraction: f64,
    #[serde(default = "default_inventory_decay_rate")]
    pub inventory_decay_rate: f64,

    // --- Sizing ---
    #[serde(default = "default_cash_fraction")]
    pub cash_fraction: f64,
    #[serde(default = "default_min_cash_fraction")]
    pub min_cash_fraction: f64,
    #[serde(default = "default_max_cash_fraction")]
    pub max_cash_fraction: f64,
    #[serde(default = "default_dynamic_sizing")]
    pub dynamic_sizing: bool,
    #[serde(default = "default_spread_ref_bps")]
    pub spread_ref_bps: f64,
    #[serde(default = "default_vol_ref_bps")]
    pub vol_ref_bps: f64,
    #[serde(default = "default_budget_guard_threshold")]
    pub budget_guard_threshold: f64,

    // --- Safety exits ---
    #[serde(default = "default_stop_loss_bps")]
    pub stop_loss_bps: f64,
    #[serde(default = "default_panic_vol_bps")]
    pub panic_vol_bps: f64,
    #[serde(default = "default_max_hold_millis")]
    pub max_hold_millis: u64,
    #[serde(default = "default_min_exit_edge_bps")]
    pub min_exit_edge_bps: f64,

    // --- Cooldown ---
    #[serde(default = "default_entry_cooldown_millis")]
    pub entry_cooldown_millis: u64,

    // --- Volatility filter ---
    #[serde(default = "default_max_vol_bps")]
    pub max_vol_bps: f64,

    // --- Requote threshold ---
    #[serde(default = "default_requote_threshold_bps")]
    pub requote_threshold_bps: f64,

    // --- Trade window ---
    #[serde(default = "default_trade_window_millis")]
    pub trade_window_millis: u64,
    #[serde(default = "default_min_trades_in_window")]
    pub min_trades_in_window: usize,
}

fn default_bid_half_spread_bps() -> f64 { 2.0 }
fn default_ask_half_spread_bps() -> f64 { 2.0 }
fn default_min_half_spread_bps() -> f64 { 0.5 }
fn default_max_half_spread_bps() -> f64 { 15.0 }
fn default_vol_reference_bps() -> f64 { 25.0 }
fn default_vol_multiplier_min() -> f64 { 0.6 }
fn default_vol_multiplier_max() -> f64 { 3.5 }
fn default_flow_window_millis() -> u64 { 5_000 }
fn default_flow_imbalance_threshold() -> f64 { 0.15 }
fn default_flow_skew_factor() -> f64 { 0.5 }
fn default_microprice_edge_threshold_bps() -> f64 { 0.5 }
fn default_use_microprice_fair() -> bool { true }
fn default_inventory_skew_factor() -> f64 { 0.7 }
fn default_max_inventory_fraction() -> f64 { 0.4 }
fn default_inventory_decay_rate() -> f64 { 0.0 }
fn default_cash_fraction() -> f64 { 0.05 }
fn default_min_cash_fraction() -> f64 { 0.01 }
fn default_max_cash_fraction() -> f64 { 0.12 }
fn default_dynamic_sizing() -> bool { true }
fn default_spread_ref_bps() -> f64 { 80.0 }
fn default_vol_ref_bps() -> f64 { 40.0 }
fn default_budget_guard_threshold() -> f64 { 0.5 }
fn default_stop_loss_bps() -> f64 { 400.0 }
fn default_panic_vol_bps() -> f64 { 1500.0 }
fn default_max_hold_millis() -> u64 { 180_000 }
fn default_min_exit_edge_bps() -> f64 { 5.0 }
fn default_entry_cooldown_millis() -> u64 { 3_000 }
fn default_max_vol_bps() -> f64 { 800.0 }
fn default_requote_threshold_bps() -> f64 { 0.3 }
fn default_trade_window_millis() -> u64 { 30_000 }
fn default_min_trades_in_window() -> usize { 3 }

impl Default for MarketMakerBidAskConfig {
    fn default() -> Self {
        Self {
            bid_half_spread_bps: default_bid_half_spread_bps(),
            ask_half_spread_bps: default_ask_half_spread_bps(),
            min_half_spread_bps: default_min_half_spread_bps(),
            max_half_spread_bps: default_max_half_spread_bps(),
            vol_reference_bps: default_vol_reference_bps(),
            vol_multiplier_min: default_vol_multiplier_min(),
            vol_multiplier_max: default_vol_multiplier_max(),
            flow_window_millis: default_flow_window_millis(),
            flow_imbalance_threshold: default_flow_imbalance_threshold(),
            flow_skew_factor: default_flow_skew_factor(),
            microprice_edge_threshold_bps: default_microprice_edge_threshold_bps(),
            use_microprice_fair: default_use_microprice_fair(),
            inventory_skew_factor: default_inventory_skew_factor(),
            max_inventory_fraction: default_max_inventory_fraction(),
            inventory_decay_rate: default_inventory_decay_rate(),
            cash_fraction: default_cash_fraction(),
            min_cash_fraction: default_min_cash_fraction(),
            max_cash_fraction: default_max_cash_fraction(),
            dynamic_sizing: default_dynamic_sizing(),
            spread_ref_bps: default_spread_ref_bps(),
            vol_ref_bps: default_vol_ref_bps(),
            budget_guard_threshold: default_budget_guard_threshold(),
            stop_loss_bps: default_stop_loss_bps(),
            panic_vol_bps: default_panic_vol_bps(),
            max_hold_millis: default_max_hold_millis(),
            min_exit_edge_bps: default_min_exit_edge_bps(),
            entry_cooldown_millis: default_entry_cooldown_millis(),
            max_vol_bps: default_max_vol_bps(),
            requote_threshold_bps: default_requote_threshold_bps(),
            trade_window_millis: default_trade_window_millis(),
            min_trades_in_window: default_min_trades_in_window(),
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct BidAskDiagnostics {
    total_decisions: usize,
    blocked_no_quote: usize,
    blocked_vol_high: usize,
    blocked_min_trades: usize,
    blocked_cooldown: usize,
    blocked_inventory_full: usize,
    blocked_flow_adverse: usize,
    blocked_microprice_edge: usize,

    entries_bid: usize,
    entries_ask: usize,
    exits_taker_stop_loss: usize,
    exits_taker_panic_vol: usize,
    exits_taker_max_hold: usize,
    exits_passive: usize,

    last_bid_spread_bps: f64,
    last_ask_spread_bps: f64,
    last_mid_price: f64,
    last_fair_price: f64,
    last_flow_imbalance: f64,
    last_inventory_ratio: f64,
    last_effective_cf: f64,
    last_buy_quantity: f64,
    last_sell_quantity: f64,

    last_report_total: usize,
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

pub struct MarketMakerBidAskStrategy {
    config: MarketMakerBidAskConfig,
    logger: NoOpStrategyLogger,
    last_exit_time_millis: Option<u64>,
    was_in_position: bool,
    last_buy_quote: f64,
    last_sell_quote: f64,
    diagnostics: BidAskDiagnostics,
}

impl MarketMakerBidAskStrategy {
    fn load_config<P: AsRef<Path>>(
        path: P,
    ) -> Result<MarketMakerBidAskConfig, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: MarketMakerBidAskConfig = toml::from_str(&content)?;
        Ok(config)
    }

    fn in_cooldown(&self, market_state: &MarketState) -> bool {
        if self.config.entry_cooldown_millis == 0 {
            return false;
        }
        let Some(now) = market_state.last_event_time_millis() else {
            return true;
        };
        match self.last_exit_time_millis {
            Some(exit_time) => now.saturating_sub(exit_time) < self.config.entry_cooldown_millis,
            None => false,
        }
    }

    fn fair_price(&self, market_state: &MarketState) -> Option<f64> {
        if self.config.use_microprice_fair {
            let micro = market_state.micro();
            if micro.mid_initialized() && micro.ema_mid_price > f64::EPSILON {
                return Some(micro.ema_mid_price);
            }
        }

        if let Some(book) = market_state.top_of_book() {
            let bid = book.bid.price;
            let ask = book.ask.price;
            if bid > f64::EPSILON && ask > bid {
                return Some((bid + ask) / 2.0);
            }
        }

        market_state.last_price()
    }

    fn book_prices(&self, market_state: &MarketState) -> Option<(f64, f64)> {
        if let Some(book) = market_state.top_of_book() {
            let bid = book.bid.price;
            let ask = book.ask.price;
            if bid > f64::EPSILON && ask > bid {
                return Some((bid, ask));
            }
        }
        None
    }

    fn flow_imbalance(&self, market_state: &MarketState) -> f64 {
        let stats = market_state.trade_window_stats();
        let total = stats.buyer_initiated_volume + stats.seller_initiated_volume;
        if total <= f64::EPSILON {
            return 0.0;
        }
        (stats.buyer_initiated_volume - stats.seller_initiated_volume) / total
    }

    fn dynamic_half_spreads(&self, market_state: &MarketState) -> (f64, f64) {
        let vol_bps = market_state.micro().realized_vol_bps.max(1.0);
        let reference = self.config.vol_reference_bps.max(1.0);
        let vol_multiplier = (vol_bps / reference).clamp(
            self.config.vol_multiplier_min,
            self.config.vol_multiplier_max,
        );

        let base_bid = self.config.bid_half_spread_bps * vol_multiplier;
        let base_ask = self.config.ask_half_spread_bps * vol_multiplier;

        let bid = base_bid.clamp(self.config.min_half_spread_bps, self.config.max_half_spread_bps);
        let ask = base_ask.clamp(self.config.min_half_spread_bps, self.config.max_half_spread_bps);

        (bid, ask)
    }

    fn apply_flow_skew(&self, bid_spread: f64, ask_spread: f64, flow_imb: f64) -> (f64, f64) {
        if self.config.flow_skew_factor <= 0.0 {
            return (bid_spread, ask_spread);
        }

        let skew = flow_imb * self.config.flow_skew_factor;
        let bid = (bid_spread * (1.0 - skew)).clamp(self.config.min_half_spread_bps, self.config.max_half_spread_bps);
        let ask = (ask_spread * (1.0 + skew)).clamp(self.config.min_half_spread_bps, self.config.max_half_spread_bps);

        (bid, ask)
    }

    fn apply_inventory_skew(&self, bid_spread: f64, ask_spread: f64, inventory_ratio: f64) -> (f64, f64) {
        if self.config.inventory_skew_factor <= 0.0 {
            return (bid_spread, ask_spread);
        }

        let skew = self.config.inventory_skew_factor.clamp(0.0, 0.95);
        let bid = (bid_spread * (1.0 + inventory_ratio * skew)).clamp(self.config.min_half_spread_bps, self.config.max_half_spread_bps);
        let ask = (ask_spread * (1.0 - inventory_ratio * skew)).clamp(self.config.min_half_spread_bps, self.config.max_half_spread_bps);

        (bid, ask)
    }

    fn max_inventory_notional(&self, context: &StrategyContext) -> f64 {
        let fraction_cap = if context.initial_capital > f64::EPSILON {
            context.initial_capital * self.config.max_inventory_fraction
        } else {
            0.0
        };

        match (
            fraction_cap > f64::EPSILON,
            context.max_position_notional > f64::EPSILON,
        ) {
            (true, true) => fraction_cap.min(context.max_position_notional),
            (true, false) => fraction_cap,
            (false, true) => context.max_position_notional,
            (false, false) => 0.0,
        }
    }

    fn compute_dynamic_cf(&self, market_state: &MarketState, context: &StrategyContext) -> f64 {
        let base_cf = self.config.cash_fraction;

        let spread_bps = if let Some(book) = market_state.top_of_book() {
            let mid = (book.bid.price + book.ask.price) / 2.0;
            if mid > f64::EPSILON {
                (book.ask.price - book.bid.price) / mid * 10_000.0
            } else {
                self.config.spread_ref_bps
            }
        } else {
            let micro = market_state.micro();
            if micro.ema_spread_bps > 0.0 {
                micro.ema_spread_bps
            } else {
                self.config.spread_ref_bps
            }
        };
        let spread_factor = (spread_bps / self.config.spread_ref_bps).clamp(0.5, 2.0);

        let vol_bps = market_state.micro().realized_vol_bps;
        let vol_factor = if vol_bps > 1.0 {
            (self.config.vol_ref_bps / vol_bps).clamp(0.5, 1.5)
        } else {
            1.0
        };

        let budget_factor = if context.initial_capital > 0.0 {
            let budget_floor = context.initial_capital * self.config.budget_guard_threshold;
            if budget_floor > 0.0 {
                (context.available_cash / budget_floor).clamp(0.3, 1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };

        let effective_cf = base_cf * spread_factor * vol_factor * budget_factor;
        effective_cf.clamp(self.config.min_cash_fraction, self.config.max_cash_fraction)
    }

    fn check_emergency_exit(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> Option<(&'static str, f64)> {
        if context.current_position.quantity <= 0.0 {
            return None;
        }

        let entry_price = context.current_position.entry_price;
        let current_price = market_state
            .top_of_book()
            .map(|b| b.bid.price)
            .or_else(|| market_state.last_price())?;

        let pnl_bps = (current_price - entry_price) / entry_price * 10_000.0;

        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_taker_stop_loss += 1;
            return Some(("stop_loss", current_price));
        }

        let micro = market_state.micro();
        if micro.realized_vol_bps > self.config.panic_vol_bps {
            self.diagnostics.exits_taker_panic_vol += 1;
            return Some(("panic_vol", current_price));
        }

        let now = market_state.last_event_time_millis()?;
        if context.current_position.entry_time > 0.0 {
            let held_millis = now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
            if held_millis >= self.config.max_hold_millis {
                self.diagnostics.exits_taker_max_hold += 1;
                return Some(("max_hold_time", current_price));
            }
        }

        None
    }

    fn passive_exit_floor_bps(&self, market_state: &MarketState, context: &StrategyContext) -> f64 {
        let base_floor_bps = self.config.min_exit_edge_bps;
        let Some(now) = market_state.last_event_time_millis() else {
            return base_floor_bps;
        };
        if context.current_position.entry_time <= 0.0 || self.config.max_hold_millis == 0 {
            return base_floor_bps;
        }

        let held_millis = now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
        let max_hold = self.config.max_hold_millis;

        if held_millis >= max_hold {
            return base_floor_bps.min(-3.0);
        }

        let held_ratio = held_millis as f64 / max_hold as f64;
        if held_ratio < 0.5 {
            return base_floor_bps;
        }

        let unwind_ratio = ((held_ratio - 0.5) / 0.5).clamp(0.0, 1.0);
        let target_floor_bps = base_floor_bps.min(-3.0);
        base_floor_bps + (target_floor_bps - base_floor_bps) * unwind_ratio
    }
}

#[async_trait::async_trait]
impl Strategy for MarketMakerBidAskStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        &self.logger
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let config = Self::load_config(config_path)?;
        Ok(Self {
            config,
            logger: NoOpStrategyLogger,
            last_exit_time_millis: None,
            was_in_position: false,
            last_buy_quote: 0.0,
            last_sell_quote: 0.0,
            diagnostics: BidAskDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "MarketMakerBidAskStrategy - independent bid/ask quoting with flow and inventory skew".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert("ba.total_decisions".into(), self.diagnostics.total_decisions);
        counters.insert("ba.blocked_no_quote".into(), self.diagnostics.blocked_no_quote);
        counters.insert("ba.blocked_vol_high".into(), self.diagnostics.blocked_vol_high);
        counters.insert("ba.blocked_min_trades".into(), self.diagnostics.blocked_min_trades);
        counters.insert("ba.blocked_cooldown".into(), self.diagnostics.blocked_cooldown);
        counters.insert("ba.blocked_inventory_full".into(), self.diagnostics.blocked_inventory_full);
        counters.insert("ba.blocked_flow_adverse".into(), self.diagnostics.blocked_flow_adverse);
        counters.insert("ba.blocked_microprice_edge".into(), self.diagnostics.blocked_microprice_edge);
        counters.insert("ba.entries_bid".into(), self.diagnostics.entries_bid);
        counters.insert("ba.entries_ask".into(), self.diagnostics.entries_ask);
        counters.insert("ba.exits_taker_stop_loss".into(), self.diagnostics.exits_taker_stop_loss);
        counters.insert("ba.exits_taker_panic_vol".into(), self.diagnostics.exits_taker_panic_vol);
        counters.insert("ba.exits_taker_max_hold".into(), self.diagnostics.exits_taker_max_hold);
        counters.insert("ba.exits_passive".into(), self.diagnostics.exits_passive);

        let mut gauges = BTreeMap::new();
        gauges.insert("ba.last_bid_spread_bps".into(), self.diagnostics.last_bid_spread_bps);
        gauges.insert("ba.last_ask_spread_bps".into(), self.diagnostics.last_ask_spread_bps);
        gauges.insert("ba.last_mid_price".into(), self.diagnostics.last_mid_price);
        gauges.insert("ba.last_fair_price".into(), self.diagnostics.last_fair_price);
        gauges.insert("ba.last_flow_imbalance".into(), self.diagnostics.last_flow_imbalance);
        gauges.insert("ba.last_inventory_ratio".into(), self.diagnostics.last_inventory_ratio);
        gauges.insert("ba.last_effective_cf".into(), self.diagnostics.last_effective_cf);
        gauges.insert("ba.last_buy_quantity".into(), self.diagnostics.last_buy_quantity);
        gauges.insert("ba.last_sell_quantity".into(), self.diagnostics.last_sell_quantity);

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, _event: &MarketEvent, _market_state: &MarketState) {}

    fn decide(&mut self, market_state: &MarketState, context: &StrategyContext) -> StrategyDecision {
        self.diagnostics.total_decisions += 1;

        let since_last = self.diagnostics.total_decisions - self.diagnostics.last_report_total;
        if since_last >= 50 && context.current_position.quantity <= 0.0 {
            self.diagnostics.last_report_total = self.diagnostics.total_decisions;
        }

        let in_position = context.current_position.quantity > 0.0;
        if self.was_in_position && !in_position {
            self.last_exit_time_millis = market_state.last_event_time_millis();
        }
        self.was_in_position = in_position;

        // === Emergency taker exit ===
        if let Some((rationale, exit_price)) = self.check_emergency_exit(market_state, context) {
            self.last_exit_time_millis = market_state.last_event_time_millis();
            return StrategyDecision {
                confidence: 1.0,
                intent: OrderIntent::Place {
                    side: Side::Sell,
                    order_type: OrderType::Taker,
                    price: Some(exit_price),
                    quantity: context.current_position.quantity,
                    time_in_force: TimeInForce::Ioc,
                    rationale,
                    expected_edge_bps: 0.0,
                },
                metrics: vec![DecisionMetric {
                    name: "exit_price",
                    value: exit_price,
                }],
            };
        }

        // === Get market data ===
        let Some((best_bid, best_ask)) = self.book_prices(market_state) else {
            self.diagnostics.blocked_no_quote += 1;
            return StrategyDecision::no_action();
        };

        let mid = (best_bid + best_ask) / 2.0;
        if mid <= f64::EPSILON {
            self.diagnostics.blocked_no_quote += 1;
            return StrategyDecision::no_action();
        }

        let fair_price = self.fair_price(market_state).unwrap_or(mid);
        let tick_size = context.tick_size.max(f64::EPSILON);

        // === Volatility filter ===
        let micro = market_state.micro();
        if micro.realized_vol_bps > self.config.max_vol_bps {
            self.diagnostics.blocked_vol_high += 1;
            return StrategyDecision::no_action();
        }

        // === Minimum trades filter ===
        let stats = market_state.trade_window_stats();
        if stats.trade_count < self.config.min_trades_in_window {
            self.diagnostics.blocked_min_trades += 1;
            return StrategyDecision::no_action();
        }

        // === Cooldown ===
        if self.in_cooldown(market_state) {
            self.diagnostics.blocked_cooldown += 1;
            return StrategyDecision::no_action();
        }

        // === Compute base spreads ===
        let (mut bid_spread, mut ask_spread) = self.dynamic_half_spreads(market_state);

        // === Flow imbalance ===
        let flow_imb = self.flow_imbalance(market_state);
        (bid_spread, ask_spread) = self.apply_flow_skew(bid_spread, ask_spread, flow_imb);

        // === Inventory ===
        let max_inventory_value = self.max_inventory_notional(context);
        let inventory_value = context.current_position.quantity * mid;
        let inventory_ratio = if max_inventory_value > f64::EPSILON {
            (inventory_value / max_inventory_value).clamp(0.0, 1.0)
        } else {
            0.0
        };

        (bid_spread, ask_spread) = self.apply_inventory_skew(bid_spread, ask_spread, inventory_ratio);

        // === HFT quoting: anchor to book, not to mid ===
        // For HFT market making, we quote at (or 1 tick inside) the best bid/ask.
        // The half-spread bps controls how aggressively we join the queue:
        //   0 bps = exactly at best bid/ask (join queue)
        //   >0 bps = 1+ ticks inside the spread (price improvement, lower fill rate)
        let spread_bps = if best_ask > f64::EPSILON {
            (best_ask - best_bid) / best_ask * 10_000.0
        } else {
            0.0
        };

        let half_spread_bps = spread_bps / 2.0;

        // If our configured spread is less than or equal to the book half-spread,
        // we quote at the book edge (HFT mode). Otherwise, we quote inside the spread.
        let mut buy_price = if bid_spread <= half_spread_bps {
            best_bid
        } else {
            let offset_from_bid = (bid_spread - half_spread_bps) / 10_000.0 * mid;
            (best_bid + offset_from_bid).min(mid)
        };

        let mut sell_price = if ask_spread <= half_spread_bps {
            best_ask
        } else {
            let offset_from_ask = (ask_spread - half_spread_bps) / 10_000.0 * mid;
            (best_ask - offset_from_ask).max(mid)
        };

        // === Flow micro-adjustment: shift 1 tick toward mid on adverse flow ===
        if flow_imb < -self.config.flow_imbalance_threshold {
            buy_price = (buy_price - tick_size).max(best_bid - tick_size * 3.0);
        } else if flow_imb > self.config.flow_imbalance_threshold {
            sell_price = (sell_price + tick_size).min(best_ask + tick_size * 3.0);
        }

        // === Spread-crossing guard ===
        if buy_price >= best_ask {
            buy_price = best_ask - tick_size;
        }
        if sell_price <= best_bid {
            sell_price = best_bid + tick_size;
        }
        if sell_price <= buy_price {
            sell_price = buy_price + tick_size;
        }

        // === Entry-price floor for sell side ===
        let entry_price = context.current_position.entry_price;
        if entry_price > 0.0 && context.current_position.quantity > f64::EPSILON {
            let exit_floor_bps = self.passive_exit_floor_bps(market_state, context);
            let min_sell = entry_price * (1.0 + exit_floor_bps / 10_000.0);
            sell_price = sell_price.max(min_sell);
        }

        // === Microprice edge filter for bid entry ===
        if !in_position {
            let microprice = market_state.micro().ema_mid_price;
            if microprice > f64::EPSILON {
                let bid_edge_bps = (microprice - buy_price) / microprice * 10_000.0;
                if bid_edge_bps < self.config.microprice_edge_threshold_bps {
                    self.diagnostics.blocked_microprice_edge += 1;
                }
            }

            // === Flow adverse filter ===
            if flow_imb < -self.config.flow_imbalance_threshold {
                self.diagnostics.blocked_flow_adverse += 1;
            }
        }

        // === Requote threshold ===
        if self.last_buy_quote > 0.0 && self.last_sell_quote > 0.0 {
            let buy_change_bps = ((buy_price - self.last_buy_quote) / self.last_buy_quote * 10_000.0).abs();
            let sell_change_bps = ((sell_price - self.last_sell_quote) / self.last_sell_quote * 10_000.0).abs();
            if buy_change_bps < self.config.requote_threshold_bps
                && sell_change_bps < self.config.requote_threshold_bps
            {
                // Prices stable, keep existing quotes
            }
        }
        self.last_buy_quote = buy_price;
        self.last_sell_quote = sell_price;

        // === Compute quantities ===
        let effective_cf = if self.config.dynamic_sizing {
            self.compute_dynamic_cf(market_state, context)
        } else {
            self.config.cash_fraction
        };
        self.diagnostics.last_effective_cf = effective_cf;

        let remaining_inventory_notional = (max_inventory_value - inventory_value).max(0.0);
        let scale = 1.0 - inventory_ratio;
        let scaled_cf = (effective_cf * scale).max(0.0);

        let buy_quantity = if inventory_ratio >= 1.0 {
            self.diagnostics.blocked_inventory_full += 1;
            0.0
        } else {
            context
                .capped_entry_quantity(buy_price, scaled_cf, Some(remaining_inventory_notional))
                .unwrap_or(0.0)
        };

        let sell_quantity = if in_position {
            context.current_position.quantity
        } else {
            0.0
        };

        if buy_quantity <= f64::EPSILON && sell_quantity <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        // === Track entry side ===
        if buy_quantity > f64::EPSILON {
            self.diagnostics.entries_bid += 1;
        }
        if sell_quantity > f64::EPSILON && in_position {
            self.diagnostics.exits_passive += 1;
        }

        let quoted_spread_bps = (sell_price - buy_price) / mid * 10_000.0;

        // === Diagnostics ===
        self.diagnostics.last_bid_spread_bps = bid_spread;
        self.diagnostics.last_ask_spread_bps = ask_spread;
        self.diagnostics.last_mid_price = mid;
        self.diagnostics.last_fair_price = fair_price;
        self.diagnostics.last_flow_imbalance = flow_imb;
        self.diagnostics.last_inventory_ratio = inventory_ratio;
        self.diagnostics.last_buy_quantity = buy_quantity;
        self.diagnostics.last_sell_quantity = sell_quantity;

        StrategyDecision {
            confidence: 1.0,
            intent: OrderIntent::QuoteBothSides {
                buy_price,
                buy_quantity,
                sell_price,
                sell_quantity,
                rationale: "bid_ask_independent_quote",
                expected_edge_bps: quoted_spread_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "mid_price",
                    value: mid,
                },
                DecisionMetric {
                    name: "fair_price",
                    value: fair_price,
                },
                DecisionMetric {
                    name: "bid_spread_bps",
                    value: bid_spread,
                },
                DecisionMetric {
                    name: "ask_spread_bps",
                    value: ask_spread,
                },
                DecisionMetric {
                    name: "flow_imbalance",
                    value: flow_imb,
                },
                DecisionMetric {
                    name: "inventory_ratio",
                    value: inventory_ratio,
                },
                DecisionMetric {
                    name: "buy_price",
                    value: buy_price,
                },
                DecisionMetric {
                    name: "sell_price",
                    value: sell_price,
                },
                DecisionMetric {
                    name: "buy_quantity",
                    value: buy_quantity,
                },
                DecisionMetric {
                    name: "sell_quantity",
                    value: sell_quantity,
                },
                DecisionMetric {
                    name: "quoted_spread_bps",
                    value: quoted_spread_bps,
                },
            ],
        }
    }
}
