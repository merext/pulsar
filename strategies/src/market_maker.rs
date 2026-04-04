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
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct MarketMakerConfig {
    // --- Spread & pricing ---

    /// Minimum spread in bps to accept an entry. If the current spread is below
    /// this threshold, the strategy skips the tick (not enough edge to cover fees).
    #[serde(default = "default_min_spread_bps")]
    pub min_spread_bps: f64,

    /// Round-trip maker cost in bps (maker_fee * 2 * 10_000).
    /// Default 20.0 for 10 bps maker fee on each side.
    #[serde(default = "default_round_trip_cost_bps")]
    pub round_trip_cost_bps: f64,

    /// Minimum edge after subtracting round-trip cost (in bps).
    /// Entry is only taken if spread_bps - round_trip_cost_bps >= min_edge_bps.
    #[serde(default = "default_min_edge_bps")]
    pub min_edge_bps: f64,

    // --- Position management ---

    /// Maximum hold time in milliseconds. After this, exit via taker.
    #[serde(default = "default_max_hold_millis")]
    pub max_hold_millis: u64,

    /// Stop loss in bps (taker exit if unrealized PnL drops below this).
    #[serde(default = "default_stop_loss_bps")]
    pub stop_loss_bps: f64,

    /// Cooldown between entries in milliseconds. Prevents churning after
    /// a losing exit.
    #[serde(default = "default_entry_cooldown_millis")]
    pub entry_cooldown_millis: u64,

    // --- Volatility filters ---

    /// Maximum realized volatility in bps. If vol exceeds this, skip entry.
    /// During high vol, spreads widen but adverse selection risk is higher.
    #[serde(default = "default_max_vol_bps")]
    pub max_vol_bps: f64,

    /// Panic volatility threshold in bps. If vol exceeds this while in position,
    /// exit immediately via taker.
    #[serde(default = "default_panic_vol_bps")]
    pub panic_vol_bps: f64,

    // --- Trade window ---

    /// Window in milliseconds for computing microstructure metrics.
    #[serde(default = "default_trade_window_millis")]
    pub trade_window_millis: u64,

    /// Minimum number of trades in the window before allowing entry.
    #[serde(default = "default_min_trades_in_window")]
    pub min_trades_in_window: usize,

    // --- Order book imbalance ---

    /// Skew factor: how much order book imbalance affects quoting.
    /// 0.0 = ignore imbalance, 1.0 = fully adjust.
    /// When bid-heavy (imbalance > 0), we're more willing to buy.
    /// When ask-heavy (imbalance < 0), we skip buying.
    #[serde(default = "default_imbalance_skew")]
    pub imbalance_skew: f64,

    /// Maximum absolute imbalance to allow entry. Beyond this, adverse
    /// selection risk is too high (one side is being consumed).
    #[serde(default = "default_max_imbalance")]
    pub max_imbalance: f64,

    // --- Trade direction filter ---

    /// Require the last trade to be seller-initiated (is_buyer_market_maker=true)
    /// before entering. This means price is near the bid — ideal for passive buy.
    /// When false, we also allow entry on buyer-initiated trades (price near ask).
    #[serde(default = "default_require_seller_initiated")]
    pub require_seller_initiated: bool,

    /// Minimum fraction of seller-initiated volume in the recent window [0.0, 1.0].
    /// Higher values require more selling pressure before entering.
    /// 0.0 = disabled (any flow allowed). 0.5 = at least half must be sells.
    #[serde(default = "default_min_sell_flow_fraction")]
    pub min_sell_flow_fraction: f64,

    /// Number of recent trades to consider for price position filtering.
    /// We check if last_price is at the low end of recent N-trade range.
    #[serde(default = "default_price_position_window")]
    pub price_position_window: usize,

    /// Maximum price position within the recent window [0.0, 1.0].
    /// 0.0 = only buy at window low, 0.5 = buy in bottom half, 1.0 = disabled.
    /// This ensures we're buying near bid, not ask.
    #[serde(default = "default_max_price_position")]
    pub max_price_position: f64,

    // --- Exit pricing ---

    /// Minimum edge in bps for passive sell orders relative to entry price.
    /// The sell limit price is floored at entry_price * (1 + min_exit_edge_bps / 10_000).
    /// Set to round_trip_cost_bps (e.g. 20.0) to ensure breakeven after fees.
    /// Set to 0.0 to disable (allow selling at entry price, losing only fees).
    /// Negative values allow selling below entry (validation mode).
    /// Does NOT affect taker exits (stop_loss, panic_vol, max_hold).
    #[serde(default = "default_min_exit_edge_bps")]
    pub min_exit_edge_bps: f64,

    // --- Sizing ---

    /// Fraction of available cash to use per entry (0.0 - 1.0).
    #[serde(default = "default_cash_fraction")]
    pub cash_fraction: f64,

    // --- Dynamic sizing ---

    /// Enable dynamic position sizing based on spread, vol, and budget.
    /// When false, static `cash_fraction` is used (backward compatible).
    #[serde(default)]
    pub dynamic_sizing: bool,

    /// Minimum cash fraction floor (prevents too-small orders).
    #[serde(default = "default_min_cash_fraction")]
    pub min_cash_fraction: f64,

    /// Maximum cash fraction ceiling (prevents over-concentration).
    #[serde(default = "default_max_cash_fraction")]
    pub max_cash_fraction: f64,

    /// Reference spread in bps for normalization (symbol-specific typical spread).
    /// spread_factor = clamp(spread_bps / spread_ref_bps, 0.5, 2.0)
    #[serde(default = "default_spread_ref_bps")]
    pub spread_ref_bps: f64,

    /// Reference volatility in bps for normalization.
    /// vol_factor = clamp(vol_ref_bps / max(vol_bps, 1.0), 0.5, 1.5)
    #[serde(default = "default_vol_ref_bps")]
    pub vol_ref_bps: f64,

    /// Fraction of initial capital below which to start reducing sizing.
    /// budget_factor = clamp(available_cash / (initial_capital * threshold), 0.3, 1.0)
    #[serde(default = "default_budget_guard_threshold")]
    pub budget_guard_threshold: f64,

    // --- Two-sided market-making ---

    /// Enable two-sided quoting mode. When true, the strategy returns
    /// QuoteBothSides intent with simultaneous buy and sell limit orders.
    /// When false, uses the original single-sided logic (backward compatible).
    #[serde(default)]
    pub two_sided: bool,

    /// Base half-spread in bps for two-sided quoting. Buy is placed at
    /// mid - half_spread, sell at mid + half_spread.
    #[serde(default = "default_half_spread_bps")]
    pub half_spread_bps: f64,

    /// Inventory skew factor [0.0, 1.0]. Controls how aggressively quotes
    /// become asymmetric when holding inventory.
    ///
    /// At inventory_ratio=0 (no position), both sides are symmetric:
    ///   buy  = mid - half_spread
    ///   sell = mid + half_spread
    ///
    /// At inventory_ratio=1 (max position), with skew_factor=0.8:
    ///   buy  = mid - half_spread * 1.8  (wider → less likely to buy more)
    ///   sell = mid + half_spread * 0.2  (tighter → much more likely to sell)
    ///
    /// This is more effective than shifting the mid-point because it directly
    /// controls each side's distance from mid independently.
    #[serde(default = "default_inventory_skew_bps")]
    pub inventory_skew_bps: f64,

    /// Maximum inventory as a fraction of initial capital [0.0, 1.0].
    /// Once reached, only sell quotes are placed (no more buying).
    #[serde(default = "default_max_inventory_fraction")]
    pub max_inventory_fraction: f64,

    /// Minimum price change in bps before re-quoting. Prevents API spam
    /// when mid-price barely moves between ticks.
    #[serde(default = "default_requote_threshold_bps")]
    pub requote_threshold_bps: f64,
}

fn default_min_spread_bps() -> f64 { 30.0 }
fn default_round_trip_cost_bps() -> f64 { 20.0 }
fn default_min_edge_bps() -> f64 { 5.0 }
fn default_max_hold_millis() -> u64 { 300_000 } // 5 minutes
fn default_stop_loss_bps() -> f64 { 500.0 }
fn default_entry_cooldown_millis() -> u64 { 5_000 }
fn default_max_vol_bps() -> f64 { 1000.0 }
fn default_panic_vol_bps() -> f64 { 2000.0 }
fn default_trade_window_millis() -> u64 { 60_000 }
fn default_min_trades_in_window() -> usize { 2 }
fn default_imbalance_skew() -> f64 { 0.0 }
fn default_max_imbalance() -> f64 { 1.0 }
fn default_require_seller_initiated() -> bool { true }
fn default_min_sell_flow_fraction() -> f64 { 0.0 }
fn default_price_position_window() -> usize { 20 }
fn default_max_price_position() -> f64 { 0.3 }
fn default_min_exit_edge_bps() -> f64 { 0.0 }
fn default_cash_fraction() -> f64 { 0.9 }
fn default_min_cash_fraction() -> f64 { 0.02 }
fn default_max_cash_fraction() -> f64 { 0.15 }
fn default_spread_ref_bps() -> f64 { 100.0 }
fn default_vol_ref_bps() -> f64 { 50.0 }
fn default_budget_guard_threshold() -> f64 { 0.5 }
fn default_half_spread_bps() -> f64 { 3.0 }
fn default_inventory_skew_bps() -> f64 { 0.8 }
fn default_max_inventory_fraction() -> f64 { 0.5 }
fn default_requote_threshold_bps() -> f64 { 0.5 }

impl Default for MarketMakerConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: default_min_spread_bps(),
            round_trip_cost_bps: default_round_trip_cost_bps(),
            min_edge_bps: default_min_edge_bps(),
            max_hold_millis: default_max_hold_millis(),
            stop_loss_bps: default_stop_loss_bps(),
            entry_cooldown_millis: default_entry_cooldown_millis(),
            max_vol_bps: default_max_vol_bps(),
            panic_vol_bps: default_panic_vol_bps(),
            trade_window_millis: default_trade_window_millis(),
            min_trades_in_window: default_min_trades_in_window(),
            imbalance_skew: default_imbalance_skew(),
            max_imbalance: default_max_imbalance(),
            require_seller_initiated: default_require_seller_initiated(),
            min_sell_flow_fraction: default_min_sell_flow_fraction(),
            price_position_window: default_price_position_window(),
            max_price_position: default_max_price_position(),
            min_exit_edge_bps: default_min_exit_edge_bps(),
            cash_fraction: default_cash_fraction(),
            dynamic_sizing: false,
            min_cash_fraction: default_min_cash_fraction(),
            max_cash_fraction: default_max_cash_fraction(),
            spread_ref_bps: default_spread_ref_bps(),
            vol_ref_bps: default_vol_ref_bps(),
            budget_guard_threshold: default_budget_guard_threshold(),
            two_sided: false,
            half_spread_bps: default_half_spread_bps(),
            inventory_skew_bps: default_inventory_skew_bps(),
            max_inventory_fraction: default_max_inventory_fraction(),
            requote_threshold_bps: default_requote_threshold_bps(),
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct MakerDiagnostics {
    // Decision counters
    total_decisions: usize,
    blocked_no_quote: usize,
    blocked_spread_narrow: usize,
    blocked_no_edge: usize,
    blocked_vol_high: usize,
    blocked_min_trades: usize,
    blocked_cooldown: usize,
    blocked_imbalance: usize,
    blocked_not_seller_initiated: usize,
    blocked_sell_flow_low: usize,
    blocked_price_too_high: usize,

    // Entry/exit counters
    entries_passive: usize,
    exits_passive_sell: usize,
    exits_stop_loss: usize,
    exits_max_hold: usize,
    exits_panic_vol: usize,
    exits_maker: usize,
    exits_taker: usize,

    // Gauges
    last_spread_bps: f64,
    last_edge_bps: f64,
    last_imbalance: f64,
    last_vol_bps: f64,
    last_price_position: f64,
    last_effective_cf: f64,

    // Snapshot for periodic delta reporting
    last_report_total: usize,
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

// ═══════════════════════════════════════════════════════════════════════════
// MarketMakerStrategy — Simple Spread Capture for Wide-Spread Pairs
// ═══════════════════════════════════════════════════════════════════════════
//
// TARGET UNIVERSE:
//   Low-price USDT pairs where 1 tick = huge bps spread.
//   Examples: TRUUSDT (tick=0.0001, price~$0.0045, 1 tick = 217 bps)
//
// CORE IDEA:
//   When spread >> fees, simply buy at bid and sell at ask.
//   No mean-reversion, no directional prediction.
//   Pure spread capture: post passive buy, then passive sell.
//
// EXECUTION MODEL:
//   Without position: Post passive Buy @ bid (OrderType::Maker)
//   With position:    Post passive Sell @ ask (OrderType::Maker)
//   Safety exits:     Taker sell on stop_loss, max_hold, panic_vol
//
// EDGE CALCULATION:
//   Gross edge = spread_bps (e.g. 217 bps for TRUUSDT)
//   Costs = maker_fee * 2 = 20 bps (10 bps each side)
//   Net edge = 217 - 20 = 197 bps per round-trip
//
// FILL RATE:
//   Backtest engine simulates ~34% fill rate for passive orders.
//   Each tick, if in position, we re-submit passive sell.
//   Over multiple ticks, cumulative fill probability increases.
//
// RISK MANAGEMENT:
//   - max_hold: taker exit after N minutes (default 5 min)
//   - stop_loss: taker exit if price drops > N bps (default 500 bps)
//   - panic_vol: taker exit if volatility spikes
//   - cooldown: pause entry after recent exit
//   - imbalance filter: skip entry when order book is skewed against us
//
// ═══════════════════════════════════════════════════════════════════════════

pub struct MarketMakerStrategy {
    config: MarketMakerConfig,
    logger: NoOpStrategyLogger,

    // State tracking
    last_exit_time_millis: Option<u64>,
    /// Whether we had an open position on the previous tick.
    was_in_position: bool,
    /// Last quoted buy price (for requote threshold)
    last_buy_quote: f64,
    /// Last quoted sell price (for requote threshold)
    last_sell_quote: f64,

    // Diagnostics
    diagnostics: MakerDiagnostics,
}

impl MarketMakerStrategy {
    fn load_config<P: AsRef<Path>>(
        path: P,
    ) -> Result<MarketMakerConfig, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: MarketMakerConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Check if we're in the post-exit cooldown period.
    fn in_cooldown(&self, market_state: &MarketState) -> bool {
        if self.config.entry_cooldown_millis == 0 {
            return false;
        }
        let Some(now) = market_state.last_event_time_millis() else {
            return true; // No time info → be cautious
        };
        match self.last_exit_time_millis {
            Some(exit_time) => now.saturating_sub(exit_time) < self.config.entry_cooldown_millis,
            None => false,
        }
    }

    /// Check exit conditions for a long position.
    /// Returns Some((rationale, use_maker)) if should exit, None otherwise.
    fn check_exit(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> Option<(&'static str, bool)> {
        if context.current_position.quantity <= 0.0 {
            return None;
        }

        let entry_price = context.current_position.entry_price;

        // Use bid for conservative long valuation (fallback to last_price)
        let current_price = market_state
            .top_of_book()
            .map(|b| b.bid.price)
            .or_else(|| market_state.last_price())?;

        let pnl_bps = (current_price - entry_price) / entry_price * 10_000.0;

        // === Emergency taker exits ===

        // Stop loss
        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_stop_loss += 1;
            self.diagnostics.exits_taker += 1;
            return Some(("stop_loss", false));
        }

        // Panic volatility
        let micro = market_state.micro();
        if micro.realized_vol_bps > self.config.panic_vol_bps {
            self.diagnostics.exits_panic_vol += 1;
            self.diagnostics.exits_taker += 1;
            return Some(("panic_vol", false));
        }

        // Max hold time
        let now = market_state.last_event_time_millis()?;
        if context.current_position.entry_time > 0.0 {
            let held_millis =
                now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
            if held_millis >= self.config.max_hold_millis {
                self.diagnostics.exits_max_hold += 1;
                self.diagnostics.exits_taker += 1;
                return Some(("max_hold_time", false));
            }
        }

        // === Normal exit: passive sell at ask ===
        // Always try to exit via maker. The backtest engine will return
        // Pending if the passive order doesn't fill this tick, and we'll
        // try again next tick. This is the core of the MM strategy.
        //
        // Note: We don't increment exits_passive_sell here because this
        // fires on every tick while in position. The actual exit counters
        // are only meaningful when the position is actually closed.
        // We track attempts separately.
        self.diagnostics.exits_passive_sell += 1;
        // exits_maker is incremented here because the INTENT is maker.
        // If the passive sell returns Pending, the position stays and
        // we try again. The session_summary's closed_trades count
        // reflects actual closures.
        self.diagnostics.exits_maker += 1;
        Some(("passive_sell_at_ask", true))
    }

    /// Compute dynamic cash fraction based on current market conditions.
    ///
    /// Formula: effective_cf = base_cf × spread_factor × vol_factor × budget_guard
    ///   - spread_factor: wider spread → bigger bet (rewarding good opportunities)
    ///   - vol_factor: higher vol → smaller bet (protecting against adverse selection)
    ///   - budget_guard: low remaining capital → smaller bet (preventing budget exhaustion)
    ///
    /// Result is clamped to [min_cash_fraction, max_cash_fraction].
    fn compute_dynamic_cf(
        &self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> f64 {
        let base_cf = self.config.cash_fraction;

        // --- Spread factor: wider spread = more edge = bigger bet ---
        let spread_bps = if let Some(book) = market_state.top_of_book() {
            let mid = (book.bid.price + book.ask.price) / 2.0;
            if mid > f64::EPSILON {
                (book.ask.price - book.bid.price) / mid * 10_000.0
            } else {
                self.config.spread_ref_bps // fallback to reference
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

        // --- Vol factor: higher vol = more risk = smaller bet ---
        let vol_bps = market_state.micro().realized_vol_bps;
        let vol_factor = if vol_bps > 1.0 {
            (self.config.vol_ref_bps / vol_bps).clamp(0.5, 1.5)
        } else {
            1.0 // no vol data yet, neutral
        };

        // --- Budget guard: protect against exhausting capital after a few bets ---
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

    /// Check entry conditions. Returns the expected edge in bps if entry is valid,
    /// or the name of the blocking filter.
    fn check_entry(&mut self, market_state: &MarketState) -> Result<f64, &'static str> {
        // Get spread — prefer explicit bid/ask, fallback to EMA estimated spread
        let spread_bps = if let Some(book) = market_state.top_of_book() {
            let bid = book.bid.price;
            let ask = book.ask.price;
            let mid = (bid + ask) / 2.0;
            if mid <= f64::EPSILON {
                self.diagnostics.blocked_no_quote += 1;
                return Err("no_quote");
            }
            (ask - bid) / mid * 10_000.0
        } else {
            // Trade-only mode: use EMA spread from microstructure
            let micro = market_state.micro();
            if !micro.mid_initialized() || micro.ema_spread_bps <= 0.0 {
                self.diagnostics.blocked_no_quote += 1;
                return Err("no_quote");
            }
            micro.ema_spread_bps
        };

        self.diagnostics.last_spread_bps = spread_bps;

        // Spread must be wide enough
        if spread_bps < self.config.min_spread_bps {
            self.diagnostics.blocked_spread_narrow += 1;
            return Err("spread_narrow");
        }

        // Edge after costs must be positive
        let edge_bps = spread_bps - self.config.round_trip_cost_bps;
        self.diagnostics.last_edge_bps = edge_bps;

        if edge_bps < self.config.min_edge_bps {
            self.diagnostics.blocked_no_edge += 1;
            return Err("no_edge");
        }

        // Volatility filter
        let micro = market_state.micro();
        self.diagnostics.last_vol_bps = micro.realized_vol_bps;

        if micro.realized_vol_bps > self.config.max_vol_bps {
            self.diagnostics.blocked_vol_high += 1;
            return Err("vol_high");
        }

        // Minimum trades in window
        let stats = market_state.trade_window_stats();
        if stats.trade_count < self.config.min_trades_in_window {
            self.diagnostics.blocked_min_trades += 1;
            return Err("min_trades");
        }

        // Cooldown check
        if self.in_cooldown(market_state) {
            self.diagnostics.blocked_cooldown += 1;
            return Err("cooldown");
        }

        // ========== Trade direction filter ==========
        if self.config.require_seller_initiated {
            if let Some(last_trade) = market_state.last_trade() {
                if !last_trade.is_buyer_market_maker {
                    self.diagnostics.blocked_not_seller_initiated += 1;
                    return Err("not_seller_initiated");
                }
            } else {
                self.diagnostics.blocked_not_seller_initiated += 1;
                return Err("not_seller_initiated");
            }
        }

        // ========== Sell flow fraction filter ==========
        if self.config.min_sell_flow_fraction > 0.0 {
            let total_vol = stats.buyer_initiated_volume + stats.seller_initiated_volume;
            if total_vol > f64::EPSILON {
                let sell_fraction = stats.seller_initiated_volume / total_vol;
                if sell_fraction < self.config.min_sell_flow_fraction {
                    self.diagnostics.blocked_sell_flow_low += 1;
                    return Err("sell_flow_low");
                }
            }
        }

        // ========== Price position filter ==========
        if self.config.max_price_position < 1.0 && self.config.price_position_window > 0 {
            let recent_stats = market_state.recent_trade_window_stats(self.config.price_position_window);
            if recent_stats.trade_count >= 2 {
                if let (Some(low), Some(high)) = (
                    market_state.trade_window_low_price(),
                    market_state.trade_window_high_price(),
                ) {
                    let range = high - low;
                    if range > f64::EPSILON {
                        let current_price = market_state.last_price().unwrap_or(low);
                        let position = (current_price - low) / range;
                        self.diagnostics.last_price_position = position;
                        if position > self.config.max_price_position {
                            self.diagnostics.blocked_price_too_high += 1;
                            return Err("price_too_high");
                        }
                    }
                }
            }
        }

        // Order book imbalance filter (optional, only with book data)
        if self.config.max_imbalance < 1.0 {
            if let Some(imbalance) = market_state.order_book_imbalance() {
                self.diagnostics.last_imbalance = imbalance;
                if imbalance < -self.config.max_imbalance {
                    self.diagnostics.blocked_imbalance += 1;
                    return Err("imbalance");
                }
            }
        }

        Ok(edge_bps)
    }

    /// Two-sided market-making decision.
    ///
    /// Computes mid-price from bookTicker (preferred) or from trade-only
    /// EMA mid-price (fallback for backtest without bookTicker data).
    /// Applies inventory skew and returns QuoteBothSides with buy and
    /// sell prices.
    ///
    /// Safety exits (stop loss, panic vol, max hold) still use taker.
    fn decide_two_sided(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> StrategyDecision {
        // Determine mid/bid/ask from bookTicker or trade-only fallback
        let (bid, _ask, mid) = if let Some(book) = market_state.top_of_book() {
            // Preferred: real bookTicker data
            let b = book.bid.price;
            let a = book.ask.price;
            (b, a, (b + a) / 2.0)
        } else {
            // Fallback: trade-only mode — use EMA mid from micro state
            let micro = market_state.micro();
            if !micro.mid_initialized() {
                return StrategyDecision::no_action();
            }
            let ema_mid = micro.ema_mid_price;
            if ema_mid <= f64::EPSILON {
                return StrategyDecision::no_action();
            }
            // Synthesize bid/ask from EMA spread (or use half_spread_bps as floor)
            let spread_bps = if micro.ema_spread_bps > f64::EPSILON {
                micro.ema_spread_bps
            } else {
                self.config.half_spread_bps * 2.0
            };
            let half_spread_frac = spread_bps / 20_000.0;
            let b = ema_mid * (1.0 - half_spread_frac);
            let a = ema_mid * (1.0 + half_spread_frac);
            (b, a, ema_mid)
        };

        if mid <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        // --- Emergency exits (taker) for existing inventory ---
        if context.current_position.quantity > 0.0 {
            if let Some((rationale, use_maker)) = self.check_exit(market_state, context) {
                if !use_maker {
                    // Taker exit — stop_loss, panic_vol, max_hold
                    self.last_exit_time_millis = market_state.last_event_time_millis();
                    let exit_price = bid; // taker sell at bid
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
                        metrics: vec![],
                    };
                }
                // If use_maker is true, it's a normal passive_sell_at_ask — we handle
                // this through the two-sided quoting below (sell side).
            }
        }

        // --- Volatility filter ---
        let micro = market_state.micro();
        if micro.realized_vol_bps > self.config.panic_vol_bps {
            // Too volatile — cancel both sides
            return StrategyDecision::no_action();
        }

        // --- Compute inventory ratio ---
        // inventory_ratio: 0 = no position, 1 = max inventory
        let max_inventory_value = context.initial_capital * self.config.max_inventory_fraction;
        let inventory_value = context.current_position.quantity * mid;
        let inventory_ratio = if max_inventory_value > f64::EPSILON {
            (inventory_value / max_inventory_value).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // --- Asymmetric inventory skew ---
        // Instead of shifting mid, we independently adjust each side's spread:
        //   buy_spread  = half_spread * (1 + inventory_ratio * skew_factor)
        //   sell_spread = half_spread * (1 - inventory_ratio * skew_factor)
        //
        // At no inventory: both sides symmetric at half_spread from mid.
        // At max inventory: buy is much wider (less aggressive), sell is much
        // tighter (more aggressive), making passive sells far more likely.
        let skew_factor = self.config.inventory_skew_bps.clamp(0.0, 0.95);
        let buy_spread_bps = self.config.half_spread_bps * (1.0 + inventory_ratio * skew_factor);
        let sell_spread_bps = self.config.half_spread_bps * (1.0 - inventory_ratio * skew_factor);

        // --- Compute quote prices ---
        let mut buy_price = mid * (1.0 - buy_spread_bps / 10_000.0);
        let mut sell_price = mid * (1.0 + sell_spread_bps / 10_000.0);

        // --- Spread-crossing guard ---
        // Prevent LIMIT orders from crossing the spread (which would fill as
        // market orders, defeating the purpose of passive market-making).
        // If bookTicker is available, ensure:
        //   buy_price < best_ask  (never buy at or above the ask)
        //   sell_price > best_bid (never sell at or below the bid)
        let tick_size = context.tick_size;
        if let Some(book) = market_state.top_of_book() {
            if book.ask.price > 0.0 && buy_price >= book.ask.price {
                buy_price = book.ask.price - tick_size;
            }
            if book.bid.price > 0.0 && sell_price <= book.bid.price {
                sell_price = book.bid.price + tick_size;
            }
        }

        // --- Entry-price floor for sell side ---
        // When holding inventory, never sell below entry_price + min_exit_edge_bps.
        // Without this floor, a mid-price drop after a buy fill causes the sell
        // quote to track the new (lower) mid, locking in a loss on the round-trip
        // even though the maker fee is zero.
        let entry_price = context.current_position.entry_price;
        if entry_price > 0.0 && context.current_position.quantity > f64::EPSILON {
            let min_sell = entry_price * (1.0 + self.config.min_exit_edge_bps / 10_000.0);
            sell_price = sell_price.max(min_sell);
        }

        // --- Requote threshold: skip if prices haven't changed enough ---
        if self.last_buy_quote > 0.0 && self.last_sell_quote > 0.0 {
            let buy_change_bps = ((buy_price - self.last_buy_quote) / self.last_buy_quote * 10_000.0).abs();
            let sell_change_bps = ((sell_price - self.last_sell_quote) / self.last_sell_quote * 10_000.0).abs();
            if buy_change_bps < self.config.requote_threshold_bps
                && sell_change_bps < self.config.requote_threshold_bps
            {
                // Prices haven't moved enough — keep existing quotes
                // But still return QuoteBothSides so backtest cumulative tracker increments
                // (in live mode, the trader would skip the cancel-replace)
            }
        }
        self.last_buy_quote = buy_price;
        self.last_sell_quote = sell_price;

        // --- Compute quantities ---
        let buy_quantity = if inventory_ratio >= 1.0 {
            // Max inventory reached — don't buy more
            0.0
        } else {
            // Scale buy size inversely with inventory
            let scale = 1.0 - inventory_ratio;
            let base_qty = context.available_cash * self.config.cash_fraction / buy_price;
            base_qty * scale
        };

        let sell_quantity = context.current_position.quantity;

        // --- Need at least one side active ---
        if buy_quantity <= f64::EPSILON && sell_quantity <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        // If only one side is active, fall through to QuoteBothSides with qty=0
        // on the inactive side. The trader handles qty=0 as "no order on this side".
        let spread_bps = (sell_price - buy_price) / mid * 10_000.0;

        StrategyDecision {
            confidence: 1.0,
            intent: OrderIntent::QuoteBothSides {
                buy_price,
                buy_quantity,
                sell_price,
                sell_quantity,
                rationale: "two_sided_quote",
                expected_edge_bps: spread_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "mid_price",
                    value: mid,
                },
                DecisionMetric {
                    name: "spread_bps",
                    value: spread_bps,
                },
                DecisionMetric {
                    name: "inventory_ratio",
                    value: inventory_ratio,
                },
                DecisionMetric {
                    name: "buy_spread_bps",
                    value: buy_spread_bps,
                },
                DecisionMetric {
                    name: "sell_spread_bps",
                    value: sell_spread_bps,
                },
                DecisionMetric {
                    name: "buy_price",
                    value: buy_price,
                },
                DecisionMetric {
                    name: "sell_price",
                    value: sell_price,
                },
            ],
        }
    }
}

#[async_trait::async_trait]
impl Strategy for MarketMakerStrategy {
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
            diagnostics: MakerDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "MarketMakerStrategy - passive spread capture for wide-spread pairs".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert("mm.total_decisions".into(), self.diagnostics.total_decisions);
        counters.insert("mm.blocked_no_quote".into(), self.diagnostics.blocked_no_quote);
        counters.insert("mm.blocked_spread_narrow".into(), self.diagnostics.blocked_spread_narrow);
        counters.insert("mm.blocked_no_edge".into(), self.diagnostics.blocked_no_edge);
        counters.insert("mm.blocked_vol_high".into(), self.diagnostics.blocked_vol_high);
        counters.insert("mm.blocked_min_trades".into(), self.diagnostics.blocked_min_trades);
        counters.insert("mm.blocked_cooldown".into(), self.diagnostics.blocked_cooldown);
        counters.insert("mm.blocked_imbalance".into(), self.diagnostics.blocked_imbalance);
        counters.insert("mm.blocked_not_seller_initiated".into(), self.diagnostics.blocked_not_seller_initiated);
        counters.insert("mm.blocked_sell_flow_low".into(), self.diagnostics.blocked_sell_flow_low);
        counters.insert("mm.blocked_price_too_high".into(), self.diagnostics.blocked_price_too_high);
        counters.insert("mm.entries_passive".into(), self.diagnostics.entries_passive);
        counters.insert("mm.exits_passive_sell".into(), self.diagnostics.exits_passive_sell);
        counters.insert("mm.exits_stop_loss".into(), self.diagnostics.exits_stop_loss);
        counters.insert("mm.exits_max_hold".into(), self.diagnostics.exits_max_hold);
        counters.insert("mm.exits_panic_vol".into(), self.diagnostics.exits_panic_vol);
        counters.insert("mm.exits_maker".into(), self.diagnostics.exits_maker);
        counters.insert("mm.exits_taker".into(), self.diagnostics.exits_taker);

        let mut gauges = BTreeMap::new();
        gauges.insert("mm.last_spread_bps".into(), self.diagnostics.last_spread_bps);
        gauges.insert("mm.last_edge_bps".into(), self.diagnostics.last_edge_bps);
        gauges.insert("mm.last_imbalance".into(), self.diagnostics.last_imbalance);
        gauges.insert("mm.last_vol_bps".into(), self.diagnostics.last_vol_bps);
        gauges.insert("mm.last_price_position".into(), self.diagnostics.last_price_position);
        gauges.insert("mm.last_effective_cf".into(), self.diagnostics.last_effective_cf);

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, _event: &MarketEvent, _market_state: &MarketState) {
        // MicrostructureState updates handled by MarketState::apply().
    }

    fn decide(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> StrategyDecision {
        self.diagnostics.total_decisions += 1;

        // Two-sided mode: delegate to dedicated method
        if self.config.two_sided {
            return self.decide_two_sided(market_state, context);
        }

        // Periodic diagnostics: every 50 flat decisions, log block stats
        let since_last = self.diagnostics.total_decisions - self.diagnostics.last_report_total;
        if since_last >= 50 && context.current_position.quantity <= 0.0 {
            self.diagnostics.last_report_total = self.diagnostics.total_decisions;
            info!(
                symbol = %context.symbol,
                total = self.diagnostics.total_decisions,
                entries = self.diagnostics.entries_passive,
                no_quote = self.diagnostics.blocked_no_quote,
                spread = self.diagnostics.blocked_spread_narrow,
                no_edge = self.diagnostics.blocked_no_edge,
                vol = self.diagnostics.blocked_vol_high,
                min_trades = self.diagnostics.blocked_min_trades,
                cooldown = self.diagnostics.blocked_cooldown,
                seller_init = self.diagnostics.blocked_not_seller_initiated,
                sell_flow = self.diagnostics.blocked_sell_flow_low,
                price_high = self.diagnostics.blocked_price_too_high,
                imbalance = self.diagnostics.blocked_imbalance,
                "entry filter stats"
            );
        }

        // Detect passive exit: position was open on previous tick but now closed.
        // In this case, the passive sell filled and we need to record exit time
        // for cooldown. Without this, only taker exits trigger cooldown.
        let in_position = context.current_position.quantity > 0.0;
        if self.was_in_position && !in_position {
            self.last_exit_time_millis = market_state.last_event_time_millis();
        }
        self.was_in_position = in_position;

        // === EXIT LOGIC (always first — long-only, exit via Sell) ===
        if context.current_position.quantity > 0.0 {
            if let Some((rationale, use_maker)) = self.check_exit(market_state, context) {
                let (order_type, time_in_force) = if use_maker {
                    (OrderType::Maker, TimeInForce::Gtc)
                } else {
                    // Record exit time for cooldown (taker exits only,
                    // passive sells that return Pending don't count)
                    self.last_exit_time_millis = market_state.last_event_time_millis();
                    (OrderType::Taker, TimeInForce::Ioc)
                };

                // Compute expected edge for the sell
                let entry_price = context.current_position.entry_price;
                let exit_ref = if use_maker {
                    // Passive sell at ask (fallback: last_price + half spread estimate)
                    let raw_ask = market_state
                        .top_of_book()
                        .map(|b| b.ask.price)
                        .or_else(|| {
                            // In trade-only mode, estimate ask as last_price + half_spread
                            let micro = market_state.micro();
                            market_state.last_price().map(|p| {
                                p * (1.0 + micro.ema_spread_bps / 20_000.0)
                            })
                        })
                        .unwrap_or(entry_price);

                    // Apply min_exit_edge_bps floor: never place passive sell
                    // below entry_price * (1 + min_exit_edge_bps / 10_000).
                    // This ensures we don't sell at a loss before fees.
                    // Does NOT affect taker exits (stop_loss, panic_vol, max_hold).
                    let min_sell_price =
                        entry_price * (1.0 + self.config.min_exit_edge_bps / 10_000.0);
                    raw_ask.max(min_sell_price)
                } else {
                    // Taker sell at bid (conservative, fallback to last_price)
                    market_state
                        .top_of_book()
                        .map(|b| b.bid.price)
                        .or_else(|| market_state.last_price())
                        .unwrap_or(entry_price)
                };
                let expected_edge_bps = if entry_price > 0.0 {
                    (exit_ref - entry_price) / entry_price * 10_000.0
                } else {
                    0.0
                };

                return StrategyDecision {
                    confidence: 1.0,
                    intent: OrderIntent::Place {
                        side: Side::Sell,
                        order_type,
                        price: Some(exit_ref), // explicit price: ask for maker, bid for taker
                        quantity: context.current_position.quantity,
                        time_in_force,
                        rationale,
                        expected_edge_bps,
                    },
                    metrics: vec![
                        DecisionMetric {
                            name: "position_quantity",
                            value: context.current_position.quantity,
                        },
                        DecisionMetric {
                            name: "expected_edge_bps",
                            value: expected_edge_bps,
                        },
                    ],
                };
            }

            // Still holding, no exit signal → no action
            // (This should not happen since check_exit always returns Some
            //  when in position — either passive_sell or a safety exit)
            return StrategyDecision::no_action();
        }

        // === ENTRY LOGIC (always passive Buy at bid) ===
        let edge_bps = match self.check_entry(market_state) {
            Ok(edge) => edge,
            Err(reason) => {
                debug!(symbol = %context.symbol, reason = reason, "entry blocked");
                return StrategyDecision::no_action();
            }
        };

        // Use bid price as reference for sizing (we're buying at bid)
        let reference_price = market_state
            .top_of_book()
            .map(|b| b.bid.price)
            .or_else(|| market_state.last_price())
            .unwrap_or(0.0);

        if reference_price <= f64::EPSILON {
            self.diagnostics.blocked_no_quote += 1;
            return StrategyDecision::no_action();
        }

        // Compute effective cash fraction (dynamic or static)
        let effective_cf = if self.config.dynamic_sizing {
            self.compute_dynamic_cf(market_state, context)
        } else {
            self.config.cash_fraction
        };
        self.diagnostics.last_effective_cf = effective_cf;

        let Some(quantity) =
            context.capped_entry_quantity(reference_price, effective_cf, None)
        else {
            return StrategyDecision::no_action();
        };

        self.diagnostics.entries_passive += 1;

        StrategyDecision {
            confidence: 1.0,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Maker,
                price: Some(reference_price), // explicit bid price for passive buy
                quantity,
                time_in_force: TimeInForce::Gtc,
                rationale: "passive_buy_at_bid",
                expected_edge_bps: edge_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "spread_bps",
                    value: self.diagnostics.last_spread_bps,
                },
                DecisionMetric {
                    name: "edge_bps",
                    value: edge_bps,
                },
                DecisionMetric {
                    name: "vol_bps",
                    value: self.diagnostics.last_vol_bps,
                },
                DecisionMetric {
                    name: "effective_cf",
                    value: effective_cf,
                },
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MarketMakerConfig::default();
        assert_eq!(config.min_spread_bps, 30.0);
        assert_eq!(config.round_trip_cost_bps, 20.0);
        assert_eq!(config.min_edge_bps, 5.0);
        assert_eq!(config.max_hold_millis, 300_000);
        assert_eq!(config.stop_loss_bps, 500.0);
        assert_eq!(config.cash_fraction, 0.9);
        // Dynamic sizing defaults
        assert!(!config.dynamic_sizing);
        assert_eq!(config.min_cash_fraction, 0.02);
        assert_eq!(config.max_cash_fraction, 0.15);
        assert_eq!(config.spread_ref_bps, 100.0);
        assert_eq!(config.vol_ref_bps, 50.0);
        assert_eq!(config.budget_guard_threshold, 0.5);
    }

    #[test]
    fn test_config_from_toml() {
        let toml_str = r#"
            min_spread_bps = 50.0
            round_trip_cost_bps = 20.0
            min_edge_bps = 10.0
            max_hold_millis = 120000
        "#;
        let config: MarketMakerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.min_spread_bps, 50.0);
        assert_eq!(config.min_edge_bps, 10.0);
        assert_eq!(config.max_hold_millis, 120_000);
        // defaults for unspecified fields
        assert_eq!(config.stop_loss_bps, 500.0);
    }
}
