// ═══════════════════════════════════════════════════════════════════════════
// MarketMakerBidAskGlmStrategy — Two-sided bid/ask market maker
// ═══════════════════════════════════════════════════════════════════════════
//
// CORE IDEA:
//   Continuously quote both sides (bid + ask) to capture the spread.
//   Profit = spread earned - adverse selection losses.
//
// KEY FEATURES:
//   1. Dynamic half-spread scaled by realized volatility
//   2. Inventory skew: widen buy / tighten sell when holding inventory
//   3. Entry-price floor: never sell below entry + min_exit_edge_bps
//   4. Time decay on exit floor: progressively reduce sell floor as
//      hold time approaches max_hold
//   5. Emergency taker exits: stop-loss, panic-vol, max-hold
//   6. Order book imbalance filter: skip buying when ask-heavy
//   7. Trade flow filter: skip buying when seller flow dominates
//   8. Dynamic sizing: vol-adjusted, spread-adjusted, budget-guarded
//   9. Cooldown after losing taker exit
//
// ═══════════════════════════════════════════════════════════════════════════

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use tracing::info;
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
pub struct MarketMakerBidAskGlmConfig {
    // --- Spread & pricing ---

    /// Base half-spread in bps. Buy at mid - half_spread, sell at mid + half_spread.
    #[serde(default = "default_base_half_spread_bps")]
    pub base_half_spread_bps: f64,

    /// Minimum half-spread clamp (prevents quoting too tight).
    #[serde(default = "default_min_half_spread_bps")]
    pub min_half_spread_bps: f64,

    /// Maximum half-spread clamp (prevents quoting too wide).
    #[serde(default = "default_max_half_spread_bps")]
    pub max_half_spread_bps: f64,

    /// Reference volatility for spread scaling (bps).
    /// vol_multiplier = clamp(realized_vol / reference, min, max).
    #[serde(default = "default_volatility_reference_bps")]
    pub volatility_reference_bps: f64,

    #[serde(default = "default_volatility_multiplier_min")]
    pub volatility_multiplier_min: f64,

    #[serde(default = "default_volatility_multiplier_max")]
    pub volatility_multiplier_max: f64,

    /// Minimum quoted full spread (buy-to-sell) in bps to emit a quote.
    #[serde(default = "default_min_edge_bps")]
    pub min_edge_bps: f64,

    // --- Inventory management ---

    /// Maximum inventory as fraction of initial_capital [0, 1].
    #[serde(default = "default_max_inventory_fraction")]
    pub max_inventory_fraction: f64,

    /// Inventory skew factor [0, 0.95]. Controls asymmetry when holding.
    /// At max inventory with skew=0.8: buy side widens by 1.8x, sell tightens to 0.2x.
    #[serde(default = "default_inventory_skew_factor")]
    pub inventory_skew_factor: f64,

    // --- Sizing ---

    /// Base cash fraction per entry [0, 1].
    #[serde(default = "default_cash_fraction")]
    pub cash_fraction: f64,

    /// Enable dynamic sizing (vol, spread, budget adjustments).
    #[serde(default)]
    pub dynamic_sizing: bool,

    /// Min cash fraction floor (dynamic sizing).
    #[serde(default = "default_min_cash_fraction")]
    pub min_cash_fraction: f64,

    /// Max cash fraction ceiling (dynamic sizing).
    #[serde(default = "default_max_cash_fraction")]
    pub max_cash_fraction: f64,

    /// Reference spread for normalization in dynamic sizing (bps).
    #[serde(default = "default_spread_ref_bps")]
    pub spread_ref_bps: f64,

    /// Reference vol for normalization in dynamic sizing (bps).
    #[serde(default = "default_vol_ref_bps")]
    pub vol_ref_bps: f64,

    /// Fraction of initial capital below which to reduce sizing.
    #[serde(default = "default_budget_guard_threshold")]
    pub budget_guard_threshold: f64,

    // --- Exit pricing ---

    /// Minimum edge in bps for passive sell relative to entry price.
    /// sell_price >= entry * (1 + min_exit_edge_bps / 10_000).
    /// Does NOT affect taker emergency exits.
    #[serde(default = "default_min_exit_edge_bps")]
    pub min_exit_edge_bps: f64,

    // --- Emergency exits ---

    /// Stop loss in bps. Taker exit if unrealized PnL drops below this.
    #[serde(default = "default_stop_loss_bps")]
    pub stop_loss_bps: f64,

    /// Panic volatility threshold. Taker exit if vol exceeds this while in position.
    #[serde(default = "default_panic_vol_bps")]
    pub panic_vol_bps: f64,

    /// Maximum hold time in milliseconds. Taker exit after this.
    #[serde(default = "default_max_hold_millis")]
    pub max_hold_millis: u64,

    /// Maximum volatility to allow new entries (quote both sides).
    /// Above this, only sell side is active (unwind inventory).
    #[serde(default = "default_max_vol_bps")]
    pub max_vol_bps: f64,

    // --- Cooldown ---

    /// Cooldown after taker exit in milliseconds. No new buys during cooldown.
    #[serde(default = "default_cooldown_millis")]
    pub cooldown_millis: u64,

    // --- Adverse selection filters ---

    /// Maximum order book imbalance to allow buying [-1, 1].
    /// Negative = ask-heavy (more selling pressure). Skip buying if imbalance < -max.
    #[serde(default = "default_max_imbalance")]
    pub max_imbalance: f64,

    /// Minimum fraction of seller-initiated volume in recent window [0, 1].
    /// Higher = require more selling pressure before entering.
    /// 0.0 = disabled.
    #[serde(default = "default_min_sell_flow_fraction")]
    pub min_sell_flow_fraction: f64,

    /// Minimum trades in the window before allowing entry.
    #[serde(default = "default_min_trades_in_window")]
    pub min_trades_in_window: usize,

    // --- Requoting ---

    /// Minimum price change in bps before re-quoting. Prevents API spam.
    #[serde(default = "default_requote_threshold_bps")]
    pub requote_threshold_bps: f64,

    // --- Market state window ---

    /// Window in milliseconds for trade_window_stats / microstructure metrics.
    #[serde(default = "default_trade_window_millis")]
    pub trade_window_millis: u64,
}

fn default_base_half_spread_bps() -> f64 { 2.0 }
fn default_min_half_spread_bps() -> f64 { 0.5 }
fn default_max_half_spread_bps() -> f64 { 10.0 }
fn default_volatility_reference_bps() -> f64 { 50.0 }
fn default_volatility_multiplier_min() -> f64 { 0.5 }
fn default_volatility_multiplier_max() -> f64 { 4.0 }
fn default_min_edge_bps() -> f64 { 0.5 }
fn default_max_inventory_fraction() -> f64 { 0.20 }
fn default_inventory_skew_factor() -> f64 { 0.8 }
fn default_cash_fraction() -> f64 { 0.10 }
fn default_min_cash_fraction() -> f64 { 0.02 }
fn default_max_cash_fraction() -> f64 { 0.15 }
fn default_spread_ref_bps() -> f64 { 5.0 }
fn default_vol_ref_bps() -> f64 { 50.0 }
fn default_budget_guard_threshold() -> f64 { 0.5 }
fn default_min_exit_edge_bps() -> f64 { 0.0 }
fn default_stop_loss_bps() -> f64 { 500.0 }
fn default_panic_vol_bps() -> f64 { 2000.0 }
fn default_max_hold_millis() -> u64 { 300_000 } // 5 minutes
fn default_max_vol_bps() -> f64 { 800.0 }
fn default_cooldown_millis() -> u64 { 5_000 }
fn default_max_imbalance() -> f64 { 1.0 }  // 1.0 = disabled
fn default_min_sell_flow_fraction() -> f64 { 0.0 }  // disabled
fn default_min_trades_in_window() -> usize { 0 }
fn default_requote_threshold_bps() -> f64 { 0.3 }
fn default_trade_window_millis() -> u64 { 30_000 }

impl Default for MarketMakerBidAskGlmConfig {
    fn default() -> Self {
        Self {
            base_half_spread_bps: default_base_half_spread_bps(),
            min_half_spread_bps: default_min_half_spread_bps(),
            max_half_spread_bps: default_max_half_spread_bps(),
            volatility_reference_bps: default_volatility_reference_bps(),
            volatility_multiplier_min: default_volatility_multiplier_min(),
            volatility_multiplier_max: default_volatility_multiplier_max(),
            min_edge_bps: default_min_edge_bps(),
            max_inventory_fraction: default_max_inventory_fraction(),
            inventory_skew_factor: default_inventory_skew_factor(),
            cash_fraction: default_cash_fraction(),
            dynamic_sizing: false,
            min_cash_fraction: default_min_cash_fraction(),
            max_cash_fraction: default_max_cash_fraction(),
            spread_ref_bps: default_spread_ref_bps(),
            vol_ref_bps: default_vol_ref_bps(),
            budget_guard_threshold: default_budget_guard_threshold(),
            min_exit_edge_bps: default_min_exit_edge_bps(),
            stop_loss_bps: default_stop_loss_bps(),
            panic_vol_bps: default_panic_vol_bps(),
            max_hold_millis: default_max_hold_millis(),
            max_vol_bps: default_max_vol_bps(),
            cooldown_millis: default_cooldown_millis(),
            max_imbalance: default_max_imbalance(),
            min_sell_flow_fraction: default_min_sell_flow_fraction(),
            min_trades_in_window: default_min_trades_in_window(),
            requote_threshold_bps: default_requote_threshold_bps(),
            trade_window_millis: default_trade_window_millis(),
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct GlmDiagnostics {
    // Decision counters
    total_decisions: usize,
    blocked_no_quote: usize,
    blocked_vol_high: usize,
    blocked_no_edge: usize,
    blocked_cooldown: usize,
    blocked_imbalance: usize,
    blocked_sell_flow: usize,
    blocked_min_trades: usize,

    // Exit counters
    exits_stop_loss: usize,
    exits_panic_vol: usize,
    exits_max_hold: usize,

    // Gauges
    last_mid_price: f64,
    last_half_spread_bps: f64,
    last_buy_price: f64,
    last_sell_price: f64,
    last_buy_quantity: f64,
    last_sell_quantity: f64,
    last_inventory_ratio: f64,
    last_quoted_spread_bps: f64,
    last_effective_cf: f64,
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

pub struct MarketMakerBidAskGlmStrategy {
    config: MarketMakerBidAskGlmConfig,
    logger: NoOpStrategyLogger,
    diagnostics: GlmDiagnostics,
    last_buy_quote: f64,
    last_sell_quote: f64,
    /// Timestamp of the last taker exit (for cooldown).
    last_exit_time_millis: Option<u64>,
    /// Slow EMA of mid price for trend detection.
    /// Alpha = 0.002 → halflife ≈ 346 ticks ≈ 35 seconds at 100ms/tick.
    /// Used to detect falling markets: if mid < slow_ema, price is trending down.
    slow_ema_mid: f64,
    /// Whether the trend filter was blocking buys on the previous tick (for change-detection logging).
    trend_filter_was_active: bool,
}

impl MarketMakerBidAskGlmStrategy {
    fn load_config<P: AsRef<Path>>(
        path: P,
    ) -> Result<MarketMakerBidAskGlmConfig, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: MarketMakerBidAskGlmConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Extract fair price and best bid/ask from bookTicker or EMA fallback.
    fn fair_price_and_book(&self, market_state: &MarketState) -> Option<(f64, f64, f64)> {
        if let Some(book) = market_state.top_of_book() {
            let bid = book.bid.price;
            let ask = book.ask.price;
            let fair = (bid + ask) / 2.0;
            if bid > f64::EPSILON && ask > bid && fair > f64::EPSILON {
                return Some((bid, ask, fair));
            }
        }

        let micro = market_state.micro();
        if !micro.mid_initialized() || micro.ema_mid_price <= f64::EPSILON {
            return None;
        }

        let fair = micro.ema_mid_price;
        let spread_bps = micro
            .ema_spread_bps
            .max(self.config.base_half_spread_bps * 2.0);
        let half_spread_frac = spread_bps / 20_000.0;
        let bid = fair * (1.0 - half_spread_frac);
        let ask = fair * (1.0 + half_spread_frac);
        Some((bid, ask, fair))
    }

    /// Compute dynamic half-spread based on realized volatility.
    fn dynamic_half_spread_bps(&self, market_state: &MarketState) -> f64 {
        let vol_bps = market_state.micro().realized_vol_bps.max(1.0);
        let reference = self.config.volatility_reference_bps.max(1.0);
        let vol_multiplier = (vol_bps / reference).clamp(
            self.config.volatility_multiplier_min,
            self.config.volatility_multiplier_max,
        );
        (self.config.base_half_spread_bps * vol_multiplier).clamp(
            self.config.min_half_spread_bps,
            self.config.max_half_spread_bps,
        )
    }

    /// Current inventory ratio [0, 1]. 0 = empty, 1 = max inventory.
    fn compute_inventory_ratio(&self, context: &StrategyContext, mid_price: f64) -> f64 {
        let max_inv = self.max_inventory_notional(context);
        if max_inv <= f64::EPSILON {
            return 0.0;
        }
        let current = context.current_position.quantity * mid_price;
        (current / max_inv).clamp(0.0, 1.0)
    }

    /// Maximum inventory in notional terms.
    /// Uses only `initial_capital * max_inventory_fraction` — NOT capped by
    /// `max_position_notional` (which is the per-ORDER size limit, not the
    /// overall inventory capacity for the MM strategy).
    fn max_inventory_notional(&self, context: &StrategyContext) -> f64 {
        if context.initial_capital > f64::EPSILON {
            context.initial_capital * self.config.max_inventory_fraction
        } else if context.max_position_notional > f64::EPSILON {
            context.max_position_notional
        } else {
            0.0
        }
    }

    /// Check if we're in the post-exit cooldown period.
    fn in_cooldown(&self, market_state: &MarketState) -> bool {
        if self.config.cooldown_millis == 0 {
            return false;
        }
        let Some(now) = market_state.last_event_time_millis() else {
            return true;
        };
        match self.last_exit_time_millis {
            Some(exit_time) => now.saturating_sub(exit_time) < self.config.cooldown_millis,
            None => false,
        }
    }

    /// Check buy-side entry filters (adverse selection protection).
    /// Returns true if buying is allowed.
    fn check_buy_filters(&mut self, market_state: &MarketState) -> bool {
        // Order book imbalance filter
        if self.config.max_imbalance < 1.0 {
            if let Some(imbalance) = market_state.order_book_imbalance() {
                // Negative imbalance = ask-heavy = selling pressure = bad for buying
                if imbalance < -self.config.max_imbalance {
                    self.diagnostics.blocked_imbalance += 1;
                    return false;
                }
            }
        }

        // Trade flow filter
        let stats = market_state.trade_window_stats();
        if self.config.min_trades_in_window > 0 && stats.trade_count < self.config.min_trades_in_window {
            self.diagnostics.blocked_min_trades += 1;
            return false;
        }

        if self.config.min_sell_flow_fraction > 0.0 {
            let total_vol = stats.buyer_initiated_volume + stats.seller_initiated_volume;
            if total_vol > f64::EPSILON {
                let sell_frac = stats.seller_initiated_volume / total_vol;
                if sell_frac < self.config.min_sell_flow_fraction {
                    self.diagnostics.blocked_sell_flow += 1;
                    return false;
                }
            }
        }

        // Cooldown
        if self.in_cooldown(market_state) {
            self.diagnostics.blocked_cooldown += 1;
            return false;
        }

        true
    }

    /// Check for emergency taker exits on existing inventory.
    /// Returns Some((rationale, exit_price)) if should exit, None otherwise.
    fn check_emergency_exit(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
        bid_price: f64,
    ) -> Option<(&'static str, f64)> {
        if context.current_position.quantity <= f64::EPSILON {
            return None;
        }

        let entry_price = context.current_position.entry_price;
        if entry_price <= f64::EPSILON {
            return None;
        }

        // Use bid for conservative long valuation
        let valuation_price = market_state
            .top_of_book()
            .map(|b| b.bid.price)
            .unwrap_or(bid_price);

        let pnl_bps = (valuation_price - entry_price) / entry_price * 10_000.0;

        // Stop loss
        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_stop_loss += 1;
            return Some(("stop_loss", valuation_price));
        }

        // Panic volatility
        let vol_bps = market_state.micro().realized_vol_bps;
        if vol_bps > self.config.panic_vol_bps {
            self.diagnostics.exits_panic_vol += 1;
            return Some(("panic_vol", valuation_price));
        }

        // Max hold time
        if let Some(now) = market_state.last_event_time_millis() {
            if context.current_position.entry_time > 0.0 && self.config.max_hold_millis > 0 {
                let held_millis =
                    now.saturating_sub((context.current_position.entry_time * 1000.0) as u64);
                if held_millis >= self.config.max_hold_millis {
                    self.diagnostics.exits_max_hold += 1;
                    return Some(("max_hold_time", valuation_price));
                }
            }
        }

        None
    }

}

// ---------------------------------------------------------------------------
// Strategy trait implementation
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl Strategy for MarketMakerBidAskGlmStrategy {
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
            diagnostics: GlmDiagnostics::default(),
            last_buy_quote: 0.0,
            last_sell_quote: 0.0,
            last_exit_time_millis: None,
            slow_ema_mid: 0.0,
            trend_filter_was_active: false,
        })
    }

    fn get_info(&self) -> String {
        "MarketMakerBidAskGlmStrategy - two-sided bid/ask maker with adverse selection filters"
            .to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert("glm.total_decisions".into(), self.diagnostics.total_decisions);
        counters.insert("glm.blocked_no_quote".into(), self.diagnostics.blocked_no_quote);
        counters.insert("glm.blocked_vol_high".into(), self.diagnostics.blocked_vol_high);
        counters.insert("glm.blocked_no_edge".into(), self.diagnostics.blocked_no_edge);
        counters.insert("glm.blocked_cooldown".into(), self.diagnostics.blocked_cooldown);
        counters.insert("glm.blocked_imbalance".into(), self.diagnostics.blocked_imbalance);
        counters.insert("glm.blocked_sell_flow".into(), self.diagnostics.blocked_sell_flow);
        counters.insert("glm.blocked_min_trades".into(), self.diagnostics.blocked_min_trades);
        counters.insert("glm.exits_stop_loss".into(), self.diagnostics.exits_stop_loss);
        counters.insert("glm.exits_panic_vol".into(), self.diagnostics.exits_panic_vol);
        counters.insert("glm.exits_max_hold".into(), self.diagnostics.exits_max_hold);

        let mut gauges = BTreeMap::new();
        gauges.insert("glm.last_mid_price".into(), self.diagnostics.last_mid_price);
        gauges.insert("glm.last_half_spread_bps".into(), self.diagnostics.last_half_spread_bps);
        gauges.insert("glm.last_buy_price".into(), self.diagnostics.last_buy_price);
        gauges.insert("glm.last_sell_price".into(), self.diagnostics.last_sell_price);
        gauges.insert("glm.last_buy_quantity".into(), self.diagnostics.last_buy_quantity);
        gauges.insert("glm.last_sell_quantity".into(), self.diagnostics.last_sell_quantity);
        gauges.insert("glm.last_inventory_ratio".into(), self.diagnostics.last_inventory_ratio);
        gauges.insert("glm.last_quoted_spread_bps".into(), self.diagnostics.last_quoted_spread_bps);
        gauges.insert("glm.last_effective_cf".into(), self.diagnostics.last_effective_cf);

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, _event: &MarketEvent, _market_state: &MarketState) {}

    fn decide(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> StrategyDecision {
        self.diagnostics.total_decisions += 1;

        // --- Get market data ---
        let Some((best_bid, best_ask, mid_price)) = self.fair_price_and_book(market_state) else {
            self.diagnostics.blocked_no_quote += 1;
            return StrategyDecision::no_action();
        };

        // --- Update slow EMA for trend detection ---
        // Alpha = 0.002 → halflife ≈ 346 ticks ≈ 35 seconds at 100ms/tick.
        const SLOW_EMA_ALPHA: f64 = 0.002;
        if self.slow_ema_mid <= f64::EPSILON {
            self.slow_ema_mid = mid_price;
        } else {
            self.slow_ema_mid = SLOW_EMA_ALPHA * mid_price + (1.0 - SLOW_EMA_ALPHA) * self.slow_ema_mid;
        }

        // --- Emergency taker exits (always checked first) ---
        if let Some((rationale, exit_price)) =
            self.check_emergency_exit(market_state, context, best_bid)
        {
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
                metrics: vec![
                    DecisionMetric {
                        name: "exit_price",
                        value: exit_price,
                    },
                    DecisionMetric {
                        name: "position_quantity",
                        value: context.current_position.quantity,
                    },
                ],
            };
        }

        // --- Volatility gate: too volatile → cancel both sides ---
        let micro = market_state.micro();
        if micro.realized_vol_bps > self.config.panic_vol_bps {
            return StrategyDecision::no_action();
        }

        // --- Avellaneda-Stoikov reservation price & inventory skewing ---
        //
        // Classic A-S model:
        //   reservation_price = mid - q * γ * σ² * τ
        //   optimal_spread = γ * σ² * τ + (2/γ) * ln(1 + γ/k)
        //
        // We simplify by using:
        //   q_normalized = inventory_ratio mapped to [-1, 1] where 0 = half-capacity
        //   γ (gamma) = inventory_skew_factor (risk aversion)
        //   σ = realized_vol in price terms
        //   τ = 1.0 (continuous market-making, no expiry)
        //
        // The reservation price shifts the "fair value" away from mid:
        //   - Long inventory → reservation price BELOW mid → tighter ask, wider bid
        //   - No inventory → reservation price = mid → symmetric quotes
        //
        // Both bid and ask are placed symmetrically around reservation_price,
        // not around mid_price. This is the key A-S insight.

        let inventory_ratio = self.compute_inventory_ratio(context, mid_price);
        // gamma (risk aversion): controls how aggressively we skew quotes
        // to mean-revert inventory. For DOGE/FDUSD with ~50 bps vol:
        // gamma=15 → ~0.35 bps shift (too weak), gamma=200 → ~4 bps shift (strong).
        // Combined with conditional join-best-price and inventory gate at 0.85.
        let gamma = self.config.inventory_skew_factor.max(0.0);

        // q_normalized: 0 (empty) → -1, 0.5 (half) → 0, 1.0 (full) → +1
        let q_normalized = 2.0 * inventory_ratio - 1.0;

        // σ in price terms (convert from bps)
        let vol_price = micro.realized_vol_bps / 10_000.0 * mid_price;

        // Reservation price shift: Δr = q * γ * σ²
        // With τ=1, this gives the inventory-adjusted fair value
        let reservation_shift = q_normalized * gamma * vol_price * vol_price;
        let reservation_price = mid_price - reservation_shift;

        // --- Compute dynamic half-spread (applied symmetrically around reservation price) ---
        let half_spread_bps = self.dynamic_half_spread_bps(market_state);

        // Both sides use the SAME half-spread, but from reservation_price (not mid)
        let mut buy_price = reservation_price * (1.0 - half_spread_bps / 10_000.0);
        let mut sell_price = reservation_price * (1.0 + half_spread_bps / 10_000.0);

        let tick_size = context.tick_size.max(f64::EPSILON);

        // --- Join-best-price: clamp quotes to best bid/ask ---
        // On tight-spread pairs (~1 bps), quoting away from the touch
        // means sitting behind the queue and rarely getting filled.
        // Pull buy up to best_bid and sell down to best_ask so we
        // compete at the top of the book.
        //
        // BID side: Only join best_bid when inventory is BELOW neutral (ratio < 0.50).
        // After rebalance (ratio ~0.50), bids barely join. As inventory grows above
        // 0.50, the A-S reservation price + inventory-aware offset push bids away
        // from best_bid, reducing aggressive buying.
        //
        // ASK side: Always join best_ask to maximize fill probability on sells.
        if buy_price < best_bid && inventory_ratio < 0.50 {
            buy_price = best_bid;
        }
        if sell_price > best_ask {
            sell_price = best_ask;
        }

        // --- Inventory-aware bid offset ---
        // When overweight (ratio > 0.55), push bid DOWN by extra bps proportional
        // to the excess.  At ratio = 0.70 → extra offset = 0.15 * 10 = 1.5 bps.
        // This stacks on top of the A-S reservation price shift.
        if inventory_ratio > 0.55 {
            let excess = inventory_ratio - 0.55;
            let extra_offset_bps = excess * 10.0; // 10 bps per 0.10 ratio above 0.55
            buy_price *= 1.0 - extra_offset_bps / 10_000.0;
        }

        // --- Spread-crossing guard ---
        if buy_price >= best_ask {
            buy_price = best_ask - tick_size;
        }
        if sell_price <= best_bid {
            sell_price = best_bid + tick_size;
        }
        if sell_price <= buy_price {
            sell_price = buy_price + tick_size;
        }

        // --- Entry-price floor: DISABLED for MM ---
        // In a two-sided MM strategy with flat sizing, there is no directional
        // "position" to protect.  The entry-price floor was preventing ASK fills
        // whenever the market dipped even 1 tick below the last buy fill, causing
        // the persistent BID-heavy fill imbalance (4:1 ratio).
        // A-S reservation price + spread skew handle inventory risk instead.

        // --- Minimum edge check ---
        let quoted_spread_bps = (sell_price - buy_price) / mid_price * 10_000.0;
        if quoted_spread_bps < self.config.min_edge_bps * 2.0 {
            self.diagnostics.blocked_no_edge += 1;
            return StrategyDecision::no_action();
        }

        // --- Buy-side filters (adverse selection protection) ---
        let buy_allowed = if micro.realized_vol_bps > self.config.max_vol_bps {
            self.diagnostics.blocked_vol_high += 1;
            false
        } else if !self.check_buy_filters(market_state) {
            false
        } else {
            // --- Trend-aware inventory gate ---
            // When price is falling (mid < slow_ema, which has ~35s halflife)
            // AND we already hold significantly above-neutral inventory, block buys.
            // Uses hysteresis to prevent rapid toggling:
            //   - Activate at ratio > 0.65 (significantly overweight)
            //   - Deactivate at ratio < 0.55 (back near neutral)
            let price_falling = mid_price < self.slow_ema_mid;
            let trend_should_block = if self.trend_filter_was_active {
                // Already blocking → keep blocking until ratio drops below 0.55
                price_falling && inventory_ratio > 0.55
            } else {
                // Not blocking → only start blocking above 0.65
                price_falling && inventory_ratio > 0.65
            };
            if trend_should_block {
                if !self.trend_filter_was_active {
                    info!(
                        mid = format!("{:.5}", mid_price),
                        slow_ema = format!("{:.5}", self.slow_ema_mid),
                        inv_ratio = format!("{:.3}", inventory_ratio),
                        "Trend filter: blocking buys (price falling + inventory above neutral)"
                    );
                    self.trend_filter_was_active = true;
                }
                self.diagnostics.blocked_vol_high += 1; // reuse counter
                false
            } else {
                if self.trend_filter_was_active {
                    info!(
                        mid = format!("{:.5}", mid_price),
                        slow_ema = format!("{:.5}", self.slow_ema_mid),
                        inv_ratio = format!("{:.3}", inventory_ratio),
                        "Trend filter: buys re-enabled"
                    );
                    self.trend_filter_was_active = false;
                }
                true
            }
        };

        // --- Requote threshold ---
        if self.last_buy_quote > 0.0 && self.last_sell_quote > 0.0 {
            let buy_change_bps =
                ((buy_price - self.last_buy_quote) / self.last_buy_quote * 10_000.0).abs();
            let sell_change_bps =
                ((sell_price - self.last_sell_quote) / self.last_sell_quote * 10_000.0).abs();
            if buy_change_bps < self.config.requote_threshold_bps
                && sell_change_bps < self.config.requote_threshold_bps
            {
                // Prices haven't moved enough. Still return QuoteBothSides
                // so the backtest cumulative tracker increments, but in live
                // mode the trader would skip the cancel-replace.
            }
        }
        self.last_buy_quote = buy_price;
        self.last_sell_quote = sell_price;

        // --- Compute quantities (flat min_qty sizing) ---
        //
        // Both sides always quote min_qty. Pure spread capture.
        // A-S reservation price handles inventory skew via PRICE, not quantity.
        // No escalation, no cycle counting.

        let min_notional = context.min_notional.unwrap_or(1.0);
        let step_size = context.step_size.unwrap_or(1.0);
        let min_qty = ((min_notional / buy_price).ceil() / step_size).ceil() * step_size;

        let mut buy_quantity = min_qty;
        let mut sell_quantity = min_qty;

        // Block buys if filters say so (trend filter, vol gate, etc.)
        if !buy_allowed {
            buy_quantity = 0.0;
        }

        // Clamp sell to available position (can't sell more DOGE than we hold on spot)
        let position = context.current_position.quantity;
        if position < min_qty {
            sell_quantity = 0.0;
        } else {
            sell_quantity = sell_quantity.min(position);
        }

        // Round to step_size
        buy_quantity = (buy_quantity / step_size).floor() * step_size;
        sell_quantity = (sell_quantity / step_size).floor() * step_size;

        // Check if we have enough balance for the buy side
        let buy_notional = buy_quantity * buy_price;
        if buy_quantity > 0.0 && buy_notional > context.available_cash {
            // Not enough FDUSD to place the buy order → need rebalance
            // For now, clamp to what we can afford or zero out
            let affordable_qty = (context.available_cash / buy_price / step_size).floor() * step_size;
            if affordable_qty < min_qty {
                buy_quantity = 0.0;
            } else {
                buy_quantity = affordable_qty;
            }
        }

        // Need at least one side active
        if buy_quantity <= f64::EPSILON && sell_quantity <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        // --- Update diagnostics ---
        self.diagnostics.last_mid_price = mid_price;
        self.diagnostics.last_half_spread_bps = half_spread_bps;
        self.diagnostics.last_buy_price = buy_price;
        self.diagnostics.last_sell_price = sell_price;
        self.diagnostics.last_buy_quantity = buy_quantity;
        self.diagnostics.last_sell_quantity = sell_quantity;
        self.diagnostics.last_inventory_ratio = inventory_ratio;
        self.diagnostics.last_quoted_spread_bps = quoted_spread_bps;

        StrategyDecision {
            confidence: 1.0,
            intent: OrderIntent::QuoteBothSides {
                buy_price,
                buy_quantity,
                sell_price,
                sell_quantity,
                rationale: "glm_bid_ask_quote",
                expected_edge_bps: quoted_spread_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "mid_price",
                    value: mid_price,
                },
                DecisionMetric {
                    name: "reservation_price",
                    value: reservation_price,
                },
                DecisionMetric {
                    name: "half_spread_bps",
                    value: half_spread_bps,
                },
                DecisionMetric {
                    name: "reservation_shift_bps",
                    value: reservation_shift / mid_price * 10_000.0,
                },
                DecisionMetric {
                    name: "inventory_ratio",
                    value: inventory_ratio,
                },
                DecisionMetric {
                    name: "quoted_spread_bps",
                    value: quoted_spread_bps,
                },
                DecisionMetric {
                    name: "buy_quantity",
                    value: buy_quantity,
                },
                DecisionMetric {
                    name: "sell_quantity",
                    value: sell_quantity,
                },
            ],
        }
    }
}
