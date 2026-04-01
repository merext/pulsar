use crate::cost_gate::{
    DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS, DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
};
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
pub struct SpreadRegimeCaptureConfig {
    /// Rolling trade window in milliseconds (for VWAP computation)
    pub trade_window_millis: u64,

    // --- Fair-value weights ---
    /// Weight of microprice in fair-value estimate
    pub fair_value_w_microprice: f64,
    /// Weight of EMA mid-price in fair-value estimate
    pub fair_value_w_ema_mid: f64,
    /// Weight of rolling VWAP in fair-value estimate
    pub fair_value_w_vwap: f64,

    // --- Adaptive entry threshold ---
    /// Base dislocation threshold in bps (minimum to act on)
    pub base_threshold_bps: f64,
    /// Scale factor for realized volatility in threshold computation
    pub vol_scale: f64,
    /// Scale factor for EMA spread in threshold computation
    pub spread_scale: f64,

    // --- Edge & cost ---
    /// Expected fraction of dislocation captured on mean reversion (0..1)
    pub mean_reversion_factor: f64,
    /// Cost of one maker leg in bps (entry side). Maker fee is typically ~1 bps.
    pub half_round_trip_cost_bps: f64,
    /// Minimum expected edge after cost to enter (bps)
    pub min_edge_after_cost_bps: f64,

    // --- Entry filters ---
    /// Minimum realized volatility to enter (need oscillation)
    pub min_vol_bps: f64,
    /// Maximum realized volatility to enter (too chaotic)
    pub max_vol_bps: f64,
    /// Maximum spread in bps for entry
    pub max_entry_spread_bps: f64,
    /// Maximum adverse depth imbalance (0..1, against trade direction)
    pub max_adverse_depth: f64,
    /// Minimum trades in window for valid statistics
    pub min_trades_in_window: usize,
    /// Minimum trade flow imbalance favoring our direction (-1 to 1)
    /// For buys: flow_imbalance should be < -min_flow_imbalance (seller-heavy = buying opp)
    /// Disabled when set to 0.0
    pub min_flow_imbalance: f64,

    // --- Exit parameters ---
    /// Stop loss in bps (taker exit)
    pub stop_loss_bps: f64,
    /// Take profit in bps (maker exit target)
    pub take_profit_bps: f64,
    /// Maximum hold time in milliseconds before forced taker exit
    pub max_hold_millis: u64,
    /// Volatility at which we panic-exit immediately (taker)
    pub panic_vol_bps: f64,
    /// Dislocation exit threshold: if dislocation reverses beyond this, exit via maker
    pub exit_dislocation_reversal_bps: f64,

    // --- Cooldown ---
    /// Minimum time between entries in milliseconds
    pub entry_cooldown_millis: u64,

    // --- Full round-trip taker cost (for cost-gate on safety exits) ---
    #[serde(default = "default_assumed_round_trip_taker_cost_bps")]
    pub assumed_round_trip_taker_cost_bps: f64,
    #[serde(default = "default_min_expected_edge_after_cost_bps")]
    pub min_expected_edge_after_cost_bps: f64,
}

fn default_assumed_round_trip_taker_cost_bps() -> f64 {
    DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS
}

fn default_min_expected_edge_after_cost_bps() -> f64 {
    DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS
}

impl Default for SpreadRegimeCaptureConfig {
    fn default() -> Self {
        Self {
            trade_window_millis: 2_000,
            fair_value_w_microprice: 0.5,
            fair_value_w_ema_mid: 0.3,
            fair_value_w_vwap: 0.2,
            base_threshold_bps: 3.0,
            vol_scale: 0.1,
            spread_scale: 0.1,
            mean_reversion_factor: 0.6,
            half_round_trip_cost_bps: 1.0, // maker fee ~1 bps per leg
            min_edge_after_cost_bps: 0.5,
            min_vol_bps: 1.0,
            max_vol_bps: 200.0,
            max_entry_spread_bps: 15.0,
            max_adverse_depth: 0.5,
            min_trades_in_window: 5,
            min_flow_imbalance: 0.0,
            stop_loss_bps: 10.0,
            take_profit_bps: 5.0,
            max_hold_millis: 10_000,
            panic_vol_bps: 300.0,
            exit_dislocation_reversal_bps: 3.0,
            entry_cooldown_millis: 500,
            assumed_round_trip_taker_cost_bps: DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS,
            min_expected_edge_after_cost_bps: DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS,
        }
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CaptureDiagnostics {
    total_decisions: usize,
    blocked_no_quote: usize,
    blocked_no_vwap: usize,
    blocked_min_trades: usize,
    blocked_spread: usize,
    blocked_vol_low: usize,
    blocked_vol_high: usize,
    blocked_dislocation: usize,
    blocked_edge: usize,
    blocked_adverse_depth: usize,
    blocked_cooldown: usize,
    blocked_flow_imbalance: usize,
    entries_long: usize,
    exits_stop_loss: usize,
    exits_take_profit: usize,
    exits_max_hold: usize,
    exits_panic_vol: usize,
    exits_dislocation_reversal: usize,
    exits_maker: usize,
    exits_taker: usize,
    last_fair_value: f64,
    last_dislocation_bps: f64,
    last_entry_threshold_bps: f64,
    last_expected_edge_bps: f64,
    last_realized_vol_bps: f64,
    last_ema_spread_bps: f64,
    last_depth_imbalance: f64,
}

// ---------------------------------------------------------------------------
// Strategy
//
// Long-only, maker-entry / maker-exit mean-reversion strategy.
//
// Signal: Detect when the last trade price dislocates below a weighted fair
//         value (microprice + EMA mid + trade VWAP). This suggests a transient
//         sell-pressure overshoot that should mean-revert.
//
// Entry:  Post a limit buy (OrderType::Maker) at the current bid price.
//         This avoids paying the spread and pays maker fee (~1 bps) instead
//         of taker fee (~10 bps). Fill rate is ~35%, so we need many signals
//         but each filled trade is very cheap to enter.
//
// Exit:   Primary: post a limit sell (OrderType::Maker) at the ask when
//         dislocation has reversed (price above fair value). This earns the
//         spread rather than paying it.
//         Safety: taker sell for stop-loss, max-hold timeout, or panic vol.
//         Taker exits are expensive but protect capital.
//
// Cost model:
//   Maker entry + maker exit = 2 * maker_fee = 2 * 1 bps = 2 bps round-trip.
//   Taker safety exit = maker_fee + taker_fee + spread ≈ 1 + 10 + 1 = 12 bps.
//   With 35% fill rate, ~65% of entry signals don't fill (no cost).
//   The strategy only pays when it fills, so the effective cost per attempt
//   is 0.35 * 2 bps = 0.7 bps.
// ---------------------------------------------------------------------------

pub struct SpreadRegimeCaptureStrategy {
    config: SpreadRegimeCaptureConfig,
    logger: NoOpStrategyLogger,
    pub last_entry_time_millis: Option<u64>,
    /// Always Some(Side::Buy) when in a position (long-only strategy)
    pub entry_side: Option<Side>,
    /// Dislocation at entry (for reversal detection)
    pub entry_dislocation_bps: f64,
    diagnostics: CaptureDiagnostics,
}

impl SpreadRegimeCaptureStrategy {
    fn load_config<P: AsRef<Path>>(
        config_path: P,
    ) -> Result<SpreadRegimeCaptureConfig, Box<dyn std::error::Error>> {
        let path = config_path.as_ref();
        if path == Path::new("/dev/null") || !path.exists() {
            return Ok(SpreadRegimeCaptureConfig::default());
        }
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str::<SpreadRegimeCaptureConfig>(&content)?;
        Ok(config)
    }

    /// Compute weighted fair value from microprice, EMA mid, and VWAP.
    /// Returns None if we don't have enough data.
    fn fair_value(&self, market_state: &MarketState) -> Option<f64> {
        let microprice = market_state.microprice()?;
        let micro = market_state.micro();

        if !micro.mid_initialized() {
            return None;
        }

        let ema_mid = micro.ema_mid_price;
        let vwap = market_state.trade_window_vwap().unwrap_or(ema_mid);

        let fv = self.config.fair_value_w_microprice * microprice
            + self.config.fair_value_w_ema_mid * ema_mid
            + self.config.fair_value_w_vwap * vwap;

        if fv > f64::EPSILON {
            Some(fv)
        } else {
            None
        }
    }

    /// Compute dislocation of last trade vs fair value, in bps.
    fn dislocation_bps(&self, market_state: &MarketState) -> Option<f64> {
        let last_price = market_state.last_price()?;
        let fv = self.fair_value(market_state)?;
        if fv <= f64::EPSILON {
            return None;
        }
        Some((last_price - fv) / fv * 10_000.0)
    }

    /// Adaptive entry threshold in bps, scaled by volatility and spread.
    fn entry_threshold_bps(&self, market_state: &MarketState) -> f64 {
        let micro = market_state.micro();
        self.config.base_threshold_bps
            + self.config.vol_scale * micro.realized_vol_bps
            + self.config.spread_scale * micro.ema_spread_bps
    }

    /// Expected edge in bps for a mean-reversion trade using maker execution.
    /// Edge = dislocation mean-reversion + spread capture - maker fees.
    /// With maker-maker: buy at bid, sell at ask → earn spread.
    /// Dislocation adds to this when price reverts toward fair value.
    fn expected_edge_bps(&self, dislocation_bps: f64, market_state: &MarketState) -> f64 {
        let spread_capture = market_state.micro().ema_spread_bps;
        dislocation_bps.abs() * self.config.mean_reversion_factor
            + spread_capture
            - self.config.half_round_trip_cost_bps * 2.0 // both legs are maker
    }

    fn in_entry_cooldown(&self, market_state: &MarketState) -> bool {
        let Some(last_entry) = self.last_entry_time_millis else {
            return false;
        };
        let Some(now) = market_state.last_event_time_millis() else {
            return false;
        };
        now.saturating_sub(last_entry) < self.config.entry_cooldown_millis
    }

    /// Check whether we should enter long on mean reversion.
    /// Only triggers when price is dislocated BELOW fair value (buying opportunity).
    /// Returns expected edge if entry conditions are met, None otherwise.
    fn check_entry(&mut self, market_state: &MarketState) -> Option<f64> {
        self.diagnostics.total_decisions += 1;

        // Need quote data for microprice, spread, and maker order pricing
        if market_state.top_of_book().is_none() {
            self.diagnostics.blocked_no_quote += 1;
            return None;
        }

        // Need enough trades for VWAP and statistics
        let stats = market_state.trade_window_stats();
        if stats.trade_count < self.config.min_trades_in_window {
            self.diagnostics.blocked_min_trades += 1;
            return None;
        }

        // Need VWAP
        if market_state.trade_window_vwap().is_none() {
            self.diagnostics.blocked_no_vwap += 1;
            return None;
        }

        // Spread filter — too wide means risk of adverse selection
        let micro = market_state.micro();
        self.diagnostics.last_ema_spread_bps = micro.ema_spread_bps;
        if micro.ema_spread_bps > self.config.max_entry_spread_bps {
            self.diagnostics.blocked_spread += 1;
            return None;
        }

        // Volatility filters
        self.diagnostics.last_realized_vol_bps = micro.realized_vol_bps;
        if micro.realized_vol_bps < self.config.min_vol_bps {
            self.diagnostics.blocked_vol_low += 1;
            return None;
        }
        if micro.realized_vol_bps > self.config.max_vol_bps {
            self.diagnostics.blocked_vol_high += 1;
            return None;
        }

        // Compute dislocation
        let dislocation = self.dislocation_bps(market_state)?;
        self.diagnostics.last_dislocation_bps = dislocation;

        // Compute adaptive threshold
        let threshold = self.entry_threshold_bps(market_state);
        self.diagnostics.last_entry_threshold_bps = threshold;

        // Long-only: only enter when price is below fair value by at least threshold
        // dislocation < 0 means price is below fair value
        if dislocation > -threshold {
            self.diagnostics.blocked_dislocation += 1;
            return None;
        }

        // Depth imbalance check: for a buy, adverse = strongly ask-heavy (sellers dominating)
        self.diagnostics.last_depth_imbalance = micro.depth_imbalance;
        if micro.depth_imbalance < -self.config.max_adverse_depth {
            self.diagnostics.blocked_adverse_depth += 1;
            return None;
        }

        // Flow imbalance check (optional): we want to buy when there's been selling pressure
        // (which creates the dislocation), but not when momentum is strongly against us.
        // trade_flow_imbalance > 0 = buyer-heavy, < 0 = seller-heavy
        if self.config.min_flow_imbalance > 0.0 {
            let flow = market_state.trade_flow_imbalance();
            // We're buying: we want sellers to have pushed price down (flow < 0),
            // but not TOO much selling (which might be informed flow, not noise).
            // Block if flow is strongly buyer-heavy (dislocation is not from selling)
            if flow > self.config.min_flow_imbalance {
                self.diagnostics.blocked_flow_imbalance += 1;
                return None;
            }
        }

        // Expected edge check
        let edge = self.expected_edge_bps(dislocation, market_state);
        self.diagnostics.last_expected_edge_bps = edge;
        if edge < self.config.min_edge_after_cost_bps {
            self.diagnostics.blocked_edge += 1;
            return None;
        }

        // Cooldown check
        if self.in_entry_cooldown(market_state) {
            self.diagnostics.blocked_cooldown += 1;
            return None;
        }

        if let Some(fv) = self.fair_value(market_state) {
            self.diagnostics.last_fair_value = fv;
        }

        Some(edge)
    }

    /// Check exit conditions for an open long position.
    /// Returns (rationale, use_maker) — use_maker=true for favorable exits,
    /// false for emergency taker exits.
    fn check_exit(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> Option<(&'static str, bool)> {
        if context.current_position.quantity <= 0.0 {
            return None;
        }

        let entry_price = context.current_position.entry_price;

        // Use bid price for long position valuation (conservative)
        let reference_price = market_state
            .top_of_book()
            .map(|b| b.bid.price)
            .or_else(|| market_state.last_price())?;

        // PnL computation (long-only: profit when price goes up)
        let pnl_bps = (reference_price - entry_price) / entry_price * 10_000.0;

        // === Emergency taker exits (capital protection, immediate) ===

        // Stop loss — must exit NOW via taker
        if pnl_bps <= -self.config.stop_loss_bps {
            self.diagnostics.exits_stop_loss += 1;
            self.diagnostics.exits_taker += 1;
            return Some(("stop_loss", false));
        }

        // Panic volatility exit — market too chaotic, exit via taker
        let micro = market_state.micro();
        if micro.realized_vol_bps > self.config.panic_vol_bps {
            self.diagnostics.exits_panic_vol += 1;
            self.diagnostics.exits_taker += 1;
            return Some(("panic_vol", false));
        }

        // Max hold time — exit via taker to avoid overnight/extended risk
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

        // === Favorable maker exits (capture profit at ask) ===

        // Take profit — price has risen enough, post limit sell at ask
        if pnl_bps >= self.config.take_profit_bps {
            self.diagnostics.exits_take_profit += 1;
            self.diagnostics.exits_maker += 1;
            return Some(("take_profit", true));
        }

        // Dislocation reversal — price moved above fair value, mean reversion complete
        if let Some(current_dislocation) = self.dislocation_bps(market_state) {
            if current_dislocation > self.config.exit_dislocation_reversal_bps {
                self.diagnostics.exits_dislocation_reversal += 1;
                self.diagnostics.exits_maker += 1;
                return Some(("dislocation_reversal", true));
            }
        }

        None
    }
}

#[async_trait::async_trait]
impl Strategy for SpreadRegimeCaptureStrategy {
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
            last_entry_time_millis: None,
            entry_side: None,
            entry_dislocation_bps: 0.0,
            diagnostics: CaptureDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "SpreadRegimeCaptureStrategy - long-only maker mean-reversion".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert("src.total_decisions".into(), self.diagnostics.total_decisions);
        counters.insert("src.blocked_no_quote".into(), self.diagnostics.blocked_no_quote);
        counters.insert("src.blocked_no_vwap".into(), self.diagnostics.blocked_no_vwap);
        counters.insert("src.blocked_min_trades".into(), self.diagnostics.blocked_min_trades);
        counters.insert("src.blocked_spread".into(), self.diagnostics.blocked_spread);
        counters.insert("src.blocked_vol_low".into(), self.diagnostics.blocked_vol_low);
        counters.insert("src.blocked_vol_high".into(), self.diagnostics.blocked_vol_high);
        counters.insert("src.blocked_dislocation".into(), self.diagnostics.blocked_dislocation);
        counters.insert("src.blocked_edge".into(), self.diagnostics.blocked_edge);
        counters.insert("src.blocked_adverse_depth".into(), self.diagnostics.blocked_adverse_depth);
        counters.insert("src.blocked_cooldown".into(), self.diagnostics.blocked_cooldown);
        counters.insert("src.blocked_flow_imbalance".into(), self.diagnostics.blocked_flow_imbalance);
        counters.insert("src.entries_long".into(), self.diagnostics.entries_long);
        counters.insert("src.exits_stop_loss".into(), self.diagnostics.exits_stop_loss);
        counters.insert("src.exits_take_profit".into(), self.diagnostics.exits_take_profit);
        counters.insert("src.exits_max_hold".into(), self.diagnostics.exits_max_hold);
        counters.insert("src.exits_panic_vol".into(), self.diagnostics.exits_panic_vol);
        counters.insert("src.exits_dislocation_reversal".into(), self.diagnostics.exits_dislocation_reversal);
        counters.insert("src.exits_maker".into(), self.diagnostics.exits_maker);
        counters.insert("src.exits_taker".into(), self.diagnostics.exits_taker);

        let mut gauges = BTreeMap::new();
        gauges.insert("src.last_fair_value".into(), self.diagnostics.last_fair_value);
        gauges.insert("src.last_dislocation_bps".into(), self.diagnostics.last_dislocation_bps);
        gauges.insert("src.last_entry_threshold_bps".into(), self.diagnostics.last_entry_threshold_bps);
        gauges.insert("src.last_expected_edge_bps".into(), self.diagnostics.last_expected_edge_bps);
        gauges.insert("src.last_realized_vol_bps".into(), self.diagnostics.last_realized_vol_bps);
        gauges.insert("src.last_ema_spread_bps".into(), self.diagnostics.last_ema_spread_bps);
        gauges.insert("src.last_depth_imbalance".into(), self.diagnostics.last_depth_imbalance);

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        self.config.trade_window_millis
    }

    async fn on_event(&mut self, _event: &MarketEvent, _market_state: &MarketState) {
        // MicrostructureState updates are handled by MarketState::apply().
        // No additional per-event work needed here.
    }

    fn decide(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> StrategyDecision {
        // --- Exit logic first (long-only: always exit via Sell) ---
        if context.current_position.quantity > 0.0 {
            if let Some((rationale, use_maker)) = self.check_exit(market_state, context) {
                // Clear position tracking
                self.entry_side = None;
                self.entry_dislocation_bps = 0.0;

                let (order_type, time_in_force) = if use_maker {
                    (OrderType::Maker, TimeInForce::Gtc)
                } else {
                    (OrderType::Taker, TimeInForce::Ioc)
                };

                return StrategyDecision {
                    confidence: 1.0,
                    intent: OrderIntent::Place {
                        side: Side::Sell, // Long-only: exit is always sell
                        order_type,
                        price: None,
                        quantity: context.current_position.quantity,
                        time_in_force,
                        rationale,
                        expected_edge_bps: 0.0,
                    },
                    metrics: vec![DecisionMetric {
                        name: "position_quantity",
                        value: context.current_position.quantity,
                    }],
                };
            }

            // Still holding, no exit signal -> no action
            return StrategyDecision::no_action();
        }

        // --- Entry logic (long-only: always Buy via Maker) ---
        let Some(expected_edge) = self.check_entry(market_state) else {
            return StrategyDecision::no_action();
        };

        let reference_price = market_state
            .mid_price()
            .or_else(|| market_state.last_price())
            .unwrap_or(0.0);
        if reference_price <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        let Some(quantity) = context.capped_entry_quantity(reference_price, 0.9, None) else {
            return StrategyDecision::no_action();
        };

        // Record entry state
        self.last_entry_time_millis = market_state.last_event_time_millis();
        self.entry_side = Some(Side::Buy);
        self.entry_dislocation_bps = self.diagnostics.last_dislocation_bps;
        self.diagnostics.entries_long += 1;

        let dislocation = self.diagnostics.last_dislocation_bps;
        let threshold = self.diagnostics.last_entry_threshold_bps;
        let micro = market_state.micro();

        StrategyDecision {
            confidence: 0.8,
            intent: OrderIntent::Place {
                side: Side::Buy,
                order_type: OrderType::Maker, // Passive limit buy at bid
                price: None,
                quantity,
                time_in_force: TimeInForce::Gtc, // Limit order rests on book
                rationale: "spread_regime_capture_entry",
                expected_edge_bps: expected_edge,
            },
            metrics: vec![
                DecisionMetric {
                    name: "dislocation_bps",
                    value: dislocation,
                },
                DecisionMetric {
                    name: "entry_threshold_bps",
                    value: threshold,
                },
                DecisionMetric {
                    name: "expected_edge_bps",
                    value: expected_edge,
                },
                DecisionMetric {
                    name: "realized_vol_bps",
                    value: micro.realized_vol_bps,
                },
                DecisionMetric {
                    name: "ema_spread_bps",
                    value: micro.ema_spread_bps,
                },
                DecisionMetric {
                    name: "depth_imbalance",
                    value: micro.depth_imbalance,
                },
            ],
        }
    }
}
