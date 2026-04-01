# Spread-Regime Capture Strategy

## Problem Statement

All four existing taker strategies produce zero or near-zero trades because the round-trip
taker cost on Binance DOGEUSDT is approximately **22.7 bps** (0.1% fee each side + spread +
slippage). Simple trade-flow momentum, sweep reversal, and reclaim signals almost never
generate predicted edge above this threshold.

The core issue is structural: a taker-only strategy on a liquid low-spread altcoin needs
>23 bps of directional edge per trade, which is extremely difficult to extract from
microstructure signals alone.

## Strategy Design

### Core Idea: Volatility-Regime-Aware Mean Reversion with Adaptive Execution

Instead of chasing directional momentum, this strategy exploits **transient price
dislocations** relative to a fast-moving fair value estimate. The key insight is:

1. Price oscillates around microprice on sub-second timescales
2. When spread widens or volatility spikes, these oscillations become larger
3. A properly calibrated mean-reversion signal can capture these oscillations
4. By using **taker entry at favorable dislocation + limit exit at fair value**, we halve
   the round-trip cost from ~22.7 bps to ~11.5 bps (one taker leg + zero-cost maker leg)

### Mathematical Foundation

#### Fair Value Estimation

We use a weighted combination of:

```
fair_value = w1 * microprice + w2 * ema_mid + w3 * trade_vwap
```

Where:
- `microprice = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)` — already in MarketState
- `ema_mid` — exponentially weighted moving average of mid prices (alpha ~0.02, fast-tracking)
- `trade_vwap` — rolling VWAP from recent trades (already in MarketState)
- Weights: `w1=0.5, w2=0.3, w3=0.2` (microprice dominates as it's most responsive)

#### Dislocation Signal

```
dislocation_bps = (last_trade_price - fair_value) / fair_value * 10_000
```

When `|dislocation_bps|` exceeds a threshold (calibrated per regime), we expect
mean reversion.

#### Volatility Estimation

We maintain an EMA of squared returns to estimate instantaneous volatility:

```
realized_vol = sqrt(ema(return_i^2)) * sqrt(trades_per_second) * 10_000  // in bps
```

Where `return_i = (price_i - price_{i-1}) / price_{i-1}` for consecutive trades.

This gives us annualized-equivalent volatility scaled to our observation frequency.

#### Spread Regime Detection

Track the EMA of the quoted spread:

```
ema_spread_bps = ema(spread_bps, alpha=0.01)
```

Regime classification:
- **tight**: `ema_spread_bps < 3.0` — typical liquid conditions, small opportunity
- **normal**: `3.0 <= ema_spread_bps < 8.0` — moderate opportunity
- **wide**: `ema_spread_bps >= 8.0` — large opportunity, but also higher risk

#### Depth Imbalance (Multi-Level)

When depth snapshots are available, compute weighted imbalance across the first N levels:

```
depth_imbalance = sum(bid_qty_i * w_i) - sum(ask_qty_i * w_i)
                  / (sum(bid_qty_i * w_i) + sum(ask_qty_i * w_i))
```

Where `w_i = 1 / (1 + i)` gives more weight to levels closer to the top of book.

#### Entry Threshold (Adaptive)

The dislocation threshold adapts to current volatility and spread:

```
entry_threshold_bps = base_threshold + vol_scale * realized_vol + spread_scale * ema_spread_bps
```

Default parameters:
- `base_threshold = 5.0 bps` — minimum dislocation to act on
- `vol_scale = 0.3` — increase threshold when volatile (avoid noise entries)
- `spread_scale = 0.2` — increase threshold when spread is wide (wider cost)

The entry fires when:
```
|dislocation_bps| > entry_threshold_bps + half_round_trip_cost_bps
```

Where `half_round_trip_cost_bps ≈ 11.35` (taker fee + estimated slippage for one leg).

#### Expected Edge Calculation

```
expected_edge_bps = |dislocation_bps| * mean_reversion_factor - half_round_trip_cost_bps
```

Where `mean_reversion_factor` is an empirically calibrated parameter (default 0.7 — we
expect to capture ~70% of the dislocation as the price reverts to fair value).

The strategy only enters if `expected_edge_bps > min_edge_after_cost_bps` (default 1.0 bps).

### Entry Logic

**Buy entry** (mean reversion from below):
1. `dislocation_bps < -entry_threshold_bps` (price below fair value by enough)
2. `depth_imbalance > -max_adverse_depth` (depth doesn't strongly oppose the trade)
3. `ema_spread_bps < max_entry_spread_bps` (spread isn't so wide we can't capture)
4. `realized_vol > min_vol_bps` (need some volatility for oscillation)
5. `realized_vol < max_vol_bps` (not extreme volatility — chaotic, not mean-reverting)
6. Position is flat or short
7. Not in cooldown period
8. Expected edge after cost > 0

**Sell entry** (mean reversion from above):
- Mirror of buy conditions with `dislocation_bps > +entry_threshold_bps`

### Exit Logic

This strategy uses a **dual exit mechanism**:

1. **Primary exit: Limit order at fair value** (maker — zero/negative cost)
   - Place a limit order at `fair_value` or slightly inside (crossing expected reversion)
   - This is the profitable exit — we earn spread rather than paying it
   - The backtest engine's `simulate_passive_order()` models fill probability

2. **Safety exits (taker)**:
   - **Stop loss**: if `pnl_bps < -stop_loss_bps` (default 15 bps — tight, since this is
     a scalping strategy)
   - **Max hold time**: if position held > `max_hold_millis` (default 5000 ms)
   - **Dislocation reversal**: if dislocation flips sign and exceeds exit threshold
   - **Volatility spike**: if `realized_vol` exceeds `panic_vol_bps`, exit immediately

### Position Sizing

Use `StrategyContext::capped_entry_quantity()` with a high cash fraction (0.9) since this
is a short-duration scalping strategy. Small positions relative to capital reduce risk per
trade while maintaining high frequency.

### Frequency Target

This strategy should generate **50-200+ trades per day** on DOGEUSDT, compared to the
0-19 trades from existing strategies. The mean reversion approach with adaptive thresholds
should find many more entry opportunities because:
- We trade both directions (long AND short mean reversion)
- We use adaptive thresholds that lower the bar in high-volatility regimes
- We don't require a sustained directional flow — just a momentary dislocation

## Implementation Plan

### Step 1: Enhance MarketState

Add the following computed fields to `MarketState`:

```rust
// New fields in MarketState struct
ema_mid_price: f64,           // EMA of mid prices
ema_spread_bps: f64,          // EMA of quoted spread
realized_vol_bps: f64,        // EMA-based realized volatility
prev_trade_price: Option<f64>, // previous trade price for return calc
ema_sq_return: f64,           // EMA of squared returns
trade_rate_per_second: f64,    // smoothed trade arrival rate
depth_imbalance: f64,          // multi-level depth imbalance
```

These fields update incrementally in `apply()` — no batch recomputation needed.

### Step 2: Create Strategy Module

File: `strategies/src/spread_regime_capture.rs`

- Config struct with all tunable parameters (TOML-loadable)
- Internal state: fair value, dislocation tracking, regime classification
- Implements `Strategy` trait with `on_event()` + `decide()`
- Comprehensive diagnostics counters and gauges

### Step 3: Configuration

File: `config/strategies/spread_regime_capture.toml`

### Step 4: Registration

Add to `strategies/src/lib.rs` and `bots/binance/src/main.rs`.

### Step 5: Testing

- Unit tests for all mathematical functions
- Backtest on 3 daily archives
- Validate trade frequency and PnL

## Parameters Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| trade_window_millis | 2000 | Rolling trade window |
| ema_alpha_mid | 0.02 | EMA decay for mid price |
| ema_alpha_spread | 0.01 | EMA decay for spread |
| ema_alpha_vol | 0.05 | EMA decay for volatility |
| ema_alpha_trade_rate | 0.01 | EMA decay for trade rate |
| fair_value_w_microprice | 0.5 | Weight of microprice in FV |
| fair_value_w_ema_mid | 0.3 | Weight of EMA mid in FV |
| fair_value_w_vwap | 0.2 | Weight of VWAP in FV |
| base_threshold_bps | 5.0 | Base dislocation threshold |
| vol_scale | 0.3 | Vol scaling for threshold |
| spread_scale | 0.2 | Spread scaling for threshold |
| mean_reversion_factor | 0.7 | Expected reversion fraction |
| min_edge_after_cost_bps | 1.0 | Minimum edge to enter |
| half_round_trip_cost_bps | 11.35 | Taker cost for one leg |
| min_vol_bps | 3.0 | Min volatility to enter |
| max_vol_bps | 200.0 | Max volatility to enter |
| max_entry_spread_bps | 15.0 | Max spread for entry |
| max_adverse_depth | 0.5 | Max adverse depth imbalance |
| stop_loss_bps | 15.0 | Stop loss |
| take_profit_bps | 12.0 | Take profit (safety, below maker target) |
| max_hold_millis | 5000 | Max hold time |
| panic_vol_bps | 300.0 | Emergency exit volatility |
| entry_cooldown_millis | 500 | Min time between entries |
| min_trades_in_window | 5 | Minimum trades for valid stats |

## Known Risks

- Mean reversion assumption breaks during strong trends — mitigated by volatility caps
  and adaptive thresholds
- Maker exit fill probability is uncertain in backtest — use conservative fill model
- DOGEUSDT-specific calibration may not transfer to other pairs
- Short hold times require low-latency execution in live mode
- Parameter overfitting risk — validate on held-out data

## Success Criteria

1. Positive total PnL across the 3-day test set
2. At least 30 trades per day on average
3. Win rate above 50% (mean reversion should win more often than lose)
4. Maximum drawdown under 5% of initial capital
5. Expected edge per trade > 1 bps after costs
