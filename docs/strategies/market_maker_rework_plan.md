# Market Maker Rework Plan

## Goal

Replace the current inventory-aware long-only maker flow with a true continuous two-sided market maker for Binance Spot:

- always maintain one active maker `bid` and one active maker `ask` when risk allows,
- derive both quotes from a common fair price,
- use inventory skew to shift both sides instead of disabling one side by default,
- adapt spread to short-horizon volatility and adverse-selection risk,
- keep inventory bounded and capital recyclable over multi-day runs.

This plan is intentionally implementation-oriented and refers to the current codebase.

## Current Problem

The current `market_maker.rs` behavior is not a classical two-sided MM:

- without inventory, it mostly behaves like `post bid and wait`,
- after a buy fill, it mostly behaves like `post ask and unwind`,
- this creates a `buy -> sell -> buy -> sell` cycle instead of persistent spread capture,
- capital can remain trapped in inventory tails,
- the system underuses the natural spread during periods when both sides could rest in the book.

Observed consequence in live runs:

- `entries > closed_trades` in many sessions,
- repeated inventory tails at session end,
- low fill ratio despite active cancel-replace behavior,
- the strategy behaves more like maker entry/exit than like a continuous market maker.

## Target Strategy

### Quote Model

At each decision tick, compute:

- fair price `f`,
- dynamic half-spread `h`,
- inventory skew adjustment `s`,
- final bid/ask quotes:

`bid = f - h - s`

`ask = f + h - s`

Sign convention:

- positive inventory means long base,
- when inventory is positive, both quotes shift downward:
  - bid becomes less aggressive,
  - ask becomes more aggressive,
  - this encourages inventory reduction without turning the strategy into sell-only mode.

### Fair Price

Use top-of-book as the primary signal.

Base version:

`f = (best_bid + best_ask) / 2`

Optional extension after the first rework pass:

weighted mid / microprice using top-of-book sizes.

For the first implementation pass, plain mid is enough and keeps the change minimal.

### Dynamic Half-Spread

Start from a configured baseline half-spread and scale it by short-horizon volatility:

`h_raw = base_half_spread_bps * vol_multiplier`

`h = clamp(h_raw, min_half_spread_bps, max_half_spread_bps)`

Where:

`vol_multiplier = clamp(current_vol_bps / vol_ref_bps, vol_mult_min, vol_mult_max)`

Design intent:

- low volatility: tighter quotes, more fill probability,
- high volatility: wider quotes, lower adverse selection,
- no dependence on long prediction horizon.

### Inventory Skew

Let:

`inventory_notional = position_qty * fair_price`

`max_inventory_notional = initial_capital * max_inventory_fraction`

`normalized_inventory = clamp(inventory_notional / max_inventory_notional, -1.0, 1.0)`

Then:

`s = skew_strength * normalized_inventory * h`

Behavior:

- flat inventory: `s ~= 0`, quotes are symmetric,
- long inventory: quotes shift lower,
- near max long inventory: bid becomes much less competitive, ask moves toward the market.

This preserves two-sided quoting while still controlling inventory.

## Implementation Phases

## Phase 1: Replace Gated Buy-Only / Sell-Only Logic

### Objective

Remove the structural asymmetry where one side often disappears for business-logic reasons rather than risk reasons.

### Changes

In `strategies/src/market_maker.rs`:

1. Refactor `decide_two_sided()` so both sides are computed from the same fair price every tick.
2. Keep emergency exits (`stop_loss`, `panic_vol`, hard `max_hold`) intact.
3. Remove the current dependency where the buy side is effectively driven by `check_entry()` while the sell side is driven by inventory presence.
4. Replace that with side-specific enable/disable rules:
   - disable `bid` only when inventory is at or above the long risk cap,
   - disable `ask` only when there is not enough inventory to sell,
   - otherwise keep both sides live.

### Result

Expected default state becomes:

- flat: both `bid` and `ask` active,
- long but under cap: both active, skewed downward,
- at max long: ask-only until inventory normalizes.

## Phase 2: Introduce Dynamic Spread as First-Class Logic

### Objective

Make quoting width respond to micro-volatility instead of relying on one static spread plus ad hoc guards.

### Changes

Extend `MarketMakerConfig` with:

- `base_half_spread_bps`
- `min_half_spread_bps`
- `max_half_spread_bps`
- `volatility_reference_bps`
- `volatility_multiplier_min`
- `volatility_multiplier_max`

Use existing realized volatility from `market_state.micro()` as the first volatility source.

### Result

The strategy becomes safer in fast markets and more competitive in calm ones without changing architecture again.

## Phase 3: Reinterpret Inventory Skew Around the Fair Price

### Objective

Use inventory skew as a quote translation mechanism, not as a side suppression mechanism.

### Changes

Replace the current spread asymmetry logic with explicit fair-price shift math:

- compute `h`,
- compute `s`,
- derive both quotes from `f, h, s`.

Keep tick-size and non-crossing protections.

### Result

Inventory control becomes continuous and easier to reason about mathematically.

## Phase 4: Adverse Selection Guard

### Objective

Avoid leaving stale quotes exposed when the market jumps through them.

### Changes

Add a local guard before returning quotes:

- if current fair price has moved away from last quote center by more than `k * h`,
- return no quotes for a short stabilization interval, or force a hard reprice.

Minimal initial version:

- hard reprice immediately,
- optional cooldown only if repeated quote failures occur.

### Result

Reduced stale-fill risk during short bursts and price jumps.

## Phase 5: Inventory Aging and Flatten Mode

### Objective

Prevent inventory from becoming a multi-hour directional bet while still allowing normal market-making.

### Changes

Keep the newly added age-based passive unwind, but make it subordinate to the new quote engine:

- fresh inventory: normal skew behavior,
- aging inventory: increasingly aggressive downward skew,
- stale inventory beyond threshold: flatten mode disables bid and prioritizes ask.

Flatten mode should be entered when one or more conditions are met:

- inventory age exceeds threshold,
- normalized inventory exceeds a high watermark,
- volatility exceeds a panic threshold,
- repeated cancel-replace failures suggest unstable execution.

### Result

The bot remains a market maker most of the time, but still has an escape hatch when inventory risk dominates spread capture.

## Config Changes

Add or rename the following strategy config fields:

- `base_half_spread_bps`
- `min_half_spread_bps`
- `max_half_spread_bps`
- `volatility_reference_bps`
- `volatility_multiplier_min`
- `volatility_multiplier_max`
- `skew_strength`
- `max_inventory_fraction`
- `flatten_inventory_fraction`
- `flatten_max_age_millis`
- `adverse_selection_mult`

Keep existing fields temporarily for backward compatibility during migration, then remove dead ones after the new path is verified.

Fields likely to become obsolete or secondary:

- `require_seller_initiated`
- `min_sell_flow_fraction`
- `max_price_position`

These were useful for a directional maker-entry model, but they should not be primary controls in a continuous two-sided MM.

## Trader-Level Requirements

The strategy rewrite depends on preserving one active bid and one active ask.

Trader expectations in `exchanges/binance/src/trader.rs`:

- one bid order slot,
- one ask order slot,
- cancel-replace each side independently,
- if one side fills, cancel the opposite side immediately and request fresh quotes,
- support temporary ask-only or bid-only states only when strategy risk rules explicitly demand it.

The current trader already supports most of this. The main work is strategy-side quote generation, not execution plumbing.

## Validation Plan

## Stage A: Structural Validation

Confirm from logs that the strategy actually behaves as a continuous MM:

- most ticks should have both sides active when flat or lightly long,
- fills should show more alternating bid/ask activity without long dead periods,
- fewer sessions should end with orphan inventory.

Primary metrics:

- `closed_trades / entries`
- average inventory age
- fraction of runtime spent in one-sided quoting
- count of flatten-mode activations

## Stage B: Risk Validation

Confirm that inventory remains bounded:

- max inventory notional,
- max inventory age,
- drawdown while long inventory,
- count of stop-loss and hard-flatten events.

## Stage C: Profitability Validation

Only after structural correctness is confirmed, optimize for PnL.

Primary metrics:

- realized PnL,
- ending equity,
- average edge captured per closed round-trip,
- fill ratio,
- maker-only close rate.

## Recommended Execution Order

Implement in this exact order:

1. Rewrite `decide_two_sided()` around fair price + spread + skew.
2. Keep both sides active by default.
3. Add dynamic spread scaling from realized volatility.
4. Add flatten mode and integrate inventory aging with the new quoting model.
5. Remove now-obsolete directional entry filters from the critical path.
6. Re-tune config only after the new structure is proven in logs.

## Non-Goals for the First Rework Pass

Do not include these in the first rewrite:

- predictive alpha model,
- multi-level ladder quoting,
- queue-position simulation,
- microprice using full depth imbalance beyond top-of-book,
- portfolio-level cross-symbol inventory netting.

The first pass should only fix the structural mismatch between the current bot and a true two-sided market maker.

## Expected Outcome

If implemented correctly, the strategy should move from:

- discrete maker entry/exit behavior,
- frequent inventory tails,
- underutilized spread,

to:

- persistent two-sided quoting,
- smoother inventory control,
- higher probability of repeated spread capture over multi-day operation,
- cleaner separation between normal market making and explicit risk-off flattening.
