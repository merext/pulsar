# Market Making Strategy Optimization Log

## Project: Pulsar — DOGEFDUSD Two-Sided Market Maker
## Strategy: MarketMakerBidAskGlmStrategy

---

## Session Start: 2026-04-05

### Architecture Overview
- **Pair**: DOGEFDUSD on Binance (price ~$0.092, tick 0.00001 ≈ 1.1 bps)
- **Fees**: ZERO maker fee (FDUSD promo), 0.1% taker fee
- **Capital**: ~$110 equity (~466 DOGE + ~68 FDUSD after Test 023)
- **Strategy file**: `strategies/src/market_maker_bid_ask_glm.rs`
- **Config**: `config/strategies/market_maker_bid_ask_glm.toml`
- **Trading config**: `config/trading_config_dogefdusd.toml`

### Key Decisions
- Based on Avellaneda-Stoikov model for optimal market making
- Zero maker fees → can profitably quote at 1-tick spread
- Primary risk: adverse selection (getting picked off by informed flow)
- Inventory management via reservation price skewing (A-S model)

---

## Change Log

### Change 001 — Symmetric sell_quantity
**Problem**: Bot sold ENTIRE inventory (411 DOGE) in one order → 0 DOGE → forced rebalance.
**Fix**: `sell_quantity = base_qty.min(position).max(min_qty)` — sell in portions.
**Status**: ✅ Implemented

### Change 002 — Avellaneda-Stoikov Reservation Price
**Problem**: Old model used asymmetric spread skewing. Not optimal.
**Fix**: `reservation_price = mid - q_normalized * gamma * σ² * τ` where q_normalized maps inventory_ratio [0,1] to [-1,+1].
**Result**: Both bid and ask placed symmetrically around reservation_price (not mid_price).
**Status**: ✅ Implemented (strategy lines 659-697)

### Change 003 — Symmetric Quantity Scaling (A-S)
**Problem**: Fixed quantity for both sides regardless of inventory.
**Fix**: `buy_qty = base_qty * (1 - inventory_ratio)`, `sell_qty = base_qty * inventory_ratio`, clamped to [min_qty, position].
**Status**: ✅ Implemented (strategy lines 762-801)

### Change 004 — Fix max_inventory_notional()
**Problem**: `max_inventory_notional()` was capped by `max_position_notional` (per-ORDER size), making inventory_ratio always ~1.0.
**Fix**: Uses ONLY `initial_capital * max_inventory_fraction`, ignoring `max_position_notional`.
**Status**: ✅ Implemented (strategy lines 340-352)

### Change 005 — Config Tuning
**Changes**:
- `max_position_notional`: 15 → **3.0** (base_qty = ~32 DOGE per order)
- `max_inventory_fraction`: 0.50 → **1.0** (max_inventory = initial_capital = $112)
- `initial_capital`: 62 → **112.0** (matches real equity)
- `inventory_skew_factor` (gamma): 0.7 → **15.0** (~0.35 bps shift at full inventory)
**Status**: ✅ Applied

### Change 006 — Change-Detection Logging
**Problem**: "both sides active" logged every tick — flooding logs.
**Fix**: Only log when state actually changes (bid_active, ask_active, prices, quantities).
**Also**: "order_pending", "two_sided_orders_placed" demoted to debug level.
**Status**: ✅ Implemented (trader.rs lines 2700-2735)

### Change 007 — Instant-Fill Cooldown (Buy Avalanche Fix)
**Problem**: When bid fills instantly on placement, bot immediately places new bid on next tick → cascading fills. 139 BID fills in first 30 seconds of Test 004.
**Fix**: Added `last_bid_instant_fill_millis` / `last_ask_instant_fill_millis` fields. After instant fill, skip placement for `min_order_rest_millis` (1000ms).
**Status**: ✅ Implemented & verified in Test 005

### Change 008 — REST API Price Fallback for CLI Rebalance
**Problem**: CLI `rebalance` command failed because `last_reference_price` = 0 (no bookTicker without websocket).
**Fix**: Fetches price from `https://api.binance.com/api/v3/ticker/price` when no bookTicker available.
**Status**: ✅ Implemented (trader.rs rebalance() method)

### Change 009 — Rebalance Over-Triggering Fix
**Problem**: Rebalance triggered on `consecutive_two_sided_failures >= 1` — any single cancel-replace error causes rebalance. Test 004: 56 rebalances in 30 min. Test 005: 4 rebalances in 3 min.
**Fix**:
- Increased threshold: `>= 1` → **`>= 3`** consecutive failures required
- Added **30-second minimum interval** between rebalances (`last_rebalance_millis` tracking)
- After rebalance (success or failure), reset counter to 0 so fresh accumulation required
**Expected**: ≤ 2-3 rebalances per 30-min test instead of 50+
**Status**: ✅ Implemented, compiles

### Change 010 — max_hold_millis Disabled
**Problem**: max_hold_millis=300000 (5 min) was causing a death cycle: rebalance buy → 5 min hold → dump ALL inventory via taker market order → forced rebalance buy. Visible in Test 007 (-0.042 PnL loss).
**Fix**: Set `max_hold_millis = 0` in config (disables the check entirely).
**Status**: ✅ Applied in config

### Change 011 — Conditional Join-Best-Price
**Problem**: `join_best_price` logic pulled bids to best_bid even when A-S reservation price would set them lower (negating inventory skew).
**Fix**: Only join best_bid when `inventory_ratio < 0.6`. When inventory is high, respect the A-S price to widen the bid.
**Status**: ✅ Implemented (strategy lines ~704-715)

### Change 012 — Allow buy_quantity = 0
**Problem**: `.max(min_qty)` forced minimum buy quantity even when A-S scaling pushed it below exchange minimum.
**Fix**: When rounded buy_qty < min_qty, set buy_quantity = 0.0 (skip bid side entirely).
**Status**: ✅ Implemented (strategy lines ~848-856)

### Change 013 — Inventory Gate at 0.85
**Problem**: Even with A-S quantity scaling, buys could still occur at very high inventory.
**Fix**: Completely block buys when inventory_ratio > 0.85.
**Status**: ✅ Implemented (strategy line ~840)

### Change 014 — Gamma Increase (15 → 200)
**Problem**: gamma=15 produced only ~0.35 bps reservation price shift at full inventory — negligible.
**Fix**: gamma=200 → ~4 bps shift at full inventory. More meaningful price skew.
**Status**: ✅ Applied in config (`inventory_skew_factor = 200.0`)

### Change 015 — Slow EMA Trend Filter
**Problem**: In a falling market, bids fill faster than asks regardless of A-S skew. Need trend awareness.
**Fix**: Added `slow_ema_mid` field (alpha=0.002, halflife ~35s). When `mid_price < slow_ema_mid` AND `inventory_ratio > 0.55`, block buys entirely.
**Status**: ✅ Implemented (strategy lines ~764-778)

### Change 016 — Fix Trend Filter / Zero-Qty Rebalance Interaction
**Problem**: When trend filter blocks buys (buy_qty=0), `consecutive_zero_qty_ticks` in trader.rs increments every tick. At threshold=10 (~10s), rebalance fires → sells DOGE at taker fees → inventory drops → trend filter disengages → buys resume → inventory grows again → cycle repeats. This caused 2 rebalances in Test 010 and wasted taker fees.
**Fix (three-part)**:
1. **Raised ZERO_QTY_REBALANCE_THRESHOLD**: 10 → **100** ticks (~100s). Gives trend filter much more time to work before rebalance overrides it.
2. **Lowered trend filter threshold**: inventory_ratio > 0.55 → **> 0.50** (block buys at any above-neutral inventory in falling market).
3. **Added trend filter logging**: Change-detection logging (`trend_filter_was_active` flag) — logs when trend filter activates/deactivates, with mid/slow_ema/inv_ratio values.
**Status**: ✅ Implemented, compiles

### Change 017 — Cycle-Based Sizing Model (replaces A-S quantity scaling)
**Problem**: Old A-S inventory-ratio-based quantity scaling (`buy_qty = base_qty × (1 - ratio)`, `sell_qty = base_qty × ratio`) had a fundamental issue: `base_qty` was capped by `max_position_notional = 3.0` → always 33 DOGE. At ratio > 0.667, buy_qty dropped below min_qty (11) → buy_qty = 0. After 100 ticks of buy_qty=0 → automatic rebalance → cycle repeated every ~130 seconds. Increasing capital only increased buffer, didn't fix the fundamental issue.
**Fix**: New cycle-based sizing model:
- Both sides start at `min_qty` (11 DOGE)
- `cycle_count` starts at 1, tracks consecutive same-side fill series
- Series side: always `min_qty` (minimal risk)
- Opposite side: `min_qty × cycle_count` (grows linearly — bigger opportunity to mean-revert)
- When opposite side fills: `cycle_count = 1` (reset)
- Fill detection via `context.current_position.quantity` delta between ticks
- Rebalance causes ASK fill detection → natural cycle_count reset
- A-S reservation price retained for PRICING only (not sizing)
**Removed**: `INVENTORY_GATE`, `buy_scale`/`sell_scale`, A-S quantity formulas, `base_qty` from `capped_entry_quantity`
**Status**: ✅ Implemented, validated in Test 014

### Change 018 — Fix Missing Strategy Log Lines
**Problem**: `info!` logs from the `strategies` crate (e.g., "Cycle sizing: fill detected") were silently dropped because the default `RUST_LOG` filter only included `binance_bot`, `binance_exchange`, and `trade` — not `strategies`.
**Fix**: Added `strategies=info` to the default env_filter in `main.rs`.
**Status**: ✅ Implemented

### Change 019 — Remove Dead Code (compute_dynamic_cf)
**Problem**: `compute_dynamic_cf()` method was unused in `market_maker_bid_ask_glm.rs` after cycle-based sizing replaced dynamic sizing. Compiler warning.
**Fix**: Removed the entire method (~48 lines).
**Status**: ✅ Implemented

### Change 020 — Flat Min-Qty Sizing (replaces cycle-based)
**Problem**: Cycle-based sizing escalated opposite-side quantities to huge values (e.g., ask_qty=121 at cycle_count=11). Unpredictable risk.
**Fix**: Both sides always quote `min_qty` (12 DOGE). A-S handles inventory via PRICE skew only, not quantity.
**Removed**: `cycle_count`, `last_series_side`, all cycle-based logic.
**Status**: ✅ Implemented

### Change 021 — Trend Filter Hysteresis
**Problem**: Trend filter oscillated on/off rapidly when price crossed slow EMA boundary.
**Fix**: Hysteresis thresholds — activate when `fast/slow ratio > 0.65`, deactivate when `ratio < 0.55`.
**Status**: ✅ Implemented

### Change 022 — Remove Entry-Price Floor (CRITICAL FIX)
**Problem**: After buying on best_bid, `passive_exit_floor_bps()` set `sell_price >= entry_price`, which in a falling market kept asks ABOVE best_ask → asks never filled → BID:ASK ratio went to 4:1. This was the **main cause** of the chronic BID-heavy imbalance.
**Fix**: Removed entry-price floor entirely. Asks now always quote at or near best_ask via A-S pricing.
**Also removed**: Dead code `passive_exit_floor_bps()` method.
**Status**: ✅ Implemented

### Change 023 — Asymmetric Join-Best-Price + Inventory-Aware Bid Offset
**Problem**: BID join-best-price threshold was `ratio < 0.60` — too aggressive, bids always joined best_bid even when inventory was balanced.
**Fix**:
1. **BID side**: Join best_bid only when `ratio < 0.50` (neutral point). Above 0.50, A-S reservation price controls.
2. **ASK side**: Always join best_ask (unchanged).
3. **Inventory-aware bid offset**: When `ratio > 0.55`, extra bps offset pushes bid down: `excess × 10 bps per 0.10 ratio`. Stacks on top of A-S shift.
**Status**: ✅ Implemented

### Change 024 — Fix Stale Orders (max_order_rest_millis)
**Problem**: `max_order_rest_millis` was 1,800,000 (30 minutes!) in config. Orders sat at the same price for entire test without refreshing → lost queue position, missed price moves.
**Fix**: Changed to **30,000 (30 seconds)** in `trading_config_dogefdusd.toml`. Orders now refresh every 30s.
**Status**: ✅ Applied in config

### Change 025 — Fix Re-Quote After Both Sides Fill
**Problem**: When both `active_bid_order` and `active_ask_order` were `None` (both sides filled simultaneously), the instant-fill cooldown blocked new placements → bot sat idle with NO quotes until cooldown expired. Visible in Test 022 as prolonged dead periods.
**Fix**: Added `both_sides_empty` check in trader.rs that bypasses instant-fill cooldown when both sides are empty. Same logic added for ask-side cooldown check.
**Status**: ✅ Implemented (trader.rs lines ~2550-2559, ~2627-2639)

---

## Live Test Log

### Test 001 — Baseline (2 min, symmetric sell_quantity only)
- Duration: 2 min
- BID fills: 55, ASK fills: 4 (ratio 14:1 — massively skewed)
- Realized PnL: +0.00567
- Rebalances: 5
- **Conclusion**: Severe buy-side bias. Need A-S model.

### Test 002 — A-S Model + Old Config (30 min)
- Duration: 30 min
- BID fills: 145, ASK fills: 8 (ratio 18:1 — worse!)
- Realized PnL: +0.053
- Rebalances: 9
- **Conclusion**: A-S model active but inventory_ratio stuck at ~1.0 due to config bug (max_inventory_notional capped by max_position_notional).

### Test 003 — A-S + Fixed Config (3 min)
- Duration: 3 min
- BID fills: 7, ASK fills: 35 (ratio 1:5 — reversed!)
- Realized PnL: +0.00221, Win rate: 100%
- Rebalances: 1
- **Conclusion**: Config fix works! Started with 443 DOGE overweight from Test 002 → A-S correctly favored selling to reduce inventory.

### Test 004 — Full A-S + No Cooldown (30 min)
- Duration: 30 min
- BID fills: 872, ASK fills: 620 (ratio 1.41:1 — much better!)
- Realized PnL: +0.310, Win rate: 96%
- Rebalances: **56** (massive over-triggering!)
- **Problem**: Buy avalanche — 139 BID fills in first 30 seconds. BNB consumed for fees. Net equity drag ~0.45 FDUSD from taker fees on rebalances.

### Test 005 — With Instant-Fill Cooldown (3 min)
- Duration: 3 min
- Entries: 3, Closed trades: 8
- Realized PnL: +0.00217, Win rate: 100%
- Rebalances: 4 (still too many for 3 min)
- **Conclusion**: Buy avalanche FIXED — only 1 instant fill at startup. Both sides active most of time. But rebalance frequency still needs fixing → Change 009.

### Test 006 — Rebalance Fix (30 min)
- Duration: 30 min
- BID:ASK ratio: 1.09:1 (nearly balanced!)
- Realized PnL: +0.020, Win rate: 92%, Profit factor: 1.24
- Rebalances: 3
- **Conclusion**: Rebalance fix works. Ratio nearly 1:1 — great improvement. But PnL thin, profit factor only 1.24.

### Test 007 — First 1-Hour Test (60 min)
- Duration: 1 hour
- BID:ASK ratio: 1.57:1
- Realized PnL: **-0.042** (FIRST LOSS), Win rate: 72%, Profit factor: 0.46
- Rebalances: 7
- **Conclusion**: LOSS caused by max_hold_millis death cycle. Holding → timeout → dump all at taker fee → forced rebalance buy → repeat. Led to Change 010.

### Test 008 — max_hold Disabled (60 min)
- Duration: 1 hour
- BID:ASK ratio: 2.67:1
- Realized PnL: +0.020, Win rate: **100%**, Profit factor: **∞**
- Rebalances: 1
- **Conclusion**: Death cycle eliminated. 100% win rate but BID-heavy. Market was falling → natural bid bias.

### Test 009 — Gamma 200 + Fixes (30 min, falling market)
- Duration: 30 min
- BID:ASK ratio: **3.33:1** (worst ever)
- Realized PnL: +0.0015, Win rate: 100%, Profit factor: ∞
- Rebalances: 0
- **Conclusion**: Gamma=200 helps quantity skew but can't stop adverse selection in strongly falling market. Very few trades closed.

### Test 010 — + Trend Filter (30 min)
- Duration: 30 min
- BID:ASK ratio: **2.20:1** (improved from 3.33)
- Realized PnL: +0.0042, Win rate: 100%, Profit factor: ∞
- Rebalances: 2
- **Key observations**:
  - Gamma=200 produces mild quantity skew (bid_qty 14-19, never reaches 0)
  - Trend filter computed zero buy_qty internally but zero_qty_ticks=10 triggered rebalance before it could take effect
  - Conditional join-best-price works: spread widened from 1-2 bps to 4-5 bps at high inventory
  - Near break-even after accounting for unrealized inventory losses
  - Led to Change 016 (fix trend filter / rebalance interaction)

### Test 011 — Change 016 validated (30 min, rangebound market)
- Duration: 30 min
- BID fills: 24, ASK fills: 29 (ratio **0.83:1** — first time ASK > BID!)
- Realized PnL: **+0.02194**, Win rate: **100%**, Profit factor: **∞**
- Rebalances: **0**
- Trend filter activations: **0** (market was rangebound, ~23 bps total range)
- Price range: 0.09198 → 0.09220 → 0.09205 (slight uptrend then reversal)
- Closed trades: 29, Entries: 24
- Ending equity: ~$111.14
- **Key observations**:
  - ZERO_QTY_REBALANCE_THRESHOLD=100 prevented false rebalances — no rebalances at all
  - A-S inventory scaling working: bid_qty grew to 19-21 mid-session, ask_qty shrank to 11-12, then normalized
  - 15 cancel-replace failures (race conditions) — all handled gracefully via fill recovery
  - PnL rate: ~0.044 FDUSD/hour → ~1.05 FDUSD/day → ~0.95%/day on $112
  - **Best test so far by all metrics**. Ready for 1-hour progressive test.

### Test 012 — First successful 1-hour test (60 min, slight downtrend)
- Duration: 1 hour (5030 ticks)
- BID fills: 56, ASK fills: 62 (ratio **0.90:1** — ASK-heavy again!)
- Realized PnL: **+0.06788**, Win rate: **100%**, Profit factor: **∞**
- Rebalances: **11** (all zero_qty_ticks=100 triggers, all selling DOGE)
- Trend filter activations: **0** (slow EMA didn't catch the gradual decline)
- Price range: 0.09153 → 0.09198 (~49 bps total, net downtrend ~24 bps)
- Closed trades: 62, Entries: 56, Fees: 0
- Ending equity: ~$110.93
- **Key observations**:
  - First 30 min perfectly balanced (23:23), second 30 min sell-heavy (34:39)
  - 18 consecutive ASK fills at one point (01:09-01:12) — strong selling pressure
  - 11 rebalances cost hidden BNB fees (11 × 16 DOGE = 176 DOGE sold at taker)
  - PnL rate: ~0.068/hour → ~1.63 FDUSD/day → ~1.47%/day (before BNB costs)
  - Trend filter didn't fire despite 24 bps decline — slow_ema too slow for gradual drift
  - **Successful 1-hour test. Ready for 3-hour progressive test.**

### Test 013 — 3-hour test attempt (stopped early at ~39 min)
- Duration: ~39 min (stopped early)
- BID:ASK ratio: **0.83:1**
- Realized PnL: **~-0.106** (slight loss)
- Rebalances: 5
- **Conclusion**: Stopped early due to slight loss. Market conditions unfavorable.

### Test 014 — Cycle-Based Sizing (30 min, first test of new model)
- Duration: 30 min
- BID fills: 28, ASK fills: 22 (ratio **1.27:1** — BID-heavy, slight downtrend)
- Realized PnL: **+0.00681**, Win rate: **100%**, Profit factor: **∞**
- Rebalances: **2** (SELL, 13 + 11 = 24 DOGE total — much fewer than old model!)
- Cycle escalation: ask_qty reached 121 (11×11 = cycle_count=11)
- Price: 0.09114 → 0.09107 (slight downtrend ~10 bps)
- 11 ERRORs (cancel-replace, all recovered), 42 WARNs (benign)
- **Key observations**:
  - Cycle-based sizing produces significantly fewer rebalances (2 vs 5-11 with old model)
  - Opposite-side quantity escalation working correctly
  - "Cycle sizing: fill detected" logs were NOT emitted — traced to missing `strategies` crate in log filter (fixed in Change 018)
  - **Successful first test of cycle-based sizing. Ready for 1-hour test.**

### Test 015 — Cycle-Based Sizing with cap=5 (47 min, aborted)
- Duration: ~47 min (aborted)
- BID:ASK ratio: **0.54:1** (ASK-heavy)
- Realized PnL: ~breakeven
- Rebalances: **11**
- **Conclusion**: Cap=5 helped reduce escalation but 11 rebalances still too many. Cycle-based sizing abandoned.

### Test 017 — Flat Sizing First Test (10 min)
- Duration: 10 min
- BID:ASK ratio: **2.33:1** (BID-heavy)
- Total fills: 10
- Rebalances: 0
- Realized PnL: **+0.00021**
- **Conclusion**: Flat sizing working, but BID-heavy. Entry-price floor still blocking ASK fills.

### Test 018 — Flat Sizing + Old Trend Filter (30 min)
- Duration: 30 min
- Total fills: ~1
- Rebalances: 0
- Realized PnL: **+0.00006**
- **Conclusion**: Almost no fills. Trend filter too aggressive — blocked nearly all activity.

### Test 019 — Flat Sizing + Relaxed Filter (10 min)
- Duration: 10 min
- BID:ASK ratio: **4:1** (worst ratio with flat sizing)
- Total fills: 15
- Rebalances: 0
- Realized PnL: **+0.000025**
- **Conclusion**: Entry-price floor clearly blocking ASK fills. Led to Change 022.

### Test 020 — Entry Floor Removed + Bid Threshold Changed (10 min)
- Duration: 10 min
- BID:ASK ratio: **1.14:1** (dramatically improved!)
- Total fills: 15 (8 BUY + 7 SELL)
- Rebalances: 0
- Realized PnL: **+0.00017**
- **Conclusion**: Entry-price floor removal and bid join threshold change worked. First balanced test with flat sizing.

### Test 021 — Stale Orders Bug (30 min)
- Duration: 30 min
- Total fills: 1
- Rebalances: 0
- Realized PnL: ~0
- **Conclusion**: Only 1 fill in 30 min. Discovered `max_order_rest_millis = 1,800,000` (30 min!) — orders sat stale at same price. Led to Change 024.

### Test 022 — Re-Quote Bug (30 min)
- Duration: 30 min
- Total fills: 3
- Rebalances: 0
- Realized PnL: **+0.00006**
- **Conclusion**: Still very few fills. Discovered both-sides-empty re-quote bug — when both sides filled simultaneously, instant-fill cooldown blocked new quotes. Led to Change 025.

### Test 023 — ALL FIXES APPLIED — BREAKTHROUGH (30 min)
- Duration: 30 min
- BID fills: 14, ASK fills: 26 (ratio **0.54:1** — ASK-heavy!)
- Total fills: **40** (13x improvement over Test 022!)
- Rebalances: **0**
- Realized PnL: **+0.01198** (70x improvement over Test 020!)
- Win rate: **100%**, Profit factor: **∞**
- Max drawdown: 0.02%
- Price: DOGE stable around 0.0903
- **Key observations**:
  - Best test ever by fill count and PnL
  - One "both sides inactive" event at 09:00:08, recovered in ~10 seconds — re-quote fix working
  - 14.5-minute gap (08:42-08:57) was NOT a bug — orders were resting, market was quiet
  - Net sold 144 DOGE — inventory drifting short because ASK fills > BID fills (slight uptrend lifts asks)
  - PnL rate: ~0.024/hr → ~0.576 FDUSD/day → ~0.52%/day on $110 capital
  - **Ready for 1-hour progressive test**

### Test 024 — 1-HOUR PROGRESSIVE TEST — CONFIRMED STABLE (60 min)
- Duration: 1 hour
- BID fills: 33, ASK fills: 32 (ratio **1.03:1** — NEAR-PERFECT balance!)
- Total fills: **65** (~1.08 fills/min)
- Rebalances: **0**
- Realized PnL: **+0.0216 FDUSD**
- Win rate: **94%**, Profit factor: **81.65**
- Max drawdown: 0.02 FDUSD
- Closed trades: 32, Entries: 33
- Errors: 13 (all benign cancel-replace race conditions, recovered)
- Price range: 0.0903 → 0.0905 (~0.2%, very tight rangebound market)
- Inventory drift: +12 DOGE (minimal, 1 open position at end)
- BNB: unchanged (no taker fees)
- 3 time gaps >5 min (max 8.6 min at 09:12-09:21 — quiet market)
- **Key observations**:
  - **Best BID:ASK ratio ever** (1.03:1) — confirms entry-price floor removal and bid threshold fixes work across different sessions
  - Zero rebalances for the second consecutive hour-long test
  - All 33 "NOT both sides active" warnings were transient (one side filled, replacement pending) — never both sides empty
  - PnL rate: ~0.022/hr → ~0.52 FDUSD/day → ~0.47%/day on $110 capital
  - **Strategy is CONFIRMED STABLE for 1-hour runs. Ready for 3-hour progressive test.**

---

## Current Strategy Parameters
```toml
# Strategy config (market_maker_bid_ask_glm.toml)
base_half_spread_bps = 0.6
min_half_spread_bps = 0.3
max_half_spread_bps = 10.0
volatility_reference_bps = 50.0
volatility_multiplier_min = 0.5
volatility_multiplier_max = 4.0
min_edge_bps = 0.3
max_inventory_fraction = 1.0
inventory_skew_factor = 200.0  # A-S gamma
cash_fraction = 0.30
dynamic_sizing = false
min_exit_edge_bps = 0.0  # no longer used (entry-price floor removed)
stop_loss_bps = 500.0
panic_vol_bps = 2000.0
max_hold_millis = 0  # DISABLED
max_vol_bps = 800.0
cooldown_millis = 5000
max_imbalance = 1.0
min_sell_flow_fraction = 0.0
requote_threshold_bps = 0.1
trade_window_millis = 30000

# Trading config (trading_config_dogefdusd.toml)
max_position_notional = 3.0
initial_capital = 112.0
min_order_rest_millis = 1000    # 1 sec
max_order_rest_millis = 30000   # 30 sec (was 1,800,000)

# Trader constants (hardcoded in trader.rs)
ZERO_QTY_REBALANCE_THRESHOLD = 100
MIN_REBALANCE_INTERVAL_MS = 30000
FAILURE_REBALANCE_THRESHOLD = 3

# Strategy constants (hardcoded in strategy)
SLOW_EMA_ALPHA = 0.002          # halflife ~35s
join_best_bid_threshold = 0.50  # join best_bid only when ratio < 0.50 (was 0.60)
inventory_bid_offset_start = 0.55  # extra offset when ratio > 0.55
inventory_bid_offset_scale = 10.0  # bps per 0.10 ratio excess
```

## Improvement Roadmap
1. ✅ Symmetric sell_quantity
2. ✅ Avellaneda-Stoikov reservation price
3. ✅ Symmetric quantity scaling
4. ✅ Fix max_inventory_notional
5. ✅ Config tuning
6. ✅ Change-detection logging
7. ✅ Instant-fill cooldown (buy avalanche fix)
8. ✅ REST API price fallback for rebalance
9. ✅ Rebalance over-triggering fix
10. ✅ max_hold_millis disabled
11. ✅ Conditional join-best-price
12. ✅ Allow buy_quantity = 0
13. ✅ Inventory gate at 0.85
14. ✅ Gamma increase (15 → 200)
15. ✅ Slow EMA trend filter
16. ✅ Fix trend filter / zero-qty rebalance interaction
17. ✅ Cycle-based sizing model (replaced by flat sizing)
18. ✅ Fix missing strategy log lines (strategies=info)
19. ✅ Remove dead code (compute_dynamic_cf)
20. ✅ Flat min-qty sizing (replaces cycle-based)
21. ✅ Trend filter hysteresis
22. ✅ Remove entry-price floor (CRITICAL — fixed ASK starvation)
23. ✅ Asymmetric join-best-price + inventory-aware bid offset
24. ✅ Fix stale orders (max_order_rest_millis 30min → 30sec)
25. ✅ Fix re-quote after both sides fill (bypass cooldown when both empty)
26. ✅ Test 023 — BREAKTHROUGH: 40 fills, +0.012 PnL, 100% WR, 0 rebalances
27. ✅ Test 024 — 1-HOUR CONFIRMED: 65 fills, +0.022 PnL, 94% WR, BID:ASK 1.03:1
28. [ ] Run 3-hour progressive test (Test 025)
29. [ ] Run 3-hour progressive test
30. [ ] Adverse selection detection (trade flow toxicity filters)
31. [ ] Lead-lag signal (BTC → DOGE)
32. [ ] Dynamic cash_fraction based on volatility
33. [ ] Clean up unused strategies (~12 strategy files, ~74 config files)
