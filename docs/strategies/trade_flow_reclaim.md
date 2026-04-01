# Trade-Flow Reclaim Taker

## Hypothesis

After a local pullback from the rolling high, price can reclaim from the local low with renewed buyer flow. If that reclaim is early enough and flow confirms, a short taker continuation may still cover taker costs.

## Data Requirements

- required: trade stream
- optional but preferred: `bookTicker`
- not required: depth for the first baseline version

This is a trade-first taker model. It can run on trade archives, but quote-aware replay still improves spread realism.

## Inputs

- rolling trade-window high and low
- trade-flow imbalance
- recent sub-window trade-flow imbalance
- pullback from local high in bps
- reclaim from local low in bps
- spread filter when quote data is available

## Entry Logic

Enter long with a taker buy when all of the following hold:

- enough trades exist in the rolling window
- current price sits inside a bounded pullback band below the local high
- price has reclaimed enough from the local low
- full-window buyer flow exceeds threshold
- recent sub-window buyer flow also confirms
- spread is not too wide when quotes exist
- no open position and cooldown expired

## Exit Logic

Exit with a taker sell when any of the following hold:

- stop loss is hit
- take profit is hit
- recent flow reverses negative enough
- maximum hold time expires

## Current Parameters

- window: `2000 ms`
- minimum trades: `14`
- pullback band: `4 .. 28 bps`
- minimum reclaim from low: `3 bps`
- minimum flow imbalance: `0.10`
- minimum recent flow imbalance: `0.18`
- maximum spread: `12 bps`
- stop loss: `14 bps`
- take profit: `18 bps`
- max hold: `4000 ms`
- cooldown: `2000 ms`
- assumed round-trip taker cost gate: `22.7 bps`
- minimum expected edge after cost: `0 bps`

## Known Risks

- current default is very selective and may win ranking only by not trading
- pullback and reclaim geometry are defined from trade-window extrema, not full depth depletion/recovery
- no short-side logic yet
- no volatility normalization or regime throttle yet

## Validation Notes

- validate on the same multi-day set as momentum and sweep
- compare both total PnL and trade count; low activity alone is not evidence of edge
- use `strategy-diagnostics` before loosening thresholds so the next tuning pass targets the real blockers

## Initial Multi-Day Results

Current three-day DOGEUSDT compare with default parameters:

- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
  - trades: `0`
  - realized PnL: `0.0`
  - fees: `0.0`
- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`
  - trades: `1`
  - realized PnL: `-0.0435852762`
  - fees: `0.0045409312`
- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
  - trades: `0`
  - realized PnL: `0.0`
  - fees: `0.0`

Aggregate ranking on the same three-day batch:

- rank: `1`
- total realized PnL: `-0.0435852762`
- total closed trades: `1`
- mean realized PnL: `-0.0145284254`
- worst max drawdown: `0.0004358528`

## Interpretation

- this model is currently the least bad strategy on the tested three-day batch
- that ranking is mostly driven by inactivity, not by demonstrated repeatable edge
- default gates are likely too tight for current DOGE trade-only replay

## Attribution And Diagnostics

- `trade-attribution` exports per-trade rows with rationale, confidence, expected edge, requested/executed size, cost components, hold time, and exit reason
- `strategy-diagnostics` exports reclaim-specific counters and gauges such as `reclaim.blocked_min_trades`, `reclaim.blocked_pullback_band`, `reclaim.blocked_reclaim`, `reclaim.blocked_flow`, and latest pullback/reclaim measurements
- diagnostics now also expose `reclaim.blocked_cost_gate`, `reclaim.last_expected_edge_bps`, and `reclaim.last_edge_after_cost_bps` so reclaim geometry can be separated from modeled taker-cost insufficiency
- on the latest diagnostics run over `DOGEUSDT-trades-2025-06-28.zip`, the dominant blockers were `blocked_min_trades`, `blocked_pullback_band`, and `blocked_reclaim`

## Latest Optimization Snapshot

- aggregated three-day optimization currently ranks `min_reclaim_from_low_bps = 4.0` best among the tested values with total realized PnL `-0.0346988558`
- `min_reclaim_from_low_bps = 2.0` ranked second with total realized PnL `-0.0396207844`
- `min_reclaim_from_low_bps = 3.0` matched the old default result at `-0.0435852762`
- `min_pullback_from_high_bps = 8.0` improved the batch slightly to `-0.0416030303`
- the first tested `min_trade_flow_imbalance` values `0.04`, `0.08`, `0.10`, and `0.14` tied on the tested batch
- first walk-forward validation trained on the first two daily archives, selected `4.0`, and then produced `0` held-out trades / `0` realized PnL on the final day

## Cost-Aware Gate Result

- after adding a hard taker-cost gate based on the current modeled round-trip drag, this strategy also dropped to `0` entries and `0` closed trades on the same three-day batch
- `reclaim.blocked_cost_gate` appeared mainly on `2025-08-08`, while `blocked_min_trades` and `blocked_pullback_band` still dominated overall
- this confirms reclaim was not a hidden profitable candidate; its earlier "least bad" ranking came from low activity, and the few surviving setups also fail honest taker-cost screening

## Next Revision Targets

- re-run full compare with `min_reclaim_from_low_bps = 4.0` before changing multiple dimensions at once
- use diagnostics on the tuned config to see whether `min_trades_in_window` or pullback geometry is the next real bottleneck
- only treat reclaim as a lead candidate if it stays least bad while trading enough to be meaningful out of sample
