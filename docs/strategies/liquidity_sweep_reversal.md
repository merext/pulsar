# Liquidity Sweep Reversal Taker

## Hypothesis

A fast downside sweep caused by aggressive selling can briefly exhaust local liquidity. If price reclaims upward
from the local low while buyer-initiated flow turns positive, a short rebound may be tradable even after taker costs.

## Data Requirements

- required: trade stream
- optional but preferred: `bookTicker`
- not required: depth for the first baseline version

## Inputs

- rolling trade window high/low prices
- trade-flow imbalance
- recent sub-window trade-flow imbalance
- reclaim from local low in bps
- reclaim relative to rolling VWAP
- sweep drop from local high in bps
- ratio of unusually large trades inside the window
- spread filter when quote data is available
- order book imbalance when quotes are available

## Entry Logic

Enter long with a taker buy when all of the following hold:

- enough trades exist in the rolling window
- a meaningful downward sweep from the local high occurred
- price has reclaimed upward from the local low, but not too far already
- buyer-initiated flow has turned positive enough
- recent sub-window buyer flow also confirms reclaim continuation
- reclaim has not stretched too far above rolling VWAP
- large-trade ratio suggests the move was a real local flush rather than noise
- order book should not lean against the bounce when quote data exists
- spread remains acceptable when quotes exist
- no open position and cooldown expired

## Exit Logic

Exit with a taker sell when any of the following hold:

- stop loss is hit
- take profit is hit
- buyer reclaim fails and flow flips negative again
- maximum hold time expires

## Current Parameters

- window: `2000 ms`
- minimum trades: `18`
- minimum sweep drop: `10 bps`
- buyer reclaim imbalance: `0.12`
- recent buyer imbalance: `0.20`
- reclaim band: `2 .. 20 bps`
- max reclaim above VWAP: `6 bps`
- max spread: `14 bps`
- minimum large-trade ratio: `0.12`
- minimum order book imbalance: `0.05`
- stop loss: `16 bps`
- take profit: `20 bps`
- max hold: `5000 ms`
- cooldown: `3000 ms`
- assumed round-trip taker cost gate: `22.7 bps`
- minimum expected edge after cost: `0 bps`

## Known Risks

- sweep detection uses only trades, not full depth depletion
- large-trade ratio may be symbol-specific and regime-sensitive
- no explicit volatility normalization yet
- long-only baseline for now

## Validation Notes

- validate on the same multi-day set as trade-flow momentum
- compare trade frequency, net PnL, and cost drag against the momentum baseline
- if this outperforms momentum materially, prioritize sweep logic refinement before deeper momentum tuning

## Initial Multi-Day Results

First baseline replay on the same three DOGEUSDT daily archives:

- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
  - trades: `1`
  - realized PnL: `-0.0504`
  - fees: `0.0045`
  - win rate: `0.0`
- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`
  - trades: `2`
  - realized PnL: `-0.1268`
  - fees: `0.0091`
  - win rate: `0.0`
- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
  - trades: `1`
  - realized PnL: `-0.0397`
  - fees: `0.0045`
  - win rate: `0.0`

### Interpretation

- baseline sweep logic trades far less often than momentum
- on the tested set it still loses money, but loses less than momentum on two of three days
- this makes it a better candidate for near-term refinement than adding deeper momentum sub-variants right now

## First Refinement Pass

One major refinement pass was added without branching into sub-variants:

- require recent sub-window buyer imbalance, not only full-window imbalance
- reject entries that reclaim too far above rolling VWAP
- require non-negative quote-side support through order book imbalance when quotes are available

### Result

- this was a structurally correct improvement to entry quality filters
- on the current three-day batch it did not materially change realized outcomes yet
- the strategy remains in the pipeline and is not rejected; deeper tuning and live emulation are still required before any discard decision

## Latest Optimization Snapshot

- aggregated three-day optimization currently ranks `min_sweep_drop_bps = 12.0` best among the tested values with total realized PnL `-0.0703276302`
- `min_sweep_drop_bps = 10.0` ranked second with total realized PnL `-0.2152563187`
- `min_sweep_drop_bps = 8.0` ranked third with total realized PnL `-0.2211656508`
- `min_recent_buyer_imbalance = 0.16`, `0.20`, and `0.24` tied on the tested three-day batch
- the first walk-forward test trained on the first two daily archives, selected `12.0`, and then produced `0` held-out trades / `0` realized PnL on the final day
- strategy entry sizing now uses shared `StrategyContext::capped_entry_quantity(...)` sizing rather than local ad hoc notional logic

## Attribution And Diagnostics

- `trade-attribution` now exports one CSV row per closed trade with rationale, confidence, expected edge, requested/executed size, cost components, hold time, and exit reason
- `strategy-diagnostics` now exports counters and gauges from this model, including `sweep.blocked_min_trades`, `sweep.blocked_sweep_drop`, `sweep.blocked_reclaim_band`, `sweep.entries`, and exit-reason counts
- diagnostics now also expose `sweep.blocked_cost_gate`, `sweep.last_expected_edge_bps`, and `sweep.last_edge_after_cost_bps` so we can separate geometry blockers from pure cost insufficiency
- on the latest single-day diagnostics run over `DOGEUSDT-trades-2025-06-28.zip`, the dominant blockers were `blocked_min_trades` and `blocked_sweep_drop`
- these exports are now the preferred way to decide whether the next refinement should tighten sweep detection, reclaim geometry, or flow confirmation

## Cost-Aware Gate Result

- after adding a hard taker-cost gate based on the current modeled round-trip drag, this strategy produced `0` entries and `0` closed trades on the same three-day DOGEUSDT batch
- `sweep.blocked_cost_gate` appeared on all three days, but remained much smaller than `blocked_min_trades` and `blocked_sweep_drop`
- this strengthens the current research view: even when sweep geometry appears acceptable, the rebound edge is usually still too small for honest taker round trips
