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
