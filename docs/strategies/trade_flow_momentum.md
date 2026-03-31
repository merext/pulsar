# Trade-Flow Momentum Taker

## Hypothesis

Short bursts of buyer-initiated aggressive flow, confirmed by positive short-window price drift,
can produce a brief continuation move large enough to cover taker fees, spread, slippage, and latency.

## Data Requirements

- required: trade stream
- optional but preferred: `bookTicker`
- not required: depth

This is a trade-first taker model. It can be validated on existing trade archives and becomes more realistic
in live/emulated mode when top-of-book quotes are available.

## Inputs

- rolling trade window statistics from `MarketState`
- trade-flow imbalance
- short-window price drift in bps
- trade burst intensity in trades per second
- spread filter when quote data is available

## Entry Logic

Enter long with a taker buy when all of the following hold:

- enough trades accumulated in the rolling window
- buyer-initiated flow imbalance exceeds threshold
- short-window price drift is positive and within a bounded chase range
- burst rate exceeds threshold
- spread is not too wide when quotes exist
- no open position
- entry cooldown has expired

## Exit Logic

Exit with a taker sell when any of the following hold:

- stop loss in bps is hit
- take profit in bps is hit
- flow reverses beyond threshold
- maximum hold time expires

## Current Parameters

- window: `1500 ms`
- minimum trades: `12`
- minimum flow imbalance: `0.18`
- drift band: `6 .. 35 bps`
- maximum spread: `12 bps`
- minimum burst: `10 trades/sec`
- cooldown: `2500 ms`
- max hold: `4000 ms`
- stop loss: `18 bps`
- take profit: `24 bps`

## Known Risks

- trade-only replay cannot fully validate quoted spread conditions
- burst filters may overfit to DOGE-specific activity regimes
- no short-side logic yet
- no daily loss lockout or regime-aware throttling yet

## Validation Notes

- must be tested on multiple daily archives
- results must be compared against the next taker model on the same day set
- live/emulated behavior should be checked with `trade + bookTicker` to confirm spread filtering and quote-aware fills

## Initial Multi-Day Results

First implementation was replayed on three DOGEUSDT daily archives.

- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
  - trades: `1`
  - realized PnL: `-0.0405`
  - fees: `0.0045`
  - win rate: `0.0`
- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`
  - trades: `13`
  - realized PnL: `-0.6938`
  - fees: `0.0589`
  - win rate: `0.0`
- `data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
  - trades: `5`
  - realized PnL: `-0.2450`
  - fees: `0.0227`
  - win rate: `0.0`

### Interpretation

- current entry trigger is able to find bursty momentum windows
- current exits and taker costs dominate the edge in replay
- the model is not deployable in current form

### Relative Standing vs Sweep Baseline

- worse than `liquidity_sweep_reversal` on `2025-08-08` and `2026-03-30`
- slightly better than `liquidity_sweep_reversal` on `2025-06-28`
- currently better treated as an exploratory baseline than as the lead candidate

### Next Revision Targets

- tighten entry quality with stronger drift and burst confirmation
- add a no-trade filter for weak or decelerating continuation after entry
- revisit sizing so strategy intent can use shared position sizing instead of ad hoc notional sizing
- compare the revised model against the planned liquidity sweep taker on the same three-day batch
