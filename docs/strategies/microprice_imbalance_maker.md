# Microprice Imbalance Maker

## Hypothesis

When the top of book becomes bid-heavy and microprice shifts above mid, passive bid placement may capture a short-lived favorable fill and a small mean move without paying taker spread costs.

## Data Requirements

- required: `bookTicker`
- preferred: depth
- optional: trade stream for replay context

This model is quote-aware and intended only for quote/depth-enabled replay or live emulate. It is not honest on trade-only archives.

## Inputs

- top-of-book bid / ask
- microprice relative to mid
- order book imbalance
- spread in bps
- position state and hold time

## Entry Logic

Enter with a passive maker buy when all of the following hold:

- spread is below a bounded threshold
- order book imbalance is sufficiently positive
- microprice edge above mid exceeds threshold
- no open position
- entry cooldown expired

## Exit Logic

Exit with a passive maker sell when any of the following hold:

- stop loss is hit
- take profit is hit
- order book imbalance turns negative
- maximum hold time expires

## Current Parameters

- min order book imbalance: `0.08`
- min microprice edge: `1.5 bps`
- max spread: `8 bps`
- max quote notional: `20`
- max hold: `3000 ms`
- stop loss: `10 bps`
- take profit: `12 bps`
- cooldown: `2000 ms`

## Realism Notes

- this model depends on queue and passive fill approximation; it should not be treated as production-realistic maker validation yet
- current queue model is only `v1` baseline scaffolding
- passive fills remain an approximation until deeper queue/fill research is completed

## Validation Notes

- use only captured quote-aware datasets
- compare only on identical quote-aware batches
- treat current results as maker-baseline research, not production evidence
- current verified quote-aware replays on `session_20260331_v3.jsonl`, `session_20260331_v4.jsonl`, `smoke_batch_part_001.jsonl`, and `smoke_batch_part_002.jsonl` produced `0` entries / `0` trades
- fresh live-validation captures on `live_validation_1m_20260401.jsonl`, `live_validation_5m_20260401.jsonl`, and `live_validation_10m_20260401.jsonl` also produced `0` entries / `0` trades
- the live-validation captures contained quote and depth coverage throughout replay, so the current issue is not simply missing quote-aware input
- next useful step is longer quote-aware capture collection and entry-condition attribution before threshold loosening or deeper maker tuning

## Attribution And Diagnostics

- maker execution reports now carry rationale, confidence, expected edge, and decision metrics into shared attribution plumbing
- `strategy-diagnostics` can export maker counters/gauges, and decision metrics are emitted on maker entry/exit decisions when they occur
- current zero-entry live-validation runs mean there are no trade-attribution rows yet for this model on the tested live captures
- maker diagnostics remain informative only on quote/depth-aware datasets and must not be used to overstate realism beyond the explicit queue/fill approximations
