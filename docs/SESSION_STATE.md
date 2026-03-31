# Session State

## Current Status

The repository has been reset for a fundamental redesign.

Completed:

- Removed all legacy strategies and their configs.
- Live market data path is websocket-only.
- Historical trade archive validation still works.
- Budget-aware account and execution simulation exists, but it is still tied to the older signal-based design.
- Added first version of the new event-driven foundation:
  - `trade/src/market.rs`
  - `trade/src/execution.rs`
  - refactored `Strategy` to `on_event + decide`
  - refactored `Trader` to `OrderIntent` / `ExecutionReport`
  - added `strategies::NullStrategy` for infrastructure validation
  - migrated Binance trader to the new event-driven path
- Added quote-aware websocket ingestion:
  - `BinanceClient::market_event_stream()` now multiplexes `trade` + `bookTicker`
  - `MarketState` now supports top-of-book derived metrics such as spread, imbalance, and microprice
  - tests added for quote-aware state behavior
- Emulated execution now prefers top-of-book quotes when available:
  - taker buys reference ask instead of last trade
  - taker sells reference bid instead of last trade
  - synthetic spread widening is only added when no explicit quote is available
- Added first real strategy implementation:
  - `strategies::TradeFlowMomentumStrategy`
  - config file: `config/strategies/trade_flow_momentum.toml`
  - strategy spec: `docs/strategies/trade_flow_momentum.md`
- Added second baseline taker strategy:
  - `strategies::LiquiditySweepReversalStrategy`
  - config file: `config/strategies/liquidity_sweep_reversal.toml`
  - strategy spec: `docs/strategies/liquidity_sweep_reversal.md`
- Added rule that every strategy must be documented under `docs/strategies/`
- Backtest CLI now supports strategy selection instead of hardwiring a single model
- Multi-day replay was run on three DOGEUSDT archives and the first model is currently unprofitable on all three days
- Comparative replay now exists for two taker baselines over the same three-day batch
- Added minimal compare workflow in CLI:
  - `compare --uris ... --strategies ...`
  - emits CSV summary for identical strategy/dataset batch runs
- Added first major refinement pass to `LiquiditySweepReversalStrategy`:
  - recent-flow confirmation
  - VWAP reclaim filter
  - quote-side order book support filter
- The first major sweep refinement did not materially improve the current three-day compare batch
- Added first major refinement pass to `TradeFlowMomentumStrategy`:
  - recent-flow confirmation
  - VWAP stretch filter
  - quote-side order book support filter
- The first major momentum refinement slightly improved `2026-03-30` and left the rest of the three-day batch broadly unchanged
- Live and emulated CLI orchestration now uses quote-aware mixed market events:
  - `BinanceClient::market_event_stream()` feeds `trade + bookTicker`
  - `trade` command uses real execution path on the mixed stream
  - `emulate` command uses simulated execution path on the mixed stream
  - trader now logs market data source status for backtest vs live/emulate paths
- Added initial quote/depth capture pipeline:
  - `BinanceClient::market_event_stream_with_depth()` now multiplexes `trade + bookTicker + depth`
  - CLI command `capture` writes raw JSONL under user-selected path
  - short capture run verified with real mixed events written to disk
  - pipeline documented in `docs/CAPTURE_PIPELINE.md`
- Added captured JSONL mixed-event replay support:
  - `BinanceClient::market_event_data_from_uri()` now loads captured JSONL as normalized `MarketEvent` replay
  - `backtest` and `compare` now auto-detect `.jsonl` / `.ndjson` and use mixed-event replay instead of trade-only replay
  - replay logging now distinguishes captured historical replay from trade-only historical replay
  - short end-to-end replay on `/tmp/pulsar_capture.jsonl` verified successfully for both baseline taker models
- Added normalized mixed-event capture ordering and replay summaries:
  - capture records now store `capture_sequence` and `captured_at_ms` for replay-safe ordering
  - exchange-native timing fields are preserved as optional raw fields instead of being treated as authoritative replay time
  - replay summary now reports event mix, parse errors, capture-order regressions, capture-time regressions, and event-time regressions
  - verified longer dataset capture at `data/binance/capture/DOGEUSDT/session_20260331_v3.jsonl`
  - verified longer captured replay end-to-end with both taker baselines
- Added capture metadata sidecar support:
  - each capture now writes `<capture>.metadata.json` next to the JSONL dataset
  - sidecar records dataset-level counts, timing bounds, schema version, and replay ordering semantics
  - verified sidecar generation on `data/binance/capture/DOGEUSDT/session_20260331_v4.jsonl.metadata.json`
- Added batch captured-dataset index support:
  - new CLI command `capture-index --root data/binance/capture`
  - index is built from metadata sidecars without rescanning full JSONL payloads
  - verified on current captured sessions under `data/binance/capture/`
- Added batch replay orchestration over indexed captured datasets:
  - new CLI command `capture-compare --root data/binance/capture ...`
  - batch replay uses indexed sidecars to discover datasets, then runs normal replay/backtest flow on the resolved JSONL files
  - verified on indexed DOGEUSDT capture dataset batch
- Added legacy capture metadata backfill support:
  - new CLI command `capture-backfill --input ...`
  - backfill derives sidecar metadata from existing captured JSONL replay summaries
  - verified on `session_20260331_v2.jsonl` and `session_20260331_v3.jsonl`
- Added minimal indexed dataset filters:
  - `capture-index` and `capture-compare` now support minimum event-count filters and capture-time presence/time filters
  - verified that quote-aware filters exclude the trade-only backfilled dataset and retain mixed-event captures
- Added minimal sequential capture-batch orchestration:
  - new CLI command `capture-batch --batch-id ... --parts ... --duration-secs ...`
  - batch capture reuses the normal capture path and writes sequentially named JSONL + sidecar files
  - verified on a two-part DOGEUSDT smoke batch

## Last Verified Commands

- `cargo check`
- `cargo test`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- --strategy trade-flow-momentum backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
- `cargo run -p binance-bot -- --strategy liquidity-sweep-reversal backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`
- `cargo run -p binance-bot -- compare --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo test -p binance-exchange -p trade -p strategies`
- `cargo run -p binance-bot -- capture --output /tmp/pulsar_capture.jsonl --duration-secs 2`
- `cargo test -p binance-exchange --test client_parse_tests`
- `cargo test -p binance-bot`
- `cargo run -p binance-bot -- --strategy trade-flow-momentum backtest --uri /tmp/pulsar_capture.jsonl`
- `cargo run -p binance-bot -- --strategy liquidity-sweep-reversal backtest --uri /tmp/pulsar_capture.jsonl`
- `cargo run -p binance-bot -- compare --uris /tmp/pulsar_capture.jsonl --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/session_20260331_v3.jsonl --duration-secs 10 --depth-levels 5`
- `cargo run -p binance-bot -- compare --uris data/binance/capture/DOGEUSDT/session_20260331_v3.jsonl --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/session_20260331_v4.jsonl --duration-secs 5 --depth-levels 5`
- `cargo run -p binance-bot -- compare --uris data/binance/capture/DOGEUSDT/session_20260331_v4.jsonl --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- capture-index --root data/binance/capture`
- `cargo run -p binance-bot -- capture-compare --root data/binance/capture --symbol DOGEUSDT --limit 1 --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- capture-backfill --input data/binance/capture/DOGEUSDT/session_20260331_v2.jsonl --duration-secs 10 --depth-levels 5`
- `cargo run -p binance-bot -- capture-backfill --input data/binance/capture/DOGEUSDT/session_20260331_v3.jsonl --duration-secs 10 --depth-levels 5`
- `cargo run -p binance-bot -- capture-compare --root data/binance/capture --symbol DOGEUSDT --limit 3 --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- capture-index --root data/binance/capture --symbol DOGEUSDT --min-book-ticker-events 1 --require-captured-at`
- `cargo run -p binance-bot -- capture-compare --root data/binance/capture --symbol DOGEUSDT --min-book-ticker-events 1 --require-captured-at --limit 5 --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- capture-batch --batch-id smoke_batch --parts 2 --duration-secs 3 --depth-levels 5`

## Important Constraints

- Do not reintroduce REST polling fallback for live HFT logic.
- Do not restore deleted RSI / mean reversion / advanced-order strategies.
- Do not pretend maker backtests are realistic without quote/depth history and queue logic.
- Every strategy must be documented under `docs/strategies/`.
- Do not discard a model after shallow iteration; require deep tuning and live emulation evidence before rejecting it.

## Immediate Next Steps

1. Add new shared domain modules for:
   - market events
   - market state
   - execution intents / reports
2. Promote `LiquiditySweepReversalStrategy` as the current lead taker baseline for the next refinement loop.
3. Keep `TradeFlowMomentumStrategy` refinements at the major-filter level, not deep sub-variant exploration.
4. Extend the new compare workflow only if a major architecture need appears; keep secondary reporting wishes in backlog.
5. Keep both models alive until deep tuning plus live emulation justify rejection.
6. Next major architecture step after this is using batch capture plus filters to build materially longer quote-aware validation sets before any maker-research transition.

## Architectural Guidance

- Keep major architecture changes immediate when they unblock multiple models.
- Push lower-value refinements into roadmap backlog instead of deepening local optimizations too early.

## Multi-Day Testing Requirement

All future strategy validation must use multiple daily datasets, not isolated one-day anecdotes.

Minimum acceptable research batch:

- at least 3 to 5 daily archives
- mixed regimes if possible

## Current Research Readout

Initial three-day validation for `TradeFlowMomentumStrategy`:

- `2025-06-28`: `1` closed trade, realized PnL `-0.0405`
- `2025-08-08`: `13` closed trades, realized PnL `-0.6938`
- `2026-03-30`: `5` closed trades, realized PnL `-0.2450`

Interpretation:

- current burst-following entry is too weak after taker costs
- the first implemented model is a usable baseline, not a valid production candidate
- next research should focus on stricter entry quality and comparative testing versus liquidity sweep logic

Initial three-day validation for `LiquiditySweepReversalStrategy`:

- `2025-06-28`: `1` closed trade, realized PnL `-0.0504`
- `2025-08-08`: `2` closed trades, realized PnL `-0.1268`
- `2026-03-30`: `1` closed trade, realized PnL `-0.0397`

Comparison takeaway:

- both baselines are still negative after realistic taker costs
- liquidity sweep baseline currently looks more promising because it loses less while trading less often
- next iteration should focus on major-quality filters and better comparison/reporting, not local micro-tuning

Refinement takeaway:

- first major refinement of liquidity sweep improved structure but not the current three-day results
- this is evidence to continue disciplined tuning, not to discard the model
- first major refinement of momentum improved one day slightly but did not change the broader ordering of the two models
- captured mixed-event replay now works end-to-end, but the verified fixture was only a short 32-event sample and produced zero trades for both baselines
- next validation should use materially longer captured sessions before drawing any quote-aware model conclusions
- longer normalized capture replay now also works end-to-end on a 617-event sample, and both baselines still produced zero trades on that sample
- replay summary shows capture ordering is stable on the normalized dataset, while exchange/native event-time ordering can still regress across mixed event types
- this confirms replay must rely on capture-order semantics for mixed captured datasets rather than raw exchange timing across event classes
- metadata sidecar now gives a stable dataset contract for future batch tooling without forcing replay code to rescan JSONL for basic dataset facts
- batch index now provides the first lightweight dataset-discovery layer needed for quote-aware validation over multiple captured sessions
- indexed batch replay now works end-to-end, but current indexed coverage is still thin because only the latest capture has a sidecar
- legacy backfill removed that immediate coverage bottleneck for the current DOGEUSDT captures, expanding indexed coverage from one dataset to three
- minimal filtering now lets batch replay focus on quote-aware datasets instead of mixing in trade-only legacy captures
- sequential capture-batch now expands quote-aware indexed coverage without manual per-file capture invocation

## Resume Checklist

- Read `docs/AGENT_PLAYBOOK.md`
- Read `docs/ARCHITECTURE.md`
- Read `docs/ROADMAP.md`
- Read this file
- Run `cargo check`
- Continue from the next unchecked roadmap item in Phase 2 / Phase 4
