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
- Completed Phase 2 latency and impact accounting cleanup:
  - backtest execution now tracks synthetic spread, slippage, latency impact, and market impact as separate components
  - latency impact is now separated from market impact instead of double-counting shared impact terms
  - execution reports now expose per-component bps fields so later attribution work has a cleaner base
  - verified with focused backtest tests and downstream bot tests
- Completed the remaining Phase 2 horizontal items:
  - trade-only taker execution realism now uses separate slippage, latency, and market-impact components with explicit attribution fields
  - shared backtest/emulation flow now applies inventory and drawdown guardrails before allowing further entries
  - performance metrics now track fill ratio, rejection rate, average latency, and average execution attribution components
  - replay observability now reports event-mix diagnostics, including trade/quote/depth counts and stale/missing quote relationships
- Added Phase 3 parameter search scaffolding:
  - new CLI command `search --strategy ... --parameter ... --values ... --uris ...`
  - search reuses the existing backtest flow and injects one parameter override at a time via temporary strategy config materialization
  - smoke-tested on `trade-flow-momentum` with `min_price_drift_bps` over a daily DOGEUSDT trade dataset
- Added Phase 5 maker baseline:
  - `BacktestEngine` now exposes queue-model-v1 estimates via `estimate_passive_fill(...)`
  - maker backtest/emulation now routes through `simulate_passive_order(...)` and can return `Pending`, `PartiallyFilled`, or `Filled`
  - passive-fill tests were added in `trade/tests/backtest_tests.rs`
  - new `MicropriceImbalanceMakerStrategy` was added with config and strategy doc
  - live maker orders remain disabled intentionally and return `live_maker_not_enabled`
- Added Phase 6 per-model parameter optimization baseline:
  - new CLI command `optimize --strategy ... --parameter ... --values ... --uris ...`
  - optimization reuses `search` results and emits one aggregated ranking row per parameter value across the full dataset batch
  - ranking currently sorts by total realized PnL, then mean realized PnL, then worst max drawdown
- Added shared strategy sizing helper:
  - `StrategyContext::capped_entry_quantity(...)` now centralizes capped entry sizing in `trade/src/strategy.rs`
  - `trade-flow-momentum`, `liquidity-sweep-reversal`, and `microprice-imbalance-maker` now reuse the same helper instead of duplicating local notional math
- Added replay regime tagging and aggregate compare reporting:
  - replay/backtest summaries now infer `source:*`, `activity:*`, `book:*`, and `quote_coverage:*` tags from dataset format and event coverage
  - `compare` and `capture-compare` now emit both per-dataset rows and aggregated per-strategy ranking rows
- Added Phase 6 walk-forward validation:
  - new CLI command `walk-forward --strategy ... --parameter ... --values ... --uris ... --min-train-size ... --test-size ...`
  - each fold optimizes on the train window, then evaluates the selected value on the next held-out test window
- Added Phase 6 ML-ready feature export:
  - new CLI command `features --uri ... --output ... --lookahead-events ...`
  - exported rows include event kind/time, quote/depth presence, rolling trade stats, microstructure features, forward-return target, and inferred regime tags
- Completed honest Phase 5 live validation on fresh quote-aware captures:
  - captured `live_validation_1m_20260401.jsonl` with `1590` events
  - captured `live_validation_5m_20260401.jsonl` with `8860` events
  - captured `live_validation_10m_20260401.jsonl` with `26647` events
  - `microprice-imbalance-maker` still produced `0` entries / `0` trades on all three live-validation replays
- Latest optimization and research findings:
  - `trade-flow-momentum.min_price_drift_bps = 12.0` is currently least bad on the tested three-day batch with total realized PnL `-0.0542378713`
  - `trade-flow-momentum.min_trade_flow_imbalance = 0.18`, `0.22`, and `0.26` tied on the tested batch
  - `liquidity-sweep-reversal.min_sweep_drop_bps = 12.0` is currently least bad on the tested three-day batch with total realized PnL `-0.0703276302`
  - `liquidity-sweep-reversal.min_recent_buyer_imbalance = 0.16`, `0.20`, and `0.24` tied on the tested batch
  - walk-forward on `liquidity-sweep-reversal.min_sweep_drop_bps` trained on the first two daily archives, selected `12.0`, and produced `0` test trades / `0` realized PnL on the final held-out day
  - feature export wrote `405` rows to `data/binance/features/DOGEUSDT/session_20260331_v4_features.csv`
- Added richer replay attribution and diagnostics exports:
  - new CLI command `trade-attribution --uris ... --strategies ...`
  - new CLI command `strategy-diagnostics --uris ... --strategies ...`
  - exported trade rows now include trade id, symbol, gross/net PnL, fees, expected edge, rationale, confidence, requested vs executed size, execution-cost components, hold time, and exit reason
  - strategies can now emit structured counters/gauges and decision metrics for replay decomposition
- Added richer trade/execution attribution plumbing:
  - `ExecutionReport` now carries symbol, rationale, decision confidence, and decision metrics
  - `TradeRecord` now carries trade id, expected edge, attribution fields, hold time, exit reason, and entry price
  - pending entry/exit attribution now propagates through shared trade management and replay metrics
- Added deeper taker-model decomposition:
  - `trade-flow-momentum` now reports blocked-entry counters plus latest signal gauges
  - `liquidity-sweep-reversal` now reports blocked-entry counters plus latest signal gauges
  - `microprice-imbalance-maker` now emits decision metrics on maker entry/exit decisions
- Added new taker model `TradeFlowReclaimStrategy`:
  - config file: `config/strategies/trade_flow_reclaim.toml`
  - strategy file: `strategies/src/trade_flow_reclaim.rs`
  - strategy doc must exist at `docs/strategies/trade_flow_reclaim.md`
  - current default behavior is intentionally selective and often blocked by `min_trades`, pullback-band, and reclaim checks
- Added capture-integrity and replay-support groundwork:
  - captured replay loaders can now read gzipped JSONL files
  - capture sidecars/index rows now store `data_size_bytes`, `data_sha256`, `sidecar_verified`, `capture_time_quality`, `quote_presence_quality`, and `depth_presence_quality`
  - capture-batch now supports `--gap-secs`
  - `capture-index` and `capture-compare` now expose `--min-quote-quality`, `--min-depth-quality`, and `--require-verified-sidecar`
  - rotation/compression is wired through `config/capture_rotation.toml` and now validated on a live smoke batch that produced rotated `.jsonl.gz` parts with sidecars
- Fixed replay source detection for gzipped captured datasets:
  - `.jsonl.gz` and `.ndjson.gz` now route into captured mixed-event replay instead of falling back to trade CSV parsing
  - this fixed a real bug discovered during rotated-capture validation
- Latest three-strategy compare on the tested DOGEUSDT three-day batch ranked:
  - `trade-flow-reclaim`: rank `1`, total realized PnL `-0.0435852762`, `1` closed trade across `3` datasets
  - `liquidity-sweep-reversal`: rank `2`, total realized PnL `-0.2152563187`, `4` closed trades across `3` datasets
  - `trade-flow-momentum`: rank `3`, total realized PnL `-0.9666310465`, `19` closed trades across `3` datasets
  - this ranking is descriptive only; `trade-flow-reclaim` is currently least bad mainly because it barely trades, not because it has demonstrated robust edge
- Latest reclaim optimization and walk-forward findings:
  - `trade-flow-reclaim.min_reclaim_from_low_bps = 4.0` is currently least bad on the tested three-day batch with total realized PnL `-0.0346988558`
  - `min_reclaim_from_low_bps = 2.0` ranked second at `-0.0396207844`
  - `min_reclaim_from_low_bps = 3.0` ranked third at `-0.0435852762`
  - `min_pullback_from_high_bps = 8.0` improved the batch slightly to `-0.0416030303`
  - `min_trade_flow_imbalance = 0.04`, `0.08`, `0.10`, and `0.14` tied on the tested batch
  - walk-forward on `min_reclaim_from_low_bps` trained on the first two daily archives, selected `4.0`, and then produced `0` held-out trades / `0` realized PnL on the final day
- Rotated capture validation findings:
  - a live 5-second smoke run with temporary `config/capture_rotation.toml` settings `max_events_per_file = 100` and `gzip = true` produced `/tmp/pulsar_rotation_batch_part_001.jsonl.gz`, `_002`, and `_003` plus matching sidecars
  - `capture-index --root /tmp --symbol DOGEUSDT --min-quote-quality dense --min-depth-quality mixed --require-verified-sidecar` correctly discovered all rotated parts
  - `compare` over the rotated `.jsonl.gz` parts now runs through captured replay successfully after the gzip-detection fix
  - `microprice-imbalance-maker` still produced `0` entries / `0` trades on the rotated smoke batch
- Added cost-aware taker entry gating across all three taker strategies:
  - `trade-flow-momentum`, `liquidity-sweep-reversal`, and `trade-flow-reclaim` now expose `assumed_round_trip_taker_cost_bps` and `min_expected_edge_after_cost_bps`
  - entry decisions now reject setups whose raw expected edge does not clear the modeled round-trip taker drag
  - diagnostics now expose per-model `blocked_cost_gate`, `last_expected_edge_bps`, and `last_edge_after_cost_bps`
  - the updated three-day DOGEUSDT compare produced `0` entries / `0` closed trades for all three taker strategies, which is scientifically consistent with earlier attribution showing edge below cost
  - updated diagnostics show the cost gate matters, but `blocked_min_trades` still dominates, so the next profitable direction likely needs stronger regimes or different execution, not just one more threshold tweak
- Added a dedicated model-testing explainer document:
  - `docs/MODEL_TESTING.md` now summarizes the current model roster, command usage, cost-gate effect, and honest testing interpretation in a table-first format
- Validated the newest live-like capture dataset `live_train_20260401_210321.jsonl` against the baseline model set:
  - `trade-flow-momentum`, `liquidity-sweep-reversal`, `trade-flow-reclaim`, and `microprice-imbalance-maker` all produced `0` entries / `0` closed trades
  - replay summary reported dense quotes and depth presence, so this was not a missing-data failure
  - diagnostics showed `trade-flow-momentum` got closest to firing, but expected edge after cost was still negative
  - diagnostics showed `liquidity-sweep-reversal` and `trade-flow-reclaim` mostly failed earlier structural filters such as sweep-drop and pullback-band conditions

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
- `cargo test -p trade --test backtest_tests`
- `cargo test -p trade`
- `cargo test -p binance-bot`
- `cargo run -p binance-bot -- search --strategy trade-flow-momentum --parameter min_price_drift_bps --values 6.0,9.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
- `cargo test -p strategies --test microprice_imbalance_maker_tests`
- `cargo run -p binance-bot -- compare --uris data/binance/capture/DOGEUSDT/session_20260331_v3.jsonl data/binance/capture/DOGEUSDT/session_20260331_v4.jsonl data/binance/capture/DOGEUSDT/smoke_batch_part_001.jsonl data/binance/capture/DOGEUSDT/smoke_batch_part_002.jsonl --strategies microprice-imbalance-maker`
- `cargo run -p binance-bot -- optimize --strategy liquidity-sweep-reversal --parameter min_sweep_drop_bps --values 8.0,10.0,12.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- optimize --strategy trade-flow-momentum --parameter min_price_drift_bps --values 6.0,9.0,12.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- optimize --strategy trade-flow-momentum --parameter min_trade_flow_imbalance --values 0.18,0.22,0.26 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- optimize --strategy liquidity-sweep-reversal --parameter min_recent_buyer_imbalance --values 0.16,0.20,0.24 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- walk-forward --strategy liquidity-sweep-reversal --parameter min_sweep_drop_bps --values 8.0,10.0,12.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip --min-train-size 2 --test-size 1`
- `cargo run -p binance-bot -- compare --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip --strategies trade-flow-momentum,liquidity-sweep-reversal`
- `cargo run -p binance-bot -- features --uri data/binance/capture/DOGEUSDT/session_20260331_v4.jsonl --output data/binance/features/DOGEUSDT/session_20260331_v4_features.csv --lookahead-events 16`
- `cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/live_validation_1m_20260401.jsonl --duration-secs 60 --depth-levels 5`
- `cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/live_validation_5m_20260401.jsonl --duration-secs 300 --depth-levels 5`
- `cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/live_validation_10m_20260401.jsonl --duration-secs 600 --depth-levels 5`
- `cargo run -p binance-bot -- compare --uris data/binance/capture/DOGEUSDT/live_validation_1m_20260401.jsonl data/binance/capture/DOGEUSDT/live_validation_5m_20260401.jsonl data/binance/capture/DOGEUSDT/live_validation_10m_20260401.jsonl --strategies microprice-imbalance-maker`
- `cargo check -p trade`
- `cargo check -p strategies`
- `cargo check -p binance-exchange`
- `cargo check -p binance-bot`
- `cargo test -p trade`
- `cargo test -p strategies`
- `cargo test -p binance-exchange`
- `cargo test -p binance-bot`
- `cargo run -p binance-bot -- trade-attribution --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim`
- `cargo run -p binance-bot -- strategy-diagnostics --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim`
- `cargo run -p binance-bot -- compare --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip --strategies trade-flow-momentum,liquidity-sweep-reversal,trade-flow-reclaim`
- `cargo run -p binance-bot -- optimize --strategy trade-flow-reclaim --parameter min_pullback_from_high_bps --values 2.0,4.0,6.0,8.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- optimize --strategy trade-flow-reclaim --parameter min_reclaim_from_low_bps --values 1.0,2.0,3.0,4.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- optimize --strategy trade-flow-reclaim --parameter min_trade_flow_imbalance --values 0.04,0.08,0.10,0.14 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- walk-forward --strategy trade-flow-reclaim --parameter min_reclaim_from_low_bps --values 2.0,3.0,4.0 --uris data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip --min-train-size 2 --test-size 1`
- `cargo run -p binance-bot -- capture --output /tmp/pulsar_rotation_batch.jsonl --duration-secs 5 --depth-levels 5`
- `cargo run -p binance-bot -- capture-index --root /tmp --symbol DOGEUSDT --min-quote-quality dense --min-depth-quality mixed --require-verified-sidecar`
- `cargo run -p binance-bot -- compare --uris /tmp/pulsar_rotation_batch_part_001.jsonl.gz /tmp/pulsar_rotation_batch_part_002.jsonl.gz /tmp/pulsar_rotation_batch_part_003.jsonl.gz --strategies microprice-imbalance-maker`

## Important Constraints

- Do not reintroduce REST polling fallback for live HFT logic.
- Do not restore deleted RSI / mean reversion / advanced-order strategies.
- Do not pretend maker backtests are realistic without quote/depth history and queue logic.
- Every strategy must be documented under `docs/strategies/`.
- Do not discard a model after shallow iteration; require deep tuning and live emulation evidence before rejecting it.

## Immediate Next Steps

1. Focus the next research pass on changed execution assumptions such as `taker entry + maker exit` or stricter regime gating.
2. Expand quote-aware capture coverage across more sessions and days before drawing conclusions from a single low-activity live-like sample.
3. Keep using `trade-attribution`, `strategy-diagnostics`, `optimize`, `walk-forward`, and `features` as the default Phase 6 loop instead of ad hoc shell sweeps.
4. Prepare the next Russian research report from the updated docs and verified command outputs.

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

- `2025-06-28`: `1` closed trade, realized PnL `-0.0401218776`
- `2025-08-08`: `13` closed trades, realized PnL `-0.6884182115`
- `2026-03-30`: `5` closed trades, realized PnL `-0.2380909574`

Interpretation:

- current burst-following entry is too weak after taker costs
- the first implemented model is a usable baseline, not a valid production candidate
- next research should focus on stricter entry quality and comparative testing versus liquidity sweep logic

Initial three-day validation for `LiquiditySweepReversalStrategy`:

- `2025-06-28`: `1` closed trade, realized PnL `-0.0500030732`
- `2025-08-08`: `2` closed trades, realized PnL `-0.1259625565`
- `2026-03-30`: `1` closed trade, realized PnL `-0.0392906890`

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
- attribution exports now make it possible to inspect why taker models lose after costs without hand-parsing logs
- diagnostics exports now show that current momentum gating is dominated by `min_trades`, flow, and drift-band blocks, while reclaim is mostly blocked by `min_trades`, pullback-band, and reclaim conditions
- the latest three-strategy compare ranks `trade-flow-reclaim` least bad, but only because it traded once across the whole batch; it is not yet a proven lead strategy
- indexed batch replay now works end-to-end, but current indexed coverage is still thin because only the latest capture has a sidecar
- legacy backfill removed that immediate coverage bottleneck for the current DOGEUSDT captures, expanding indexed coverage from one dataset to three
- minimal filtering now lets batch replay focus on quote-aware datasets instead of mixing in trade-only legacy captures
- sequential capture-batch now expands quote-aware indexed coverage without manual per-file capture invocation
- maker baseline now exists end-to-end, but the tested quote-aware capture sessions still produced `0` entries / `0` trades for `microprice-imbalance-maker`
- queue-model-v1 and passive-fill behavior are explicit approximations only; Phase 5 should not be interpreted as honest live-maker realism yet
- aggregated optimization confirms `liquidity-sweep-reversal.min_sweep_drop_bps = 12.0` is currently least bad over the tested three-day batch because it cuts activity sharply, but it is still negative overall
- the new optimization command removes the earlier shell-output truncation problem by collapsing multi-day results into one ranked table per tested value
- `trade-flow-momentum.min_price_drift_bps = 12.0` is currently the least bad tested momentum setting, materially better than `9.0` and `6.0` on the same three-day batch
- the first tested `trade-flow-momentum.min_trade_flow_imbalance` and `liquidity-sweep-reversal.min_recent_buyer_imbalance` values tied, which suggests those dimensions are not yet the next high-signal tuning axes
- walk-forward validation now exists in the CLI and immediately showed that the current best sweep threshold did not generalize into held-out trading activity on the final day
- ML-ready feature export now exists, which gives a clean path for attribution studies and future supervised experiments without bypassing the event-driven market-state layer
- honest live-validation captures for `1m`, `5m`, and `10m` are now complete, and all three still produced `0` maker entries / `0` trades despite quote and depth availability
- the newer live-like capture `live_train_20260401_210321.jsonl` also failed to produce entries for all four baseline models, which reinforces that quote/depth availability alone does not create edge
- the next profitable direction should therefore change execution style or regime selectivity, not just invent more taker entry triggers over the same short-horizon noise

## Resume Checklist

- Read `docs/AGENT_PLAYBOOK.md`
- Read `docs/ARCHITECTURE.md`
- Read `docs/ROADMAP.md`
- Read this file
- Run `cargo check`
- Continue from the next unchecked roadmap item or highest-value backlog item
