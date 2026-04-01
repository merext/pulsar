# Architecture

## Target Design

The project is evolving into a layered HFT research platform.

### Layers

1. Market Data Ingestion
   - Binance websocket adapters
   - historical archive loaders
   - live capture writers for raw datasets
   - live trade/emulate now consume mixed `trade + bookTicker` streams through shared orchestration
   - capture pipeline now supports mixed `trade + bookTicker + depth` JSONL output for future replay
   - historical replay now supports both trade-only archives and captured mixed-event JSONL datasets
   - captured replay loaders can now read both plain JSONL and gzipped captured JSONL datasets
   - captured mixed-event datasets now include local capture ordering (`capture_sequence`) and capture timestamps (`captured_at_ms`) for replay-safe sequencing
   - captured datasets now also emit metadata sidecars for batch tooling and dataset introspection
   - sidecars now record payload integrity fields such as `data_size_bytes` and `data_sha256`
   - indexed dataset entries now carry capture-time, quote-presence, and depth-presence quality classes plus sidecar verification status
   - captured dataset discovery can now be built from sidecars via CLI without scanning all event payload files
   - indexed captured datasets can now feed shared batch replay orchestration without custom per-file wiring
   - legacy captured datasets can now be upgraded into the same indexed flow through sidecar backfill
   - indexed dataset selection now supports minimal quality/time filters before replay orchestration
   - sequential batch capture can now populate the indexed quote-aware dataset pool through repeated standard captures
   - rotation/compression now works through `config/capture_rotation.toml`, producing rotated `.jsonl.gz` capture parts with sidecars and integrity metadata

2. Normalized Event Layer
   - `MarketEvent`
   - trade events
   - top-of-book quote events
   - depth updates

3. Market State Layer
   - top of book state
   - rolling trade statistics
   - short-horizon flow imbalance
   - microprice and spread state
   - optional local depth book

4. Strategy Layer
     - consumes `MarketState`
     - produces `OrderIntent`
     - no exchange-specific logic
     - selected by orchestration CLI so multiple models can share the same replay and live plumbing
     - shared capped entry sizing now lives in `StrategyContext::capped_entry_quantity(...)` so strategies do not duplicate ad hoc target-notional logic
     - strategies can now emit structured `DecisionMetric` values and `StrategyDiagnostics` counters/gauges for replay analysis

5. Execution / Simulation Layer
   - validates intents
   - simulates fills, latency, queue effects, and fees
   - now separates synthetic spread, slippage, latency impact, and market impact for cleaner execution attribution
   - now enforces inventory and drawdown guardrails in shared backtest/emulation orchestration
   - returns richer `ExecutionReport` values including rationale, confidence, expected edge, and decision metrics needed for attribution export

6. Evaluation Layer
      - PnL attribution
      - drawdown / exposure / fill quality
      - per-trade attribution export with rationale, sizing, execution cost components, and hold time
      - per-strategy diagnostics export with counters and gauges from strategy decision paths
      - regime analysis
      - multi-day comparison
      - orchestration-level batch comparison across multiple strategies and identical datasets
     - aggregated per-strategy compare ranking across identical dataset batches
     - parameter optimization and walk-forward summaries over ordered replay datasets
     - ML-ready feature export for downstream modeling and attribution work
     - replay observability now includes event-mix diagnostics for trade/quote/depth coverage quality

## Current Repository Roles

- `trade/`
  - shared domain types, strategy interfaces, market state, execution simulation, metrics
- `exchanges/binance/`
  - websocket clients, archive parsing, captured JSONL replay parsing, real exchange adapter, live capture integration
- `bots/binance/`
  - orchestration CLI for replay, emulate, live validation, dataset capture
  - now also provides minimal parameter-search scaffolding over the shared backtest flow
  - now also provides aggregated parameter-optimization ranking over multi-day replay batches
  - now also owns cross-strategy batch comparison for identical replay inputs and replay source selection
  - now also provides replay regime tagging, aggregated compare ranking, walk-forward validation, and feature export
  - now also provides `trade-attribution` and `strategy-diagnostics` research exports plus capture metadata/index workflows
- `strategies/`
  - future strategy implementations only; currently reset for redesign
- `docs/strategies/`
  - strategy specifications and research notes
  - each strategy must have a dedicated document kept in sync with implementation and validation

## Refactor Direction

### Old model

- strategy saw only trades
- strategy returned `Signal`
- trader converted `Signal` into orders

### New model

- strategy sees normalized market state
- strategy returns `OrderIntent`
- execution layer simulates or routes intents
- trader/orchestrator manages streams and reports

## Replay Modes

- trade-only archive replay remains the baseline validation path for taker models on large historical batches
- captured JSONL replay is now the quote/depth-aware validation path for mixed-event datasets captured from live websocket flow
- captured replay can now be loaded from gzipped JSONL payloads as well as plain JSONL, including rotated `.jsonl.gz` capture parts
- maker research still remains blocked on queue/fill realism even though quote/depth replay now exists
- maker baseline now exists with explicit queue-model-v1 and passive-fill approximations, but it is still research scaffolding rather than production-realistic maker simulation
- mixed captured replay should trust capture ordering for sequence semantics; raw exchange timing fields can remain informative but are not authoritative across event classes

## Key Shared Types To Maintain

- `MarketEvent`
- `TradeTick`
- `BookTicker`
- `DepthUpdate`
- `MarketState`
- `OrderIntent`
- `ExecutionReport`
- `StrategyDecision`
- `StrategyContext`

## Realism Principles

### Taker realism

- execute against bid/ask, not last trade
- model latency drift
- size-based slippage
- budget constraints
- fill rejection / partial fill possibilities

### Maker realism

- queue-position approximation
- passive fill probability based on order flow and time at level
- cancel / replace latency
- adverse selection tracking
- current implementation only satisfies the first baseline item with simple quote-aware heuristics and partial-fill estimates; deeper fill-process realism is still pending

## Validation Modes

- historical replay over multiple days
- paper/live emulate on websocket stream
- controlled parameter sweeps
- aggregated multi-day parameter optimization ranking
- walk-forward evaluation
- per-event feature export with forward-return targets and regime tags
