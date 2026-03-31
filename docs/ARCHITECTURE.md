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

5. Execution / Simulation Layer
   - validates intents
   - simulates fills, latency, queue effects, and fees
   - returns `ExecutionReport`

6. Evaluation Layer
   - PnL attribution
   - drawdown / exposure / fill quality
   - regime analysis
   - multi-day comparison
   - orchestration-level batch comparison across multiple strategies and identical datasets

## Current Repository Roles

- `trade/`
  - shared domain types, strategy interfaces, market state, execution simulation, metrics
- `exchanges/binance/`
  - websocket clients, archive parsing, captured JSONL replay parsing, real exchange adapter, live capture integration
- `bots/binance/`
  - orchestration CLI for replay, emulate, live validation, dataset capture
  - now also owns cross-strategy batch comparison for identical replay inputs and replay source selection
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
- maker research still remains blocked on queue/fill realism even though quote/depth replay now exists

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

## Validation Modes

- historical replay over multiple days
- paper/live emulate on websocket stream
- controlled parameter sweeps
- walk-forward evaluation
