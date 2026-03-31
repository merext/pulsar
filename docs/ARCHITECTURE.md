# Architecture

## Target Design

The project is evolving into a layered HFT research platform.

### Layers

1. Market Data Ingestion
   - Binance websocket adapters
   - historical archive loaders
   - live capture writers for raw datasets

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

5. Execution / Simulation Layer
   - validates intents
   - simulates fills, latency, queue effects, and fees
   - returns `ExecutionReport`

6. Evaluation Layer
   - PnL attribution
   - drawdown / exposure / fill quality
   - regime analysis
   - multi-day comparison

## Current Repository Roles

- `trade/`
  - shared domain types, strategy interfaces, market state, execution simulation, metrics
- `exchanges/binance/`
  - websocket clients, archive parsing, real exchange adapter, live capture integration
- `bots/binance/`
  - orchestration CLI for replay, emulate, live validation, dataset capture
- `strategies/`
  - future strategy implementations only; currently reset for redesign

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
