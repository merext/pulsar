# Roadmap

## Phase 0 - Reset and Cleanup

- [x] Remove legacy strategies
- [x] Enforce websocket-only live market data
- [x] Preserve historical data validation path
- [x] Add persistent project documentation under `docs/`

## Phase 1 - Shared Event-Driven Foundation

- [x] Add normalized market event domain types
- [x] Add market state reconstruction layer
- [x] Add execution intent / report layer
- [x] Refactor shared strategy interface to consume market state
- [x] Keep build green during migration
- [x] Add quote (`bookTicker`) ingestion and normalized events
- [ ] Route live runs through mixed trade + quote streams

## Phase 2 - Emulator V2

- [ ] Trade-only taker execution realism improvements
- [x] Quote-aware execution pricing at bid/ask
- [ ] Latency and impact accounting cleanup
- [ ] Inventory and drawdown guardrails
- [ ] Better evaluation metrics and attribution

## Phase 3 - First Research Models

- [x] Create `docs/strategies/` documentation for every implemented strategy
- [x] Add CLI-level strategy selection so the same orchestration can run multiple models
- [ ] Trade-flow momentum taker refinement after first multi-day replay
- [x] Liquidity sweep / exhaustion taker baseline
- [ ] Multi-day comparison harness
- [ ] Parameter search scaffolding

## Backlog - Secondary Improvements

- [ ] Shared position-sizing adapter inside strategies to avoid duplicate ad hoc target notional logic
- [ ] Richer per-trade attribution export for replay analysis
- [ ] Regime tags on replay runs

## Phase 4 - Quote/Depth Infrastructure

- [ ] Add `bookTicker` websocket ingestion
- [ ] Add depth ingestion and capture
- [ ] Store raw quote/depth datasets under `data/`
- [ ] Build quote-aware replay support

## Phase 5 - Maker Research

- [ ] Microprice imbalance maker
- [ ] Queue model v1
- [ ] Passive fill realism tests
- [ ] Live 1m / 5m / 10m validation

## Phase 6 - Optimization and Research Loop

- [ ] Per-model parameter optimization
- [ ] Regime segmentation
- [ ] Walk-forward validation
- [ ] ML-ready feature pipeline if justified by results
