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
- [x] Route live/emulate runs through mixed trade + quote streams

## Phase 2 - Emulator V2

- [ ] Trade-only taker execution realism improvements
- [x] Quote-aware execution pricing at bid/ask
- [ ] Latency and impact accounting cleanup
- [ ] Inventory and drawdown guardrails
- [ ] Better evaluation metrics and attribution
- [ ] Quote/depth replay observability and event mix diagnostics

## Phase 3 - First Research Models

- [x] Create `docs/strategies/` documentation for every implemented strategy
- [x] Add CLI-level strategy selection so the same orchestration can run multiple models
- [x] Trade-flow momentum taker first major refinement pass
- [x] Liquidity sweep / exhaustion taker baseline
- [x] Liquidity sweep / exhaustion taker first major refinement pass
- [x] Multi-day comparison harness as shared orchestration workflow
- [ ] Parameter search scaffolding

## Backlog - Secondary Improvements

- [ ] Shared position-sizing adapter inside strategies to avoid duplicate ad hoc target notional logic
- [ ] Richer per-trade attribution export for replay analysis
- [ ] Regime tags on replay runs
- [ ] Derived aggregate compare metrics in CLI output (per-strategy totals, means, ranking)
- [ ] More detailed sweep decomposition metrics before any second sweep refinement pass
- [ ] More detailed momentum decomposition metrics before any second momentum refinement pass
- [ ] Capture file rotation/compression after replay format stabilizes

## Phase 4 - Quote/Depth Infrastructure

- [x] Add `bookTicker` websocket ingestion
- [x] Add depth ingestion and capture
- [x] Store raw quote/depth datasets under `data/` via JSONL capture pipeline
- [x] Build quote-aware replay support
- [ ] Add normalized captured dataset layout and metadata sidecar
- [ ] Add replay-time event integrity checks and dataset summaries

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
