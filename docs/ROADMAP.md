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

- [x] Trade-only taker execution realism improvements
- [x] Quote-aware execution pricing at bid/ask
- [x] Latency and impact accounting cleanup
- [x] Inventory and drawdown guardrails
- [x] Better evaluation metrics and attribution
- [x] Quote/depth replay observability and event mix diagnostics

## Phase 3 - First Research Models

- [x] Create `docs/strategies/` documentation for every implemented strategy
- [x] Add CLI-level strategy selection so the same orchestration can run multiple models
- [x] Trade-flow momentum taker first major refinement pass
- [x] Liquidity sweep / exhaustion taker baseline
- [x] Liquidity sweep / exhaustion taker first major refinement pass
- [x] Multi-day comparison harness as shared orchestration workflow
- [x] Parameter search scaffolding

## Backlog - Secondary Improvements

- [ ] Shared position-sizing adapter inside strategies to avoid duplicate ad hoc target notional logic
- [ ] Richer per-trade attribution export for replay analysis
- [ ] Regime tags on replay runs
- [ ] Derived aggregate compare metrics in CLI output (per-strategy totals, means, ranking)
- [ ] More detailed sweep decomposition metrics before any second sweep refinement pass
- [ ] More detailed momentum decomposition metrics before any second momentum refinement pass
- [ ] Capture file rotation/compression after replay format stabilizes
- [ ] Metadata sidecar checksum/file-size verification
- [x] Batch dataset index builder over captured sessions under `data/binance/capture/`
- [x] Batch replay launcher over indexed captured datasets
- [x] Legacy capture backfill into sidecar/index format only if needed
- [x] Indexed dataset filters by capture time window and minimum event counts
- [x] Minimal sequential capture-batch orchestration for quote-aware dataset collection
- [ ] Sidecar inference quality hints for backfilled datasets with missing capture timestamps
- [ ] Indexed dataset filters for quote/depth presence quality classes
- [ ] Capture-batch pause/gap controls between sequential segments

## Phase 4 - Quote/Depth Infrastructure

- [x] Add `bookTicker` websocket ingestion
- [x] Add depth ingestion and capture
- [x] Store raw quote/depth datasets under `data/` via JSONL capture pipeline
- [x] Build quote-aware replay support
- [x] Add normalized capture ordering and local capture timestamps for mixed-event datasets
- [x] Add normalized captured dataset layout and metadata sidecar
- [x] Add replay-time event integrity checks and dataset summaries

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
