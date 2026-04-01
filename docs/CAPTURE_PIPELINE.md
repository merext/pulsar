# Capture Pipeline

## Purpose

This pipeline captures raw live Binance market data for later quote-aware and depth-aware replay.

Initial scope:

- trade events
- top-of-book `bookTicker`
- shallow depth snapshots from Binance multiplexed websocket stream

## Command

```bash
cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/live.jsonl --duration-secs 60
```

Optional depth size:

```bash
cargo run -p binance-bot -- capture --output data/binance/capture/DOGEUSDT/live.jsonl --duration-secs 60 --depth-levels 5
```

## Output Format

The capture writes JSONL records, one event per line.

Alongside the JSONL file, the pipeline now writes a metadata sidecar:

- `<capture>.metadata.json`

Event types:

- `trade`
- `book_ticker`
- `depth`

Each record now also carries replay-safe capture metadata:

- `capture_sequence` - monotonic local sequence number within the file
- `captured_at_ms` - local wall-clock capture timestamp in milliseconds
- exchange-native timing/update fields when available

The sidecar stores dataset-level metadata such as:

- schema version
- symbol
- requested duration and depth levels
- total event counts by class
- first/last capture sequence
- first/last local capture timestamps
- file size in bytes
- SHA-256 checksum of the payload file
- replay ordering semantics

Indexed dataset rows derived from sidecars now also track:

- `sidecar_verified`
- `capture_time_quality`
- `quote_presence_quality`
- `depth_presence_quality`

The CLI can now build a batch index from sidecars:

```bash
cargo run -p binance-bot -- capture-index --root data/binance/capture
```

The CLI can also launch compare runs over indexed captured datasets:

```bash
cargo run -p binance-bot -- capture-compare --root data/binance/capture --symbol DOGEUSDT --limit 1 --strategies trade-flow-momentum,liquidity-sweep-reversal
```

Index and batch replay now support minimal filters such as:

- `--min-total-events`
- `--min-book-ticker-events`
- `--min-depth-events`
- `--require-captured-at`
- `--since-captured-at-ms`
- `--min-quote-quality`
- `--min-depth-quality`
- `--require-verified-sidecar`

Legacy captured JSONL files can be backfilled with sidecars:

```bash
cargo run -p binance-bot -- capture-backfill --input data/binance/capture/DOGEUSDT/session_20260331_v3.jsonl --duration-secs 10 --depth-levels 5
```

Sequential capture batches can now be collected with one command:

```bash
cargo run -p binance-bot -- capture-batch --batch-id smoke_batch --parts 2 --duration-secs 3 --depth-levels 5
```

Optional pause between parts:

```bash
cargo run -p binance-bot -- capture-batch --batch-id smoke_batch --parts 2 --duration-secs 3 --depth-levels 5 --gap-secs 2
```

Captured replay loading now also supports gzipped JSONL payloads.

Rotation/compression scaffolding is configured through `config/capture_rotation.toml`:

```toml
max_events_per_file = 0
gzip = false
```

This path is now validated end-to-end on a live smoke batch, producing multiple rotated `.jsonl.gz` parts with sidecars.

This keeps the file append-friendly while making mixed-event replay deterministic.

## Current Guarantees

- websocket-only live capture
- mixed `trade + bookTicker + depth` stream
- output directory is created automatically
- short capture runs already verified locally
- captured JSONL replay reader now exists for `backtest` and `compare`
- capture ordering and local capture timestamps are stored for replay-safe sequencing
- metadata sidecar is written automatically next to the JSONL capture
- metadata sidecar records payload checksum and byte size for later integrity checks
- batch dataset index can be generated from available sidecars without scanning every JSONL file
- batch compare runs can now be launched from indexed captured datasets
- legacy captured JSONL files can be promoted into the indexed dataset set via sidecar backfill
- indexed dataset selection can now exclude weak or incomplete captures before replay
- sequential capture batches can now generate multiple sidecar-compatible quote-aware sessions in one run
- capture-batch can now pause between parts with `--gap-secs`
- captured replay loaders can now open gzipped captured JSONL files
- rotated `.jsonl.gz` capture parts now replay correctly through normal `compare`/`backtest` captured-data flow

## Current Limitations

- depth is current shallow snapshot/update payload from Binance, not a reconstructed local book history format yet
- raw exchange timing across mixed event types is informative but not authoritative for replay ordering
- quote/depth quality labels are simple heuristics for dataset selection, not a guarantee of production-quality market microstructure fidelity

## Next Planned Steps

- keep expanding quote/depth capture coverage for honest maker research
