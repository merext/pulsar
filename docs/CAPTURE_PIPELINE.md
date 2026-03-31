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

Event types:

- `trade`
- `book_ticker`
- `depth`

This is intentionally raw and append-friendly. It is not yet a replay-optimized normalized archive format.

## Current Guarantees

- websocket-only live capture
- mixed `trade + bookTicker + depth` stream
- output directory is created automatically
- short capture runs already verified locally
- captured JSONL replay reader now exists for `backtest` and `compare`

## Current Limitations

- no file rotation yet
- no compression yet
- no metadata sidecar yet
- depth is current shallow snapshot/update payload from Binance, not a reconstructed local book history format yet

## Next Planned Steps

- define normalized archival layout under `data/`
- add dataset metadata and replay-time integrity summaries
- add optional rotation/compression only if needed after core replay support exists
