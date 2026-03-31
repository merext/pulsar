# Pulsar Trading Bot

Rust workspace for crypto trading experiments with Binance market data, a shared trading core, replayable backtests, and HFT infrastructure research.

## What is in the repo

- `trade/` - shared domain layer: strategy trait, metrics, backtest execution, signals, models
- `strategies/` - reserved for future strategy research; legacy strategies were removed
- `exchanges/binance/` - Binance market data client and live execution adapter
- `bots/binance/` - CLI entrypoint for live, emulated, and backtest runs
- `config/` - runtime configs for the bot, exchange, and strategies

## Strategy status

- All previous strategies were removed after failing realistic backtest/live validation
- The repo is now in a clean state for designing new HFT strategies from scratch
- Live market data must use Binance websocket only; REST fallback is intentionally not allowed for strategy execution

## Current status

- Workspace builds with `cargo check`
- Test suite runs with `cargo test`
- Backtest mode now uses event time, fee-aware metrics, and budget-aware simulation
- Live market data path is websocket-only for HFT correctness

## Quick start

### Historical data validation

```bash
cargo run --bin binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip
```

### Live market-data connectivity debugging

```bash
RUST_LOG=debug cargo run --bin binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip
```

## CLI

```bash
cargo run --bin binance-bot -- <trade|emulate|backtest> [--uri PATH] [--duration-secs N]
```

## Data formats

Backtest input supports:

- Binance-style CSV rows: `trade_id,price,quantity,,timestamp,is_buyer_market_maker`
- Simple CSV rows: `timestamp,price,quantity`
- ZIP files containing a supported CSV

Primary runtime configs:

- `config/trading_config.toml`
- `config/binance_exchange.toml`

## Development

```bash
cargo check
cargo test
```

Useful validation commands:

```bash
RUST_LOG=debug cargo run --bin binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip
```

## Notes

- No active strategy implementations remain in the repository
- `trade` and `emulate` are intentionally blocked until new websocket-native strategies are implemented
- Backtest realism is improved, but order book simulation is still incomplete

## Risk warning

This project is experimental. Do not use it for unattended real trading without further validation, risk controls, and exchange-level safeguards.
