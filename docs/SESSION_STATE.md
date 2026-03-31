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

## Last Verified Commands

- `cargo check`
- `cargo test`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`

## Important Constraints

- Do not reintroduce REST polling fallback for live HFT logic.
- Do not restore deleted RSI / mean reversion / advanced-order strategies.
- Do not pretend maker backtests are realistic without quote/depth history and queue logic.

## Immediate Next Steps

1. Add new shared domain modules for:
   - market events
   - market state
   - execution intents / reports
2. Route live orchestration to `market_event_stream()` where appropriate.
3. Improve emulator so decisions use quote-aware pricing instead of trade-price fallback where available.
4. Implement first real taker model: trade-flow momentum taker.

## Multi-Day Testing Requirement

All future strategy validation must use multiple daily datasets, not isolated one-day anecdotes.

Minimum acceptable research batch:

- at least 3 to 5 daily archives
- mixed regimes if possible

## Resume Checklist

- Read `docs/AGENT_PLAYBOOK.md`
- Read `docs/ARCHITECTURE.md`
- Read `docs/ROADMAP.md`
- Read this file
- Run `cargo check`
- Continue from the first unchecked item in Phase 1
