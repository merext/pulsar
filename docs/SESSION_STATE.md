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
- Emulated execution now prefers top-of-book quotes when available:
  - taker buys reference ask instead of last trade
  - taker sells reference bid instead of last trade
  - synthetic spread widening is only added when no explicit quote is available
- Added first real strategy implementation:
  - `strategies::TradeFlowMomentumStrategy`
  - config file: `config/strategies/trade_flow_momentum.toml`
  - strategy spec: `docs/strategies/trade_flow_momentum.md`
- Added second baseline taker strategy:
  - `strategies::LiquiditySweepReversalStrategy`
  - config file: `config/strategies/liquidity_sweep_reversal.toml`
  - strategy spec: `docs/strategies/liquidity_sweep_reversal.md`
- Added rule that every strategy must be documented under `docs/strategies/`
- Backtest CLI now supports strategy selection instead of hardwiring a single model
- Multi-day replay was run on three DOGEUSDT archives and the first model is currently unprofitable on all three days
- Comparative replay now exists for two taker baselines over the same three-day batch

## Last Verified Commands

- `cargo check`
- `cargo test`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`
- `cargo run -p binance-bot -- backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2026-03-30.zip`
- `cargo run -p binance-bot -- --strategy trade-flow-momentum backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip`
- `cargo run -p binance-bot -- --strategy liquidity-sweep-reversal backtest --uri data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-08-08.zip`

## Important Constraints

- Do not reintroduce REST polling fallback for live HFT logic.
- Do not restore deleted RSI / mean reversion / advanced-order strategies.
- Do not pretend maker backtests are realistic without quote/depth history and queue logic.
- Every strategy must be documented under `docs/strategies/`.

## Immediate Next Steps

1. Add new shared domain modules for:
   - market events
   - market state
   - execution intents / reports
2. Route live orchestration to `market_event_stream()` where appropriate.
3. Promote `LiquiditySweepReversalStrategy` as the current lead taker baseline for the next refinement loop.
4. Refine `TradeFlowMomentumStrategy` only at the major-filter level, not by deep sub-variant exploration.
5. Add a simple same-dataset comparison harness/reporting flow.

## Architectural Guidance

- Keep major architecture changes immediate when they unblock multiple models.
- Push lower-value refinements into roadmap backlog instead of deepening local optimizations too early.

## Multi-Day Testing Requirement

All future strategy validation must use multiple daily datasets, not isolated one-day anecdotes.

Minimum acceptable research batch:

- at least 3 to 5 daily archives
- mixed regimes if possible

## Current Research Readout

Initial three-day validation for `TradeFlowMomentumStrategy`:

- `2025-06-28`: `1` closed trade, realized PnL `-0.0405`
- `2025-08-08`: `13` closed trades, realized PnL `-0.6938`
- `2026-03-30`: `5` closed trades, realized PnL `-0.2450`

Interpretation:

- current burst-following entry is too weak after taker costs
- the first implemented model is a usable baseline, not a valid production candidate
- next research should focus on stricter entry quality and comparative testing versus liquidity sweep logic

Initial three-day validation for `LiquiditySweepReversalStrategy`:

- `2025-06-28`: `1` closed trade, realized PnL `-0.0504`
- `2025-08-08`: `2` closed trades, realized PnL `-0.1268`
- `2026-03-30`: `1` closed trade, realized PnL `-0.0397`

Comparison takeaway:

- both baselines are still negative after realistic taker costs
- liquidity sweep baseline currently looks more promising because it loses less while trading less often
- next iteration should focus on major-quality filters and better comparison/reporting, not local micro-tuning

## Resume Checklist

- Read `docs/AGENT_PLAYBOOK.md`
- Read `docs/ARCHITECTURE.md`
- Read `docs/ROADMAP.md`
- Read this file
- Run `cargo check`
- Continue from the first unchecked item in Phase 1
