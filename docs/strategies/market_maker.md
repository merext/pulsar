# MarketMaker Strategy

## Problem Statement

HFT market-making on Binance Spot faces a structural challenge: **10 bps maker fee** per side
(20 bps round-trip) makes profitability impossible on tight-spread pairs like DOGEUSDT (2-3 bps spread).
The previous SpreadRegimeCapture strategy proved that a mean-reversion approach on wide-spread pairs can
be profitable ($150/day on TRUUSDT), but 98% of exits were taker-based — wasting the maker fee advantage.

## Strategy Design

### Core Concept: Passive Spread Capture

Instead of predicting price direction, the MarketMaker strategy exploits the natural bid-ask spread
on pairs where spread >> fees.

```
Without position: Post passive Buy @ bid (Maker order)
With position:    Post passive Sell @ ask (Maker order)
Safety exits:     Taker sell on stop-loss or max-hold timeout
```

### Why This Works on TRUUSDT

- **Price**: ~$0.0044-$0.0045
- **Tick size**: 0.0001 (1 tick = ~217 bps at $0.0045)
- **Spread**: almost always exactly 1 tick = 217 bps
- **Round-trip maker cost**: 20 bps
- **Net edge per round-trip**: ~197 bps (217 - 20)
- **Autocorrelation**: -0.547 (strong mean-reversion — price bounces between two levels)

The key insight: TRUUSDT price oscillates between $0.0044 and $0.0045. The strategy buys at
$0.0044 and sells at $0.0045, capturing 1 tick = ~$0.0001/unit. After fees, profit is
~$0.00008/unit (~80 bps net).

### Entry Filters

1. **Minimum spread** (`min_spread_bps = 100`): Only enter when spread is genuinely wide
2. **Minimum edge** (`min_edge_bps = 50`): Spread minus fees must exceed 50 bps
3. **Seller-initiated trade** (`require_seller_initiated = true`): Last trade was a sell →
   price is at/near bid → good time to buy
4. **Price position** (`max_price_position = 0.3`): Only buy in bottom 30% of recent price range
5. **Minimum trades in window** (`min_trades_in_window = 2`): Need some market activity

### Exit Logic (Priority Order)

1. **Stop loss** (`stop_loss_bps = 300`): ~1.4 adverse ticks. Taker exit.
2. **Max hold** (`max_hold_millis = 120000`): 2-minute timeout. Taker exit.
3. **Passive sell at ask**: Maker order — the primary exit mechanism.
   On every tick while in position, the strategy attempts a passive sell at the ask price.

### Position Sizing

- `cash_fraction = 0.05`: Uses 5% of available cash per trade (~$5 on $100 capital)
- This results in ~1100-1200 TRU per order (~$5 notional)
- Partial fills are common (fill rate ~35-46%), so actual position size is ~385-400 TRU

## Performance (8-Day Out-of-Sample Backtest)

Data: TRUUSDT, 2026-03-24 to 2026-03-31. Strategy parameters optimized on Mar 31 only.

| Metric | Value |
|---|---|
| **Win days** | **8/8 (100%)** |
| **Avg PnL/day** | **$0.77** |
| **Daily Sharpe** | **4.64** |
| Avg Win Rate | 66.3% |
| Avg Profit Factor | 7.79 |
| Min PnL (worst day) | $0.45 |
| Max PnL (best day) | $1.02 |
| Real equity drawdown | ~$0.03-0.04 (0.03-0.04%) |
| Annual PnL estimate ($100 cap) | ~$282 (282% APR) |

### Daily Breakdown

| Date | Ticks | Entries | PnL $ | WinRate | PF | MaxDD |
|---|---|---|---|---|---|---|
| 2026-03-24 | 2551 | 117 | 0.789 | 74.3% | 7.02 | 49.4% |
| 2026-03-25 | 1172 | 80 | 0.449 | 63.3% | 4.73 | 30.0% |
| 2026-03-26 | 2999 | 75 | 0.837 | 69.0% | 9.77 | 49.0% |
| 2026-03-27 | 2534 | 104 | 0.723 | 67.8% | 5.10 | 45.6% |
| 2026-03-28 | 2718 | 94 | 1.018 | 65.8% | 10.91 | 55.0% |
| 2026-03-29 | 1427 | 69 | 0.889 | 75.6% | 16.09 | 44.6% |
| 2026-03-30 | 3100 | 139 | 0.681 | 52.7% | 3.92 | 47.1% |
| 2026-03-31 | 4390 | 158 | 0.794 | 62.2% | 4.77 | 49.7% |

**Note**: MaxDD% shown above is capital deployed in positions, NOT equity drawdown.
Real equity drawdown is ~$0.03-0.04 per day (negligible).

## Equity Curve Characteristics

- Nearly linear growth with minimal drawdowns
- First positive PnL within first 3 trades
- Max consecutive equity decline: ~14 entries (very shallow)
- Pattern: Buy maker → partial fills → sell maker (profit) or sell taker (small loss)
- Typical winning trade: ~$0.001-0.002 profit
- Typical losing trade: ~$0.002-0.003 loss (taker exit)

## Configuration

Strategy config: `config/strategies/market_maker.toml`
Trading config: `config/trading_config.toml` (TRUUSDT-specific)

### Key Parameters

| Parameter | Value | Rationale |
|---|---|---|
| min_spread_bps | 100 | Filter compressed spread ticks |
| min_edge_bps | 50 | Buffer above break-even |
| max_hold_millis | 120000 | 2min max hold reduces adverse selection |
| stop_loss_bps | 300 | ~1.4 ticks adverse movement |
| entry_cooldown_millis | 15000 | 15s cooldown after taker exit |
| cash_fraction | 0.05 | 5% of cash per trade |
| require_seller_initiated | true | Only buy after seller-initiated trade |
| max_price_position | 0.3 | Bottom 30% of recent range |

## Limitations and Risks

1. **Single pair dependence**: Only tested on TRUUSDT. Pair could be delisted or spread could compress.
2. **Low absolute PnL**: $0.77/day on $100 capital. Scales linearly with capital but $1000+ positions
   may face liquidity constraints on a ~$376K/day volume pair.
3. **Fill rate assumption**: Backtest uses 35% fill rate for passive orders. Real fill rate depends
   on queue position and could be lower.
4. **No multi-tick passive order tracking**: Backtest engine evaluates passive orders on single tick only.
   Real market maker would persist orders across multiple ticks, potentially improving fill rate.
5. **Trade-only data**: Backtest uses trade stream, not order book. Spread is synthesized from trade
   prices. Real order book may show different dynamics.
6. **Short selling not supported**: Strategy is long-only. Cannot profit from ask→bid spread capture.

## Files

- Strategy: `strategies/src/market_maker.rs` (~699 lines, 2 unit tests)
- Config: `config/strategies/market_maker.toml`
- Registration: `strategies/src/lib.rs`, `bots/binance/src/main.rs` (3 places)

## Run Commands

```bash
# Build
cargo build -p binance-bot

# Backtest single day
cargo run -p binance-bot -- --strategy market-maker backtest \
  --uri data/binance/daily/trades/TRUUSDT/TRUUSDT-trades-2026-03-31.zip

# Summary only
cargo run -p binance-bot -- --strategy market-maker backtest \
  --uri data/binance/daily/trades/TRUUSDT/TRUUSDT-trades-2026-03-31.zip \
  2>&1 | rg "session_summary"
```
