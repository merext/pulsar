#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Pulsar Multi-Symbol Backtest Runner
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs backtest for all 19 portfolio symbols across all available data days.
# Outputs per-symbol and aggregate PnL summary.
#
# Usage:
#   ./scripts/backtest_portfolio.sh                  # all symbols, all days
#   ./scripts/backtest_portfolio.sh 2026-03-31       # all symbols, one day
#   ./scripts/backtest_portfolio.sh all TRUUSDT       # one symbol, all days
#
# Requirements: cargo build --release -p binance-bot

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PULSAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PULSAR_DIR/target/release/binance-bot"
DATA_DIR="$PULSAR_DIR/data/binance/daily/trades"

# 19 portfolio symbols (validated 8/8 profitable days)
SYMBOLS=(
    TRUUSDT
    ARKMUSDT
    PHBUSDT
    ACTUSDT
    IOUSDT
    WUSDT
    ACEUSDT
    GLMRUSDT
    GTCUSDT
    WIFUSDT
    HFTUSDT
    MBOXUSDT
    HIGHUSDT
    ARUSDT
    SXTUSDT
    EIGENUSDT
    ATAUSDT
    RAREUSDT
    1000CATUSDT
)

# Parse arguments
FILTER_DATE="${1:-}"
FILTER_SYMBOL="${2:-}"

# Check binary exists
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found. Run: cargo build --release -p binance-bot"
    exit 1
fi

# Strip ANSI escape codes and extract realized_pnl from session_summary
extract_pnl() {
    sed 's/\x1b\[[0-9;]*m//g' | grep 'session_summary' | sed 's/.*realized_pnl=\([^ ]*\).*/\1/'
}

# Get config paths for a symbol
get_configs() {
    local sym="$1"
    local sym_lower
    sym_lower=$(echo "$sym" | tr '[:upper:]' '[:lower:]')

    local trading_config strategy_config

    if [ "$sym" = "TRUUSDT" ]; then
        trading_config="$PULSAR_DIR/config/trading_config.toml"
        strategy_config="$PULSAR_DIR/config/strategies/market_maker.toml"
    else
        trading_config="$PULSAR_DIR/config/trading_config_${sym_lower}.toml"
        strategy_config="$PULSAR_DIR/config/strategies/market_maker_${sym_lower}.toml"
    fi

    if [ ! -f "$trading_config" ]; then
        echo "WARN: Missing trading config for $sym: $trading_config" >&2
        return 1
    fi
    if [ ! -f "$strategy_config" ]; then
        echo "WARN: Missing strategy config for $sym: $strategy_config" >&2
        return 1
    fi

    echo "$trading_config $strategy_config"
}

# Run backtest for one symbol, one day
run_one() {
    local sym="$1"
    local data_file="$2"
    local configs
    configs=$(get_configs "$sym") || return 1

    local trading_config strategy_config
    trading_config=$(echo "$configs" | cut -d' ' -f1)
    strategy_config=$(echo "$configs" | cut -d' ' -f2)

    local pnl
    if [ "$sym" = "TRUUSDT" ]; then
        pnl=$("$BINARY" --strategy market-maker backtest --uri "$data_file" 2>&1 | extract_pnl)
    else
        pnl=$("$BINARY" --strategy market-maker --config "$trading_config" --strategy-config "$strategy_config" backtest --uri "$data_file" 2>&1 | extract_pnl)
    fi

    echo "$pnl"
}

# Main
echo "═══════════════════════════════════════════════════════════════"
echo " Pulsar Portfolio Backtest"
echo "═══════════════════════════════════════════════════════════════"
echo ""

portfolio_total=0
portfolio_days=0
symbol_results=""

for sym in "${SYMBOLS[@]}"; do
    # Filter by symbol if specified
    if [ -n "$FILTER_SYMBOL" ] && [ "$FILTER_SYMBOL" != "all" ] && [ "$sym" != "$FILTER_SYMBOL" ]; then
        continue
    fi

    sym_dir="$DATA_DIR/$sym"
    if [ ! -d "$sym_dir" ]; then
        echo "SKIP $sym: no data directory"
        continue
    fi

    sym_total=0
    sym_days=0

    for data_file in "$sym_dir"/${sym}-trades-*.zip; do
        [ -f "$data_file" ] || continue

        day=$(basename "$data_file" | sed "s/${sym}-trades-\\(.*\\)\\.zip/\\1/")

        # Filter by date if specified
        if [ -n "$FILTER_DATE" ] && [ "$FILTER_DATE" != "all" ] && [ "$day" != "$FILTER_DATE" ]; then
            continue
        fi

        pnl=$(run_one "$sym" "$data_file")
        if [ -n "$pnl" ]; then
            sym_total=$(echo "$sym_total + $pnl" | bc -l)
            sym_days=$((sym_days + 1))
        else
            echo "  $sym $day: FAILED"
        fi
    done

    if [ "$sym_days" -gt 0 ]; then
        sym_avg=$(echo "$sym_total / $sym_days" | bc -l)
        # Format to 4 decimal places
        sym_total_fmt=$(printf "%.4f" "$sym_total")
        sym_avg_fmt=$(printf "%.4f" "$sym_avg")
        echo "$sym: total=\$${sym_total_fmt}  avg/day=\$${sym_avg_fmt}  days=$sym_days"
        portfolio_total=$(echo "$portfolio_total + $sym_total" | bc -l)
        portfolio_days=$sym_days
        symbol_results="$symbol_results\n$sym_avg_fmt $sym"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
if [ "$portfolio_days" -gt 0 ]; then
    portfolio_avg=$(printf "%.4f" "$(echo "$portfolio_total / $portfolio_days" | bc -l)")
    portfolio_total_fmt=$(printf "%.4f" "$portfolio_total")
    portfolio_annual=$(printf "%.0f" "$(echo "$portfolio_total / $portfolio_days * 365" | bc -l)")
    echo " PORTFOLIO: total=\$${portfolio_total_fmt}  avg/day=\$${portfolio_avg}  annual=~\$${portfolio_annual}"
else
    echo " No results."
fi
echo "═══════════════════════════════════════════════════════════════"
