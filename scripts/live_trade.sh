#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Pulsar Live Trading Launcher
# ──────────────────────────────────────────────────────────────────────
# Usage:
#   ./scripts/live_trade.sh              # Real trading (TRUUSDT, default config)
#   ./scripts/live_trade.sh emulate      # Live data, simulated execution
#   ./scripts/live_trade.sh trade ARKMUSDT  # Real trading with specific symbol
#
# Environment variables (required for 'trade' mode):
#   BINANCE_API_KEY    — Binance API key
#   BINANCE_API_SECRET — Binance API secret
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-trade}"
SYMBOL="${2:-}"

# Validate mode
if [[ "$MODE" != "trade" && "$MODE" != "emulate" ]]; then
    echo "ERROR: Unknown mode '$MODE'. Use 'trade' or 'emulate'."
    exit 1
fi

# Check API credentials for real trading
if [[ "$MODE" == "trade" ]]; then
    if [[ -z "${BINANCE_API_KEY:-}" ]]; then
        echo "ERROR: BINANCE_API_KEY not set."
        echo "  export BINANCE_API_KEY='your-api-key'"
        exit 1
    fi
    if [[ -z "${BINANCE_API_SECRET:-}" ]]; then
        echo "ERROR: BINANCE_API_SECRET not set."
        echo "  export BINANCE_API_SECRET='your-api-secret'"
        exit 1
    fi
fi

# Determine config files based on symbol
TRADING_CONFIG="config/trading_config.toml"
STRATEGY_CONFIG="config/strategies/market_maker.toml"

if [[ -n "$SYMBOL" && "$SYMBOL" != "TRUUSDT" ]]; then
    # Use symbol-specific config if it exists
    SYM_LOWER=$(echo "$SYMBOL" | tr '[:upper:]' '[:lower:]')
    SYM_TRADING="config/trading_config_${SYM_LOWER}.toml"
    SYM_STRATEGY="config/strategies/market_maker_${SYM_LOWER}.toml"

    if [[ -f "$PROJECT_DIR/$SYM_TRADING" ]]; then
        TRADING_CONFIG="$SYM_TRADING"
    else
        echo "WARNING: No trading config for $SYMBOL at $SYM_TRADING, using default."
    fi

    if [[ -f "$PROJECT_DIR/$SYM_STRATEGY" ]]; then
        STRATEGY_CONFIG="$SYM_STRATEGY"
    else
        echo "WARNING: No strategy config for $SYMBOL at $SYM_STRATEGY, using default."
    fi
fi

echo "═══════════════════════════════════════════════════"
echo " Pulsar Live Trading"
echo "═══════════════════════════════════════════════════"
echo " Mode:     $MODE"
echo " Symbol:   ${SYMBOL:-TRUUSDT (default)}"
echo " Config:   $TRADING_CONFIG"
echo " Strategy: $STRATEGY_CONFIG"
echo "═══════════════════════════════════════════════════"

# Build first (release mode for performance)
echo "Building..."
cargo build --release -p binance-bot 2>&1 | tail -1

# Launch
echo "Starting..."
cd "$PROJECT_DIR"
exec cargo run --release -p binance-bot -- --strategy market-maker "$MODE"
