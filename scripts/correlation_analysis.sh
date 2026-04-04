#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Pulsar Portfolio Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════
#
# Collects daily PnL for all 19 symbols into a CSV, then computes
# pairwise correlations and portfolio statistics.
#
# Output: scripts/correlation_report.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PULSAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PULSAR_DIR/target/release/binance-bot"
DATA_DIR="$PULSAR_DIR/data/binance/daily/trades"
CSV_FILE="$SCRIPT_DIR/daily_pnl.csv"
REPORT_FILE="$SCRIPT_DIR/correlation_report.txt"

SYMBOLS=(
    TRUUSDT ARKMUSDT PHBUSDT ACTUSDT IOUSDT WUSDT ACEUSDT
    GLMRUSDT GTCUSDT WIFUSDT HFTUSDT MBOXUSDT HIGHUSDT
    ARUSDT SXTUSDT EIGENUSDT ATAUSDT RAREUSDT 1000CATUSDT
)

DATES=(2026-03-24 2026-03-25 2026-03-26 2026-03-27 2026-03-28 2026-03-29 2026-03-30 2026-03-31)

extract_pnl() {
    sed 's/\x1b\[[0-9;]*m//g' | grep 'session_summary' | sed 's/.*realized_pnl=\([^ ]*\).*/\1/'
}

get_configs() {
    local sym="$1"
    local sym_lower
    sym_lower=$(echo "$sym" | tr '[:upper:]' '[:lower:]')
    if [ "$sym" = "TRUUSDT" ]; then
        echo "$PULSAR_DIR/config/trading_config.toml $PULSAR_DIR/config/strategies/market_maker.toml"
    else
        echo "$PULSAR_DIR/config/trading_config_${sym_lower}.toml $PULSAR_DIR/config/strategies/market_maker_${sym_lower}.toml"
    fi
}

run_one() {
    local sym="$1"
    local data_file="$2"
    local configs
    configs=$(get_configs "$sym")
    local trading_config strategy_config
    trading_config=$(echo "$configs" | cut -d' ' -f1)
    strategy_config=$(echo "$configs" | cut -d' ' -f2)

    if [ "$sym" = "TRUUSDT" ]; then
        "$BINARY" --strategy market-maker backtest --uri "$data_file" 2>&1 | extract_pnl
    else
        "$BINARY" --strategy market-maker --config "$trading_config" --strategy-config "$strategy_config" backtest --uri "$data_file" 2>&1 | extract_pnl
    fi
}

echo "Collecting daily PnL data..."

# CSV header
header="date"
for sym in "${SYMBOLS[@]}"; do
    header="$header,$sym"
done
echo "$header" > "$CSV_FILE"

# Collect PnL for each date
for date in "${DATES[@]}"; do
    row="$date"
    for sym in "${SYMBOLS[@]}"; do
        data_file="$DATA_DIR/$sym/${sym}-trades-${date}.zip"
        if [ -f "$data_file" ]; then
            pnl=$(run_one "$sym" "$data_file")
            [ -z "$pnl" ] && pnl="0"
        else
            pnl="0"
        fi
        row="$row,$pnl"
    done
    echo "$row" >> "$CSV_FILE"
    echo "  $date done"
done

echo "Daily PnL CSV saved to: $CSV_FILE"
echo ""

# Now compute correlation matrix using awk
echo "Computing correlations..."

cat > "$REPORT_FILE" << 'HEADER'
═══════════════════════════════════════════════════════════════
 Pulsar Portfolio Correlation Analysis
═══════════════════════════════════════════════════════════════

HEADER

# Use awk to compute correlation matrix
awk -F',' '
BEGIN {
    # Will be filled from header
}
NR == 1 {
    # Parse header
    ncols = NF - 1
    for (i = 2; i <= NF; i++) {
        name[i-1] = $i
    }
    next
}
{
    nrows++
    for (i = 2; i <= NF; i++) {
        val[nrows][i-1] = $i + 0
        sum[i-1] += $i
    }
}
END {
    n = nrows

    # Compute means
    for (i = 1; i <= ncols; i++) {
        mean[i] = sum[i] / n
    }

    # Compute std devs and covariances
    for (i = 1; i <= ncols; i++) {
        for (j = i; j <= ncols; j++) {
            cov = 0
            for (k = 1; k <= n; k++) {
                cov += (val[k][i] - mean[i]) * (val[k][j] - mean[j])
            }
            cov /= (n - 1)
            covariance[i][j] = cov
            covariance[j][i] = cov
        }
        stddev[i] = sqrt(covariance[i][i])
    }

    # Print summary statistics
    print "=== Per-Symbol Statistics (8 days) ==="
    print ""
    printf "%-14s %8s %8s %8s %8s\n", "Symbol", "Mean", "StdDev", "Min", "Max"
    printf "%-14s %8s %8s %8s %8s\n", "------", "----", "------", "---", "---"

    portfolio_mean = 0
    for (i = 1; i <= ncols; i++) {
        mn = 999; mx = -999
        for (k = 1; k <= n; k++) {
            if (val[k][i] < mn) mn = val[k][i]
            if (val[k][i] > mx) mx = val[k][i]
        }
        printf "%-14s %8.4f %8.4f %8.4f %8.4f\n", name[i], mean[i], stddev[i], mn, mx
        portfolio_mean += mean[i]
    }

    print ""
    printf "Portfolio daily mean: $%.4f\n", portfolio_mean

    # Portfolio variance (sum of all covariances)
    portfolio_var = 0
    for (i = 1; i <= ncols; i++) {
        for (j = 1; j <= ncols; j++) {
            portfolio_var += covariance[i][j]
        }
    }
    portfolio_std = sqrt(portfolio_var)
    printf "Portfolio daily std:  $%.4f\n", portfolio_std
    if (portfolio_std > 0) {
        printf "Portfolio Sharpe (daily): %.2f\n", portfolio_mean / portfolio_std
        printf "Portfolio Sharpe (annual): %.2f\n", (portfolio_mean / portfolio_std) * sqrt(365)
    }

    # Average pairwise correlation
    print ""
    print "=== Correlation Matrix (top-10 highest pairs) ==="
    print ""

    # Collect all pairs
    npairs = 0
    for (i = 1; i <= ncols; i++) {
        for (j = i+1; j <= ncols; j++) {
            if (stddev[i] > 0 && stddev[j] > 0) {
                npairs++
                corr = covariance[i][j] / (stddev[i] * stddev[j])
                pair_corr[npairs] = corr
                pair_name[npairs] = name[i] " / " name[j]
            }
        }
    }

    # Simple sort (selection sort) to find top-10
    for (p = 1; p <= npairs; p++) {
        sorted_idx[p] = p
    }
    for (p = 1; p <= npairs - 1; p++) {
        max_idx = p
        for (q = p + 1; q <= npairs; q++) {
            if (pair_corr[sorted_idx[q]] > pair_corr[sorted_idx[max_idx]]) {
                max_idx = q
            }
        }
        if (max_idx != p) {
            tmp = sorted_idx[p]
            sorted_idx[p] = sorted_idx[max_idx]
            sorted_idx[max_idx] = tmp
        }
    }

    printf "%-35s %8s\n", "Pair", "Corr"
    printf "%-35s %8s\n", "----", "----"
    limit = (npairs < 10) ? npairs : 10
    for (p = 1; p <= limit; p++) {
        idx = sorted_idx[p]
        printf "%-35s %8.4f\n", pair_name[idx], pair_corr[idx]
    }

    # Bottom 10
    print ""
    print "=== Bottom-10 (lowest/negative correlations) ==="
    print ""
    printf "%-35s %8s\n", "Pair", "Corr"
    printf "%-35s %8s\n", "----", "----"
    start = (npairs - limit + 1 > 0) ? npairs - limit + 1 : 1
    for (p = npairs; p >= start; p--) {
        idx = sorted_idx[p]
        printf "%-35s %8.4f\n", pair_name[idx], pair_corr[idx]
    }

    # Average correlation
    sum_corr = 0
    for (p = 1; p <= npairs; p++) {
        sum_corr += pair_corr[p]
    }
    avg_corr = sum_corr / npairs
    print ""
    printf "Average pairwise correlation: %.4f\n", avg_corr
    printf "Number of pairs: %d\n", npairs

    # Diversification ratio
    sum_individual_var = 0
    for (i = 1; i <= ncols; i++) {
        sum_individual_var += covariance[i][i]
    }
    if (portfolio_var > 0) {
        div_ratio = sqrt(sum_individual_var) / sqrt(portfolio_var)
        printf "Diversification ratio: %.2f (>1 = diversification benefit)\n", div_ratio
    }
}
' "$CSV_FILE" >> "$REPORT_FILE"

echo ""
cat "$REPORT_FILE"
