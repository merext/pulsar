#!/bin/bash
# Compute correlation matrix from daily_pnl.csv using macOS-compatible awk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_FILE="$SCRIPT_DIR/daily_pnl.csv"
REPORT_FILE="$SCRIPT_DIR/correlation_report.txt"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE not found. Run correlation_analysis.sh first."
    exit 1
fi

awk -F',' '
NR == 1 {
    ncols = NF - 1
    for (i = 2; i <= NF; i++) {
        name[i-1] = $i
    }
    next
}
{
    nrows++
    for (i = 2; i <= NF; i++) {
        v = $i + 0
        # linear index: val_ROW_COL
        key = nrows "_" (i-1)
        val[key] = v
        sum[i-1] += v
    }
}
END {
    n = nrows

    # Means
    for (i = 1; i <= ncols; i++) {
        mean[i] = sum[i] / n
    }

    # Covariance matrix (symmetric)
    for (i = 1; i <= ncols; i++) {
        for (j = i; j <= ncols; j++) {
            cov = 0
            for (k = 1; k <= n; k++) {
                ki = k "_" i
                kj = k "_" j
                cov += (val[ki] - mean[i]) * (val[kj] - mean[j])
            }
            cov /= (n - 1)
            cov_key_ij = i "_" j
            cov_key_ji = j "_" i
            covar[cov_key_ij] = cov
            covar[cov_key_ji] = cov
        }
        sd_key = i "_" i
        stddev[i] = sqrt(covar[sd_key])
    }

    # Print header
    print "═══════════════════════════════════════════════════════════════"
    print " Pulsar Portfolio Correlation Analysis (8 days, 19 symbols)"
    print "═══════════════════════════════════════════════════════════════"
    print ""

    # Per-symbol stats
    print "=== Per-Symbol Statistics ==="
    print ""
    printf "%-14s %8s %8s %8s %8s %8s\n", "Symbol", "Mean", "StdDev", "Min", "Max", "CV"
    printf "%-14s %8s %8s %8s %8s %8s\n", "------", "----", "------", "---", "---", "--"

    portfolio_mean = 0
    for (i = 1; i <= ncols; i++) {
        mn = 999; mx = -999
        for (k = 1; k <= n; k++) {
            key = k "_" i
            if (val[key] < mn) mn = val[key]
            if (val[key] > mx) mx = val[key]
        }
        cv = (mean[i] > 0) ? stddev[i] / mean[i] : 0
        printf "%-14s %8.4f %8.4f %8.4f %8.4f %8.2f\n", name[i], mean[i], stddev[i], mn, mx, cv
        portfolio_mean += mean[i]
    }

    print ""
    printf "Portfolio daily mean:   $%.4f\n", portfolio_mean

    # Portfolio variance
    portfolio_var = 0
    for (i = 1; i <= ncols; i++) {
        for (j = 1; j <= ncols; j++) {
            key = i "_" j
            portfolio_var += covar[key]
        }
    }
    portfolio_std = sqrt(portfolio_var)
    printf "Portfolio daily std:    $%.4f\n", portfolio_std

    if (portfolio_std > 0) {
        daily_sharpe = portfolio_mean / portfolio_std
        annual_sharpe = daily_sharpe * sqrt(365)
        printf "Portfolio Sharpe (daily):  %.2f\n", daily_sharpe
        printf "Portfolio Sharpe (annual): %.2f\n", annual_sharpe
    }

    # Sum of individual variances (for diversification ratio)
    sum_ind_var = 0
    for (i = 1; i <= ncols; i++) {
        key = i "_" i
        sum_ind_var += covar[key]
    }
    if (portfolio_var > 0) {
        div_ratio = sqrt(sum_ind_var) / sqrt(portfolio_var)
        printf "Diversification ratio:    %.2f (>1 = diversification benefit)\n", div_ratio
    }

    # Collect all pairwise correlations
    npairs = 0
    for (i = 1; i <= ncols; i++) {
        for (j = i+1; j <= ncols; j++) {
            if (stddev[i] > 0 && stddev[j] > 0) {
                npairs++
                key = i "_" j
                pc = covar[key] / (stddev[i] * stddev[j])
                pcorr[npairs] = pc
                pname[npairs] = name[i] " / " name[j]
                pidx[npairs] = npairs
            }
        }
    }

    # Selection sort by correlation (descending)
    for (p = 1; p <= npairs - 1; p++) {
        max_p = p
        for (q = p + 1; q <= npairs; q++) {
            if (pcorr[pidx[q]] > pcorr[pidx[max_p]]) {
                max_p = q
            }
        }
        if (max_p != p) {
            tmp = pidx[p]
            pidx[p] = pidx[max_p]
            pidx[max_p] = tmp
        }
    }

    print ""
    print "=== Top-15 Highest Correlations ==="
    print ""
    printf "%-35s %8s\n", "Pair", "Corr"
    printf "%-35s %8s\n", "----", "----"
    limit = (npairs < 15) ? npairs : 15
    for (p = 1; p <= limit; p++) {
        idx = pidx[p]
        printf "%-35s %+8.4f\n", pname[idx], pcorr[idx]
    }

    print ""
    print "=== Bottom-15 Lowest Correlations ==="
    print ""
    printf "%-35s %8s\n", "Pair", "Corr"
    printf "%-35s %8s\n", "----", "----"
    start = (npairs - 14 > 0) ? npairs - 14 : 1
    for (p = npairs; p >= start; p--) {
        idx = pidx[p]
        printf "%-35s %+8.4f\n", pname[idx], pcorr[idx]
    }

    # Average correlation
    sum_corr = 0
    for (p = 1; p <= npairs; p++) {
        sum_corr += pcorr[p]
    }
    avg_corr = sum_corr / npairs
    print ""
    printf "Average pairwise correlation: %+.4f\n", avg_corr
    printf "Number of pairs: %d\n", npairs

    # Count negative correlations
    neg_count = 0
    for (p = 1; p <= npairs; p++) {
        if (pcorr[p] < 0) neg_count++
    }
    printf "Negative correlations: %d / %d (%.0f%%)\n", neg_count, npairs, (neg_count/npairs)*100
}
' "$CSV_FILE" | tee "$REPORT_FILE"
