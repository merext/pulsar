#!/bin/bash

# Strategy Optimization Test Script
# This script tests all strategies and records their P&L performance

echo "=== Strategy Performance Optimization Test ==="
echo "Testing all strategies with optimized configurations..."
echo ""

# Function to test a strategy and extract final P&L
test_strategy() {
    local strategy_name=$1
    local strategy_class=$2
    local config_file=$3
    
    echo "Testing: $strategy_name"
    
    # Update main.rs to use this strategy
    sed -i.bak "s/use strategies::.*::.*Strategy;/use strategies::$strategy_class::${strategy_class}Strategy;/" src/main.rs
    sed -i.bak "s/let strategy = .*Strategy::new();/let strategy = ${strategy_class}Strategy::new();/" src/main.rs
    
    # Run backtest and extract final P&L
    local result=$(RUST_LOG=error cargo run backtest --path ../../data/DOGEUSDT-trades-2025-05-28.zip 2>&1 | grep "place_sell_order" | tail -1 | grep -o 'profit=[^[:space:]]*' | cut -d'=' -f2)
    
    if [ -z "$result" ]; then
        result="0.000000"
    fi
    
    echo "$strategy_name: $result"
    echo "$strategy_name,$result" >> strategy_results.csv
    echo ""
}

# Create results file
echo "Strategy,PnL" > strategy_results.csv

# Test all strategies
test_strategy "HFT Ultra Fast" "hft_ultra_fast_strategy" "config/hft_ultra_fast_strategy.toml"
test_strategy "HFT Market Maker" "hft_market_maker_strategy" "config/hft_market_maker_strategy.toml"
test_strategy "RSI" "rsi_strategy" "config/rsi_strategy.toml"
test_strategy "Mean Reversion" "mean_reversion" "config/mean_reversion_strategy.toml"
test_strategy "Momentum Scalping" "momentum_scalping" "config/momentum_scalping_strategy.toml"
test_strategy "Kalman Filter" "kalman_filter_strategy" "config/kalman_filter_strategy.toml"
test_strategy "Order Book Imbalance" "order_book_imbalance_strategy" "config/order_book_imbalance_strategy.toml"
test_strategy "Spline" "spline_strategy" "config/spline_strategy.toml"
test_strategy "VWAP Deviation" "vwap_deviation_strategy" "config/vwap_deviation_strategy.toml"
test_strategy "Z-Score" "zscore_strategy" "config/zscore_strategy.toml"
test_strategy "Fractal Approximation" "fractal_approximation_strategy" "config/fractal_approximation_strategy.toml"
test_strategy "Adaptive Multi-Factor" "adaptive_multi_factor_strategy" "config/adaptive_multi_factor_strategy.toml"
test_strategy "Neural Market Microstructure" "neural_market_microstructure_strategy" "config/neural_market_microstructure_strategy.toml"

echo "=== Final Results ==="
echo "Strategy,PnL"
sort -t',' -k2 -nr strategy_results.csv

# Cleanup
rm -f src/main.rs.bak 