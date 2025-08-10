#!/bin/bash

# Test all strategies and compare performance
echo "Testing all strategies on full dataset..."
echo "========================================"

# List of strategies to test
strategies=(
    "PulsarAlphaStrategy"
    "MeanReversionStrategy" 
    "StochasticHftStrategy"
    "EnhancedPulsarStrategy"
    "SentimentAnalysisStrategy"
    "PulsarTradingStrategy"
)

# Test each strategy
for strategy in "${strategies[@]}"; do
    echo ""
    echo "Testing $strategy..."
    echo "-------------------"
    
    # Update the strategy in main.rs
    sed -i '' "s/use strategies::[A-Za-z]*Strategy;/use strategies::${strategy};/" bots/binance/src/main.rs
    sed -i '' "s/\/\/ Create [A-Za-z ]* strategy instance/\/\/ Create ${strategy} strategy instance/" bots/binance/src/main.rs
    sed -i '' "s/let strategy = [A-Za-z]*Strategy::new();/let strategy = ${strategy}::new();/" bots/binance/src/main.rs
    
    # Run backtest on the largest dataset
    echo "Running backtest on DOGEUSDT-trades-2025-08-09.zip..."
    RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-08-09.zip 2>/dev/null | tail -1
    
    echo ""
done

echo ""
echo "Strategy comparison complete!"
