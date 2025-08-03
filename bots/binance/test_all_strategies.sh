#!/bin/bash

# Test All Strategies Script
# This script tests all available strategies and collects their performance metrics

echo "=== STRATEGY PERFORMANCE TESTING ==="
echo "Testing all strategies on DOGEUSDT data..."
echo ""

# Print header
printf "%-35s %-8s %-8s %-12s %-8s\n" "Strategy" "Trades" "Win Rate" "Net PnL" "Max DD"
echo "-------------------------------------------------------------------------------"

# Function to test a strategy
test_strategy() {
    local strategy_name=$1
    local strategy_import=$2
    local strategy_constructor=$3
    
    # Create temporary main.rs with the strategy
    cat > temp_main.rs << EOF
use ::trade::trader::TradeMode;
use ::trade::trader::Trader;
use clap::{Parser, Subcommand};
use std::env;
use std::error::Error;
use strategies::strategy::Strategy;
use tracing::info;

mod backtest;
mod trade;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Trade,
    Emulate,
    Backtest {
        #[arg(short, long)]
        url: Option<String>,
        #[arg(short, long)]
        path: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    
    $strategy_import
    let strategy = $strategy_constructor;
    
    let trading_symbol = "DOGEUSDT";
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");
    
    info!("Testing strategy: {}", strategy.get_info());
    
    let trade_mode = match cli.command {
        Commands::Trade => TradeMode::Real,
        Commands::Emulate => TradeMode::Emulated,
        Commands::Backtest { .. } => TradeMode::Emulated,
    };
    
    match cli.command {
        Commands::Trade => {
            info!("Starting live trading...");
            // Live trading implementation would go here
        }
        Commands::Emulate => {
            info!("Starting emulated trading...");
            // Emulated trading implementation would go here
        }
        Commands::Backtest { path, url } => {
            if let Some(data_path) = path {
                info!("Starting backtest with data from: {}", data_path);
                backtest::run_backtest(&data_path, strategy, trading_symbol).await?;
            } else if let Some(ws_url) = url {
                info!("Starting backtest with WebSocket data from: {}", ws_url);
                backtest::run_backtest(&ws_url, strategy, trading_symbol).await?;
            } else {
                return Err("No data source specified for backtest".into());
            }
        }
    }
    
    Ok(())
}
EOF
    
    # Replace main.rs temporarily
    cp src/main.rs src/main.rs.backup
    cp temp_main.rs src/main.rs
    
    # Run the backtest and capture the result
    RUST_LOG=info cargo run backtest --path ../../data/DOGEUSDT-trades-2025-05-28.zip > temp_output.txt 2>&1
    
    # Extract and parse the result
    if grep -q "=== FINAL BACKTEST RESULTS ===" temp_output.txt; then
        result_line=$(grep -A 1 "=== FINAL BACKTEST RESULTS ===" temp_output.txt | tail -1 | sed 's/.*INFO binance_bot::backtest: //')
        
        # Parse the metrics using sed
        trades=$(echo "$result_line" | sed -n 's/.*Trades: \([0-9]*\).*/\1/p')
        win_rate=$(echo "$result_line" | sed -n 's/.*Win Rate: \([0-9.]*\)%.*/\1/p')
        net_pnl=$(echo "$result_line" | sed -n 's/.*Net PnL: \([-0-9.]*\).*/\1/p')
        max_dd=$(echo "$result_line" | sed -n 's/.*Max Drawdown: \([0-9.]*\)%.*/\1/p')
        
        # Set defaults if parsing failed
        trades=${trades:-0}
        win_rate=${win_rate:-0}
        net_pnl=${net_pnl:-0.000000}
        max_dd=${max_dd:-0}
        
        # Format the output
        printf "%-35s %-8s %-8s %-12s %-8s\n" "$strategy_name" "$trades" "${win_rate}%" "$net_pnl" "${max_dd}%"
    else
        printf "%-35s %-8s %-8s %-12s %-8s\n" "$strategy_name" "0" "0%" "0.000000" "0%"
    fi
    
    # Clean up
    rm temp_output.txt
    rm temp_main.rs
    
    # Restore original main.rs
    mv src/main.rs.backup src/main.rs
}

# Test all strategies
test_strategy "RSI Strategy" "use strategies::rsi_strategy::RsiStrategy;" "RsiStrategy::new()"
test_strategy "Kalman Filter Strategy" "use strategies::kalman_filter_strategy::KalmanFilterStrategy;" "KalmanFilterStrategy::new()"
test_strategy "Mean Reversion Strategy" "use strategies::mean_reversion_strategy::MeanReversionStrategy;" "MeanReversionStrategy::new()"
test_strategy "Momentum Scalping Strategy" "use strategies::momentum_scalping_strategy::MomentumScalping;" "MomentumScalping::new()"
test_strategy "Order Book Imbalance Strategy" "use strategies::order_book_imbalance_strategy::OrderBookImbalance;" "OrderBookImbalance::new()"
test_strategy "Spline Strategy" "use strategies::spline_strategy::SplineStrategy;" "SplineStrategy::new()"
test_strategy "VWAP Deviation Strategy" "use strategies::vwap_deviation_strategy::VwapDeviationStrategy;" "VwapDeviationStrategy::new()"
test_strategy "Z-Score Strategy" "use strategies::zscore_strategy::ZScoreStrategy;" "ZScoreStrategy::new()"
test_strategy "Ultra Fast Strategy" "use strategies::ultra_fast_strategy::UltraFastStrategy;" "UltraFastStrategy::new()"
test_strategy "Market Maker Strategy" "use strategies::market_maker_strategy::MarketMakerStrategy;" "MarketMakerStrategy::new()"
test_strategy "Fractal Approximation Strategy" "use strategies::fractal_approximation_strategy::FractalApproximationStrategy;" "FractalApproximationStrategy::new()"
test_strategy "Neural Market Microstructure Strategy" "use strategies::neural_market_microstructure_strategy::NeuralMarketMicrostructureStrategy;" "NeuralMarketMicrostructureStrategy::new()"
test_strategy "Advanced Momentum Strategy" "use strategies::advanced_momentum_strategy::AdvancedMomentumStrategy;" "AdvancedMomentumStrategy::new()"
test_strategy "Trend Volume Strategy" "use strategies::trend_volume_strategy::TrendVolumeStrategy;" "TrendVolumeStrategy::new()"
test_strategy "Enhanced RSI Strategy" "use strategies::enhanced_rsi_strategy::EnhancedRsiStrategy;" "EnhancedRsiStrategy::new()"
test_strategy "Refined RSI Strategy" "use strategies::refined_rsi_strategy::RefinedRsiStrategy;" "RefinedRsiStrategy::new()"

echo ""
echo "=== TESTING COMPLETE ==="
echo "Legend: Trades = Number of trades, Win Rate = Percentage of winning trades"
echo "Net PnL = Net profit/loss, Max DD = Maximum drawdown percentage" 