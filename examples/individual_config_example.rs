//! # Individual Strategy Configuration Example
//! 
//! This example demonstrates how each strategy now has its own separate config file
//! and can be initialized without any parameters using `Strategy::new()`.
//! 
//! Each strategy loads its configuration from:
//! `config/{strategy_name}.toml`

use strategies::{
    config::StrategyConfig,
    strategy::Strategy,
    rsi_strategy::RsiStrategy,
    hft_ultra_fast_strategy::HftUltraFastStrategy,
    adaptive_multi_factor_strategy::AdaptiveMultiFactorStrategy,
    neural_market_microstructure_strategy::NeuralMarketMicrostructureStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Individual Strategy Configuration Example");
    println!("==========================================\n");

    // Example 1: RSI Strategy
    println!("📊 Example 1: RSI Strategy");
    let rsi_strategy = RsiStrategy::new(); // Loads from config/rsi_strategy.toml
    println!("   - Info: {}", rsi_strategy.get_info());
    println!("   - Config file: config/rsi_strategy.toml");
    println!();

    // Example 2: HFT Ultra-Fast Strategy
    println!("⚡ Example 2: HFT Ultra-Fast Strategy");
    let hft_strategy = HftUltraFastStrategy::new(); // Loads from config/hft_ultra_fast_strategy.toml
    println!("   - Info: {}", hft_strategy.get_info());
    println!("   - Config file: config/hft_ultra_fast_strategy.toml");
    println!();

    // Example 3: Adaptive Multi-Factor Strategy
    println!("🧠 Example 3: Adaptive Multi-Factor Strategy");
    let adaptive_strategy = AdaptiveMultiFactorStrategy::new(); // Loads from config/adaptive_multi_factor_strategy.toml
    println!("   - Info: {}", adaptive_strategy.get_info());
    println!("   - Config file: config/adaptive_multi_factor_strategy.toml");
    println!();

    // Example 4: Neural Market Microstructure Strategy
    println!("🕸️  Example 4: Neural Market Microstructure Strategy");
    let neural_strategy = NeuralMarketMicrostructureStrategy::new(); // Loads from config/neural_market_microstructure_strategy.toml
    println!("   - Info: {}", neural_strategy.get_info());
    println!("   - Config file: config/neural_market_microstructure_strategy.toml");
    println!();

    // Example 5: Show all available config files
    println!("📋 Example 5: Available Configuration Files");
    println!("Each strategy has its own config file:");
    println!("   - config/rsi_strategy.toml");
    println!("   - config/mean_reversion_strategy.toml");
    println!("   - config/momentum_scalping_strategy.toml");
    println!("   - config/kalman_filter_strategy.toml");
    println!("   - config/order_book_imbalance_strategy.toml");
    println!("   - config/spline_strategy.toml");
    println!("   - config/vwap_deviation_strategy.toml");
    println!("   - config/zscore_strategy.toml");
    println!("   - config/fractal_approximation_strategy.toml");
    println!("   - config/hft_ultra_fast_strategy.toml");
    println!("   - config/hft_market_maker_strategy.toml");
    println!("   - config/adaptive_multi_factor_strategy.toml");
    println!("   - config/neural_market_microstructure_strategy.toml");
    println!();

    // Example 6: Show how to switch strategies
    println!("🔄 Example 6: Strategy Switching");
    println!("To switch strategies in main.rs, simply change:");
    println!("   // Current: HFT Ultra-Fast Strategy");
    println!("   let strategy = HftUltraFastStrategy::new();");
    println!();
    println!("   // To RSI Strategy:");
    println!("   // let strategy = RsiStrategy::new();");
    println!();
    println!("   // To Adaptive Multi-Factor Strategy:");
    println!("   // let strategy = AdaptiveMultiFactorStrategy::new();");
    println!();

    // Example 7: Show configuration loading
    println!("⚙️  Example 7: Configuration Loading");
    println!("Each strategy automatically loads its config:");
    println!("   - No parameters needed in constructor");
    println!("   - Config file is loaded automatically");
    println!("   - Default values used if config file missing");
    println!("   - Easy to modify parameters without code changes");
    println!();

    println!("✅ All strategies now use individual configuration files!");
    println!("📁 Config directory: config/");

    Ok(())
} 