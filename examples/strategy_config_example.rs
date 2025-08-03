//! # Strategy Configuration Example
//! 
//! This example demonstrates how to use the centralized configuration file
//! to initialize different trading strategies with their parameters.
//! 
//! All strategies now use the same configuration file:
//! `config/advanced_strategies_config.toml`

use strategies::{
    mean_reversion_strategy::MeanReversionStrategy,
    momentum_scalping_strategy::MomentumScalping,
    rsi_strategy::RsiStrategy,
    order_book_imbalance_strategy::OrderBookImbalance,
    zscore_strategy::ZScoreStrategy,
    kalman_filter_strategy::KalmanFilterStrategy,
    vwap_deviation_strategy::VwapDeviationStrategy,
    spline_strategy::SplineStrategy,
    fractal_approximation_strategy::FractalApproximationStrategy,
    adaptive_multi_factor_strategy::AdaptiveMultiFactorStrategy,
    neural_market_microstructure_strategy::NeuralMarketMicrostructureStrategy,
    hft_ultra_fast_strategy::HftUltraFastStrategy,
    hft_market_maker_strategy::HftMarketMakerStrategy,
};
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Strategy Configuration Example");
    println!("================================\n");

    // Load configuration from file
    let config = StrategyConfig::from_file("../../config/advanced_strategies_config.toml")?;
    println!("✅ Configuration loaded successfully\n");

    // Example 1: RSI Strategy with config
    println!("📊 Example 1: RSI Strategy");
    let rsi_config = config.section("rsi_strategy").unwrap_or_else(|| {
        println!("⚠️  Using default RSI configuration");
        StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
    });
    let rsi_period = rsi_config.get_or("period", DefaultConfig::rsi_period());
    let rsi_overbought = rsi_config.get_or("overbought", DefaultConfig::rsi_overbought());
    let rsi_oversold = rsi_config.get_or("oversold", DefaultConfig::rsi_oversold());
    let rsi_scale = rsi_config.get_or("scale", DefaultConfig::rsi_scale());
    
    let rsi_strategy = RsiStrategy::new(rsi_period, rsi_overbought, rsi_oversold, rsi_scale);
    println!("   - Period: {}", rsi_period);
    println!("   - Overbought: {}", rsi_overbought);
    println!("   - Oversold: {}", rsi_oversold);
    println!("   - Scale: {}", rsi_scale);
    println!("   - Info: {}", rsi_strategy.get_info());
    println!();

    // Example 2: HFT Ultra-Fast Strategy with config
    println!("⚡ Example 2: HFT Ultra-Fast Strategy");
    let hft_config = config.section("hft_ultra_fast").unwrap_or_else(|| {
        println!("⚠️  Using default HFT configuration");
        StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
    });
    let hft_signal_threshold = hft_config.get_or("signal_threshold", DefaultConfig::hft_signal_threshold());
    let hft_buffer_size = hft_config.get_or("buffer_size", 64);
    let hft_fast_ema_alpha = hft_config.get_or("fast_ema_alpha", 0.1);
    
    let hft_strategy = HftUltraFastStrategy::new();
    println!("   - Signal threshold: {:.1}%", hft_signal_threshold * 100.0);
    println!("   - Buffer size: {}", hft_buffer_size);
    println!("   - Fast EMA alpha: {}", hft_fast_ema_alpha);
    println!("   - Info: {}", hft_strategy.get_info());
    println!();

    // Example 3: Adaptive Multi-Factor Strategy with config
    println!("🧠 Example 3: Adaptive Multi-Factor Strategy");
    let adaptive_config = config.section("adaptive_multi_factor").unwrap_or_else(|| {
        println!("⚠️  Using default Adaptive configuration");
        StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
    });
    let adaptive_short_window = adaptive_config.get_or("short_window", DefaultConfig::adaptive_short_window());
    let adaptive_long_window = adaptive_config.get_or("long_window", DefaultConfig::adaptive_long_window());
    let adaptive_volatility_window = adaptive_config.get_or("volatility_window", DefaultConfig::adaptive_volatility_window());
    let adaptive_volume_window = adaptive_config.get_or("volume_window", DefaultConfig::adaptive_volume_window());
    let adaptive_signal_threshold = adaptive_config.get_or("signal_threshold", DefaultConfig::adaptive_signal_threshold());
    
    let adaptive_strategy = AdaptiveMultiFactorStrategy::new(
        adaptive_short_window,
        adaptive_long_window,
        adaptive_volatility_window,
        adaptive_volume_window,
        adaptive_signal_threshold,
    );
    println!("   - Short window: {}", adaptive_short_window);
    println!("   - Long window: {}", adaptive_long_window);
    println!("   - Volatility window: {}", adaptive_volatility_window);
    println!("   - Volume window: {}", adaptive_volume_window);
    println!("   - Signal threshold: {:.1}%", adaptive_signal_threshold * 100.0);
    println!("   - Info: {}", adaptive_strategy.get_info());
    println!();

    // Example 4: Neural Market Microstructure Strategy with config
    println!("🕸️  Example 4: Neural Market Microstructure Strategy");
    let neural_config = config.section("neural_market_microstructure").unwrap_or_else(|| {
        println!("⚠️  Using default Neural configuration");
        StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
    });
    let neural_short_window = neural_config.get_or("short_window", DefaultConfig::neural_short_window());
    let neural_medium_window = neural_config.get_or("medium_window", DefaultConfig::neural_medium_window());
    let neural_long_window = neural_config.get_or("long_window", DefaultConfig::neural_long_window());
    let neural_micro_window = neural_config.get_or("micro_window", DefaultConfig::neural_micro_window());
    let neural_signal_threshold = neural_config.get_or("signal_threshold", DefaultConfig::neural_signal_threshold());
    
    let neural_strategy = NeuralMarketMicrostructureStrategy::new(
        neural_short_window,
        neural_medium_window,
        neural_long_window,
        neural_micro_window,
        neural_signal_threshold,
    );
    println!("   - Short window: {}", neural_short_window);
    println!("   - Medium window: {}", neural_medium_window);
    println!("   - Long window: {}", neural_long_window);
    println!("   - Micro window: {}", neural_micro_window);
    println!("   - Signal threshold: {:.1}%", neural_signal_threshold * 100.0);
    println!("   - Info: {}", neural_strategy.get_info());
    println!();

    // Example 5: Show how to switch between strategies
    println!("🔄 Example 5: Strategy Switching");
    println!("To switch strategies, uncomment the desired strategy in main.rs:");
    println!("   - RSI Strategy: Uncomment lines 42-49");
    println!("   - HFT Ultra-Fast: Currently active (lines 58-65)");
    println!("   - Adaptive Multi-Factor: Uncomment lines 95-102");
    println!("   - Neural Market Microstructure: Uncomment lines 104-111");
    println!("   - All strategies use the same config file!");
    println!();

    // Example 6: Show configuration sections
    println!("📋 Example 6: Available Configuration Sections");
    println!("The config file contains sections for:");
    println!("   - [rsi_strategy]");
    println!("   - [mean_reversion_strategy]");
    println!("   - [momentum_scalping_strategy]");
    println!("   - [kalman_filter_strategy]");
    println!("   - [order_book_imbalance_strategy]");
    println!("   - [spline_strategy]");
    println!("   - [vwap_deviation_strategy]");
    println!("   - [zscore_strategy]");
    println!("   - [fractal_approximation_strategy]");
    println!("   - [hft_ultra_fast]");
    println!("   - [hft_market_maker]");
    println!("   - [adaptive_multi_factor]");
    println!("   - [neural_market_microstructure]");
    println!("   - [risk_management]");
    println!("   - [performance_monitoring]");
    println!("   - [scenarios]");
    println!();

    println!("✅ All strategies now use centralized configuration!");
    println!("📁 Config file: ../../config/advanced_strategies_config.toml");

    Ok(())
} 