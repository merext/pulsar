use ::trade::trader::TradeMode;
use ::trade::trader::Trader;
use clap::{Parser, Subcommand};
use std::env;
use std::error::Error;
use tracing::info;
use strategies::strategy::Strategy;

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
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    // Instantiate the strategy with required state
    // Only one strategy should be uncommented at a time.

    // RSI Strategy
    // use strategies::rsi_strategy::RsiStrategy;
    // let period = 14;
    // let overbought = 70.0;
    // let oversold = 30.0;
    // let strategy = RsiStrategy::new(period, overbought, oversold);

    // Kalman Filter Strategy
    // use strategies::kalman_filter_strategy::KalmanFilterStrategy;
    // let signal_threshold = 0.00001; // Example threshold, needs tuning
    // let strategy = KalmanFilterStrategy::new(signal_threshold);

    // Mean Reversion Strategy
    // use strategies::mean_reversion::MeanReversionStrategy;
    // let window_size = 20; // Example window size
    // let max_trade_window = 10; // Example max trade window
    // let strategy = MeanReversionStrategy::new(window_size, max_trade_window);

    // Momentum Scalping Strategy
    // use strategies::momentum_scalping::MomentumScalping;
    // let trade_window_size = 5; // Example window size
    // let price_change_threshold = 0.00001; // Example threshold
    // let strategy = MomentumScalping::new(trade_window_size, price_change_threshold);

    // HFT Ultra-Fast Strategy
    use strategies::hft_ultra_fast_strategy::HftUltraFastStrategy;
    let strategy = HftUltraFastStrategy::new();

    info!("Using strategy: {}", strategy.get_info());

    // Order Book Imbalance Strategy
    // use strategies::order_book_imbalance::OrderBookImbalance;
    // let period = 10; // Example period (number of trades)
    // let buy_threshold = 0.00001; // Example buy threshold
    // let sell_threshold = -0.00001; // Example sell threshold
    // let strategy = OrderBookImbalance::new(period, buy_threshold, sell_threshold);

    // Spline Strategy
    // use strategies::spline_strategy::{Interpolation, SplineStrategy};
    // let window_size = 5; // Example window size (reduced for testing)
    // let interpolation = Interpolation::Linear; // Example interpolation type
    // let derivative_buy_threshold = 0.000001; // Example threshold, needs tuning
    // let derivative_sell_threshold = -0.000001; // Example threshold, needs tuning
    // let strategy = SplineStrategy::new(
    //     window_size,
    //     interpolation,
    //     derivative_buy_threshold,
    //     derivative_sell_threshold,
    // );

    // VWAP Deviation Strategy (Placeholder - requires re-implementation)
    // use strategies::vwap_deviation_strategy::VwapDeviationStrategy;
    // let period = 10; // Example period
    // let deviation_threshold = 0.00001; // Example threshold
    // let strategy = VwapDeviationStrategy::new(period, deviation_threshold);

    // Z-Score Strategy (Currently active)
    // use strategies::zscore_strategy::ZScoreStrategy;
    // let period = 50; // Example period
    // let buy_threshold = -0.00001; // Example buy threshold
    // let sell_threshold = 0.00001; // Example sell threshold
    // let strategy = ZScoreStrategy::new(period, buy_threshold, sell_threshold);

    // HFT Strategies (Ultra-low latency optimized)
    // HFT Ultra-Fast Strategy (Currently active)
    // use strategies::hft_ultra_fast_strategy::HftUltraFastStrategy;
    // let strategy = HftUltraFastStrategy::new();

    // HFT Market Maker Strategy
    // use strategies::hft_market_maker_strategy::HftMarketMakerStrategy;
    // let strategy = HftMarketMakerStrategy::new();

    // Advanced Strategies (Medium-term trading)
    // Adaptive Multi-Factor Strategy
    // use strategies::adaptive_multi_factor_strategy::AdaptiveMultiFactorStrategy;
    // let short_window = 10;
    // let long_window = 50;
    // let volatility_window = 20;
    // let volume_window = 30;
    // let strategy = AdaptiveMultiFactorStrategy::new(short_window, long_window, volatility_window, volume_window);

    // Neural Market Microstructure Strategy
    // use strategies::neural_market_microstructure_strategy::NeuralMarketMicrostructureStrategy;
    // let short_window = 5;
    // let medium_window = 20;
    // let long_window = 100;
    // let micro_window = 10;
    // let strategy = NeuralMarketMicrostructureStrategy::new(short_window, medium_window, long_window, micro_window);

    // Initialize trader with API credentials
    let trading_symbol = "DOGEUSDT";
    // IMPORTANT: Replace with your actual Binance API Key and Secret
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set in the environment");
    let api_secret =
        env::var("BINANCE_API_SECRET").expect("API_SECRET must be set in the environment");

    // HFT-specific configuration
    info!("HFT Ultra-Fast Strategy Configuration:");
    info!("  - Latency target: < 1 microsecond");
    info!("  - Buffer size: 64 ticks");
    info!("  - Fast EMA alpha: 0.1 (10-period equivalent)");
    info!("  - Slow EMA alpha: 0.05 (20-period equivalent)");
    info!("  - Buy threshold: 0.05% price change");
    info!("  - Stop loss: 0.2%");
    info!("  - Take profit: 0.2%");
    info!("  - Config file: ../../config/advanced_strategies_config.toml");

    let trade_mode = match cli.command {
        Commands::Trade => TradeMode::Real,
        Commands::Emulate => TradeMode::Emulated,
        Commands::Backtest { .. } => TradeMode::Emulated,
    };

    let mut binance_trader = binance_exchange::trader::BinanceTrader::new(
        &trading_symbol,
        &api_key,
        &api_secret,
        trade_mode,
    )
    .await;
    binance_trader.account_status().await?;

    match cli.command {
        Commands::Trade => {
            info!("Starting live trading...");
            trade::run_trade(
                &trading_symbol,
                &api_key,
                &api_secret,
                strategy,
                &mut binance_trader,
                TradeMode::Real,
            )
            .await?;
        }
        Commands::Emulate => {
            info!("Starting emulation trading...");
            trade::run_trade(
                &trading_symbol,
                &api_key,
                &api_secret,
                strategy,
                &mut binance_trader,
                TradeMode::Emulated,
            )
            .await?;
        }
        Commands::Backtest { url, path } => {
            if let Some(url) = url {
                info!("Starting backtest with URL: {}", url);
                backtest::run_backtest(&url, strategy.clone(), &mut binance_trader).await?;
            } else if let Some(path) = path {
                info!("Starting backtest with path: {}", path);
                backtest::run_backtest(&path, strategy.clone(), &mut binance_trader).await?;
            } else {
                return Err("Either --url or --path must be provided for backtest".into());
            }
        }
    }

    Ok(())
}
