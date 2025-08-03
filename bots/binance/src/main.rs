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
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    // Configuration is now loaded by each strategy individually

    // Instantiate the strategy with required state
    // Only one strategy should be uncommented at a time.

    // RSI Strategy
    // use strategies::rsi_strategy::RsiStrategy;
    // let strategy = RsiStrategy::new(); // Loads config from config/rsi_strategy.toml

    // Kalman Filter Strategy
    // use strategies::kalman_filter_strategy::KalmanFilterStrategy;
    // let strategy = KalmanFilterStrategy::new(); // Loads config from config/kalman_filter_strategy.toml

    // Mean Reversion Strategy
    // use strategies::mean_reversion::MeanReversionStrategy;
    // let strategy = MeanReversionStrategy::new(); // Loads config from config/mean_reversion_strategy.toml

    // Momentum Scalping Strategy
    // use strategies::momentum_scalping::MomentumScalping;
    // let strategy = MomentumScalping::new(); // Loads config from config/momentum_scalping_strategy.toml

    // Order Book Imbalance Strategy
    // use strategies::order_book_imbalance::OrderBookImbalance;
    // let strategy = OrderBookImbalance::new(); // Loads config from config/order_book_imbalance_strategy.toml

    // Spline Strategy
    // use strategies::spline_strategy::{Interpolation, SplineStrategy};
    // let strategy = SplineStrategy::new(); // Loads config from config/spline_strategy.toml

    // VWAP Deviation Strategy (Placeholder - requires re-implementation)
    // use strategies::vwap_deviation_strategy::VwapDeviationStrategy;
    // let strategy = VwapDeviationStrategy::new(); // Loads config from config/vwap_deviation_strategy.toml

    // Z-Score Strategy (Currently active)
    // use strategies::zscore_strategy::ZScoreStrategy;
    // let strategy = ZScoreStrategy::new(); // Loads config from config/zscore_strategy.toml

    // HFT Strategies (Ultra-low latency optimized)
    // HFT Ultra-Fast Strategy
    // use strategies::hft_ultra_fast_strategy::HftUltraFastStrategy;
    // let strategy = HftUltraFastStrategy::new();

    // Mean Reversion Strategy (Currently active - Optimizing)
    // use strategies::mean_reversion_strategy::MeanReversionStrategy;
    // let strategy = MeanReversionStrategy::new();

    // HFT Market Maker Strategy
    // use strategies::hft_market_maker_strategy::HftMarketMakerStrategy;
    // let strategy = HftMarketMakerStrategy::new(); // Loads config from config/hft_market_maker_strategy.toml

    // RSI Strategy
    // use strategies::rsi_strategy::RsiStrategy;
    // let strategy = RsiStrategy::new(); // Loads config from config/rsi_strategy.toml

    // Mean Reversion Strategy
    // use strategies::mean_reversion_strategy::MeanReversionStrategy;
    // let strategy = MeanReversionStrategy::new(); // Loads config from config/mean_reversion_strategy.toml

    // Momentum Scalping Strategy
    // use strategies::momentum_scalping_strategy::MomentumScalping;
    // let strategy = MomentumScalping::new();

    // Adaptive Multi-Factor Strategy
    // use strategies::adaptive_multi_factor_strategy::AdaptiveMultiFactorStrategy;
    // let strategy = AdaptiveMultiFactorStrategy::new();

    // Z-Score Strategy
    // use strategies::zscore_strategy::ZScoreStrategy;
    // let strategy = ZScoreStrategy::new();

    // Momentum Scalping Strategy
    // use strategies::momentum_scalping_strategy::MomentumScalping;
    // let strategy = MomentumScalping::new();

    // Mean Reversion Strategy
    use strategies::mean_reversion_strategy::MeanReversionStrategy;
    let strategy = MeanReversionStrategy::new();

    // Z-Score Strategy
    // use strategies::zscore_strategy::ZScoreStrategy;
    // let strategy = ZScoreStrategy::new();

    // Kalman Filter Strategy
    // use strategies::kalman_filter_strategy::KalmanFilterStrategy;
    // let strategy = KalmanFilterStrategy::new();

    // RSI Strategy
    // use strategies::rsi_strategy::RsiStrategy;
    // let strategy = RsiStrategy::new();

    // HFT Market Maker Strategy
    use strategies::hft_market_maker_strategy::HftMarketMakerStrategy;
    let strategy = HftMarketMakerStrategy::new();

    // Z-Score Strategy (Currently active - Optimizing)
    // use strategies::zscore_strategy::ZScoreStrategy;
    // let strategy = ZScoreStrategy::new();

    // Advanced Strategies (Medium-term trading)
    // Adaptive Multi-Factor Strategy
    // use strategies::adaptive_multi_factor_strategy::AdaptiveMultiFactorStrategy;
    // let strategy = AdaptiveMultiFactorStrategy::new(); // Loads config from config/adaptive_multi_factor_strategy.toml

    // Neural Market Microstructure Strategy
    // use strategies::neural_market_microstructure_strategy::NeuralMarketMicrostructureStrategy;
    // let strategy = NeuralMarketMicrostructureStrategy::new(); // Loads config from config/neural_market_microstructure_strategy.toml

    // RSI Strategy
    // use strategies::rsi_strategy::RsiStrategy;
    // let strategy = RsiStrategy::new();

    // Spline Strategy
    // use strategies::spline_strategy::SplineStrategy;
    // let strategy = SplineStrategy::new();

    // Order Book Imbalance Strategy
    // use strategies::order_book_imbalance_strategy::OrderBookImbalance;
    // let strategy = OrderBookImbalance::new();

    // Fractal Approximation Strategy
    // use strategies::fractal_approximation_strategy::FractalApproximationStrategy;
    // let strategy = FractalApproximationStrategy::new();

    // VWAP Deviation Strategy
    use strategies::vwap_deviation_strategy::VwapDeviationStrategy;
    let strategy = VwapDeviationStrategy::new();

    // Kalman Filter Strategy
    // use strategies::kalman_filter_strategy::KalmanFilterStrategy;
    // let strategy = KalmanFilterStrategy::new();

    // Adaptive Multi-Factor Strategy
    // use strategies::adaptive_multi_factor_strategy::AdaptiveMultiFactorStrategy;
    // let strategy = AdaptiveMultiFactorStrategy::new();

    // Neural Market Microstructure Strategy
    // use strategies::neural_market_microstructure_strategy::NeuralMarketMicrostructureStrategy;
    // let strategy = NeuralMarketMicrostructureStrategy::new();

    // Initialize trader with API credentials
    let trading_symbol = "DOGEUSDT";
    // IMPORTANT: Replace with your actual Binance API Key and Secret
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set in the environment");
    let api_secret =
        env::var("BINANCE_API_SECRET").expect("API_SECRET must be set in the environment");

    // HFT Market Maker configuration
    info!("HFT Market Maker Strategy Configuration:");
    info!("  - Latency target: < 1 microsecond");
    info!("  - Spread capture: Dynamic spread calculation");
    info!("  - Inventory management: Real-time position tracking");
    info!("  - Risk controls: Max inventory and loss limits");
    info!("  - Order book analysis: Bid/ask placement");
    info!("  - Config file: ../../config/hft_market_maker_strategy.toml");
    info!("  - Each strategy has its own config file");

    info!("Using strategy: {}", strategy.get_info());

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
