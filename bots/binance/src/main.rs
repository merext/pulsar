use ::trade::trader::TradeMode;
use clap::{Parser, Subcommand};
use std::env;
use std::error::Error;
use strategies::config::StrategyConfig;
use strategies::SentimentAnalysisStrategy;
use strategies::strategy::Strategy;
use tracing::info;

mod backtest;
mod trade;

use trade::TradeConfig;

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
    // Configure logging with specific levels for different modules
    let env_filter = if let Ok(rust_log) = std::env::var("RUST_LOG") {
        // If RUST_LOG is set, use it but force binance_sdk to warn level
        format!("{rust_log},binance_sdk=warn")
    } else {
        // Default configuration
        "binance_bot=info,binance_sdk=warn,binance_exchange=info".to_string()
    };

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let cli = Cli::parse();

    // Create Sentiment Analysis strategy instance
    let strategy = SentimentAnalysisStrategy::new();
    
    // Load trading configuration
    let trading_config =
        StrategyConfig::load_trading_config().expect("Failed to load trading configuration");
    let position_sizing = trading_config
        .section("position_sizing")
        .expect("Position sizing configuration not found");

    let trading_symbol = position_sizing.get_or("default_trading_symbol", "DOGEUSDT".to_string());
    let trading_size_min = position_sizing.get_or("trading_size_min", 10.0);
    let trading_size_max = position_sizing.get_or("trading_size_max", 50.0);
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");

    info!("Trading strategy: {}", strategy.get_info());

    match cli.command {
        Commands::Trade => {
            info!("Starting live trading...");
            let mut binance_trader = binance_exchange::trader::BinanceTrader::new(
                &trading_symbol,
                &api_key,
                &api_secret,
                TradeMode::Real,
            )
            .await?;

            let config = TradeConfig {
                trading_symbol: trading_symbol.clone(),
                trade_mode: TradeMode::Real,
                trading_size_min,
                trading_size_max,
            };

            trade::run_trade(config, &api_key, &api_secret, strategy, &mut binance_trader).await?;
        }
        Commands::Emulate => {
            info!("Starting emulated trading...");
            let mut binance_trader = binance_exchange::trader::BinanceTrader::new(
                &trading_symbol,
                &api_key,
                &api_secret,
                TradeMode::Emulated,
            )
            .await?;

            let config = TradeConfig {
                trading_symbol: trading_symbol.clone(),
                trade_mode: TradeMode::Emulated,
                trading_size_min,
                trading_size_max,
            };

            trade::run_trade(config, &api_key, &api_secret, strategy, &mut binance_trader).await?;
        }
        Commands::Backtest { path, url } => {
            if let Some(data_path) = path {
                info!("Starting backtest with data from: {}", data_path);
                backtest::run_backtest(
                    &data_path,
                    strategy,
                    &trading_symbol,
                    trading_size_min,
                    trading_size_max,
                )
                .await?;
            } else if let Some(ws_url) = url {
                info!("Starting backtest with WebSocket data from: {}", ws_url);
                backtest::run_backtest(
                    &ws_url,
                    strategy,
                    &trading_symbol,
                    trading_size_min,
                    trading_size_max,
                )
                .await?;
            } else {
                return Err("No data source specified for backtest".into());
            }
        }
    }

    Ok(())
}
