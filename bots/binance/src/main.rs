use ::trade::trader::TradeMode;
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
    // Configure logging with specific levels for different modules
    let env_filter = if let Ok(rust_log) = std::env::var("RUST_LOG") {
        // If RUST_LOG is set, use it but force binance_sdk to warn level
        format!("{},binance_sdk=warn", rust_log)
    } else {
        // Default configuration
        "binance_bot=info,binance_sdk=warn,binance_exchange=info".to_string()
    };
    
    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .init();
    
    let cli = Cli::parse();
    
    use strategies::quantum_hft_strategy::QuantumHftStrategy;
    let strategy = QuantumHftStrategy::new();
    
    let trading_symbol = "DOGEUSDT";
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");
    
    info!("Trading strategy: {}", strategy.get_info());
    
    match cli.command {
        Commands::Trade => {
            info!("Starting live trading...");
            let mut binance_trader = binance_exchange::trader::BinanceTrader::new(
                trading_symbol,
                &api_key,
                &api_secret,
                TradeMode::Real,
            ).await?;
            
            trade::run_trade(
                trading_symbol,
                &api_key,
                &api_secret,
                strategy,
                &mut binance_trader,
                TradeMode::Real,
            ).await?;
        }
        Commands::Emulate => {
            info!("Starting emulated trading...");
            let mut binance_trader = binance_exchange::trader::BinanceTrader::new(
                trading_symbol,
                &api_key,
                &api_secret,
                TradeMode::Emulated,
            ).await?;
            
            trade::run_trade(
                trading_symbol,
                &api_key,
                &api_secret,
                strategy,
                &mut binance_trader,
                TradeMode::Emulated,
            ).await?;
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
