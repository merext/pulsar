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
    
    use strategies::rsi_strategy::RsiStrategy;
    let strategy = RsiStrategy::new();
    
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
                backtest::run_backtest(strategy, data_path).await?;
            } else if let Some(ws_url) = url {
                info!("Starting backtest with WebSocket data from: {}", ws_url);
                backtest::run_backtest(strategy, ws_url).await?;
            } else {
                return Err("No data source specified for backtest".into());
            }
        }
    }
    
    Ok(())
}
