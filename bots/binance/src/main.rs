use std::error::Error;
use std::fs;
use std::env;
use toml::Value;
use clap::{Parser, Subcommand};
use strategies::StochasticHftStrategy;
use strategies::strategy::Strategy;
use tracing::info;
use ::trade::trader::TradeMode;
use binance_exchange::BinanceClient;


mod backtest;
mod trade;



fn load_config<P: AsRef<std::path::Path>>(config_path: P) -> Result<Value, Box<dyn Error>> {
    let content = fs::read_to_string(config_path)?;
    let config = content.parse::<Value>()?;
    Ok(config)
}

fn get_config_value<T: ConfigValue>(config: &Value, key: &str) -> Option<T> {
    T::from_config_value(config, key)
}

trait ConfigValue: Sized {
    fn from_config_value(config: &Value, key: &str) -> Option<Self>;
}

impl ConfigValue for f64 {
    fn from_config_value(config: &Value, key: &str) -> Option<Self> {
        let keys: Vec<&str> = key.split('.').collect();
        let mut current = config;
        
        for key in keys {
            current = current.get(key)?;
        }
        
        current.as_float()
    }
}

impl ConfigValue for String {
    fn from_config_value(config: &Value, key: &str) -> Option<Self> {
        let keys: Vec<&str> = key.split('.').collect();
        let mut current = config;
        
        for key in keys {
            current = current.get(key)?;
        }
        
        current.as_str().map(std::string::ToString::to_string)
    }
}

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
    let env_filter = std::env::var("RUST_LOG").map_or_else(
        |_| "binance_bot=info,binance_sdk=warn,binance_exchange=info".to_string(),
        |rust_log| format!("{rust_log},binance_sdk=warn"),
    );

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let cli = Cli::parse();

    // Load trading configuration
    let trading_config = load_config("config/trading_config.toml")
        .expect("Failed to load trading configuration");
    
    let trading_symbol = get_config_value(&trading_config, "position_sizing.trading_symbol")
        .unwrap_or_else(|| "DOGEUSDT".to_string());
    
    // Create StochasticHftStrategy strategy instance
    let strategy = StochasticHftStrategy::new();
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");

    info!("Trading strategy: {}", strategy.get_info());

    match cli.command {
        Commands::Trade => {
            let binance_trader = binance_exchange::trader::BinanceTrader::new(
                &trading_symbol,
                &api_key,
                &api_secret,
                TradeMode::Real,
            ).await?;
            
            info!("Starting live trading for {}...", trading_symbol);
            
            // Create live WebSocket trade stream for real trading
            let binance_client = BinanceClient::new().await?;
            let live_trade_stream = binance_client.trade_stream(&trading_symbol).await?;
            
            // Run live trading with real-time WebSocket stream
            trade::run_trade_loop(Box::new(strategy), binance_trader, live_trade_stream, TradeMode::Real).await?;
        }
        Commands::Emulate => {
            let binance_trader = binance_exchange::trader::BinanceTrader::new(
                &trading_symbol,
                &api_key,
                &api_secret,
                TradeMode::Emulated,
            ).await?;
            
            info!("Starting LIVE EMULATION with real WebSocket stream for {}...", trading_symbol);
            
            // Create live WebSocket trade stream for continuous emulation
            let binance_client = BinanceClient::new().await?;
            let live_trade_stream = binance_client.trade_stream(&trading_symbol).await?;
            
            // Run live emulation with real-time WebSocket stream
            trade::run_trade_loop(Box::new(strategy), binance_trader, live_trade_stream, TradeMode::Emulated).await?;
        }
        Commands::Backtest { path, url } => {
            if let Some(data_path) = path {
                info!("Starting backtest with data from: {}", data_path);
                backtest::run_backtest(&data_path, strategy, &trading_symbol, "config/trading_config.toml")
                .await?;
            } else if let Some(ws_url) = url {
                info!("Starting backtest with WebSocket data from: {}", ws_url);
                backtest::run_backtest(&ws_url, strategy, &trading_symbol, "config/trading_config.toml")
                .await?;
            } else {
                return Err("No data source specified for backtest".into());
            }
        }
    }

    Ok(())
}
