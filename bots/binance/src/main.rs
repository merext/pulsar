use std::error::Error;
use std::fs;
use std::env;
use toml::Value;
use clap::{Parser, Subcommand};
use strategies::StochasticHftStrategy;
use strategies::strategy::Strategy;
use tracing::info;
use ::trade::trader::TradeMode;
use binance_exchange::trader::BinanceTrader;





#[allow(clippy::too_many_lines)]


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
        uri: String,
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
    
    // Create strategy and trader once
    let mut strategy = Box::new(
        StochasticHftStrategy::from_file("config/stochastic_hft_strategy.toml")
            .expect("Failed to load StochasticHftStrategy configuration")
    );
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");
    
    info!("Trading strategy: {}", strategy.get_info());

    // Create trader once for all modes
    let mut binance_trader = BinanceTrader::new(
        &api_key,
        &api_secret
    ).await?;

    match cli.command {
        Commands::Trade => {
            info!("Starting live trading for {}...", trading_symbol);
            
            binance_trader.run_trading_loop(
                &mut *strategy,
                &trading_symbol,
                TradeMode::Real,
                ""
            ).await?;
        }
        Commands::Emulate => {
            info!("Starting live emulation with real WebSocket stream for {}...", trading_symbol);
            
            binance_trader.run_trading_loop(
                &mut *strategy,
                &trading_symbol,
                TradeMode::Emulated,
                ""
            ).await?;
        }
        Commands::Backtest { uri } => {
            info!("Starting backtest with data from: {}", uri);
            
            binance_trader.run_trading_loop(
                &mut *strategy,
                &trading_symbol,
                TradeMode::Backtest,
                &uri
            ).await?;
        }
    }

    Ok(())
}
