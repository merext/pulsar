
use ::trade::trader::{TradeMode, Trader};
use binance_exchange::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use clap::{Parser, Subcommand};
use std::error::Error;
use std::fs;
use strategies::strategy::Strategy;
use toml::Value;
use tracing::info;

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
    let trading_config =
        load_config("config/trading_config.toml").expect("Failed to load trading configuration");

    let trading_symbol: String = get_config_value(&trading_config, "position_sizing.trading_symbol")
        .ok_or("Trading symbol not defined in configuration")?;


    let mut strategy = strategies::MeanReversionHftStrategy::from_file("config/mean_reversion_hft_strategy.toml")
        .expect("Failed to load MeanReversionHftStrategy configuration");
        
    info!("Trading strategy: {}", strategy.get_info());

    // Create trader once for all modes
    let mut binance_trader = BinanceTrader::new().await?;

    match cli.command {
        Commands::Trade => {
            info!("Starting live trading for {}...", trading_symbol);

            // Get live WebSocket stream
            let binance_client = BinanceClient::new().await?;
            let trading_stream = binance_client.trade_stream(&trading_symbol).await?;

            binance_trader
                .trade(
                    trading_stream,
                    &mut strategy,
                    &trading_symbol,
                    TradeMode::Real,
                )
                .await?;
        }
        Commands::Emulate => {
            info!(
                "Starting live emulation with real WebSocket stream for {}...",
                trading_symbol
            );

            // Get live WebSocket stream
            let binance_client = BinanceClient::new().await?;
            let trading_stream = binance_client.trade_stream(&trading_symbol).await?;

            binance_trader
                .trade(
                    trading_stream,
                    &mut strategy,
                    &trading_symbol,
                    TradeMode::Emulated,
                )
                .await?;
        }
        Commands::Backtest { uri } => {
            info!("Starting backtest with data from: {}", uri);

            // Get historical data stream
            let trading_stream = BinanceClient::trade_data_from_uri(&uri).await?;

            binance_trader
                .trade(
                    trading_stream,
                    &mut strategy,
                    &trading_symbol,
                    TradeMode::Backtest,
                )
                .await?;
        }
    }

    Ok(())
}
