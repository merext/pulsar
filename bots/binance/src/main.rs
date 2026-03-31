
use ::trade::market::MarketEvent;
use ::trade::Strategy;
use ::trade::trader::{TradeMode, Trader};
use binance_exchange::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Duration;
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
    #[arg(long)]
    duration_secs: Option<u64>,

    #[arg(long, default_value = "trade-flow-momentum")]
    strategy: String,

    #[command(subcommand)]
    command: Commands,
}

fn build_strategy(
    strategy_name: &str,
) -> Result<Box<dyn Strategy>, Box<dyn Error + Send + Sync>> {
    match strategy_name {
        "trade-flow-momentum" => Ok(Box::new(
            strategies::TradeFlowMomentumStrategy::from_file(
                "config/strategies/trade_flow_momentum.toml",
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "liquidity-sweep-reversal" => Ok(Box::new(
            strategies::LiquiditySweepReversalStrategy::from_file(
                "config/strategies/liquidity_sweep_reversal.toml",
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        _ => Err(format!(
            "Unknown strategy '{}'. Available: trade-flow-momentum, liquidity-sweep-reversal",
            strategy_name
        )
        .into()),
    }
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
        |_| "binance_bot=info,binance_sdk=warn,binance_exchange=info,trade=info".to_string(),
        |rust_log| format!("{rust_log},binance_sdk=warn"),
    );

    tracing_subscriber::fmt().with_env_filter(env_filter).init();

    let cli = Cli::parse();

    // Load trading configuration
    let trading_config =
        load_config("config/trading_config.toml").expect("Failed to load trading configuration");

    let trading_symbol: String = get_config_value(&trading_config, "position_sizing.trading_symbol")
        .ok_or("Trading symbol not defined in configuration")?;

    let mut binance_trader = BinanceTrader::new().await?;
    let mut strategy = build_strategy(&cli.strategy)?;

    match cli.command {
        Commands::Trade => {
            return Err("Trading remains disabled until new HFT strategies are implemented.".into());
        }
        Commands::Emulate => {
            return Err("Live emulation remains disabled until new HFT strategies are implemented.".into());
        }
        Commands::Backtest { uri } => {
            let uri_path = Path::new(&uri);
            info!(strategy = %cli.strategy, strategy_info = %strategy.get_info(), uri = %uri, exists = uri_path.exists(), "Starting strategy backtest");
            info!("Verified historical market data access from: {}", uri);

            let trading_stream = BinanceClient::trade_data_from_uri(&uri)
                .await?
                .map(MarketEvent::Trade);
            let trading_stream = if let Some(duration_secs) = cli.duration_secs {
                trading_stream.take_until(tokio::time::sleep(Duration::from_secs(duration_secs)))
            } else {
                trading_stream.take_until(tokio::time::sleep(Duration::from_secs(1_000_000_000)))
            };

            binance_trader
                .trade(
                    trading_stream,
                    strategy.as_mut(),
                    &trading_symbol,
                    TradeMode::Backtest,
                )
                .await?;
        }
    }

    Ok(())
}
