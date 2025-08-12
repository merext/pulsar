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


use binance_exchange::trader::BinanceTrader;
use trade::signal::Signal;
use trade::trader::Trader;
use std::time::Instant;
use futures_util::StreamExt;



#[allow(clippy::too_many_lines)]
async fn run_trade_loop(
    mut strategy: Box<dyn Strategy>,
    trading_symbol: &str,
    trade_mode: TradeMode,
    trade_data: impl futures_util::Stream<Item = trade::models::Trade> + Unpin,
    api_key: &str,
    api_secret: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Create trader based on mode
    let mut binance_trader = BinanceTrader::new(
        trading_symbol,
        api_key,
        api_secret,
        trade_mode,
    ).await?;
    let mut current_position = trade::trader::Position {
        symbol: "DOGEUSDT".to_string(),
        quantity: 0.0,
        entry_price: 0.0,
    };



    let mut trade_stream = trade_data;
    let mut trade_count = 0;

    while let Some(trade) = trade_stream.next().await {
        trade_count += 1;
        
        // Debug logging for trade processing
        if trade_count % 1000 == 0 {
            tracing::debug!("Processing trade {}: price={:.8}, time={}", trade_count, trade.price, trade.event_time);
        }
        
        // Update strategy with trade data
        strategy.on_trade(trade.clone().into()).await;
        
        // Debug logging for strategy calls
        if trade_count % 1000 == 0 {
            tracing::debug!("Strategy on_trade called for trade {}", trade_count);
        }

        let trade_price = trade.price;
        let trade_time = trade.event_time as f64;

        // Get signal from strategy
        let (final_signal, confidence) = 
            strategy.get_signal(trade_price, trade_time, current_position.clone());
        
        // Debug logging for signal generation
        if trade_count % 1000 == 0 {
            tracing::debug!("Trade {}: signal={:?}, confidence={:.3}", trade_count, final_signal, confidence);
        }

        // Debug logging to see what signals are being generated
        tracing::debug!(
            "Trade {}: Price: {:.8}, Signal: {:?}, Confidence: {:.3}",
            trade.trade_id, trade_price, final_signal, confidence
        );

        // Calculate position size based on confidence and trading config
        let quantity_to_trade = binance_trader.calculate_trade_size(
            &current_position.symbol,
            trade_price,
            confidence,
            binance_trader.config.position_sizing.trading_size_min,
            binance_trader.config.position_sizing.trading_size_max,
            1.0, // trading_size_step - use 1.0 for DOGEUSDT
        );

        // Only log HOLD signals at debug level (no execution)
        if matches!(final_signal, Signal::Hold) {
            tracing::debug!(
                signal = %final_signal,
                confidence = %format!("{:.2}", confidence),
                position = %current_position
            );
        }

        // Log when there's insufficient data for trading (only at INFO level)
        if matches!(final_signal, Signal::Hold) && confidence < 0.1 {
            info!("Insufficient data for trading - confidence: {:.3}, waiting for more market data", confidence);
        }

        match trade_mode {
            TradeMode::Real => {
                binance_trader
                    .on_signal(final_signal, trade_price, quantity_to_trade)
                    .await;
                
                // Log executed trades for live trading with PnL information
                if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    tracing::debug!(
                        signal = %final_signal,
                        confidence = %format!("{:.2}", confidence),
                        price = format!("{:.8}", trade_price),
                        quantity = format!("{:.2}", quantity_to_trade),
                        position = %current_position,
                        unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                        realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                    );
                }
            }
            TradeMode::Emulated | TradeMode::Backtest => {
                // Record execution start time for latency measurement
                let execution_start = Instant::now();
                
                // Simulate realistic network latency (1-5ms typical for HFT)
                let network_latency = if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    // Simulate variable latency based on market conditions
                    let base_latency = 1.0; // 1ms base
                    let volatility_factor = (trade_price - current_position.entry_price).abs() / trade_price;
                    let latency_variance = (volatility_factor * 4.0).min(4.0); // 0-4ms additional
                    std::time::Duration::from_millis((base_latency + latency_variance) as u64)
                } else {
                    std::time::Duration::from_millis(0)
                };

                // Simulate network delay
                if !network_latency.is_zero() {
                    tokio::time::sleep(network_latency).await;
                }

                // Use trader for trade execution simulation
                let trade_result = binance_trader.on_emulate(final_signal, trade_price, quantity_to_trade).await;

                // Calculate total execution time including simulated latency
                let total_execution_time = execution_start.elapsed();
                let computational_latency = total_execution_time.saturating_sub(network_latency);

                // Log executed trades with detailed execution metrics
                if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    if let Some((fill_price, fees, slippage, rebates, order_type)) = trade_result {
                        tracing::debug!(
                            signal = %final_signal,
                            confidence = %format!("{:.2}", confidence),
                            price = format!("{:.8}", trade_price),
                            fill_price = format!("{:.8}", fill_price),
                            quantity = format!("{:.2}", quantity_to_trade),
                            order_type = ?order_type,
                            fees = format!("{:.8}", fees),
                            slippage = format!("{:.8}", slippage),
                            rebates = format!("{:.8}", rebates),
                            computational_latency = format!("{:?}", computational_latency),
                            network_latency = format!("{:?}", network_latency),
                            total_latency = format!("{:?}", total_execution_time),
                            position = %current_position,
                            unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                            realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                        );
                    } else {
                        // Fallback logging if trade_result is None
                        tracing::debug!(
                            signal = %final_signal,
                            confidence = %format!("{:.2}", confidence),
                            price = format!("{:.8}", trade_price),
                            quantity = format!("{:.2}", quantity_to_trade),
                            computational_latency = format!("{:?}", computational_latency),
                            network_latency = format!("{:?}", network_latency),
                            total_latency = format!("{:?}", total_execution_time),
                            position = %current_position,
                            unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                            realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                        );
                    }
                }
            }
        }

        // Update current position for display purposes
        current_position = binance_trader.position();
    }

    // Log completion message
    match trade_mode {
        TradeMode::Real => {
            info!("Live trading completed - processed {} trades", trade_count);
            info!("Final position: {}", current_position);
            info!("Final PnL: {:.6}", binance_trader.realized_pnl());
            let last_price = if current_position.entry_price > 0.0 { current_position.entry_price } else { 0.0 };
            info!("Final Unrealized PnL: {:.6}", binance_trader.unrealized_pnl(last_price));
        }
        TradeMode::Emulated | TradeMode::Backtest => {
            let mode_name = if matches!(trade_mode, TradeMode::Emulated) { "Emulation" } else { "Backtest" };
            info!("{} completed - processed {} trades", mode_name, trade_count);
            info!("Final position: {}", current_position);
            info!("Final PnL: {:.6}", binance_trader.realized_pnl());
            // For unrealized PnL, we need a current price - use the last known price or 0
            let last_price = if current_position.entry_price > 0.0 { current_position.entry_price } else { 0.0 };
            info!("Final Unrealized PnL: {:.6}", binance_trader.unrealized_pnl(last_price));
        }
    }

    Ok(())
}

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
    let strategy = Box::new(
        StochasticHftStrategy::from_file("config/stochastic_hft_strategy.toml")
            .expect("Failed to load StochasticHftStrategy configuration")
    );
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");
    
    info!("Trading strategy: {}", strategy.get_info());

    match cli.command {
        Commands::Trade => {
            info!("Starting live trading for {}...", trading_symbol);
            
            // Create live WebSocket trade stream for real trading
            let binance_client = BinanceClient::new().await?;
            let live_trade_stream = binance_client.trade_stream(&trading_symbol).await?;
            
            // Run trading loop directly
            run_trade_loop(
                strategy, 
                &trading_symbol, 
                TradeMode::Real.clone(), 
                live_trade_stream,
                &api_key,
                &api_secret
            ).await?;
        }
        Commands::Emulate => {
            info!("Starting live emulation with real WebSocket stream for {}...", trading_symbol);
            
            // Create live WebSocket trade stream for continuous emulation
            let binance_client = BinanceClient::new().await?;
            let live_trade_stream = binance_client.trade_stream(&trading_symbol).await?;
            
            // Run trading loop directly
            run_trade_loop(
                strategy, 
                &trading_symbol, 
                TradeMode::Emulated.clone(), 
                live_trade_stream,
                &api_key,
                &api_secret
            ).await?;
        }
        Commands::Backtest { uri } => {
            info!("Starting backtest with data from: {}", uri);
            
            // Create historical data stream for backtesting from URI (local file or remote URL)
            // TODO: Implement BinanceClient::trade_data_from_uri() that auto-detects schema
            let historical_trade_stream = BinanceClient::trade_data(&uri).await?;
            
            // Run trading loop directly
            run_trade_loop(
                strategy, 
                &trading_symbol, 
                TradeMode::Backtest.clone(), 
                historical_trade_stream,
                &api_key,
                &api_secret
            ).await?;
        }
    }

    Ok(())
}
