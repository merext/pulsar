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
use ::trade::Trade;

use binance_exchange::trader::BinanceTrader;
use trade::signal::Signal;
use trade::trader::Trader;
use trade::trading_engine::TradingConfig;
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

    // Create TradingEngine for emulated and backtest modes with realistic calculations
    let mut trading_engine = if matches!(trade_mode, TradeMode::Emulated | TradeMode::Backtest) {
        match TradingConfig::from_file("config/trading_config.toml") {
            Ok(config) => {
                match trade::trading_engine::TradingEngine::new_with_config("DOGEUSDT", config) {
                    Ok(engine) => Some(engine),
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let mut trade_stream = trade_data;
    let mut trade_count = 0;

    while let Some(trade) = trade_stream.next().await {
        trade_count += 1;
        
        // Debug logging for trade processing
        if trade_count % 1000 == 0 {
            info!("Processing trade {}: price={:.8}, time={}", trade_count, trade.price, trade.event_time);
        }
        
        // Update strategy with trade data
        strategy.on_trade(trade.clone().into()).await;
        
        // Debug logging for strategy calls
        if trade_count % 1000 == 0 {
            info!("Strategy on_trade called for trade {}", trade_count);
        }

        let trade_price = trade.price;
        let trade_time = trade.event_time as f64;

        // Get signal from strategy
        let (final_signal, confidence) = 
            strategy.get_signal(trade_price, trade_time, current_position.clone());
        
        // Debug logging for signal generation
        if trade_count % 1000 == 0 {
            info!("Trade {}: signal={:?}, confidence={:.3}", trade_count, final_signal, confidence);
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

        match trade_mode {
            TradeMode::Real => {
                binance_trader
                    .on_signal(final_signal, trade_price, quantity_to_trade)
                    .await;
                
                // Log executed trades for live trading with PnL information
                if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    info!(
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

                // Use TradingEngine for realistic trade execution simulation
                let trade_result = if let Some(engine) = &mut trading_engine {
                    engine.on_emulate(final_signal, trade_price, quantity_to_trade).await
                } else {
                    // Fallback to BinanceTrader if TradingEngine not available
                    binance_trader.on_emulate(final_signal, trade_price, quantity_to_trade).await
                };

                // Calculate total execution time including simulated latency
                let total_execution_time = execution_start.elapsed();
                let computational_latency = total_execution_time.saturating_sub(network_latency);

                // Log executed trades with detailed execution metrics
                if matches!(final_signal, Signal::Buy | Signal::Sell) {
                    if let Some((fill_price, fees, slippage, rebates, order_type)) = trade_result {
                        info!(
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
                            unrealized_pnl = format!("{:.6}", if let Some(engine) = &trading_engine { engine.unrealized_pnl(trade_price) } else { 0.0 }),
                            realized_pnl = format!("{:.6}", if let Some(engine) = &trading_engine { engine.realized_pnl() } else { 0.0 })
                        );
                    } else {
                        // Fallback logging if trade_result is None
                        info!(
                            signal = %final_signal,
                            confidence = %format!("{:.2}", confidence),
                            price = format!("{:.8}", trade_price),
                            quantity = format!("{:.2}", quantity_to_trade),
                            computational_latency = format!("{:?}", computational_latency),
                            network_latency = format!("{:?}", network_latency),
                            total_latency = format!("{:?}", total_execution_time),
                            position = %current_position,
                            unrealized_pnl = format!("{:.6}", if let Some(engine) = &trading_engine { engine.unrealized_pnl(trade_price) } else { 0.0 }),
                            realized_pnl = format!("{:.6}", if let Some(engine) = &trading_engine { engine.realized_pnl() } else { 0.0 })
                        );
                    }
                }
            }
        }

        // Update current position for display purposes
        match trade_mode {
            TradeMode::Real => {
                current_position = binance_trader.position();
            }
            TradeMode::Emulated | TradeMode::Backtest => {
                if let Some(engine) = &mut trading_engine {
                    current_position = engine.position();
                }
            }
        }
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
            
            // Use TradingEngine PnL if available, otherwise fallback to BinanceTrader
            if let Some(engine) = &trading_engine {
                info!("Final PnL: {:.6}", engine.realized_pnl());
                // For unrealized PnL, we need a current price - use the last known price or 0
                let last_price = if current_position.entry_price > 0.0 { current_position.entry_price } else { 0.0 };
                info!("Final Unrealized PnL: {:.6}", engine.unrealized_pnl(last_price));
            } else {
                info!("Final PnL: {:.6}", binance_trader.realized_pnl());
            }
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

fn create_strategy(_strategy_name: &str) -> Box<dyn Strategy> {
    Box::new(StochasticHftStrategy::new())
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
    
    // Create strategy and trader once
    let strategy = Box::new(StochasticHftStrategy::new());
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set");
    let api_secret = env::var("BINANCE_API_SECRET").expect("API_SECRET must be set");
    
    info!("Trading strategy: {}", strategy.get_info());

    match cli.command {
        Commands::Trade => {
            info!("Starting LIVE TRADING for {}...", trading_symbol);
            
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
            info!("Starting LIVE EMULATION with real WebSocket stream for {}...", trading_symbol);
            
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
        Commands::Backtest { path, url } => {
            if let Some(data_path) = path {
                info!("Starting BACKTEST with historical data from: {}", data_path);
                
                // Create historical data stream for backtesting
                let historical_trade_stream = BinanceClient::trade_data_from_path(&data_path).await?;
                
                // Run trading loop directly
                run_trade_loop(
                    strategy, 
                    &trading_symbol, 
                    TradeMode::Backtest.clone(), 
                    historical_trade_stream,
                    &api_key,
                &api_secret
                ).await?;
            } else if let Some(ws_url) = url {
                info!("Starting BACKTEST with WebSocket data from: {}", ws_url);
                
                // Create WebSocket data stream for backtesting
                let ws_trade_stream = BinanceClient::trade_data(&ws_url).await?;
                
                // Run trading loop directly
                run_trade_loop(
                    strategy, 
                    &trading_symbol, 
                    TradeMode::Backtest.clone(), 
                    ws_trade_stream,
                    &api_key,
                    &api_secret
                ).await?;
            } else {
                return Err("No data source specified for backtest".into());
            }
        }
    }

    Ok(())
}
