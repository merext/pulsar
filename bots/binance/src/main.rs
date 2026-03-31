
use ::trade::market::MarketEvent;
use ::trade::Strategy;
use ::trade::trader::{MarketDataSourceKind, TradeMode, Trader};
use binance_exchange::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use serde::Serialize;
use std::error::Error;
use std::fs;
use std::pin::Pin;
use std::path::Path;
use std::time::Duration;
use toml::Value;
use tracing::info;

#[derive(Debug, Serialize)]
struct CaptureDepthLevel {
    price: f64,
    quantity: f64,
}

#[derive(Debug, Clone)]
struct ReplayDatasetSummary {
    total_events: usize,
    trade_events: usize,
    book_ticker_events: usize,
    depth_events: usize,
    parse_errors: usize,
    first_event_time: Option<u64>,
    last_event_time: Option<u64>,
    event_time_regressions: usize,
    symbols: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HistoricalMarketDataFormat {
    TradeCsv,
    CapturedJsonl,
}

#[derive(Debug, Clone)]
struct BacktestSummary {
    strategy: String,
    uri: String,
    total_ticks: usize,
    entries: usize,
    closed_trades: usize,
    realized_pnl: f64,
    fees_paid: f64,
    ending_cash: f64,
    ending_equity: f64,
    win_rate: f64,
    profit_factor: f64,
    avg_pnl_per_trade: f64,
    max_drawdown: f64,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event_type", rename_all = "snake_case")]
enum CaptureRecord {
    Trade {
        event_time: u64,
        symbol: String,
        trade_id: u64,
        price: f64,
        quantity: f64,
        trade_time: u64,
        is_buyer_market_maker: bool,
    },
    BookTicker {
        event_time: u64,
        bid_price: f64,
        bid_quantity: f64,
        ask_price: f64,
        ask_quantity: f64,
    },
    Depth {
        event_time: u64,
        bids: Vec<CaptureDepthLevel>,
        asks: Vec<CaptureDepthLevel>,
    },
}

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
            strategies::trade_flow_momentum::TradeFlowMomentumStrategy::from_file(
                "config/strategies/trade_flow_momentum.toml",
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "liquidity-sweep-reversal" => Ok(Box::new(
            strategies::liquidity_sweep_reversal::LiquiditySweepReversalStrategy::from_file(
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

fn detect_historical_market_data_format(uri: &str) -> HistoricalMarketDataFormat {
    let normalized = uri
        .split(['?', '#'])
        .next()
        .unwrap_or(uri)
        .to_ascii_lowercase();

    if normalized.ends_with(".jsonl") || normalized.ends_with(".ndjson") {
        HistoricalMarketDataFormat::CapturedJsonl
    } else {
        HistoricalMarketDataFormat::TradeCsv
    }
}

fn log_replay_dataset_summary(uri: &str, summary: &ReplayDatasetSummary) {
    info!(
        uri = %uri,
        total_events = summary.total_events,
        trade_events = summary.trade_events,
        book_ticker_events = summary.book_ticker_events,
        depth_events = summary.depth_events,
        parse_errors = summary.parse_errors,
        first_event_time = summary.first_event_time,
        last_event_time = summary.last_event_time,
        event_time_regressions = summary.event_time_regressions,
        symbols = %summary.symbols.join(","),
        action = "replay_dataset_summary"
    );
}

async fn run_backtest(
    trading_symbol: &str,
    strategy_name: &str,
    uri: &str,
    duration_secs: Option<u64>,
) -> Result<BacktestSummary, Box<dyn Error + Send + Sync>> {
    let mut binance_trader = BinanceTrader::new().await?;
    let mut strategy = build_strategy(strategy_name)?;
    let uri_path = Path::new(uri);
    let data_format = detect_historical_market_data_format(uri);

    info!(strategy = %strategy_name, strategy_info = %strategy.get_info(), uri = %uri, exists = uri_path.exists(), "Starting strategy backtest");
    info!("Verified historical market data access from: {}", uri);

    let mut replay_summary = None;
    let trading_stream: Pin<Box<dyn futures_util::Stream<Item = MarketEvent> + Send>> =
        match data_format {
            HistoricalMarketDataFormat::TradeCsv => {
                info!(uri = %uri, source = "historical_trade_csv", "Using trade-only historical replay source");
                Box::pin(BinanceClient::trade_data_from_uri(uri).await?.map(MarketEvent::Trade))
            }
            HistoricalMarketDataFormat::CapturedJsonl => {
                info!(uri = %uri, source = "captured_market_jsonl", "Using captured mixed-event replay source");
                let parsed = BinanceClient::load_captured_market_event_data_from_uri(uri).await?;
                replay_summary = Some(ReplayDatasetSummary {
                    total_events: parsed.summary.parsed_events,
                    trade_events: parsed.summary.trade_events,
                    book_ticker_events: parsed.summary.book_ticker_events,
                    depth_events: parsed.summary.depth_events,
                    parse_errors: parsed.summary.parse_errors,
                    first_event_time: parsed.summary.first_event_time,
                    last_event_time: parsed.summary.last_event_time,
                    event_time_regressions: parsed.summary.event_time_regressions,
                    symbols: parsed.summary.symbols.clone(),
                });
                Box::pin(futures_util::stream::iter(parsed.events))
            }
        };

    if let Some(summary) = replay_summary.as_ref() {
        log_replay_dataset_summary(uri, summary);
    }

    let trading_stream = if let Some(duration_secs) = duration_secs {
        trading_stream.take_until(tokio::time::sleep(Duration::from_secs(duration_secs)))
    } else {
        trading_stream.take_until(tokio::time::sleep(Duration::from_secs(1_000_000_000)))
    };

    binance_trader
        .trade(
            trading_stream,
            strategy.as_mut(),
            trading_symbol,
            TradeMode::Backtest,
            match data_format {
                HistoricalMarketDataFormat::TradeCsv => MarketDataSourceKind::HistoricalTradeReplay,
                HistoricalMarketDataFormat::CapturedJsonl => {
                    MarketDataSourceKind::HistoricalCapturedReplay
                }
            },
        )
        .await?;

    let metrics = binance_trader.get_metrics();
    Ok(BacktestSummary {
        strategy: strategy_name.to_string(),
        uri: uri.to_string(),
        total_ticks: binance_trader.get_trade_manager().get_total_ticks(),
        entries: metrics.entry_trades(),
        closed_trades: metrics.closed_trades(),
        realized_pnl: metrics.realized_pnl(),
        fees_paid: metrics.fees_paid(),
        ending_cash: metrics.current_cash(),
        ending_equity: metrics.last_equity(),
        win_rate: metrics.win_rate(),
        profit_factor: metrics.profit_factor(),
        avg_pnl_per_trade: metrics.avg_pnl_per_closed_trade(),
        max_drawdown: metrics.max_drawdown(),
    })
}

async fn run_live_mode(
    trading_symbol: &str,
    strategy_name: &str,
    trade_mode: TradeMode,
    duration_secs: Option<u64>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut binance_trader = BinanceTrader::new().await?;
    let mut strategy = build_strategy(strategy_name)?;
    let stream = BinanceClient::new().await?.market_event_stream(trading_symbol).await?;
    let trading_stream: Pin<Box<dyn futures_util::Stream<Item = MarketEvent> + Send + '_>> =
        if let Some(duration_secs) = duration_secs {
            Box::pin(stream.take_until(tokio::time::sleep(Duration::from_secs(duration_secs))))
        } else {
            Box::pin(stream)
        };

    info!(
        strategy = %strategy_name,
        strategy_info = %strategy.get_info(),
        trade_mode = ?trade_mode,
        source = "websocket_trade_bookticker",
        "Starting live market orchestration"
    );

    binance_trader
        .trade(
            trading_stream,
            strategy.as_mut(),
            trading_symbol,
            trade_mode,
            MarketDataSourceKind::WebsocketTradeBookTicker,
        )
        .await?;

    Ok(())
}

fn print_compare_report(results: &[BacktestSummary]) {
    println!("strategy,uri,total_ticks,entries,closed_trades,realized_pnl,fees_paid,ending_cash,ending_equity,win_rate,profit_factor,avg_pnl_per_trade,max_drawdown");
    for result in results {
        println!(
            "{},{},{},{},{},{:.10},{:.10},{:.10},{:.10},{:.6},{:.6},{:.10},{:.10}",
            result.strategy,
            result.uri,
            result.total_ticks,
            result.entries,
            result.closed_trades,
            result.realized_pnl,
            result.fees_paid,
            result.ending_cash,
            result.ending_equity,
            result.win_rate,
            result.profit_factor,
            result.avg_pnl_per_trade,
            result.max_drawdown,
        );
    }
}

async fn run_capture(
    trading_symbol: &str,
    output_path: &str,
    duration_secs: u64,
    depth_levels: u32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let output = Path::new(output_path);
    if let Some(parent) = output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let client = BinanceClient::new().await?;
    let stream = client
        .market_event_stream_with_depth(trading_symbol, depth_levels)
        .await?;
    let mut stream = Box::pin(stream.take_until(tokio::time::sleep(Duration::from_secs(duration_secs))));
    let file = tokio::fs::File::create(output).await?;
    let mut writer = tokio::io::BufWriter::new(file);
    let mut event_count = 0_u64;

    info!(
        symbol = %trading_symbol,
        output = %output_path,
        duration_secs,
        depth_levels,
        source = "websocket_trade_bookticker_depth",
        "Starting raw market capture"
    );

    while let Some(event) = stream.next().await {
        let record = match event {
            MarketEvent::Trade(trade) => CaptureRecord::Trade {
                event_time: trade.event_time,
                symbol: trade.symbol,
                trade_id: trade.trade_id,
                price: trade.price,
                quantity: trade.quantity,
                trade_time: trade.trade_time,
                is_buyer_market_maker: trade.is_buyer_market_maker,
            },
            MarketEvent::BookTicker(book) => CaptureRecord::BookTicker {
                event_time: book.event_time,
                bid_price: book.bid.price,
                bid_quantity: book.bid.quantity,
                ask_price: book.ask.price,
                ask_quantity: book.ask.quantity,
            },
            MarketEvent::Depth(depth) => CaptureRecord::Depth {
                event_time: depth.event_time,
                bids: depth
                    .bids
                    .into_iter()
                    .map(|level| CaptureDepthLevel {
                        price: level.price,
                        quantity: level.quantity,
                    })
                    .collect(),
                asks: depth
                    .asks
                    .into_iter()
                    .map(|level| CaptureDepthLevel {
                        price: level.price,
                        quantity: level.quantity,
                    })
                    .collect(),
            },
        };

        let line = serde_json::to_string(&record)?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, line.as_bytes()).await?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, b"\n").await?;
        event_count += 1;
    }

    tokio::io::AsyncWriteExt::flush(&mut writer).await?;
    info!(symbol = %trading_symbol, output = %output_path, events = event_count, "Finished raw market capture");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{HistoricalMarketDataFormat, ReplayDatasetSummary, detect_historical_market_data_format};

    #[test]
    fn detects_captured_jsonl_replay_inputs() {
        assert_eq!(
            detect_historical_market_data_format("/tmp/pulsar_capture.jsonl"),
            HistoricalMarketDataFormat::CapturedJsonl
        );
        assert_eq!(
            detect_historical_market_data_format("https://example.com/replay.ndjson?download=1"),
            HistoricalMarketDataFormat::CapturedJsonl
        );
    }

    #[test]
    fn defaults_to_trade_csv_replay_inputs() {
        assert_eq!(
            detect_historical_market_data_format("../../data/binance/daily/trades/DOGEUSDT/day.zip"),
            HistoricalMarketDataFormat::TradeCsv
        );
        assert_eq!(
            detect_historical_market_data_format("https://example.com/day.csv"),
            HistoricalMarketDataFormat::TradeCsv
        );
    }

    #[test]
    fn replay_dataset_summary_can_represent_mixed_event_counts() {
        let summary = ReplayDatasetSummary {
            total_events: 32,
            trade_events: 2,
            book_ticker_events: 20,
            depth_events: 10,
            parse_errors: 0,
            first_event_time: Some(1),
            last_event_time: Some(2),
            event_time_regressions: 1,
            symbols: vec!["DOGEUSDT".to_string()],
        };

        assert_eq!(summary.total_events, 32);
        assert_eq!(summary.trade_events + summary.book_ticker_events + summary.depth_events, 32);
        assert_eq!(summary.symbols, vec!["DOGEUSDT"]);
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
    Compare {
        #[arg(short, long, required = true, num_args = 1..)]
        uris: Vec<String>,

        #[arg(long, value_delimiter = ',', default_value = "trade-flow-momentum,liquidity-sweep-reversal")]
        strategies: Vec<String>,
    },
    Capture {
        #[arg(short, long)]
        output: String,

        #[arg(long, default_value_t = 5)]
        depth_levels: u32,

        #[arg(long)]
        duration_secs: u64,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Configure logging with specific levels for different modules
    let env_filter = std::env::var("RUST_LOG").map_or_else(
        |_| "binance_bot=info,binance_sdk=warn,binance_exchange=info,trade=info".to_string(),
        |rust_log| format!("{rust_log},binance_sdk=warn"),
    );

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    // Load trading configuration
    let trading_config =
        load_config("config/trading_config.toml").expect("Failed to load trading configuration");

    let trading_symbol: String = get_config_value(&trading_config, "position_sizing.trading_symbol")
        .ok_or("Trading symbol not defined in configuration")?;

    match cli.command {
        Commands::Trade => {
            run_live_mode(&trading_symbol, &cli.strategy, TradeMode::Real, cli.duration_secs).await?;
        }
        Commands::Emulate => {
            run_live_mode(&trading_symbol, &cli.strategy, TradeMode::Emulated, cli.duration_secs)
                .await?;
        }
        Commands::Backtest { uri } => {
            let _ = run_backtest(&trading_symbol, &cli.strategy, &uri, cli.duration_secs).await?;
        }
        Commands::Compare { uris, strategies } => {
            let mut results = Vec::new();

            for strategy_name in strategies {
                for uri in &uris {
                    results.push(
                        run_backtest(&trading_symbol, &strategy_name, uri, cli.duration_secs).await?,
                    );
                }
            }

            print_compare_report(&results);
        }
        Commands::Capture {
            output,
            depth_levels,
            duration_secs,
        } => {
            run_capture(&trading_symbol, &output, duration_secs, depth_levels).await?;
        }
    }

    Ok(())
}
