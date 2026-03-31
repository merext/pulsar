
use ::trade::market::MarketEvent;
use ::trade::Strategy;
use ::trade::trader::{MarketDataSourceKind, TradeMode, Trader};
use binance_exchange::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use clap::{Parser, Subcommand};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::pin::Pin;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use toml::Value;
use tracing::info;

#[derive(Debug, Serialize)]
struct CaptureDepthLevel {
    price: f64,
    quantity: f64,
}

#[derive(Debug, Serialize)]
struct CaptureMetadata {
    schema_version: u32,
    symbol: String,
    output_path: String,
    data_path: String,
    generated_at_ms: u64,
    capture_duration_secs: u64,
    depth_levels: u32,
    total_events: u64,
    trade_events: u64,
    book_ticker_events: u64,
    depth_events: u64,
    first_capture_sequence: Option<u64>,
    last_capture_sequence: Option<u64>,
    first_captured_at_ms: Option<u64>,
    last_captured_at_ms: Option<u64>,
    capture_format: &'static str,
    ordering_semantics: &'static str,
    exchange_time_semantics: &'static str,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct CaptureMetadataRecord {
    schema_version: u32,
    symbol: String,
    output_path: String,
    data_path: String,
    generated_at_ms: u64,
    capture_duration_secs: u64,
    depth_levels: u32,
    total_events: u64,
    trade_events: u64,
    book_ticker_events: u64,
    depth_events: u64,
    first_capture_sequence: Option<u64>,
    last_capture_sequence: Option<u64>,
    first_captured_at_ms: Option<u64>,
    last_captured_at_ms: Option<u64>,
    capture_format: String,
    ordering_semantics: String,
    exchange_time_semantics: String,
}

#[derive(Debug, Clone, Serialize)]
struct CapturedDatasetIndexEntry {
    metadata_path: String,
    data_path: String,
    symbol: String,
    total_events: u64,
    trade_events: u64,
    book_ticker_events: u64,
    depth_events: u64,
    capture_duration_secs: u64,
    depth_levels: u32,
    first_captured_at_ms: Option<u64>,
    last_captured_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct CapturedDatasetIndex {
    root: String,
    dataset_count: usize,
    datasets: Vec<CapturedDatasetIndexEntry>,
}

#[derive(Debug, Clone, Default)]
struct CapturedDatasetFilter {
    symbol: Option<String>,
    min_total_events: Option<u64>,
    min_book_ticker_events: Option<u64>,
    min_depth_events: Option<u64>,
    require_captured_at: bool,
    since_captured_at_ms: Option<u64>,
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
    first_capture_sequence: Option<u64>,
    last_capture_sequence: Option<u64>,
    capture_sequence_regressions: usize,
    first_captured_at_ms: Option<u64>,
    last_captured_at_ms: Option<u64>,
    captured_at_regressions: usize,
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
        capture_sequence: u64,
        captured_at_ms: u64,
        exchange_event_time: Option<u64>,
        symbol: String,
        trade_id: u64,
        price: f64,
        quantity: f64,
        trade_time: u64,
        is_buyer_market_maker: bool,
    },
    BookTicker {
        capture_sequence: u64,
        captured_at_ms: u64,
        exchange_event_time: Option<u64>,
        update_id: Option<u64>,
        bid_price: f64,
        bid_quantity: f64,
        ask_price: f64,
        ask_quantity: f64,
    },
    Depth {
        capture_sequence: u64,
        captured_at_ms: u64,
        exchange_event_time: Option<u64>,
        last_update_id: Option<u64>,
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
        first_capture_sequence = summary.first_capture_sequence,
        last_capture_sequence = summary.last_capture_sequence,
        capture_sequence_regressions = summary.capture_sequence_regressions,
        first_captured_at_ms = summary.first_captured_at_ms,
        last_captured_at_ms = summary.last_captured_at_ms,
        captured_at_regressions = summary.captured_at_regressions,
        symbols = %summary.symbols.join(","),
        action = "replay_dataset_summary"
    );
}

fn current_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time is after unix epoch")
        .as_millis() as u64
}

fn capture_metadata_path(output: &Path) -> String {
    format!("{}.metadata.json", output.display())
}

fn default_capture_batch_output_path(symbol: &str, batch_id: &str, sequence: usize) -> String {
    format!(
        "data/binance/capture/{symbol}/{batch_id}_part_{sequence:03}.jsonl",
    )
}

fn build_capture_metadata_record(
    trading_symbol: &str,
    data_path: &str,
    duration_secs: u64,
    depth_levels: u32,
    summary: &ReplayDatasetSummary,
) -> CaptureMetadataRecord {
    CaptureMetadataRecord {
        schema_version: 1,
        symbol: trading_symbol.to_string(),
        output_path: data_path.to_string(),
        data_path: data_path.to_string(),
        generated_at_ms: current_time_millis(),
        capture_duration_secs: duration_secs,
        depth_levels,
        total_events: summary.total_events as u64,
        trade_events: summary.trade_events as u64,
        book_ticker_events: summary.book_ticker_events as u64,
        depth_events: summary.depth_events as u64,
        first_capture_sequence: summary.first_capture_sequence,
        last_capture_sequence: summary.last_capture_sequence,
        first_captured_at_ms: summary.first_captured_at_ms,
        last_captured_at_ms: summary.last_captured_at_ms,
        capture_format: "binance_mixed_event_jsonl".to_string(),
        ordering_semantics: "replay_in_capture_sequence_order".to_string(),
        exchange_time_semantics:
            "exchange_timestamps_are_informative_not_authoritative_across_event_classes"
                .to_string(),
    }
}

async fn build_captured_dataset_index(
    root: &str,
) -> Result<CapturedDatasetIndex, Box<dyn Error + Send + Sync>> {
    let mut entries = Vec::new();
    let mut dir = tokio::fs::read_dir(root).await?;

    while let Some(entry) = dir.next_entry().await? {
        let path = entry.path();
        let file_type = entry.file_type().await?;

        if file_type.is_dir() {
            let nested_root = path.to_string_lossy().into_owned();
            let nested_index = Box::pin(build_captured_dataset_index(&nested_root)).await?;
            entries.extend(nested_index.datasets);
            continue;
        }

        let path_string = path.to_string_lossy().into_owned();
        if !path_string.ends_with(".metadata.json") {
            continue;
        }

        let payload = tokio::fs::read(&path).await?;
        let metadata: CaptureMetadataRecord = serde_json::from_slice(&payload)?;
        entries.push(CapturedDatasetIndexEntry {
            metadata_path: path_string,
            data_path: metadata.data_path,
            symbol: metadata.symbol,
            total_events: metadata.total_events,
            trade_events: metadata.trade_events,
            book_ticker_events: metadata.book_ticker_events,
            depth_events: metadata.depth_events,
            capture_duration_secs: metadata.capture_duration_secs,
            depth_levels: metadata.depth_levels,
            first_captured_at_ms: metadata.first_captured_at_ms,
            last_captured_at_ms: metadata.last_captured_at_ms,
        });
    }

    entries.sort_by(|left, right| {
        left.first_captured_at_ms
            .cmp(&right.first_captured_at_ms)
            .then_with(|| left.data_path.cmp(&right.data_path))
    });

    Ok(CapturedDatasetIndex {
        root: root.to_string(),
        dataset_count: entries.len(),
        datasets: entries,
    })
}

fn dataset_matches_filter(
    entry: &CapturedDatasetIndexEntry,
    filter: &CapturedDatasetFilter,
) -> bool {
    if let Some(symbol) = filter.symbol.as_deref()
        && entry.symbol != symbol
    {
        return false;
    }

    if let Some(min_total_events) = filter.min_total_events
        && entry.total_events < min_total_events
    {
        return false;
    }

    if let Some(min_book_ticker_events) = filter.min_book_ticker_events
        && entry.book_ticker_events < min_book_ticker_events
    {
        return false;
    }

    if let Some(min_depth_events) = filter.min_depth_events
        && entry.depth_events < min_depth_events
    {
        return false;
    }

    if filter.require_captured_at && entry.first_captured_at_ms.is_none() {
        return false;
    }

    if let Some(since_captured_at_ms) = filter.since_captured_at_ms
        && entry
            .first_captured_at_ms
            .is_none_or(|captured_at| captured_at < since_captured_at_ms)
    {
        return false;
    }

    true
}

fn apply_captured_dataset_filter(
    mut index: CapturedDatasetIndex,
    filter: &CapturedDatasetFilter,
) -> CapturedDatasetIndex {
    index.datasets.retain(|entry| dataset_matches_filter(entry, filter));
    index.dataset_count = index.datasets.len();
    index
}

async fn run_capture_compare(
    trading_symbol: &str,
    root: &str,
    strategies: &[String],
    filter: &CapturedDatasetFilter,
    limit: Option<usize>,
    duration_secs: Option<u64>,
) -> Result<Vec<BacktestSummary>, Box<dyn Error + Send + Sync>> {
    let index = apply_captured_dataset_filter(build_captured_dataset_index(root).await?, filter);
    let mut datasets = index.datasets;

    if let Some(limit) = limit {
        datasets.truncate(limit);
    }

    let mut results = Vec::new();
    for strategy_name in strategies {
        for dataset in &datasets {
            results.push(
                run_backtest(trading_symbol, strategy_name, &dataset.data_path, duration_secs).await?,
            );
        }
    }

    Ok(results)
}

fn print_captured_dataset_index(index: &CapturedDatasetIndex) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("{}", serde_json::to_string_pretty(index)?);
    Ok(())
}

async fn run_capture_backfill(
    trading_symbol: &str,
    input_path: &str,
    duration_secs: Option<u64>,
    depth_levels: Option<u32>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let parsed = BinanceClient::load_captured_market_event_data_from_uri(input_path).await?;
    let summary = ReplayDatasetSummary {
        total_events: parsed.summary.parsed_events,
        trade_events: parsed.summary.trade_events,
        book_ticker_events: parsed.summary.book_ticker_events,
        depth_events: parsed.summary.depth_events,
        parse_errors: parsed.summary.parse_errors,
        first_event_time: parsed.summary.first_event_time,
        last_event_time: parsed.summary.last_event_time,
        event_time_regressions: parsed.summary.event_time_regressions,
        first_capture_sequence: parsed.summary.first_capture_sequence,
        last_capture_sequence: parsed.summary.last_capture_sequence,
        capture_sequence_regressions: parsed.summary.capture_sequence_regressions,
        first_captured_at_ms: parsed.summary.first_captured_at_ms,
        last_captured_at_ms: parsed.summary.last_captured_at_ms,
        captured_at_regressions: parsed.summary.captured_at_regressions,
        symbols: parsed.summary.symbols,
    };

    let inferred_symbol = summary
        .symbols
        .first()
        .cloned()
        .unwrap_or_else(|| trading_symbol.to_string());
    let metadata = build_capture_metadata_record(
        &inferred_symbol,
        input_path,
        duration_secs.unwrap_or(0),
        depth_levels.unwrap_or(5),
        &summary,
    );
    let metadata_path = capture_metadata_path(Path::new(input_path));
    let metadata_json = serde_json::to_vec_pretty(&metadata)?;
    tokio::fs::write(&metadata_path, metadata_json).await?;

    info!(
        input = %input_path,
        metadata_path = %metadata_path,
        symbol = %metadata.symbol,
        total_events = metadata.total_events,
        "Backfilled capture metadata sidecar"
    );

    Ok(())
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
                    first_capture_sequence: parsed.summary.first_capture_sequence,
                    last_capture_sequence: parsed.summary.last_capture_sequence,
                    capture_sequence_regressions: parsed.summary.capture_sequence_regressions,
                    first_captured_at_ms: parsed.summary.first_captured_at_ms,
                    last_captured_at_ms: parsed.summary.last_captured_at_ms,
                    captured_at_regressions: parsed.summary.captured_at_regressions,
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
    let mut capture_sequence = 0_u64;
    let mut trade_events = 0_u64;
    let mut book_ticker_events = 0_u64;
    let mut depth_events = 0_u64;
    let mut first_captured_at_ms = None;
    let mut last_captured_at_ms = None;

    info!(
        symbol = %trading_symbol,
        output = %output_path,
        duration_secs,
        depth_levels,
        source = "websocket_trade_bookticker_depth",
        "Starting raw market capture"
    );

    while let Some(event) = stream.next().await {
        capture_sequence += 1;
        let captured_at_ms = current_time_millis();
        first_captured_at_ms = Some(
            first_captured_at_ms.map_or(captured_at_ms, |current: u64| current.min(captured_at_ms)),
        );
        last_captured_at_ms = Some(
            last_captured_at_ms.map_or(captured_at_ms, |current: u64| current.max(captured_at_ms)),
        );
        let record = match event {
            MarketEvent::Trade(trade) => {
                trade_events += 1;
                CaptureRecord::Trade {
                capture_sequence,
                captured_at_ms,
                exchange_event_time: Some(trade.event_time),
                symbol: trade.symbol,
                trade_id: trade.trade_id,
                price: trade.price,
                quantity: trade.quantity,
                trade_time: trade.trade_time,
                is_buyer_market_maker: trade.is_buyer_market_maker,
                }
            }
            MarketEvent::BookTicker(book) => {
                book_ticker_events += 1;
                CaptureRecord::BookTicker {
                capture_sequence,
                captured_at_ms,
                exchange_event_time: None,
                update_id: None,
                bid_price: book.bid.price,
                bid_quantity: book.bid.quantity,
                ask_price: book.ask.price,
                ask_quantity: book.ask.quantity,
                }
            }
            MarketEvent::Depth(depth) => {
                depth_events += 1;
                CaptureRecord::Depth {
                capture_sequence,
                captured_at_ms,
                exchange_event_time: None,
                last_update_id: None,
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
                }
            }
        };

        let line = serde_json::to_string(&record)?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, line.as_bytes()).await?;
        tokio::io::AsyncWriteExt::write_all(&mut writer, b"\n").await?;
        event_count += 1;
    }

    tokio::io::AsyncWriteExt::flush(&mut writer).await?;
    let metadata = CaptureMetadata {
        schema_version: 1,
        symbol: trading_symbol.to_string(),
        output_path: output_path.to_string(),
        data_path: output_path.to_string(),
        generated_at_ms: current_time_millis(),
        capture_duration_secs: duration_secs,
        depth_levels,
        total_events: event_count,
        trade_events,
        book_ticker_events,
        depth_events,
        first_capture_sequence: if event_count > 0 { Some(1) } else { None },
        last_capture_sequence: if event_count > 0 { Some(capture_sequence) } else { None },
        first_captured_at_ms,
        last_captured_at_ms,
        capture_format: "binance_mixed_event_jsonl",
        ordering_semantics: "replay_in_capture_sequence_order",
        exchange_time_semantics: "exchange_timestamps_are_informative_not_authoritative_across_event_classes",
    };
    let metadata_path = capture_metadata_path(output);
    let metadata_json = serde_json::to_vec_pretty(&metadata)?;
    tokio::fs::write(&metadata_path, metadata_json).await?;
    info!(symbol = %trading_symbol, output = %output_path, events = event_count, "Finished raw market capture");
    info!(symbol = %trading_symbol, metadata_path = %metadata_path, events = event_count, "Wrote capture metadata sidecar");
    Ok(())
}

async fn run_capture_batch(
    trading_symbol: &str,
    output_dir: Option<&str>,
    batch_id: &str,
    parts: usize,
    duration_secs: u64,
    depth_levels: u32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    for sequence in 1..=parts {
        let output_path = if let Some(output_dir) = output_dir {
            format!("{output_dir}/{batch_id}_part_{sequence:03}.jsonl")
        } else {
            default_capture_batch_output_path(trading_symbol, batch_id, sequence)
        };

        info!(
            symbol = %trading_symbol,
            batch_id = %batch_id,
            sequence,
            parts,
            output = %output_path,
            duration_secs,
            depth_levels,
            "Starting capture batch segment"
        );

        run_capture(trading_symbol, &output_path, duration_secs, depth_levels).await?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CapturedDatasetFilter, CapturedDatasetIndex, CapturedDatasetIndexEntry,
        HistoricalMarketDataFormat, ReplayDatasetSummary, apply_captured_dataset_filter,
        detect_historical_market_data_format,
    };

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
            first_capture_sequence: Some(1),
            last_capture_sequence: Some(32),
            capture_sequence_regressions: 0,
            first_captured_at_ms: Some(10),
            last_captured_at_ms: Some(20),
            captured_at_regressions: 0,
            symbols: vec!["DOGEUSDT".to_string()],
        };

        assert_eq!(summary.total_events, 32);
        assert_eq!(summary.trade_events + summary.book_ticker_events + summary.depth_events, 32);
        assert_eq!(summary.symbols, vec!["DOGEUSDT"]);
    }

    #[test]
    fn dataset_index_serializes_for_cli_output() {
        let index = CapturedDatasetIndex {
            root: "data/binance/capture".to_string(),
            dataset_count: 1,
            datasets: vec![CapturedDatasetIndexEntry {
                metadata_path: "data/binance/capture/DOGEUSDT/sample.jsonl.metadata.json".to_string(),
                data_path: "data/binance/capture/DOGEUSDT/sample.jsonl".to_string(),
                symbol: "DOGEUSDT".to_string(),
                total_events: 10,
                trade_events: 2,
                book_ticker_events: 6,
                depth_events: 2,
                capture_duration_secs: 5,
                depth_levels: 5,
                first_captured_at_ms: Some(1),
                last_captured_at_ms: Some(2),
            }],
        };

        let json = serde_json::to_string(&index).expect("index serializes");
        assert!(json.contains("sample.jsonl"));
        assert!(json.contains("DOGEUSDT"));
    }

    #[test]
    fn capture_dataset_index_entries_can_be_filtered_by_symbol() {
        let datasets = vec![
            CapturedDatasetIndexEntry {
                metadata_path: "a.metadata.json".to_string(),
                data_path: "a.jsonl".to_string(),
                symbol: "DOGEUSDT".to_string(),
                total_events: 1,
                trade_events: 1,
                book_ticker_events: 0,
                depth_events: 0,
                capture_duration_secs: 1,
                depth_levels: 5,
                first_captured_at_ms: Some(1),
                last_captured_at_ms: Some(2),
            },
            CapturedDatasetIndexEntry {
                metadata_path: "b.metadata.json".to_string(),
                data_path: "b.jsonl".to_string(),
                symbol: "BTCUSDT".to_string(),
                total_events: 1,
                trade_events: 1,
                book_ticker_events: 0,
                depth_events: 0,
                capture_duration_secs: 1,
                depth_levels: 5,
                first_captured_at_ms: Some(3),
                last_captured_at_ms: Some(4),
            },
        ];

        let filtered: Vec<_> = datasets
            .into_iter()
            .filter(|entry| entry.symbol == "DOGEUSDT")
            .collect();

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].data_path, "a.jsonl");
    }

    #[test]
    fn capture_metadata_record_can_be_built_from_replay_summary() {
        let summary = ReplayDatasetSummary {
            total_events: 3,
            trade_events: 1,
            book_ticker_events: 1,
            depth_events: 1,
            parse_errors: 0,
            first_event_time: Some(1),
            last_event_time: Some(2),
            event_time_regressions: 0,
            first_capture_sequence: Some(1),
            last_capture_sequence: Some(3),
            capture_sequence_regressions: 0,
            first_captured_at_ms: Some(10),
            last_captured_at_ms: Some(20),
            captured_at_regressions: 0,
            symbols: vec!["DOGEUSDT".to_string()],
        };

        let metadata = super::build_capture_metadata_record(
            "DOGEUSDT",
            "data/binance/capture/DOGEUSDT/sample.jsonl",
            5,
            5,
            &summary,
        );

        assert_eq!(metadata.symbol, "DOGEUSDT");
        assert_eq!(metadata.total_events, 3);
        assert_eq!(metadata.capture_duration_secs, 5);
        assert_eq!(metadata.depth_levels, 5);
    }

    #[test]
    fn dataset_filter_excludes_trade_only_backfill_when_quotes_required() {
        let index = CapturedDatasetIndex {
            root: "data/binance/capture".to_string(),
            dataset_count: 2,
            datasets: vec![
                CapturedDatasetIndexEntry {
                    metadata_path: "trade_only.metadata.json".to_string(),
                    data_path: "trade_only.jsonl".to_string(),
                    symbol: "DOGEUSDT".to_string(),
                    total_events: 65,
                    trade_events: 65,
                    book_ticker_events: 0,
                    depth_events: 0,
                    capture_duration_secs: 10,
                    depth_levels: 5,
                    first_captured_at_ms: None,
                    last_captured_at_ms: None,
                },
                CapturedDatasetIndexEntry {
                    metadata_path: "mixed.metadata.json".to_string(),
                    data_path: "mixed.jsonl".to_string(),
                    symbol: "DOGEUSDT".to_string(),
                    total_events: 617,
                    trade_events: 65,
                    book_ticker_events: 472,
                    depth_events: 80,
                    capture_duration_secs: 10,
                    depth_levels: 5,
                    first_captured_at_ms: Some(1),
                    last_captured_at_ms: Some(2),
                },
            ],
        };

        let filtered = apply_captured_dataset_filter(
            index,
            &CapturedDatasetFilter {
                min_book_ticker_events: Some(1),
                require_captured_at: true,
                ..CapturedDatasetFilter::default()
            },
        );

        assert_eq!(filtered.dataset_count, 1);
        assert_eq!(filtered.datasets[0].data_path, "mixed.jsonl");
    }

    #[test]
    fn default_capture_batch_output_path_uses_stable_sequential_naming() {
        let path = super::default_capture_batch_output_path("DOGEUSDT", "smoke_batch", 7);
        assert_eq!(
            path,
            "data/binance/capture/DOGEUSDT/smoke_batch_part_007.jsonl"
        );
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
    CaptureIndex {
        #[arg(short, long, default_value = "data/binance/capture")]
        root: String,

        #[arg(long)]
        symbol: Option<String>,

        #[arg(long)]
        min_total_events: Option<u64>,

        #[arg(long)]
        min_book_ticker_events: Option<u64>,

        #[arg(long)]
        min_depth_events: Option<u64>,

        #[arg(long, default_value_t = false)]
        require_captured_at: bool,

        #[arg(long)]
        since_captured_at_ms: Option<u64>,
    },
    CaptureCompare {
        #[arg(short, long, default_value = "data/binance/capture")]
        root: String,

        #[arg(long)]
        symbol: Option<String>,

        #[arg(long)]
        min_total_events: Option<u64>,

        #[arg(long)]
        min_book_ticker_events: Option<u64>,

        #[arg(long)]
        min_depth_events: Option<u64>,

        #[arg(long, default_value_t = false)]
        require_captured_at: bool,

        #[arg(long)]
        since_captured_at_ms: Option<u64>,

        #[arg(long)]
        limit: Option<usize>,

        #[arg(long, value_delimiter = ',', default_value = "trade-flow-momentum,liquidity-sweep-reversal")]
        strategies: Vec<String>,
    },
    CaptureBackfill {
        #[arg(short, long)]
        input: String,

        #[arg(long)]
        duration_secs: Option<u64>,

        #[arg(long)]
        depth_levels: Option<u32>,
    },
    CaptureBatch {
        #[arg(long)]
        batch_id: String,

        #[arg(long)]
        output_dir: Option<String>,

        #[arg(long)]
        parts: usize,

        #[arg(long)]
        duration_secs: u64,

        #[arg(long, default_value_t = 5)]
        depth_levels: u32,
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
        Commands::CaptureIndex {
            root,
            symbol,
            min_total_events,
            min_book_ticker_events,
            min_depth_events,
            require_captured_at,
            since_captured_at_ms,
        } => {
            let filter = CapturedDatasetFilter {
                symbol,
                min_total_events,
                min_book_ticker_events,
                min_depth_events,
                require_captured_at,
                since_captured_at_ms,
            };
            let index = apply_captured_dataset_filter(build_captured_dataset_index(&root).await?, &filter);
            print_captured_dataset_index(&index)?;
        }
        Commands::CaptureCompare {
            root,
            symbol,
            min_total_events,
            min_book_ticker_events,
            min_depth_events,
            require_captured_at,
            since_captured_at_ms,
            limit,
            strategies,
        } => {
            let filter = CapturedDatasetFilter {
                symbol,
                min_total_events,
                min_book_ticker_events,
                min_depth_events,
                require_captured_at,
                since_captured_at_ms,
            };
            let results = run_capture_compare(
                &trading_symbol,
                &root,
                &strategies,
                &filter,
                limit,
                cli.duration_secs,
            )
            .await?;
            print_compare_report(&results);
        }
        Commands::CaptureBackfill {
            input,
            duration_secs,
            depth_levels,
        } => {
            run_capture_backfill(&trading_symbol, &input, duration_secs, depth_levels).await?;
        }
        Commands::CaptureBatch {
            batch_id,
            output_dir,
            parts,
            duration_secs,
            depth_levels,
        } => {
            run_capture_batch(
                &trading_symbol,
                output_dir.as_deref(),
                &batch_id,
                parts,
                duration_secs,
                depth_levels,
            )
            .await?;
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
