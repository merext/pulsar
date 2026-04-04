
use ::trade::market::{MarketEvent, MarketState};
use ::trade::Strategy;
use ::trade::trader::{MarketDataSourceKind, TradeMode, Trader};
use binance_exchange::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use clap::{Parser, Subcommand};
use flate2::Compression;
use flate2::write::GzEncoder;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::pin::Pin;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use toml::Value;
use tracing::info;
use strategies::TradeFlowReclaimStrategy;

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
    data_size_bytes: u64,
    data_sha256: String,
    capture_format: &'static str,
    ordering_semantics: &'static str,
    exchange_time_semantics: &'static str,
    capture_time_quality: &'static str,
    quote_presence_quality: &'static str,
    depth_presence_quality: &'static str,
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
    data_size_bytes: u64,
    data_sha256: String,
    capture_format: String,
    ordering_semantics: String,
    exchange_time_semantics: String,
    capture_time_quality: String,
    quote_presence_quality: String,
    depth_presence_quality: String,
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
    data_size_bytes: u64,
    data_sha256: String,
    sidecar_verified: bool,
    capture_time_quality: String,
    quote_presence_quality: String,
    depth_presence_quality: String,
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
    min_quote_quality: Option<String>,
    min_depth_quality: Option<String>,
    require_verified_sidecar: bool,
}

#[derive(Debug, Clone, Serialize)]
struct StrategyDiagnosticRow {
    strategy: String,
    uri: String,
    key: String,
    value: String,
    value_type: String,
}

#[derive(Debug, Clone, Serialize)]
struct TradeAttributionRow {
    strategy: String,
    uri: String,
    trade_id: usize,
    symbol: String,
    signal: String,
    timestamp: f64,
    price: f64,
    quantity: f64,
    pnl: f64,
    gross_pnl: f64,
    fee_paid: f64,
    expected_edge_bps: f64,
    rationale: String,
    decision_confidence: f64,
    requested_quantity: f64,
    executed_quantity: f64,
    synthetic_half_spread_bps: f64,
    slippage_bps: f64,
    latency_impact_bps: f64,
    market_impact_bps: f64,
    hold_time_millis: u64,
    exit_reason: String,
    entry_price: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct CaptureRotationConfig {
    max_events_per_file: Option<u64>,
    gzip: Option<bool>,
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

#[derive(Debug, Clone, Serialize)]
struct ParameterSearchResult {
    strategy: String,
    uri: String,
    parameter: String,
    value: f64,
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

#[derive(Debug, Clone)]
struct ParameterSearchSpec {
    strategy: String,
    parameter: String,
    values: Vec<f64>,
    uris: Vec<String>,
}

#[derive(Debug, Clone)]
struct ParameterOptimizationSpec {
    strategy: String,
    parameter: String,
    values: Vec<f64>,
    uris: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ParameterOptimizationResult {
    strategy: String,
    parameter: String,
    value: f64,
    dataset_count: usize,
    total_ticks: usize,
    total_entries: usize,
    total_closed_trades: usize,
    total_realized_pnl: f64,
    total_fees_paid: f64,
    mean_realized_pnl: f64,
    mean_ending_cash: f64,
    mean_ending_equity: f64,
    mean_win_rate: f64,
    mean_profit_factor: f64,
    mean_avg_pnl_per_trade: f64,
    worst_max_drawdown: f64,
}

#[derive(Debug, Clone)]
struct ParameterOptimizationAccumulator {
    strategy: String,
    parameter: String,
    value: f64,
    dataset_count: usize,
    total_ticks: usize,
    total_entries: usize,
    total_closed_trades: usize,
    total_realized_pnl: f64,
    total_fees_paid: f64,
    total_ending_cash: f64,
    total_ending_equity: f64,
    total_win_rate: f64,
    total_profit_factor: f64,
    total_avg_pnl_per_trade: f64,
    worst_max_drawdown: f64,
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
    market_data_format: String,
    regime_tags: Vec<String>,
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

struct BacktestArtifacts {
    summary: BacktestSummary,
    trade_rows: Vec<TradeAttributionRow>,
    diagnostic_rows: Vec<StrategyDiagnosticRow>,
}

#[derive(Debug, Clone, Serialize)]
struct CompareAggregateResult {
    strategy: String,
    dataset_count: usize,
    total_ticks: usize,
    total_entries: usize,
    total_closed_trades: usize,
    total_realized_pnl: f64,
    total_fees_paid: f64,
    mean_realized_pnl: f64,
    mean_ending_cash: f64,
    mean_ending_equity: f64,
    mean_win_rate: f64,
    mean_profit_factor: f64,
    mean_avg_pnl_per_trade: f64,
    worst_max_drawdown: f64,
}

#[derive(Debug, Clone)]
struct CompareAggregateAccumulator {
    strategy: String,
    dataset_count: usize,
    total_ticks: usize,
    total_entries: usize,
    total_closed_trades: usize,
    total_realized_pnl: f64,
    total_fees_paid: f64,
    total_ending_cash: f64,
    total_ending_equity: f64,
    total_win_rate: f64,
    total_profit_factor: f64,
    total_avg_pnl_per_trade: f64,
    worst_max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WalkForwardFoldResult {
    fold: usize,
    strategy: String,
    parameter: String,
    selected_value: f64,
    train_dataset_count: usize,
    test_dataset_count: usize,
    train_total_realized_pnl: f64,
    train_mean_realized_pnl: f64,
    test_total_realized_pnl: f64,
    test_mean_realized_pnl: f64,
    test_total_closed_trades: usize,
    test_mean_win_rate: f64,
    test_worst_max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize)]
struct WalkForwardSummary {
    strategy: String,
    parameter: String,
    fold_count: usize,
    total_test_realized_pnl: f64,
    mean_test_realized_pnl: f64,
    total_test_closed_trades: usize,
    mean_test_win_rate: f64,
    worst_test_max_drawdown: f64,
}

#[derive(Debug, Clone, Serialize)]
struct FeatureRow {
    symbol: String,
    uri: String,
    event_index: usize,
    event_kind: String,
    event_time_ms: u64,
    has_quote: bool,
    has_depth: bool,
    trade_count_window: usize,
    trade_volume_window: f64,
    trade_notional_window: f64,
    last_price: f64,
    mid_price: f64,
    spread_bps: f64,
    microprice: f64,
    microprice_edge_bps: f64,
    order_book_imbalance: f64,
    trade_flow_imbalance: f64,
    recent_trade_flow_imbalance: f64,
    trade_window_vwap: f64,
    trade_window_low: f64,
    trade_window_high: f64,
    forward_return_bps: f64,
    regime_tags: String,
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
fn load_config<P: AsRef<std::path::Path>>(
    config_path: P,
) -> Result<Value, Box<dyn Error + Send + Sync>> {
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

    /// Path to trading configuration file (symbol-specific)
    #[arg(long, default_value = "config/trading_config.toml")]
    config: String,

    /// Path to strategy configuration file (overrides default for the chosen strategy)
    #[arg(long)]
    strategy_config: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

fn build_strategy(
    strategy_name: &str,
    strategy_config_override: Option<&str>,
) -> Result<Box<dyn Strategy>, Box<dyn Error + Send + Sync>> {
    let market_maker_config = strategy_config_override
        .map(|p| resolve_workspace_path(p))
        .unwrap_or_else(|| resolve_workspace_path("config/strategies/market_maker.toml"));
    let market_maker_ba_config = strategy_config_override
        .map(|p| resolve_workspace_path(p))
        .unwrap_or_else(|| resolve_workspace_path("config/strategies/market_maker_ba.toml"));

    match strategy_name {
        "trade-flow-momentum" => Ok(Box::new(
            strategies::trade_flow_momentum::TradeFlowMomentumStrategy::from_file(
                resolve_workspace_path("config/strategies/trade_flow_momentum.toml"),
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "trade-flow-reclaim" => Ok(Box::new(
            TradeFlowReclaimStrategy::from_file(
                resolve_workspace_path("config/strategies/trade_flow_reclaim.toml"),
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "liquidity-sweep-reversal" => Ok(Box::new(
            strategies::liquidity_sweep_reversal::LiquiditySweepReversalStrategy::from_file(
                resolve_workspace_path("config/strategies/liquidity_sweep_reversal.toml"),
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "microprice-imbalance-maker" => Ok(Box::new(
            strategies::microprice_imbalance_maker::MicropriceImbalanceMakerStrategy::from_file(
                resolve_workspace_path("config/strategies/microprice_imbalance_maker.toml"),
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "spread-regime-capture" => Ok(Box::new(
            strategies::spread_regime_capture::SpreadRegimeCaptureStrategy::from_file(
                resolve_workspace_path("config/strategies/spread_regime_capture.toml"),
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "market-maker" => Ok(Box::new(
            strategies::market_maker::MarketMakerStrategy::from_file(
                market_maker_config,
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        "market-maker-ba" => Ok(Box::new(
            strategies::market_maker_ba::MarketMakerBaStrategy::from_file(
                market_maker_ba_config,
            )
            .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        )),
        _ => Err(format!(
            "Unknown strategy '{}'. Available: trade-flow-momentum, trade-flow-reclaim, liquidity-sweep-reversal, microprice-imbalance-maker, spread-regime-capture, market-maker, market-maker-ba",
            strategy_name
        )
        .into()),
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn resolve_workspace_path(relative_path: &str) -> PathBuf {
    workspace_root().join(relative_path)
}

fn capture_rotation_config() -> CaptureRotationConfig {
    let path = resolve_workspace_path("config/capture_rotation.toml");
    if !path.exists() {
        return CaptureRotationConfig {
            max_events_per_file: None,
            gzip: Some(false),
        };
    }

    let content = fs::read_to_string(path).unwrap_or_default();
    toml::from_str(&content).unwrap_or(CaptureRotationConfig {
        max_events_per_file: None,
        gzip: Some(false),
    })
}

fn rotated_capture_output_path(base_output_path: &str, part: usize, gzip: bool) -> String {
    let suffix = if gzip { ".jsonl.gz" } else { ".jsonl" };
    if let Some(stripped) = base_output_path.strip_suffix(".jsonl") {
        format!("{stripped}_part_{part:03}{suffix}")
    } else if let Some(stripped) = base_output_path.strip_suffix(".jsonl.gz") {
        format!("{stripped}_part_{part:03}{suffix}")
    } else {
        format!("{base_output_path}_part_{part:03}{suffix}")
    }
}

fn strategy_config_path(strategy_name: &str) -> Result<&'static str, Box<dyn Error + Send + Sync>> {
    match strategy_name {
        "trade-flow-momentum" => Ok("config/strategies/trade_flow_momentum.toml"),
        "trade-flow-reclaim" => Ok("config/strategies/trade_flow_reclaim.toml"),
        "liquidity-sweep-reversal" => Ok("config/strategies/liquidity_sweep_reversal.toml"),
        "microprice-imbalance-maker" => Ok("config/strategies/microprice_imbalance_maker.toml"),
        "spread-regime-capture" => Ok("config/strategies/spread_regime_capture.toml"),
        "market-maker" => Ok("config/strategies/market_maker.toml"),
        "market-maker-ba" => Ok("config/strategies/market_maker_ba.toml"),
        _ => Err(format!("Unknown strategy '{}'.", strategy_name).into()),
    }
}

fn set_config_value(config: &mut Value, key: &str, value: f64) -> Result<(), Box<dyn Error + Send + Sync>> {
    let keys: Vec<&str> = key.split('.').collect();
    let Some((last, parents)) = keys.split_last() else {
        return Err("parameter key cannot be empty".into());
    };

    let mut current = config;
    for parent in parents {
        current = current
            .get_mut(parent)
            .ok_or_else(|| format!("Missing config section '{}'.", parent))?;
    }

    let target = current
        .get_mut(last)
        .ok_or_else(|| format!("Missing config key '{}'.", key))?;
    *target = match target {
        Value::Integer(_) => {
            if !value.is_finite() || value.fract().abs() > f64::EPSILON {
                return Err(format!("Parameter '{}' expects an integer value, got {}.", key, value).into());
            }
            Value::Integer(value as i64)
        }
        Value::Float(_) => Value::Float(value),
        _ => return Err(format!("Unsupported config type for parameter '{}'.", key).into()),
    };
    Ok(())
}

fn parse_parameter_values(raw: &str) -> Result<Vec<f64>, Box<dyn Error + Send + Sync>> {
    let mut values = Vec::new();
    for item in raw.split(',') {
        let trimmed = item.trim();
        if trimmed.is_empty() {
            continue;
        }
        values.push(trimmed.parse::<f64>()?);
    }

    if values.is_empty() {
        return Err("parameter values cannot be empty".into());
    }

    Ok(values)
}

fn build_strategy_from_search_spec(
    strategy_name: &str,
    parameter: &str,
    value: f64,
    strategy_config_override: Option<&str>,
) -> Result<Box<dyn Strategy>, Box<dyn Error + Send + Sync>> {
    let resolved_path = match strategy_config_override {
        Some(override_path) => resolve_workspace_path(override_path),
        None => resolve_workspace_path(strategy_config_path(strategy_name)?),
    };
    let mut config = load_config(resolved_path)?;
    set_config_value(&mut config, parameter, value)?;

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(format!(
        "pulsar_search_{}_{}_{}.toml",
        strategy_name.replace('-', "_"),
        parameter.replace('.', "_"),
        value.to_string().replace('.', "_")
    ));
    fs::write(&temp_path, toml::to_string_pretty(&config)?)?;

    let strategy = match strategy_name {
        "trade-flow-momentum" => Box::new(
            strategies::trade_flow_momentum::TradeFlowMomentumStrategy::from_file(&temp_path)
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        ) as Box<dyn Strategy>,
        "trade-flow-reclaim" => Box::new(
            TradeFlowReclaimStrategy::from_file(&temp_path)
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        ) as Box<dyn Strategy>,
        "liquidity-sweep-reversal" => Box::new(
            strategies::liquidity_sweep_reversal::LiquiditySweepReversalStrategy::from_file(&temp_path)
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
        ) as Box<dyn Strategy>,
            "microprice-imbalance-maker" => Box::new(
                strategies::microprice_imbalance_maker::MicropriceImbalanceMakerStrategy::from_file(
                    &temp_path,
                )
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
            ) as Box<dyn Strategy>,
            "spread-regime-capture" => Box::new(
                strategies::spread_regime_capture::SpreadRegimeCaptureStrategy::from_file(
                    &temp_path,
                )
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
            ) as Box<dyn Strategy>,
            "market-maker" => Box::new(
                strategies::market_maker::MarketMakerStrategy::from_file(
                    &temp_path,
                )
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
            ) as Box<dyn Strategy>,
            "market-maker-ba" => Box::new(
                strategies::market_maker_ba::MarketMakerBaStrategy::from_file(
                    &temp_path,
                )
                .map_err(|err| -> Box<dyn Error + Send + Sync> { err.to_string().into() })?,
            ) as Box<dyn Strategy>,
            _ => return Err(format!("Unknown strategy '{}'.", strategy_name).into()),
        };

    let _ = fs::remove_file(&temp_path);
    Ok(strategy)
}

fn detect_historical_market_data_format(uri: &str) -> HistoricalMarketDataFormat {
    let normalized = uri
        .split(['?', '#'])
        .next()
        .unwrap_or(uri)
        .to_ascii_lowercase();

    if normalized.ends_with(".jsonl")
        || normalized.ends_with(".ndjson")
        || normalized.ends_with(".jsonl.gz")
        || normalized.ends_with(".ndjson.gz")
    {
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

fn infer_regime_tags(
    data_format: HistoricalMarketDataFormat,
    total_ticks: usize,
    replay_summary: Option<&ReplayDatasetSummary>,
) -> Vec<String> {
    let mut tags = Vec::new();
    tags.push(match data_format {
        HistoricalMarketDataFormat::TradeCsv => "source:trade_only".to_string(),
        HistoricalMarketDataFormat::CapturedJsonl => "source:captured".to_string(),
    });

    tags.push(match total_ticks {
        0..=100_000 => "activity:low".to_string(),
        100_001..=700_000 => "activity:medium".to_string(),
        _ => "activity:high".to_string(),
    });

    if let Some(summary) = replay_summary {
        tags.push(if summary.depth_events > 0 {
            "book:depth".to_string()
        } else if summary.book_ticker_events > 0 {
            "book:quote".to_string()
        } else {
            "book:none".to_string()
        });

        let quote_coverage = if summary.total_events == 0 {
            0.0
        } else {
            summary.book_ticker_events as f64 / summary.total_events as f64
        };
        tags.push(match quote_coverage {
            coverage if coverage <= 0.0 => "quote_coverage:none".to_string(),
            coverage if coverage < 0.25 => "quote_coverage:sparse".to_string(),
            coverage if coverage < 0.75 => "quote_coverage:mixed".to_string(),
            _ => "quote_coverage:dense".to_string(),
        });
    } else {
        tags.push("book:none".to_string());
        tags.push("quote_coverage:none".to_string());
    }

    tags
}

fn quote_presence_quality(book_ticker_events: usize, total_events: usize) -> &'static str {
    let coverage = if total_events == 0 {
        0.0
    } else {
        book_ticker_events as f64 / total_events as f64
    };
    match coverage {
        value if value <= 0.0 => "none",
        value if value < 0.25 => "sparse",
        value if value < 0.75 => "mixed",
        _ => "dense",
    }
}

fn depth_presence_quality(depth_events: usize, total_events: usize) -> &'static str {
    let coverage = if total_events == 0 {
        0.0
    } else {
        depth_events as f64 / total_events as f64
    };
    match coverage {
        value if value <= 0.0 => "none",
        value if value < 0.10 => "sparse",
        value if value < 0.50 => "mixed",
        _ => "dense",
    }
}

fn capture_time_quality(first_captured_at_ms: Option<u64>) -> &'static str {
    if first_captured_at_ms.is_some() {
        "explicit"
    } else {
        "missing"
    }
}

fn quality_rank(quality: &str) -> usize {
    match quality {
        "none" => 0,
        "sparse" => 1,
        "mixed" => 2,
        "dense" => 3,
        _ => 0,
    }
}

async fn load_historical_market_events(
    uri: &str,
) -> Result<
    (
        HistoricalMarketDataFormat,
        Vec<MarketEvent>,
        Option<ReplayDatasetSummary>,
    ),
    Box<dyn Error + Send + Sync>,
> {
    let data_format = detect_historical_market_data_format(uri);
    match data_format {
        HistoricalMarketDataFormat::TradeCsv => {
            let trades: Vec<_> = BinanceClient::trade_data_from_uri(uri).await?.collect().await;
            Ok((
                data_format,
                trades.into_iter().map(MarketEvent::Trade).collect(),
                None,
            ))
        }
        HistoricalMarketDataFormat::CapturedJsonl => {
            let parsed = BinanceClient::load_captured_market_event_data_from_uri(uri).await?;
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
            Ok((data_format, parsed.events, Some(summary)))
        }
    }
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
    data_size_bytes: u64,
    data_sha256: String,
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
        data_size_bytes,
        data_sha256,
        capture_format: "binance_mixed_event_jsonl".to_string(),
        ordering_semantics: "replay_in_capture_sequence_order".to_string(),
        exchange_time_semantics:
            "exchange_timestamps_are_informative_not_authoritative_across_event_classes"
                .to_string(),
        capture_time_quality: capture_time_quality(summary.first_captured_at_ms).to_string(),
        quote_presence_quality: quote_presence_quality(summary.book_ticker_events, summary.total_events)
            .to_string(),
        depth_presence_quality: depth_presence_quality(summary.depth_events, summary.total_events)
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
        let integrity = BinanceClient::file_integrity_summary_from_path(&metadata.data_path).await.ok();
        let sidecar_verified = integrity.as_ref().is_some_and(|integrity| {
            integrity.size_bytes == metadata.data_size_bytes
                && integrity.sha256_hex == metadata.data_sha256
        });
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
            data_size_bytes: metadata.data_size_bytes,
            data_sha256: metadata.data_sha256,
            sidecar_verified,
            capture_time_quality: metadata.capture_time_quality,
            quote_presence_quality: metadata.quote_presence_quality,
            depth_presence_quality: metadata.depth_presence_quality,
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

    if let Some(min_quote_quality) = filter.min_quote_quality.as_deref()
        && quality_rank(&entry.quote_presence_quality) < quality_rank(min_quote_quality)
    {
        return false;
    }

    if let Some(min_depth_quality) = filter.min_depth_quality.as_deref()
        && quality_rank(&entry.depth_presence_quality) < quality_rank(min_depth_quality)
    {
        return false;
    }

    if filter.require_verified_sidecar && !entry.sidecar_verified {
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
    config_path: &std::path::Path,
    strategy_config_override: Option<&str>,
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
                run_backtest(trading_symbol, strategy_name, &dataset.data_path, duration_secs, config_path, strategy_config_override).await?,
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
    let integrity = BinanceClient::file_integrity_summary_from_path(input_path).await?;
    let metadata = build_capture_metadata_record(
        &inferred_symbol,
        input_path,
        duration_secs.unwrap_or(0),
        depth_levels.unwrap_or(5),
        &summary,
        integrity.size_bytes,
        integrity.sha256_hex,
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
    config_path: &std::path::Path,
    strategy_config_override: Option<&str>,
) -> Result<BacktestSummary, Box<dyn Error + Send + Sync>> {
    let mut binance_trader = BinanceTrader::new_with_config_path(config_path).await?;
    let mut strategy = build_strategy(strategy_name, strategy_config_override)?;
    Ok(run_backtest_with_strategy(
        &mut binance_trader,
        strategy.as_mut(),
        strategy_name,
        uri,
        trading_symbol,
        duration_secs,
    )
    .await?
    .summary)
}

async fn run_backtest_with_strategy(
    binance_trader: &mut BinanceTrader,
    strategy: &mut dyn Strategy,
    strategy_name: &str,
    uri: &str,
    trading_symbol: &str,
    duration_secs: Option<u64>,
) -> Result<BacktestArtifacts, Box<dyn Error + Send + Sync>> {
    let uri_path = Path::new(uri);

    info!(strategy = %strategy_name, strategy_info = %strategy.get_info(), uri = %uri, exists = uri_path.exists(), "Starting strategy backtest");
    info!("Verified historical market data access from: {}", uri);

    let (data_format, loaded_events, replay_summary) = load_historical_market_events(uri).await?;
    let trading_stream: Pin<Box<dyn futures_util::Stream<Item = MarketEvent> + Send>> = match data_format {
        HistoricalMarketDataFormat::TradeCsv => {
            info!(uri = %uri, source = "historical_trade_csv", "Using trade-only historical replay source");
            Box::pin(futures_util::stream::iter(loaded_events))
        }
        HistoricalMarketDataFormat::CapturedJsonl => {
            info!(uri = %uri, source = "captured_market_jsonl", "Using captured mixed-event replay source");
            Box::pin(futures_util::stream::iter(loaded_events))
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
            strategy,
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
    let summary = BacktestSummary {
        strategy: strategy_name.to_string(),
        uri: uri.to_string(),
        market_data_format: match data_format {
            HistoricalMarketDataFormat::TradeCsv => "trade_csv".to_string(),
            HistoricalMarketDataFormat::CapturedJsonl => "captured_jsonl".to_string(),
        },
        regime_tags: infer_regime_tags(data_format, binance_trader.get_trade_manager().get_total_ticks(), replay_summary.as_ref()),
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
    };

    let trade_rows = metrics
        .get_trades()
        .iter()
        .map(|trade| TradeAttributionRow {
            strategy: strategy_name.to_string(),
            uri: uri.to_string(),
            trade_id: trade.trade_id,
            symbol: trade.symbol.clone(),
            signal: format!("{:?}", trade.signal).to_lowercase(),
            timestamp: trade.timestamp,
            price: trade.price,
            quantity: trade.quantity,
            pnl: trade.pnl.unwrap_or(0.0),
            gross_pnl: trade.gross_pnl.unwrap_or(0.0),
            fee_paid: trade.fee_paid,
            expected_edge_bps: trade.expected_edge_bps,
            rationale: trade.rationale.unwrap_or("").to_string(),
            decision_confidence: trade.decision_confidence,
            requested_quantity: trade.requested_quantity,
            executed_quantity: trade.executed_quantity,
            synthetic_half_spread_bps: trade.synthetic_half_spread_bps,
            slippage_bps: trade.slippage_bps,
            latency_impact_bps: trade.latency_impact_bps,
            market_impact_bps: trade.market_impact_bps,
            hold_time_millis: trade.hold_time_millis.unwrap_or(0),
            exit_reason: trade.exit_reason.unwrap_or("").to_string(),
            entry_price: trade.entry_price.unwrap_or(0.0),
        })
        .collect();

    let diagnostics = strategy.diagnostics();
    let mut diagnostic_rows = Vec::new();
    for (key, value) in diagnostics.counters {
        diagnostic_rows.push(StrategyDiagnosticRow {
            strategy: strategy_name.to_string(),
            uri: uri.to_string(),
            key,
            value: value.to_string(),
            value_type: "counter".to_string(),
        });
    }
    for (key, value) in diagnostics.gauges {
        diagnostic_rows.push(StrategyDiagnosticRow {
            strategy: strategy_name.to_string(),
            uri: uri.to_string(),
            key,
            value: format!("{value:.10}"),
            value_type: "gauge".to_string(),
        });
    }

    Ok(BacktestArtifacts {
        summary,
        trade_rows,
        diagnostic_rows,
    })
}

async fn run_live_mode(
    trading_symbol: &str,
    strategy_name: &str,
    trade_mode: TradeMode,
    duration_secs: Option<u64>,
    config_path: &std::path::Path,
    strategy_config_override: Option<&str>,
    shutdown: tokio::sync::watch::Receiver<bool>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut binance_trader = BinanceTrader::new_with_config_path(config_path).await?;
    let mut strategy = build_strategy(strategy_name, strategy_config_override)?;
    let stream = BinanceClient::new().await?.market_event_stream(trading_symbol).await?;

    // Build a future that resolves when shutdown is requested
    let mut shutdown_rx = shutdown.clone();
    let shutdown_signal = async move {
        // Wait until value becomes true
        while !*shutdown_rx.borrow() {
            if shutdown_rx.changed().await.is_err() {
                break;
            }
        }
    };

    let trading_stream: Pin<Box<dyn futures_util::Stream<Item = MarketEvent> + Send + '_>> =
        if let Some(duration_secs) = duration_secs {
            // Stop on whichever comes first: duration or shutdown signal
            let deadline = tokio::time::sleep(Duration::from_secs(duration_secs));
            let stop = async move {
                tokio::select! {
                    _ = deadline => {},
                    _ = shutdown_signal => {},
                }
            };
            Box::pin(stream.take_until(stop))
        } else {
            Box::pin(stream.take_until(shutdown_signal))
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
    println!("strategy,uri,market_data_format,regime_tags,total_ticks,entries,closed_trades,realized_pnl,fees_paid,ending_cash,ending_equity,win_rate,profit_factor,avg_pnl_per_trade,max_drawdown");
    for result in results {
        println!(
            "{},{},{},{},{},{},{},{:.10},{:.10},{:.10},{:.10},{:.6},{:.6},{:.10},{:.10}",
            result.strategy,
            result.uri,
            result.market_data_format,
            result.regime_tags.join("|"),
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

fn aggregate_compare_results(results: &[BacktestSummary]) -> Vec<CompareAggregateResult> {
    let mut grouped: HashMap<String, CompareAggregateAccumulator> = HashMap::new();

    for result in results {
        let entry = grouped
            .entry(result.strategy.clone())
            .or_insert_with(|| CompareAggregateAccumulator {
                strategy: result.strategy.clone(),
                dataset_count: 0,
                total_ticks: 0,
                total_entries: 0,
                total_closed_trades: 0,
                total_realized_pnl: 0.0,
                total_fees_paid: 0.0,
                total_ending_cash: 0.0,
                total_ending_equity: 0.0,
                total_win_rate: 0.0,
                total_profit_factor: 0.0,
                total_avg_pnl_per_trade: 0.0,
                worst_max_drawdown: result.max_drawdown,
            });

        entry.dataset_count += 1;
        entry.total_ticks += result.total_ticks;
        entry.total_entries += result.entries;
        entry.total_closed_trades += result.closed_trades;
        entry.total_realized_pnl += result.realized_pnl;
        entry.total_fees_paid += result.fees_paid;
        entry.total_ending_cash += result.ending_cash;
        entry.total_ending_equity += result.ending_equity;
        entry.total_win_rate += result.win_rate;
        entry.total_profit_factor += result.profit_factor;
        entry.total_avg_pnl_per_trade += result.avg_pnl_per_trade;
        entry.worst_max_drawdown = entry.worst_max_drawdown.max(result.max_drawdown);
    }

    let mut aggregates: Vec<_> = grouped
        .into_values()
        .map(|entry| {
            let dataset_count = entry.dataset_count.max(1);
            CompareAggregateResult {
                strategy: entry.strategy,
                dataset_count: entry.dataset_count,
                total_ticks: entry.total_ticks,
                total_entries: entry.total_entries,
                total_closed_trades: entry.total_closed_trades,
                total_realized_pnl: entry.total_realized_pnl,
                total_fees_paid: entry.total_fees_paid,
                mean_realized_pnl: entry.total_realized_pnl / dataset_count as f64,
                mean_ending_cash: entry.total_ending_cash / dataset_count as f64,
                mean_ending_equity: entry.total_ending_equity / dataset_count as f64,
                mean_win_rate: entry.total_win_rate / dataset_count as f64,
                mean_profit_factor: entry.total_profit_factor / dataset_count as f64,
                mean_avg_pnl_per_trade: entry.total_avg_pnl_per_trade / dataset_count as f64,
                worst_max_drawdown: entry.worst_max_drawdown,
            }
        })
        .collect();

    aggregates.sort_by(|left, right| {
        cmp_f64_desc(left.total_realized_pnl, right.total_realized_pnl)
            .then_with(|| cmp_f64_desc(left.mean_realized_pnl, right.mean_realized_pnl))
            .then_with(|| cmp_f64_asc(left.worst_max_drawdown, right.worst_max_drawdown))
            .then_with(|| left.strategy.cmp(&right.strategy))
    });

    aggregates
}

fn print_compare_aggregate_report(results: &[CompareAggregateResult]) {
    println!("rank,strategy,dataset_count,total_ticks,total_entries,total_closed_trades,total_realized_pnl,total_fees_paid,mean_realized_pnl,mean_ending_cash,mean_ending_equity,mean_win_rate,mean_profit_factor,mean_avg_pnl_per_trade,worst_max_drawdown");
    for (index, result) in results.iter().enumerate() {
        println!(
            "{},{},{},{},{},{},{:.10},{:.10},{:.10},{:.10},{:.10},{:.6},{:.6},{:.10},{:.10}",
            index + 1,
            result.strategy,
            result.dataset_count,
            result.total_ticks,
            result.total_entries,
            result.total_closed_trades,
            result.total_realized_pnl,
            result.total_fees_paid,
            result.mean_realized_pnl,
            result.mean_ending_cash,
            result.mean_ending_equity,
            result.mean_win_rate,
            result.mean_profit_factor,
            result.mean_avg_pnl_per_trade,
            result.worst_max_drawdown,
        );
    }
}

fn print_parameter_search_report(results: &[ParameterSearchResult]) {
    println!("strategy,uri,parameter,value,total_ticks,entries,closed_trades,realized_pnl,fees_paid,ending_cash,ending_equity,win_rate,profit_factor,avg_pnl_per_trade,max_drawdown");
    for result in results {
        println!(
            "{},{},{},{:.10},{},{},{},{:.10},{:.10},{:.10},{:.10},{:.6},{:.6},{:.10},{:.10}",
            result.strategy,
            result.uri,
            result.parameter,
            result.value,
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

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn cmp_f64_asc(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn aggregate_parameter_search_results(
    results: &[ParameterSearchResult],
) -> Vec<ParameterOptimizationResult> {
    let mut grouped: HashMap<(String, String, u64), ParameterOptimizationAccumulator> =
        HashMap::new();

    for result in results {
        let key = (
            result.strategy.clone(),
            result.parameter.clone(),
            result.value.to_bits(),
        );
        let entry = grouped
            .entry(key)
            .or_insert_with(|| ParameterOptimizationAccumulator {
                strategy: result.strategy.clone(),
                parameter: result.parameter.clone(),
                value: result.value,
                dataset_count: 0,
                total_ticks: 0,
                total_entries: 0,
                total_closed_trades: 0,
                total_realized_pnl: 0.0,
                total_fees_paid: 0.0,
                total_ending_cash: 0.0,
                total_ending_equity: 0.0,
                total_win_rate: 0.0,
                total_profit_factor: 0.0,
                total_avg_pnl_per_trade: 0.0,
                worst_max_drawdown: result.max_drawdown,
            });

        entry.dataset_count += 1;
        entry.total_ticks += result.total_ticks;
        entry.total_entries += result.entries;
        entry.total_closed_trades += result.closed_trades;
        entry.total_realized_pnl += result.realized_pnl;
        entry.total_fees_paid += result.fees_paid;
        entry.total_ending_cash += result.ending_cash;
        entry.total_ending_equity += result.ending_equity;
        entry.total_win_rate += result.win_rate;
        entry.total_profit_factor += result.profit_factor;
        entry.total_avg_pnl_per_trade += result.avg_pnl_per_trade;
        entry.worst_max_drawdown = entry.worst_max_drawdown.max(result.max_drawdown);
    }

    let mut aggregates: Vec<_> = grouped
        .into_values()
        .map(|entry| {
            let dataset_count = entry.dataset_count.max(1);
            ParameterOptimizationResult {
                strategy: entry.strategy,
                parameter: entry.parameter,
                value: entry.value,
                dataset_count: entry.dataset_count,
                total_ticks: entry.total_ticks,
                total_entries: entry.total_entries,
                total_closed_trades: entry.total_closed_trades,
                total_realized_pnl: entry.total_realized_pnl,
                total_fees_paid: entry.total_fees_paid,
                mean_realized_pnl: entry.total_realized_pnl / dataset_count as f64,
                mean_ending_cash: entry.total_ending_cash / dataset_count as f64,
                mean_ending_equity: entry.total_ending_equity / dataset_count as f64,
                mean_win_rate: entry.total_win_rate / dataset_count as f64,
                mean_profit_factor: entry.total_profit_factor / dataset_count as f64,
                mean_avg_pnl_per_trade: entry.total_avg_pnl_per_trade / dataset_count as f64,
                worst_max_drawdown: entry.worst_max_drawdown,
            }
        })
        .collect();

    aggregates.sort_by(|left, right| {
        cmp_f64_desc(left.total_realized_pnl, right.total_realized_pnl)
            .then_with(|| cmp_f64_desc(left.mean_realized_pnl, right.mean_realized_pnl))
            .then_with(|| cmp_f64_asc(left.worst_max_drawdown, right.worst_max_drawdown))
            .then_with(|| cmp_f64_asc(left.value, right.value))
    });

    aggregates
}

fn print_parameter_optimization_report(results: &[ParameterOptimizationResult]) {
    println!("rank,strategy,parameter,value,dataset_count,total_ticks,total_entries,total_closed_trades,total_realized_pnl,total_fees_paid,mean_realized_pnl,mean_ending_cash,mean_ending_equity,mean_win_rate,mean_profit_factor,mean_avg_pnl_per_trade,worst_max_drawdown");
    for (index, result) in results.iter().enumerate() {
        println!(
            "{},{},{},{:.10},{},{},{},{},{:.10},{:.10},{:.10},{:.10},{:.10},{:.6},{:.6},{:.10},{:.10}",
            index + 1,
            result.strategy,
            result.parameter,
            result.value,
            result.dataset_count,
            result.total_ticks,
            result.total_entries,
            result.total_closed_trades,
            result.total_realized_pnl,
            result.total_fees_paid,
            result.mean_realized_pnl,
            result.mean_ending_cash,
            result.mean_ending_equity,
            result.mean_win_rate,
            result.mean_profit_factor,
            result.mean_avg_pnl_per_trade,
            result.worst_max_drawdown,
        );
    }
}

async fn run_parameter_search(
    trading_symbol: &str,
    spec: &ParameterSearchSpec,
    duration_secs: Option<u64>,
    config_path: &std::path::Path,
    strategy_config_override: Option<&str>,
) -> Result<Vec<ParameterSearchResult>, Box<dyn Error + Send + Sync>> {
    let mut results = Vec::new();

    for value in &spec.values {
        for uri in &spec.uris {
            let mut binance_trader = BinanceTrader::new_with_config_path(config_path).await?;
            let mut strategy = build_strategy_from_search_spec(&spec.strategy, &spec.parameter, *value, strategy_config_override)?;
            let artifacts = run_backtest_with_strategy(
                &mut binance_trader,
                strategy.as_mut(),
                &spec.strategy,
                uri,
                trading_symbol,
                duration_secs,
            )
            .await?;
            let summary = artifacts.summary;

            results.push(ParameterSearchResult {
                strategy: summary.strategy,
                uri: summary.uri,
                parameter: spec.parameter.clone(),
                value: *value,
                total_ticks: summary.total_ticks,
                entries: summary.entries,
                closed_trades: summary.closed_trades,
                realized_pnl: summary.realized_pnl,
                fees_paid: summary.fees_paid,
                ending_cash: summary.ending_cash,
                ending_equity: summary.ending_equity,
                win_rate: summary.win_rate,
                profit_factor: summary.profit_factor,
                avg_pnl_per_trade: summary.avg_pnl_per_trade,
                max_drawdown: summary.max_drawdown,
            });
        }
    }

    Ok(results)
}

async fn run_parameter_optimization(
    trading_symbol: &str,
    spec: &ParameterOptimizationSpec,
    duration_secs: Option<u64>,
    config_path: &std::path::Path,
    strategy_config_override: Option<&str>,
) -> Result<Vec<ParameterOptimizationResult>, Box<dyn Error + Send + Sync>> {
    let search_results = run_parameter_search(
        trading_symbol,
        &ParameterSearchSpec {
            strategy: spec.strategy.clone(),
            parameter: spec.parameter.clone(),
            values: spec.values.clone(),
            uris: spec.uris.clone(),
        },
        duration_secs,
        config_path,
        strategy_config_override,
    )
    .await?;

    Ok(aggregate_parameter_search_results(&search_results))
}

async fn run_walk_forward_validation(
    trading_symbol: &str,
    strategy: &str,
    parameter: &str,
    values: &[f64],
    uris: &[String],
    min_train_size: usize,
    test_size: usize,
    duration_secs: Option<u64>,
    config_path: &std::path::Path,
    strategy_config_override: Option<&str>,
) -> Result<(Vec<WalkForwardFoldResult>, WalkForwardSummary), Box<dyn Error + Send + Sync>> {
    let mut folds = Vec::new();
    let effective_test_size = test_size.max(1);

    if uris.len() < min_train_size + effective_test_size {
        return Err("not enough datasets for walk-forward validation".into());
    }

    let mut fold_index = 1;
    let mut train_end = min_train_size;
    while train_end + effective_test_size <= uris.len() {
        let train_uris = uris[..train_end].to_vec();
        let test_uris = uris[train_end..train_end + effective_test_size].to_vec();

        let train_results = run_parameter_optimization(
            trading_symbol,
            &ParameterOptimizationSpec {
                strategy: strategy.to_string(),
                parameter: parameter.to_string(),
                values: values.to_vec(),
                uris: train_uris,
            },
            duration_secs,
            config_path,
            strategy_config_override,
        )
        .await?;

        let Some(best_train) = train_results.first() else {
            return Err("walk-forward training produced no optimization results".into());
        };

        let test_results = run_parameter_search(
            trading_symbol,
            &ParameterSearchSpec {
                strategy: strategy.to_string(),
                parameter: parameter.to_string(),
                values: vec![best_train.value],
                uris: test_uris,
            },
            duration_secs,
            config_path,
            strategy_config_override,
        )
        .await?;
        let test_aggregate = aggregate_parameter_search_results(&test_results)
            .into_iter()
            .next()
            .ok_or("walk-forward test produced no aggregate results")?;

        folds.push(WalkForwardFoldResult {
            fold: fold_index,
            strategy: strategy.to_string(),
            parameter: parameter.to_string(),
            selected_value: best_train.value,
            train_dataset_count: best_train.dataset_count,
            test_dataset_count: test_aggregate.dataset_count,
            train_total_realized_pnl: best_train.total_realized_pnl,
            train_mean_realized_pnl: best_train.mean_realized_pnl,
            test_total_realized_pnl: test_aggregate.total_realized_pnl,
            test_mean_realized_pnl: test_aggregate.mean_realized_pnl,
            test_total_closed_trades: test_aggregate.total_closed_trades,
            test_mean_win_rate: test_aggregate.mean_win_rate,
            test_worst_max_drawdown: test_aggregate.worst_max_drawdown,
        });

        fold_index += 1;
        train_end += effective_test_size;
    }

    let fold_count = folds.len().max(1);
    let summary = WalkForwardSummary {
        strategy: strategy.to_string(),
        parameter: parameter.to_string(),
        fold_count: folds.len(),
        total_test_realized_pnl: folds.iter().map(|fold| fold.test_total_realized_pnl).sum(),
        mean_test_realized_pnl: folds
            .iter()
            .map(|fold| fold.test_mean_realized_pnl)
            .sum::<f64>()
            / fold_count as f64,
        total_test_closed_trades: folds.iter().map(|fold| fold.test_total_closed_trades).sum(),
        mean_test_win_rate: folds
            .iter()
            .map(|fold| fold.test_mean_win_rate)
            .sum::<f64>()
            / fold_count as f64,
        worst_test_max_drawdown: folds
            .iter()
            .map(|fold| fold.test_worst_max_drawdown)
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
    };

    Ok((folds, summary))
}

fn print_walk_forward_report(folds: &[WalkForwardFoldResult], summary: &WalkForwardSummary) {
    println!("fold,strategy,parameter,selected_value,train_dataset_count,test_dataset_count,train_total_realized_pnl,train_mean_realized_pnl,test_total_realized_pnl,test_mean_realized_pnl,test_total_closed_trades,test_mean_win_rate,test_worst_max_drawdown");
    for fold in folds {
        println!(
            "{},{},{},{:.10},{},{},{:.10},{:.10},{:.10},{:.10},{},{:.6},{:.10}",
            fold.fold,
            fold.strategy,
            fold.parameter,
            fold.selected_value,
            fold.train_dataset_count,
            fold.test_dataset_count,
            fold.train_total_realized_pnl,
            fold.train_mean_realized_pnl,
            fold.test_total_realized_pnl,
            fold.test_mean_realized_pnl,
            fold.test_total_closed_trades,
            fold.test_mean_win_rate,
            fold.test_worst_max_drawdown,
        );
    }
    println!(
        "summary,{},{},{},{:.10},{:.10},{},{:.6},{:.10}",
        summary.strategy,
        summary.parameter,
        summary.fold_count,
        summary.total_test_realized_pnl,
        summary.mean_test_realized_pnl,
        summary.total_test_closed_trades,
        summary.mean_test_win_rate,
        summary.worst_test_max_drawdown,
    );
}

fn print_trade_attribution_report(rows: &[TradeAttributionRow]) {
    println!("strategy,uri,trade_id,symbol,signal,timestamp,price,quantity,pnl,gross_pnl,fee_paid,expected_edge_bps,rationale,decision_confidence,requested_quantity,executed_quantity,synthetic_half_spread_bps,slippage_bps,latency_impact_bps,market_impact_bps,hold_time_millis,exit_reason,entry_price");
    for row in rows {
        println!(
            "{},{},{},{},{},{:.6},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{},{:.6},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{},{},{:.10}",
            row.strategy,
            row.uri,
            row.trade_id,
            row.symbol,
            row.signal,
            row.timestamp,
            row.price,
            row.quantity,
            row.pnl,
            row.gross_pnl,
            row.fee_paid,
            row.expected_edge_bps,
            row.rationale,
            row.decision_confidence,
            row.requested_quantity,
            row.executed_quantity,
            row.synthetic_half_spread_bps,
            row.slippage_bps,
            row.latency_impact_bps,
            row.market_impact_bps,
            row.hold_time_millis,
            row.exit_reason,
            row.entry_price,
        );
    }
}

fn print_strategy_diagnostics_report(rows: &[StrategyDiagnosticRow]) {
    println!("strategy,uri,key,value,value_type");
    for row in rows {
        println!("{},{},{},{},{}", row.strategy, row.uri, row.key, row.value, row.value_type);
    }
}

fn feature_event_kind(event: &MarketEvent) -> &'static str {
    match event {
        MarketEvent::Trade(_) => "trade",
        MarketEvent::BookTicker(_) => "book_ticker",
        MarketEvent::Depth(_) => "depth",
    }
}

fn feature_event_time(event: &MarketEvent, market_state: &MarketState) -> u64 {
    match event {
        MarketEvent::Trade(trade) => trade.trade_time,
        MarketEvent::BookTicker(book) => book.event_time,
        MarketEvent::Depth(depth) => depth.event_time,
    }
    .max(market_state.last_event_time_millis().unwrap_or(0))
}

fn compute_forward_return_bps(events: &[MarketEvent], index: usize, lookahead: usize) -> f64 {
    if lookahead == 0 {
        return 0.0;
    }

    let current_price = events[index]
        .clone();
    let current_price = match current_price {
        MarketEvent::Trade(trade) => trade.price,
        MarketEvent::BookTicker(book) => (book.bid.price + book.ask.price) / 2.0,
        MarketEvent::Depth(_) => return 0.0,
    };
    if current_price <= f64::EPSILON {
        return 0.0;
    }

    let forward_index = (index + lookahead).min(events.len().saturating_sub(1));
    let forward_price = match &events[forward_index] {
        MarketEvent::Trade(trade) => trade.price,
        MarketEvent::BookTicker(book) => (book.bid.price + book.ask.price) / 2.0,
        MarketEvent::Depth(_) => current_price,
    };

    (forward_price - current_price) / current_price * 10_000.0
}

async fn export_feature_rows(
    trading_symbol: &str,
    uri: &str,
    output: &str,
    lookahead_events: usize,
) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let (data_format, events, replay_summary) = load_historical_market_events(uri).await?;
    let mut market_state = MarketState::new(trading_symbol.to_string(), 2_000);
    let mut rows = Vec::new();

    for (index, event) in events.iter().enumerate() {
        market_state.apply(event);
        let stats = market_state.trade_window_stats();
        let last_price = market_state.last_price().unwrap_or(0.0);
        let mid_price = market_state.mid_price().unwrap_or(last_price);
        let microprice = market_state.microprice().unwrap_or(mid_price);
        let microprice_edge_bps = if mid_price <= f64::EPSILON {
            0.0
        } else {
            (microprice - mid_price) / mid_price * 10_000.0
        };

        rows.push(FeatureRow {
            symbol: trading_symbol.to_string(),
            uri: uri.to_string(),
            event_index: index,
            event_kind: feature_event_kind(event).to_string(),
            event_time_ms: feature_event_time(event, &market_state),
            has_quote: market_state.top_of_book().is_some(),
            has_depth: market_state.depth().is_some(),
            trade_count_window: stats.trade_count,
            trade_volume_window: stats.volume,
            trade_notional_window: stats.notional,
            last_price,
            mid_price,
            spread_bps: market_state.spread_bps().unwrap_or(0.0),
            microprice,
            microprice_edge_bps,
            order_book_imbalance: market_state.order_book_imbalance().unwrap_or(0.0),
            trade_flow_imbalance: market_state.trade_flow_imbalance(),
            recent_trade_flow_imbalance: market_state.recent_trade_flow_imbalance(8),
            trade_window_vwap: market_state.trade_window_vwap().unwrap_or(0.0),
            trade_window_low: market_state.trade_window_low_price().unwrap_or(0.0),
            trade_window_high: market_state.trade_window_high_price().unwrap_or(0.0),
            forward_return_bps: compute_forward_return_bps(&events, index, lookahead_events),
            regime_tags: infer_regime_tags(data_format, events.len(), replay_summary.as_ref()).join("|"),
        });
    }

    let output_path = Path::new(output);
    if let Some(parent) = output_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut csv = String::from(
        "symbol,uri,event_index,event_kind,event_time_ms,has_quote,has_depth,trade_count_window,trade_volume_window,trade_notional_window,last_price,mid_price,spread_bps,microprice,microprice_edge_bps,order_book_imbalance,trade_flow_imbalance,recent_trade_flow_imbalance,trade_window_vwap,trade_window_low,trade_window_high,forward_return_bps,regime_tags\n",
    );
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{:.10},{}\n",
            row.symbol,
            row.uri,
            row.event_index,
            row.event_kind,
            row.event_time_ms,
            row.has_quote,
            row.has_depth,
            row.trade_count_window,
            row.trade_volume_window,
            row.trade_notional_window,
            row.last_price,
            row.mid_price,
            row.spread_bps,
            row.microprice,
            row.microprice_edge_bps,
            row.order_book_imbalance,
            row.trade_flow_imbalance,
            row.recent_trade_flow_imbalance,
            row.trade_window_vwap,
            row.trade_window_low,
            row.trade_window_high,
            row.forward_return_bps,
            row.regime_tags,
        ));
    }
    tokio::fs::write(output_path, csv).await?;
    Ok(events.len())
}

async fn run_capture(
    trading_symbol: &str,
    output_path: &str,
    duration_secs: u64,
    depth_levels: u32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let rotation = capture_rotation_config();
    if rotation.max_events_per_file.unwrap_or(0) > 0 {
        return run_capture_rotated(
            trading_symbol,
            output_path,
            duration_secs,
            depth_levels,
            rotation.max_events_per_file.unwrap_or(0),
            rotation.gzip.unwrap_or(false),
        )
        .await;
    }

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
    let integrity = BinanceClient::file_integrity_summary_from_path(output_path).await?;
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
        data_size_bytes: integrity.size_bytes,
        data_sha256: integrity.sha256_hex,
        capture_format: "binance_mixed_event_jsonl",
        ordering_semantics: "replay_in_capture_sequence_order",
        exchange_time_semantics: "exchange_timestamps_are_informative_not_authoritative_across_event_classes",
        capture_time_quality: capture_time_quality(first_captured_at_ms),
        quote_presence_quality: quote_presence_quality(book_ticker_events as usize, event_count as usize),
        depth_presence_quality: depth_presence_quality(depth_events as usize, event_count as usize),
    };
    let metadata_path = capture_metadata_path(output);
    let metadata_json = serde_json::to_vec_pretty(&metadata)?;
    tokio::fs::write(&metadata_path, metadata_json).await?;
    info!(symbol = %trading_symbol, output = %output_path, events = event_count, "Finished raw market capture");
    info!(symbol = %trading_symbol, metadata_path = %metadata_path, events = event_count, "Wrote capture metadata sidecar");
    Ok(())
}

async fn run_capture_rotated(
    trading_symbol: &str,
    output_path: &str,
    duration_secs: u64,
    depth_levels: u32,
    max_events_per_file: u64,
    gzip: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let client = BinanceClient::new().await?;
    let stream = client
        .market_event_stream_with_depth(trading_symbol, depth_levels)
        .await?;
    let mut stream = Box::pin(stream.take_until(tokio::time::sleep(Duration::from_secs(duration_secs))));
    let base_output = Path::new(output_path);
    if let Some(parent) = base_output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut part = 1_usize;
    let mut current_records: Vec<CaptureRecord> = Vec::new();

    while let Some(event) = stream.next().await {
        let capture_sequence = (current_records.len() + 1) as u64;
        let captured_at_ms = current_time_millis();
        let record = match event {
            MarketEvent::Trade(trade) => CaptureRecord::Trade {
                capture_sequence,
                captured_at_ms,
                exchange_event_time: Some(trade.event_time),
                symbol: trade.symbol,
                trade_id: trade.trade_id,
                price: trade.price,
                quantity: trade.quantity,
                trade_time: trade.trade_time,
                is_buyer_market_maker: trade.is_buyer_market_maker,
            },
            MarketEvent::BookTicker(book) => CaptureRecord::BookTicker {
                capture_sequence,
                captured_at_ms,
                exchange_event_time: None,
                update_id: None,
                bid_price: book.bid.price,
                bid_quantity: book.bid.quantity,
                ask_price: book.ask.price,
                ask_quantity: book.ask.quantity,
            },
            MarketEvent::Depth(depth) => CaptureRecord::Depth {
                capture_sequence,
                captured_at_ms,
                exchange_event_time: None,
                last_update_id: None,
                bids: depth
                    .bids
                    .into_iter()
                    .map(|level| CaptureDepthLevel { price: level.price, quantity: level.quantity })
                    .collect(),
                asks: depth
                    .asks
                    .into_iter()
                    .map(|level| CaptureDepthLevel { price: level.price, quantity: level.quantity })
                    .collect(),
            },
        };
        current_records.push(record);

        if current_records.len() as u64 >= max_events_per_file {
            flush_rotated_capture_part(
                trading_symbol,
                output_path,
                part,
                duration_secs,
                depth_levels,
                gzip,
                &current_records,
            )
            .await?;
            current_records.clear();
            part += 1;
        }
    }

    if !current_records.is_empty() {
        flush_rotated_capture_part(
            trading_symbol,
            output_path,
            part,
            duration_secs,
            depth_levels,
            gzip,
            &current_records,
        )
        .await?;
    }

    Ok(())
}

async fn flush_rotated_capture_part(
    trading_symbol: &str,
    base_output_path: &str,
    part: usize,
    duration_secs: u64,
    depth_levels: u32,
    gzip: bool,
    records: &[CaptureRecord],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let output_path = rotated_capture_output_path(base_output_path, part, gzip);
    let path = Path::new(&output_path);
    let mut payload = String::new();
    let mut trade_events = 0_u64;
    let mut book_ticker_events = 0_u64;
    let mut depth_events = 0_u64;
    let mut first_captured_at_ms = None;
    let mut last_captured_at_ms = None;

    for record in records {
        match record {
            CaptureRecord::Trade { captured_at_ms, .. } => {
                trade_events += 1;
                first_captured_at_ms = Some(first_captured_at_ms.map_or(*captured_at_ms, |cur: u64| cur.min(*captured_at_ms)));
                last_captured_at_ms = Some(last_captured_at_ms.map_or(*captured_at_ms, |cur: u64| cur.max(*captured_at_ms)));
            }
            CaptureRecord::BookTicker { captured_at_ms, .. } => {
                book_ticker_events += 1;
                first_captured_at_ms = Some(first_captured_at_ms.map_or(*captured_at_ms, |cur: u64| cur.min(*captured_at_ms)));
                last_captured_at_ms = Some(last_captured_at_ms.map_or(*captured_at_ms, |cur: u64| cur.max(*captured_at_ms)));
            }
            CaptureRecord::Depth { captured_at_ms, .. } => {
                depth_events += 1;
                first_captured_at_ms = Some(first_captured_at_ms.map_or(*captured_at_ms, |cur: u64| cur.min(*captured_at_ms)));
                last_captured_at_ms = Some(last_captured_at_ms.map_or(*captured_at_ms, |cur: u64| cur.max(*captured_at_ms)));
            }
        }
        payload.push_str(&serde_json::to_string(record)?);
        payload.push('\n');
    }

    if gzip {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(payload.as_bytes())?;
        let compressed = encoder.finish()?;
        tokio::fs::write(path, compressed).await?;
    } else {
        tokio::fs::write(path, payload.as_bytes()).await?;
    }

    let integrity = BinanceClient::file_integrity_summary_from_path(&output_path).await?;
    let metadata = CaptureMetadata {
        schema_version: 1,
        symbol: trading_symbol.to_string(),
        output_path: output_path.clone(),
        data_path: output_path.clone(),
        generated_at_ms: current_time_millis(),
        capture_duration_secs: duration_secs,
        depth_levels,
        total_events: records.len() as u64,
        trade_events,
        book_ticker_events,
        depth_events,
        first_capture_sequence: if records.is_empty() { None } else { Some(1) },
        last_capture_sequence: if records.is_empty() { None } else { Some(records.len() as u64) },
        first_captured_at_ms,
        last_captured_at_ms,
        data_size_bytes: integrity.size_bytes,
        data_sha256: integrity.sha256_hex,
        capture_format: if gzip { "binance_mixed_event_jsonl_gzip" } else { "binance_mixed_event_jsonl" },
        ordering_semantics: "replay_in_capture_sequence_order",
        exchange_time_semantics: "exchange_timestamps_are_informative_not_authoritative_across_event_classes",
        capture_time_quality: capture_time_quality(first_captured_at_ms),
        quote_presence_quality: quote_presence_quality(book_ticker_events as usize, records.len()),
        depth_presence_quality: depth_presence_quality(depth_events as usize, records.len()),
    };
    let metadata_path = capture_metadata_path(path);
    let metadata_json = serde_json::to_vec_pretty(&metadata)?;
    tokio::fs::write(&metadata_path, metadata_json).await?;
    Ok(())
}

async fn run_capture_batch(
    trading_symbol: &str,
    output_dir: Option<&str>,
    batch_id: &str,
    parts: usize,
    duration_secs: u64,
    depth_levels: u32,
    gap_secs: u64,
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
        if gap_secs > 0 && sequence < parts {
            tokio::time::sleep(Duration::from_secs(gap_secs)).await;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BacktestSummary, CapturedDatasetFilter, CapturedDatasetIndex, CapturedDatasetIndexEntry,
        Cli, Commands, HistoricalMarketDataFormat, ParameterSearchResult, ReplayDatasetSummary,
        aggregate_compare_results, aggregate_parameter_search_results, apply_captured_dataset_filter,
        build_strategy_from_search_spec, detect_historical_market_data_format,
        infer_regime_tags, parse_parameter_values,
    };
    use clap::Parser;

    #[test]
    fn detects_captured_jsonl_replay_inputs() {
        assert_eq!(
            detect_historical_market_data_format("/tmp/pulsar_capture.jsonl"),
            HistoricalMarketDataFormat::CapturedJsonl
        );
        assert_eq!(
            detect_historical_market_data_format("/tmp/pulsar_capture.jsonl.gz"),
            HistoricalMarketDataFormat::CapturedJsonl
        );
        assert_eq!(
            detect_historical_market_data_format("https://example.com/replay.ndjson?download=1"),
            HistoricalMarketDataFormat::CapturedJsonl
        );
        assert_eq!(
            detect_historical_market_data_format("https://example.com/replay.ndjson.gz?download=1"),
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
                data_size_bytes: 123,
                data_sha256: "abc".to_string(),
                sidecar_verified: true,
                capture_time_quality: "explicit".to_string(),
                quote_presence_quality: "dense".to_string(),
                depth_presence_quality: "mixed".to_string(),
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
                data_size_bytes: 1,
                data_sha256: "a".to_string(),
                sidecar_verified: true,
                capture_time_quality: "explicit".to_string(),
                quote_presence_quality: "none".to_string(),
                depth_presence_quality: "none".to_string(),
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
                data_size_bytes: 1,
                data_sha256: "b".to_string(),
                sidecar_verified: true,
                capture_time_quality: "explicit".to_string(),
                quote_presence_quality: "none".to_string(),
                depth_presence_quality: "none".to_string(),
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
            128,
            "abc123".to_string(),
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
                    data_size_bytes: 10,
                    data_sha256: "tradeonly".to_string(),
                    sidecar_verified: false,
                    capture_time_quality: "missing".to_string(),
                    quote_presence_quality: "none".to_string(),
                    depth_presence_quality: "none".to_string(),
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
                    data_size_bytes: 20,
                    data_sha256: "mixed".to_string(),
                    sidecar_verified: true,
                    capture_time_quality: "explicit".to_string(),
                    quote_presence_quality: "dense".to_string(),
                    depth_presence_quality: "mixed".to_string(),
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

    #[test]
    fn parses_parameter_values_from_cli_input() {
        let values = parse_parameter_values("1.0, 2.5,3.75").expect("values parse");
        assert_eq!(values, vec![1.0, 2.5, 3.75]);
    }

    #[test]
    fn parameter_search_strategy_builds_with_overridden_value() {
        let strategy = build_strategy_from_search_spec(
            "trade-flow-momentum",
            "min_price_drift_bps",
            9.0,
            None,
        )
        .expect("strategy builds from overridden config");

        assert!(strategy.get_info().contains("TradeFlowMomentumStrategy"));
    }

    #[test]
    fn maker_strategy_builds_from_overridden_config() {
        let strategy = build_strategy_from_search_spec(
            "microprice-imbalance-maker",
            "min_microprice_edge_bps",
            2.0,
            None,
        )
        .expect("maker strategy builds from overridden config");

        assert!(strategy.get_info().contains("MicropriceImbalanceMakerStrategy"));
    }

    #[test]
    fn parameter_search_results_aggregate_and_rank_by_value() {
        let results = vec![
            ParameterSearchResult {
                strategy: "liquidity-sweep-reversal".to_string(),
                uri: "day_a".to_string(),
                parameter: "min_sweep_drop_bps".to_string(),
                value: 8.0,
                total_ticks: 100,
                entries: 3,
                closed_trades: 3,
                realized_pnl: -0.1,
                fees_paid: 0.01,
                ending_cash: 99.9,
                ending_equity: 99.9,
                win_rate: 0.33,
                profit_factor: 0.8,
                avg_pnl_per_trade: -0.0333333333,
                max_drawdown: 0.2,
            },
            ParameterSearchResult {
                strategy: "liquidity-sweep-reversal".to_string(),
                uri: "day_b".to_string(),
                parameter: "min_sweep_drop_bps".to_string(),
                value: 8.0,
                total_ticks: 120,
                entries: 2,
                closed_trades: 2,
                realized_pnl: 0.05,
                fees_paid: 0.02,
                ending_cash: 100.05,
                ending_equity: 100.05,
                win_rate: 0.5,
                profit_factor: 1.1,
                avg_pnl_per_trade: 0.025,
                max_drawdown: 0.3,
            },
            ParameterSearchResult {
                strategy: "liquidity-sweep-reversal".to_string(),
                uri: "day_a".to_string(),
                parameter: "min_sweep_drop_bps".to_string(),
                value: 10.0,
                total_ticks: 110,
                entries: 1,
                closed_trades: 1,
                realized_pnl: 0.2,
                fees_paid: 0.01,
                ending_cash: 100.2,
                ending_equity: 100.2,
                win_rate: 1.0,
                profit_factor: 1.5,
                avg_pnl_per_trade: 0.2,
                max_drawdown: 0.1,
            },
        ];

        let aggregates = aggregate_parameter_search_results(&results);

        assert_eq!(aggregates.len(), 2);
        assert_eq!(aggregates[0].value, 10.0);

        let value_eight = aggregates
            .iter()
            .find(|result| result.value == 8.0)
            .expect("8.0 aggregate exists");
        assert_eq!(value_eight.dataset_count, 2);
        assert_eq!(value_eight.total_ticks, 220);
        assert_eq!(value_eight.total_entries, 5);
        assert!((value_eight.total_realized_pnl + 0.05).abs() < 1e-9);
        assert!((value_eight.mean_realized_pnl + 0.025).abs() < 1e-9);
        assert!((value_eight.worst_max_drawdown - 0.3).abs() < 1e-9);
    }

    #[test]
    fn compare_results_aggregate_and_rank_by_strategy() {
        let results = vec![
            BacktestSummary {
                strategy: "alpha".to_string(),
                uri: "day_a".to_string(),
                market_data_format: "trade_csv".to_string(),
                regime_tags: vec!["source:trade_only".to_string()],
                total_ticks: 10,
                entries: 1,
                closed_trades: 1,
                realized_pnl: 0.3,
                fees_paid: 0.01,
                ending_cash: 100.3,
                ending_equity: 100.3,
                win_rate: 1.0,
                profit_factor: 1.5,
                avg_pnl_per_trade: 0.3,
                max_drawdown: 0.1,
            },
            BacktestSummary {
                strategy: "alpha".to_string(),
                uri: "day_b".to_string(),
                market_data_format: "trade_csv".to_string(),
                regime_tags: vec!["source:trade_only".to_string()],
                total_ticks: 12,
                entries: 2,
                closed_trades: 2,
                realized_pnl: -0.1,
                fees_paid: 0.02,
                ending_cash: 99.9,
                ending_equity: 99.9,
                win_rate: 0.0,
                profit_factor: 0.8,
                avg_pnl_per_trade: -0.05,
                max_drawdown: 0.2,
            },
            BacktestSummary {
                strategy: "beta".to_string(),
                uri: "day_a".to_string(),
                market_data_format: "trade_csv".to_string(),
                regime_tags: vec!["source:trade_only".to_string()],
                total_ticks: 11,
                entries: 1,
                closed_trades: 1,
                realized_pnl: -0.05,
                fees_paid: 0.01,
                ending_cash: 99.95,
                ending_equity: 99.95,
                win_rate: 0.0,
                profit_factor: 0.9,
                avg_pnl_per_trade: -0.05,
                max_drawdown: 0.05,
            },
        ];

        let aggregates = aggregate_compare_results(&results);
        assert_eq!(aggregates.len(), 2);
        assert_eq!(aggregates[0].strategy, "alpha");
        assert_eq!(aggregates[0].dataset_count, 2);
        assert!((aggregates[0].total_realized_pnl - 0.2).abs() < 1e-9);
    }

    #[test]
    fn inferred_regime_tags_capture_source_activity_and_quote_coverage() {
        let summary = ReplayDatasetSummary {
            total_events: 100,
            trade_events: 20,
            book_ticker_events: 70,
            depth_events: 10,
            parse_errors: 0,
            first_event_time: Some(1),
            last_event_time: Some(2),
            event_time_regressions: 0,
            first_capture_sequence: Some(1),
            last_capture_sequence: Some(100),
            capture_sequence_regressions: 0,
            first_captured_at_ms: Some(1),
            last_captured_at_ms: Some(2),
            captured_at_regressions: 0,
            symbols: vec!["DOGEUSDT".to_string()],
        };

        let tags = infer_regime_tags(HistoricalMarketDataFormat::CapturedJsonl, 500_000, Some(&summary));
        assert!(tags.iter().any(|tag| tag == "source:captured"));
        assert!(tags.iter().any(|tag| tag == "activity:medium"));
        assert!(tags.iter().any(|tag| tag == "book:depth"));
        assert!(tags.iter().any(|tag| tag == "quote_coverage:mixed"));
    }

    #[test]
    fn dataset_filter_can_require_quote_quality_and_verified_sidecar() {
        let index = CapturedDatasetIndex {
            root: "data/binance/capture".to_string(),
            dataset_count: 2,
            datasets: vec![
                CapturedDatasetIndexEntry {
                    metadata_path: "a.metadata.json".to_string(),
                    data_path: "a.jsonl".to_string(),
                    symbol: "DOGEUSDT".to_string(),
                    total_events: 10,
                    trade_events: 1,
                    book_ticker_events: 0,
                    depth_events: 0,
                    capture_duration_secs: 1,
                    depth_levels: 5,
                    first_captured_at_ms: Some(1),
                    last_captured_at_ms: Some(2),
                    data_size_bytes: 1,
                    data_sha256: "a".to_string(),
                    sidecar_verified: false,
                    capture_time_quality: "explicit".to_string(),
                    quote_presence_quality: "none".to_string(),
                    depth_presence_quality: "none".to_string(),
                },
                CapturedDatasetIndexEntry {
                    metadata_path: "b.metadata.json".to_string(),
                    data_path: "b.jsonl".to_string(),
                    symbol: "DOGEUSDT".to_string(),
                    total_events: 100,
                    trade_events: 20,
                    book_ticker_events: 70,
                    depth_events: 10,
                    capture_duration_secs: 5,
                    depth_levels: 5,
                    first_captured_at_ms: Some(3),
                    last_captured_at_ms: Some(4),
                    data_size_bytes: 2,
                    data_sha256: "b".to_string(),
                    sidecar_verified: true,
                    capture_time_quality: "explicit".to_string(),
                    quote_presence_quality: "mixed".to_string(),
                    depth_presence_quality: "mixed".to_string(),
                },
            ],
        };

        let filtered = apply_captured_dataset_filter(
            index,
            &CapturedDatasetFilter {
                min_quote_quality: Some("mixed".to_string()),
                require_verified_sidecar: true,
                ..CapturedDatasetFilter::default()
            },
        );

        assert_eq!(filtered.dataset_count, 1);
        assert_eq!(filtered.datasets[0].data_path, "b.jsonl");
    }

    #[test]
    fn capture_index_cli_accepts_quality_filters() {
        let cli = Cli::parse_from([
            "binance-bot",
            "capture-index",
            "--min-quote-quality",
            "mixed",
            "--min-depth-quality",
            "mixed",
            "--require-verified-sidecar",
        ]);

        match cli.command {
            Commands::CaptureIndex {
                min_quote_quality,
                min_depth_quality,
                require_verified_sidecar,
                ..
            } => {
                assert_eq!(min_quote_quality.as_deref(), Some("mixed"));
                assert_eq!(min_depth_quality.as_deref(), Some("mixed"));
                assert!(require_verified_sidecar);
            }
            _ => panic!("expected capture-index command"),
        }
    }

    #[test]
    fn capture_compare_cli_accepts_quality_filters() {
        let cli = Cli::parse_from([
            "binance-bot",
            "capture-compare",
            "--min-quote-quality",
            "full",
            "--min-depth-quality",
            "mixed",
            "--require-verified-sidecar",
        ]);

        match cli.command {
            Commands::CaptureCompare {
                min_quote_quality,
                min_depth_quality,
                require_verified_sidecar,
                ..
            } => {
                assert_eq!(min_quote_quality.as_deref(), Some("full"));
                assert_eq!(min_depth_quality.as_deref(), Some("mixed"));
                assert!(require_verified_sidecar);
            }
            _ => panic!("expected capture-compare command"),
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    Trade,
    Emulate,
    Rebalance,
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
    Search {
        #[arg(long)]
        strategy: String,

        #[arg(long)]
        parameter: String,

        #[arg(long)]
        values: String,

        #[arg(short, long, required = true, num_args = 1..)]
        uris: Vec<String>,
    },
    Optimize {
        #[arg(long)]
        strategy: String,

        #[arg(long)]
        parameter: String,

        #[arg(long)]
        values: String,

        #[arg(short, long, required = true, num_args = 1..)]
        uris: Vec<String>,
    },
    WalkForward {
        #[arg(long)]
        strategy: String,

        #[arg(long)]
        parameter: String,

        #[arg(long)]
        values: String,

        #[arg(short, long, required = true, num_args = 1..)]
        uris: Vec<String>,

        #[arg(long, default_value_t = 2)]
        min_train_size: usize,

        #[arg(long, default_value_t = 1)]
        test_size: usize,
    },
    Features {
        #[arg(short, long)]
        uri: String,

        #[arg(short, long)]
        output: String,

        #[arg(long, default_value_t = 32)]
        lookahead_events: usize,
    },
    TradeAttribution {
        #[arg(short, long, required = true, num_args = 1..)]
        uris: Vec<String>,

        #[arg(long, value_delimiter = ',', default_value = "trade-flow-momentum,liquidity-sweep-reversal")]
        strategies: Vec<String>,
    },
    StrategyDiagnostics {
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

        #[arg(long)]
        min_quote_quality: Option<String>,

        #[arg(long)]
        min_depth_quality: Option<String>,

        #[arg(long, default_value_t = false)]
        require_verified_sidecar: bool,
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
        min_quote_quality: Option<String>,

        #[arg(long)]
        min_depth_quality: Option<String>,

        #[arg(long, default_value_t = false)]
        require_verified_sidecar: bool,

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

        #[arg(long, default_value_t = 0)]
        gap_secs: u64,
    },
    Capture {
        #[arg(short, long)]
        output: String,

        #[arg(long, default_value_t = 5)]
        depth_levels: u32,

        #[arg(long)]
        duration_secs: u64,
    },
    /// Run live trading on multiple pairs simultaneously (TOP 5 portfolio)
    MultiTrade,
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

    // Graceful shutdown: Ctrl+C sends signal to all trading tasks
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("\n=== Ctrl+C received, shutting down gracefully... ===");
        let _ = shutdown_tx.send(true);
    });

    // Load trading configuration (supports per-symbol configs via --config flag)
    let config_path = resolve_workspace_path(&cli.config);
    let trading_config =
        load_config(&config_path).unwrap_or_else(|e| panic!("Failed to load trading configuration from {}: {}", config_path.display(), e));

    let trading_symbol: String = get_config_value(&trading_config, "position_sizing.trading_symbol")
        .ok_or("Trading symbol not defined in configuration")?;

    let strategy_config_ref = cli.strategy_config.as_deref();

    match cli.command {
        Commands::Rebalance => {
            let mut trader = BinanceTrader::new_with_config_path(&config_path).await?;
            trader.connect().await?;
            let quote_asset = trader.config.exchange.quote_asset.clone();
            let base_asset = trading_symbol.strip_suffix(quote_asset.as_str()).unwrap_or(&trading_symbol);
            let (quote_before, base_before) = trader.account_balances(base_asset).await?;
            println!("Before: {:.8} {}, {:.4} {}", base_before, base_asset, quote_before, quote_asset);

            match trader.rebalance_half_and_half(&trading_symbol, base_asset).await {
                Ok(rebalanced) => {
                    let (quote_after, base_after) = trader.account_balances(base_asset).await?;
                    println!("Rebalanced: {}", rebalanced);
                    println!("After:  {:.8} {}, {:.4} {}", base_after, base_asset, quote_after, quote_asset);
                }
                Err(e) => println!("Rebalance failed: {}", e),
            }
        }
        Commands::Trade => {
            run_live_mode(&trading_symbol, &cli.strategy, TradeMode::Real, cli.duration_secs, &config_path, strategy_config_ref, shutdown_rx.clone()).await?;
        }
        Commands::Emulate => {
            run_live_mode(&trading_symbol, &cli.strategy, TradeMode::Emulated, cli.duration_secs, &config_path, strategy_config_ref, shutdown_rx.clone())
                .await?;
        }
        Commands::Backtest { uri } => {
            let _ = run_backtest(&trading_symbol, &cli.strategy, &uri, cli.duration_secs, &config_path, strategy_config_ref).await?;
        }
        Commands::Compare { uris, strategies } => {
            let mut results = Vec::new();

            for strategy_name in strategies {
                for uri in &uris {
                    results.push(
                        run_backtest(&trading_symbol, &strategy_name, uri, cli.duration_secs, &config_path, strategy_config_ref).await?,
                    );
                }
            }

            print_compare_report(&results);
            print_compare_aggregate_report(&aggregate_compare_results(&results));
        }
        Commands::Search {
            strategy,
            parameter,
            values,
            uris,
        } => {
            let spec = ParameterSearchSpec {
                strategy,
                parameter,
                values: parse_parameter_values(&values)?,
                uris,
            };
            let results = run_parameter_search(&trading_symbol, &spec, cli.duration_secs, &config_path, strategy_config_ref).await?;
            print_parameter_search_report(&results);
        }
        Commands::Optimize {
            strategy,
            parameter,
            values,
            uris,
        } => {
            let spec = ParameterOptimizationSpec {
                strategy,
                parameter,
                values: parse_parameter_values(&values)?,
                uris,
            };
            let results =
                run_parameter_optimization(&trading_symbol, &spec, cli.duration_secs, &config_path, strategy_config_ref).await?;
            print_parameter_optimization_report(&results);
        }
        Commands::WalkForward {
            strategy,
            parameter,
            values,
            uris,
            min_train_size,
            test_size,
        } => {
            let (folds, summary) = run_walk_forward_validation(
                &trading_symbol,
                &strategy,
                &parameter,
                &parse_parameter_values(&values)?,
                &uris,
                min_train_size,
                test_size,
                cli.duration_secs,
                &config_path,
                strategy_config_ref,
            )
            .await?;
            print_walk_forward_report(&folds, &summary);
        }
        Commands::Features {
            uri,
            output,
            lookahead_events,
        } => {
            let exported = export_feature_rows(&trading_symbol, &uri, &output, lookahead_events).await?;
            println!("exported_rows,{}", exported);
        }
        Commands::TradeAttribution { uris, strategies } => {
            let mut rows = Vec::new();
            for strategy_name in strategies {
                for uri in &uris {
                    let mut binance_trader = BinanceTrader::new_with_config_path(&config_path).await?;
                    let mut strategy = build_strategy(&strategy_name, strategy_config_ref)?;
                    let artifacts = run_backtest_with_strategy(
                        &mut binance_trader,
                        strategy.as_mut(),
                        &strategy_name,
                        uri,
                        &trading_symbol,
                        cli.duration_secs,
                    )
                    .await?;
                    rows.extend(artifacts.trade_rows);
                }
            }
            print_trade_attribution_report(&rows);
        }
        Commands::StrategyDiagnostics { uris, strategies } => {
            let mut rows = Vec::new();
            for strategy_name in strategies {
                for uri in &uris {
                    let mut binance_trader = BinanceTrader::new_with_config_path(&config_path).await?;
                    let mut strategy = build_strategy(&strategy_name, strategy_config_ref)?;
                    let artifacts = run_backtest_with_strategy(
                        &mut binance_trader,
                        strategy.as_mut(),
                        &strategy_name,
                        uri,
                        &trading_symbol,
                        cli.duration_secs,
                    )
                    .await?;
                    rows.extend(artifacts.diagnostic_rows);
                }
            }
            print_strategy_diagnostics_report(&rows);
        }
        Commands::CaptureIndex {
            root,
            symbol,
            min_total_events,
            min_book_ticker_events,
            min_depth_events,
            require_captured_at,
            since_captured_at_ms,
            min_quote_quality,
            min_depth_quality,
            require_verified_sidecar,
        } => {
            let filter = CapturedDatasetFilter {
                symbol,
                min_total_events,
                min_book_ticker_events,
                min_depth_events,
                require_captured_at,
                since_captured_at_ms,
                min_quote_quality,
                min_depth_quality,
                require_verified_sidecar,
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
            min_quote_quality,
            min_depth_quality,
            require_verified_sidecar,
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
                min_quote_quality,
                min_depth_quality,
                require_verified_sidecar,
            };
            let results = run_capture_compare(
                &trading_symbol,
                &root,
                &strategies,
                &filter,
                limit,
                cli.duration_secs,
                &config_path,
                strategy_config_ref,
            )
            .await?;
            print_compare_report(&results);
            print_compare_aggregate_report(&aggregate_compare_results(&results));
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
            gap_secs,
        } => {
            run_capture_batch(
                &trading_symbol,
                output_dir.as_deref(),
                &batch_id,
                parts,
                duration_secs,
                depth_levels,
                gap_secs,
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
        Commands::MultiTrade => {
            // TOP 5 profitable pairs from 8-day multi-day backtest
            let pairs: Vec<(&str, &str, &str)> = vec![
                ("TRUUSDT",  "config/trading_config_truusdt.toml",  "config/strategies/market_maker_truusdt.toml"),
                ("GLMRUSDT", "config/trading_config_glmrusdt.toml", "config/strategies/market_maker_glmrusdt.toml"),
                ("WIFUSDT",  "config/trading_config_wifusdt.toml",  "config/strategies/market_maker_wifusdt.toml"),
                ("QIUSDT",   "config/trading_config_qiusdt.toml",   "config/strategies/market_maker_qiusdt.toml"),
                ("PHBUSDT",  "config/trading_config_phbusdt.toml",  "config/strategies/market_maker_phbusdt.toml"),
            ];

            let duration = cli.duration_secs;
            let strategy_name = cli.strategy.clone();

            println!("=== MULTI-TRADE: launching {} pairs ===", pairs.len());
            for (symbol, tc, sc) in &pairs {
                println!("  {} | {} | {}", symbol, tc, sc);
            }

            let mut handles = Vec::new();
            for (symbol, trading_cfg, strategy_cfg) in pairs {
                let sym = symbol.to_string();
                let strat = strategy_name.clone();
                let tc = trading_cfg.to_string();
                let sc = strategy_cfg.to_string();
                let shutdown = shutdown_rx.clone();

                let handle = tokio::spawn(async move {
                    let cfg_path = resolve_workspace_path(&tc);
                    let result = run_live_mode(
                        &sym,
                        &strat,
                        TradeMode::Real,
                        duration,
                        &cfg_path,
                        Some(sc.as_str()),
                        shutdown,
                    )
                    .await;
                    if let Err(ref e) = result {
                        eprintln!("[{}] ERROR: {}", sym, e);
                    }
                    (sym, result)
                });
                handles.push(handle);
            }

            let mut errors = 0u32;
            for handle in handles {
                match handle.await {
                    Ok((sym, Ok(()))) => println!("[{}] finished OK", sym),
                    Ok((sym, Err(e))) => {
                        eprintln!("[{}] trading error: {}", sym, e);
                        errors += 1;
                    }
                    Err(e) => {
                        eprintln!("[multi-trade] task panicked: {}", e);
                        errors += 1;
                    }
                }
            }

            println!("=== MULTI-TRADE: done, {} error(s) ===", errors);
        }
    }

    Ok(())
}
