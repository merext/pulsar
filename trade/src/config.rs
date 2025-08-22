use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeConfig {
    pub position_sizing: PositionSizingConfig,
    pub exchange: ExchangeConfig,
    pub slippage: SlippageConfig,
    pub order_execution: OrderExecutionConfig,
    pub risk_management: RiskManagementConfig,
    pub market_data: MarketDataConfig,
    pub performance_tracking: PerformanceTrackingConfig,
    pub backtest_settings: Option<BacktestSettingsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizingConfig {
    pub trading_size_min: f64,
    pub trading_size_max: f64,
    pub trading_symbol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub taker_fee: f64,
    pub maker_fee: f64,
    pub maker_rebate: f64,
    pub tick_size: f64,
    pub min_notional: f64,
    pub max_order_size: f64,
    pub step_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    pub min_slippage: f64,
    pub max_slippage: f64,
    pub volatility_multiplier: f64,
    pub size_multiplier: f64,
    pub market_order_volatility: f64,
    pub taker_order_volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderExecutionConfig {
    pub order_timeout: f64,
    pub max_retries: u32,
    pub retry_delay: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    pub min_trade_size: f64,
    pub max_price_change: f64,
    pub outlier_threshold: f64,
    pub trading_hours_start: u8,
    pub trading_hours_end: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagementConfig {
    pub max_position_size: f64,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
    pub max_leverage: f64,
    pub min_signal_strength: f64,
    pub max_consecutive_losses: usize,
    pub cooldown_period: u64,
    pub max_trades_per_hour: u32,
    pub initial_margin: f64,
    pub maintenance_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct PerformanceTrackingConfig {
    pub track_fill_rate: bool,
    pub track_slippage: bool,
    pub track_fees: bool,
    pub track_rebates: bool,
    pub track_order_rejections: bool,
    pub calculate_sharpe_ratio: bool,
    pub calculate_sortino_ratio: bool,
    pub calculate_max_drawdown: bool,
    pub calculate_calmar_ratio: bool,
    pub calculate_profit_factor: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct BacktestSettingsConfig {
    // Simulation parameters
    pub initial_capital: f64,
    pub commission_model: String,
    pub slippage_model: String,
    pub fill_model: String,

    // Data handling
    pub handle_data_gaps: bool,
    pub interpolate_missing_data: bool,
    pub filter_outliers: bool,

    // Bias prevention
    pub prevent_lookahead_bias: bool,
    pub use_point_in_time_data: bool,

    // Simulation features
    pub simulate_market_microstructure: bool,
    pub account_for_latency_in_signals: bool,
    pub simulate_order_queue_position: bool,
    pub simulate_partial_fills: bool,
    pub simulate_order_cancellations: bool,

    // Market impact simulation
    pub simulate_market_impact: bool,
    pub market_impact_model: String,
    pub impact_decay_rate: f64,

    // Order execution simulation
    pub simulate_order_delays: bool,
    pub simulate_exchange_errors: bool,
    pub simulate_network_timeouts: bool,

    // Fill and rejection simulation
    pub limit_order_fill_rate: f64,
    pub partial_fill_probability: f64,
    pub partial_fill_ratio: f64,
    pub invalid_order_probability: f64,
    pub rejection_reasons: Vec<String>,

    // Market events simulation
    pub handle_market_halts: bool,
    pub halt_probability: f64,
    pub halt_duration_min: u64,
    pub halt_duration_max: u64,

    // Spread and market impact simulation
    pub base_spread: f64,
    pub spread_volatility: f64,
    pub spread_widening_factor: f64,
    pub order_book_depth: u32,
    pub market_impact_factor: f64,

    // Performance tracking simulation
    pub track_queue_position: bool,
    pub track_latency: bool,
    pub track_market_impact: bool,

    // Latency simulation (20ms base latency)
    pub total_latency_model: String,
    pub propagation_delay: f64,  // 10ms
    pub transmission_delay: f64, // 2ms
    pub queuing_delay: f64,      // 5ms
    pub processing_delay: f64,   // 3ms
    pub latency_spike_probability: f64,
    pub latency_spike_multiplier: f64,
    pub congestion_latency_factor: f64,
    pub exchange_processing_time: f64,
    pub order_matching_latency: f64,
    pub market_data_latency: f64,
    pub signal_generation_time: f64,
    pub order_construction_time: f64,
    pub risk_check_time: f64,

    // Market microstructure simulation
    pub order_book_simulation: bool,
    pub bid_ask_spread_model: String,
    pub spread_volatility_factor: f64,
    pub spread_widening_events: bool,
    pub order_book_levels: u32,
    pub level_size_distribution: String,
    pub min_level_size: f64,
    pub max_level_size: f64,
    pub market_maker_presence: bool,
    pub mm_spread_tightening: f64,
    pub mm_inventory_management: bool,
    pub mm_risk_aversion: f64,
    pub hft_competition: bool,
    pub hft_latency_advantage: f64,
    pub hft_order_cancellation_rate: f64,
    pub hft_market_impact: f64,
    pub disable_drawdown_limit: bool,
    pub max_drawdown_override: Option<f64>,
}

impl TradeConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::from_file("config/trading_config.toml")
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let content = std::fs::read_to_string(path)?;
        let config: TradeConfig = toml::from_str(&content)?;
        Ok(config)
    }
}
