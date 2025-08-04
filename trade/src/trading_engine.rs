use crate::signal::Signal;
use crate::trader::{Position, Trader};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use toml;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub exchange: ExchangeConfig,
    pub slippage: SlippageConfig,
    pub order_execution: OrderExecutionConfig,
    pub risk_management: RiskManagementConfig,
    pub position_sizing: PositionSizingConfig,
    pub market_data: MarketDataConfig,
    pub performance_tracking: PerformanceTrackingConfig,
    pub backtest_settings: Option<BacktestSettingsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub taker_fee: f64,
    pub maker_fee: f64,
    pub maker_rebate: f64,
    pub tick_size: f64,
    pub min_notional: f64,
    pub max_order_size: f64,
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
pub struct PositionSizingConfig {
    // Pair-specific trade size limits (in base currency)
    // ALL PAIRS MUST BE EXPLICITLY DEFINED - no default fallback
    pub pairs: std::collections::HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub propagation_delay: f64,      // 10ms
    pub transmission_delay: f64,     // 2ms
    pub queuing_delay: f64,          // 5ms
    pub processing_delay: f64,       // 3ms
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
}



impl TradingConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Try multiple possible paths for the config file
        let possible_paths = [
            "config/trading_config.toml",
            "../../config/trading_config.toml",
            "../config/trading_config.toml",
        ];
        
        for config_path in &possible_paths {
            if let Ok(config_content) = fs::read_to_string(config_path) {
                if let Ok(config) = toml::from_str(&config_content) {
                    return Ok(config);
                }
            }
        }
        
        Err("Could not find or parse trading_config.toml".into())
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub total_fees: f64,
    pub total_rebates: f64,
    pub total_slippage: f64,
    pub average_fill_rate: f64,
    pub consecutive_losses: usize,
    pub max_drawdown: f64,
    pub peak_equity: f64,
    pub current_equity: f64,
    pub trade_history: VecDeque<TradeRecord>,
}

#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub timestamp: f64,
    pub signal: Signal,
    pub price: f64,
    pub quantity: f64,
    pub fees: f64,
    pub slippage: f64,
    pub rebates: f64,
    pub pnl: f64,
    pub order_type: OrderType,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit,
    Maker,
    Taker,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            total_fees: 0.0,
            total_rebates: 0.0,
            total_slippage: 0.0,
            average_fill_rate: 0.0,
            consecutive_losses: 0,
            max_drawdown: 0.0,
            peak_equity: 0.0,
            current_equity: 0.0,
            trade_history: VecDeque::new(),
        }
    }

    pub fn update_equity(&mut self, new_equity: f64) {
        self.current_equity = new_equity;
        if new_equity > self.peak_equity {
            self.peak_equity = new_equity;
        }
        
        // Prevent division by zero when peak_equity is 0
        if self.peak_equity > 0.0 {
            let drawdown = (self.peak_equity - new_equity) / self.peak_equity;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }
    }

    pub fn record_trade(&mut self, record: TradeRecord) {
        self.total_trades += 1;
        self.total_fees += record.fees;
        self.total_slippage += record.slippage;
        self.total_rebates += record.rebates;
        
        if record.pnl > 0.0 {
            self.winning_trades += 1;
            self.consecutive_losses = 0;
        } else {
            self.losing_trades += 1;
            self.consecutive_losses += 1;
        }
        
        self.trade_history.push_back(record);
        if self.trade_history.len() > 1000 {
            self.trade_history.pop_front();
        }
    }

    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.winning_trades as f64 / self.total_trades as f64
        }
    }

    pub fn average_trade_pnl(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.trade_history.iter().map(|t| t.pnl).sum::<f64>() / self.total_trades as f64
        }
    }

    pub fn net_pnl_after_costs(&self) -> f64 {
        self.trade_history.iter().map(|t| t.pnl).sum::<f64>() - self.total_fees + self.total_rebates
    }
}

pub struct TradingEngine {
    pub position: Position,
    pub realized_pnl: f64,
    pub config: TradingConfig,
    pub metrics: PerformanceMetrics,
    pub last_trade_time: f64,
    pub is_circuit_breaker_active: bool,
    pub circuit_breaker_start: f64,
}

impl TradingEngine {
    pub fn new(symbol: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config = TradingConfig::load()?;
        
        Ok(Self {
            position: Position {
                symbol: symbol.to_string(),
                quantity: 0.0,
                entry_price: 0.0,
            },
            realized_pnl: 0.0,
            config,
            metrics: PerformanceMetrics::new(),
            last_trade_time: 0.0,
            is_circuit_breaker_active: false,
            circuit_breaker_start: 0.0,
        })
    }

    pub fn calculate_slippage(&self, _price: f64, quantity: f64, volatility: f64) -> f64 {
        let base_slippage = self.config.slippage.min_slippage;
        let max_slippage = self.config.slippage.max_slippage;
        
        // Calculate volatility-adjusted slippage
        let volatility_factor = volatility.min(1.0);
        let volatility_slippage = base_slippage + (max_slippage - base_slippage) * volatility_factor * self.config.slippage.volatility_multiplier;
        
        // Calculate size-adjusted slippage
        let size_factor = (quantity / self.config.exchange.max_order_size).min(1.0);
        let size_slippage = base_slippage + (max_slippage - base_slippage) * size_factor * self.config.slippage.size_multiplier;
        
        // Combine volatility and size effects
        let total_slippage = (volatility_slippage + size_slippage) / 2.0;
        total_slippage.min(max_slippage)
    }

    pub fn calculate_fees(&self, price: f64, quantity: f64, is_maker: bool) -> f64 {
        let fee_rate = if is_maker {
            self.config.exchange.maker_fee
        } else {
            self.config.exchange.taker_fee
        };
        
        price * quantity * fee_rate
    }

    pub fn calculate_rebates(&self, price: f64, quantity: f64) -> f64 {
        price * quantity * self.config.exchange.maker_rebate
    }

    pub fn should_trade(&mut self, _signal: &Signal, confidence: f64, _current_price: f64, timestamp: f64) -> bool {
        // Check circuit breaker
        if self.is_circuit_breaker_active {
            if timestamp - self.circuit_breaker_start > self.config.risk_management.cooldown_period as f64 {
                self.is_circuit_breaker_active = false; // Circuit breaker expired
                return true;
            }
            return false;
        }

        // Check signal strength
        if confidence < self.config.risk_management.min_signal_strength {
            return false;
        }

        // Check consecutive losses
        if self.metrics.consecutive_losses >= self.config.risk_management.max_consecutive_losses {
            self.is_circuit_breaker_active = true;
            self.circuit_breaker_start = timestamp;
            warn!("Circuit breaker activated due to consecutive losses");
            return false;
        }

        // Check drawdown
        if self.metrics.max_drawdown > self.config.risk_management.max_drawdown {
            warn!("Trading stopped due to maximum drawdown reached");
            return false;
        }

        // Check daily loss limit
        if self.realized_pnl < -self.config.risk_management.max_daily_loss {
            warn!("Trading stopped due to daily loss limit");
            return false;
        }

        true
    }

    pub fn determine_order_type(&self, _signal: &Signal, confidence: f64, spread: f64) -> OrderType {
        // Use maker orders for high confidence signals or when spread is wide
        if confidence > self.config.risk_management.min_signal_strength || spread > self.config.exchange.tick_size * 100.0 {
            OrderType::Maker
        } else {
            OrderType::Taker
        }
    }

    pub fn simulate_fill(&self, order_type: &OrderType, price: f64, quantity: f64) -> (f64, f64, f64) {
        let (fill_price, slippage, fill_quantity) = match order_type {
            OrderType::Market => {
                let slippage = self.calculate_slippage(price, quantity, self.config.slippage.market_order_volatility);
                let fill_price = price + slippage;
                (fill_price, slippage, quantity)
            }
            OrderType::Limit => {
                // Limit orders might not fill immediately
                if let Some(ref backtest_settings) = self.config.backtest_settings {
                    if rand::random::<f64>() < backtest_settings.limit_order_fill_rate {
                        (price, 0.0, quantity)
                    } else {
                        (price, 0.0, quantity * backtest_settings.partial_fill_ratio)
                    }
                } else {
                    // Default fallback
                    if rand::random::<f64>() < 0.7 {
                        (price, 0.0, quantity)
                    } else {
                        (price, 0.0, quantity * 0.5)
                    }
                }
            }
            OrderType::Maker => {
                // Maker orders get better prices
                let _rebate = self.calculate_rebates(price, quantity);
                (price, 0.0, quantity)
            }
            OrderType::Taker => {
                let slippage = self.calculate_slippage(price, quantity, self.config.slippage.taker_order_volatility);
                let fill_price = price + slippage;
                (fill_price, slippage, quantity)
            }
        };

        (fill_price, slippage, fill_quantity)
    }

    pub fn calculate_latency(&self) -> f64 {
        // Check if we have backtest settings for latency simulation
        if let Some(ref backtest_settings) = self.config.backtest_settings {
            if backtest_settings.account_for_latency_in_signals {
                // Calculate total latency with 20ms base
                let base_latency = backtest_settings.propagation_delay 
                    + backtest_settings.transmission_delay 
                    + backtest_settings.queuing_delay 
                    + backtest_settings.processing_delay
                    + backtest_settings.exchange_processing_time
                    + backtest_settings.order_matching_latency
                    + backtest_settings.market_data_latency
                    + backtest_settings.signal_generation_time
                    + backtest_settings.order_construction_time
                    + backtest_settings.risk_check_time;
                
                // Add latency spikes (1% probability)
                let latency = if rand::random::<f64>() < backtest_settings.latency_spike_probability {
                    base_latency * backtest_settings.latency_spike_multiplier
                } else {
                    base_latency
                };
                
                // Add jitter (±20% variation)
                let jitter = latency * (rand::random::<f64>() * 0.4 - 0.2);
                latency + jitter
            } else {
                0.0 // No latency simulation
            }
        } else {
            0.0 // No backtest settings, no latency simulation
        }
    }

    pub fn calculate_position_size(&self, symbol: &str, price: f64, confidence: f64, _available_capital: f64) -> f64 {
        // Get pair-specific maximum trade size limit (MUST be explicitly defined)
        let max_trade_size = self.config.position_sizing.pairs
            .get(symbol)
            .expect(&format!("No trade size limit defined for pair: {}", symbol));
        
        // Calculate dynamic position size based on confidence
        // Higher confidence = larger position, but never exceed max_trade_size
        let base_quantity = max_trade_size * 0.1; // Start with 10% of max limit
        let confidence_multiplier = 0.5 + (confidence * 0.5); // 0.5x to 1.0x based on confidence
        let dynamic_quantity = base_quantity * confidence_multiplier;
        
        // Ensure we never exceed the maximum trade size limit
        let quantity = dynamic_quantity.min(*max_trade_size);
        
        // Apply exchange minimum notional requirement
        let min_notional = self.config.exchange.min_notional;
        let position_value = quantity * price;
        if position_value < min_notional {
            let min_quantity = min_notional / price;
            return min_quantity.min(*max_trade_size); // Still respect max limit
        }
        
        // Apply tick size rounding
        let tick_size = self.config.exchange.tick_size;
        let quantity_step = tick_size;
        let rounded_quantity = (quantity / quantity_step).ceil() * quantity_step;
        
        // Final check to ensure we don't exceed max trade size after rounding
        rounded_quantity.min(*max_trade_size)
    }
}

#[async_trait::async_trait]
impl Trader for TradingEngine {
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64) {
        // This would be implemented for live trading
        // For now, we'll use on_emulate for backtesting
        self.on_emulate(signal, price, quantity).await;
    }

    async fn on_emulate(&mut self, signal: Signal, price: f64, quantity: f64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let order_type = self.determine_order_type(&signal, 0.8, self.config.exchange.tick_size * 10.0); // Use tick size for spread
        let is_maker = matches!(order_type, OrderType::Maker);
        
        let (fill_price, slippage, fill_quantity) = self.simulate_fill(&order_type, price, quantity);
        let fees = self.calculate_fees(fill_price, fill_quantity, is_maker);
        let rebates = if is_maker { self.calculate_rebates(fill_price, fill_quantity) } else { 0.0 };

        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 && fill_quantity > 0.0 {
                    let _cost = fill_price * fill_quantity + fees - rebates;
                    self.position.quantity = fill_quantity;
                    self.position.entry_price = fill_price;
                    
                    let record = TradeRecord {
                        timestamp,
                        signal: signal.clone(),
                        price: fill_price,
                        quantity: fill_quantity,
                        fees,
                        slippage,
                        rebates,
                        pnl: 0.0, // Will be calculated on sell
                        order_type: order_type.clone(),
                    };
                    
                    self.metrics.record_trade(record);
                    
                    info!(
                        "BUY: price={:.6}, qty={:.6}, fees={:.6}, rebates={:.6}, slippage={:.6}",
                        fill_price, fill_quantity, fees, rebates, slippage
                    );
                }
            }
            Signal::Sell => {
                if self.position.quantity > 0.0 && fill_quantity > 0.0 {
                    let sell_quantity = fill_quantity.min(self.position.quantity);
                    let revenue = fill_price * sell_quantity - fees + rebates;
                    let cost = self.position.entry_price * sell_quantity;
                    let pnl = revenue - cost;
                    
                    self.realized_pnl += pnl;
                    self.position.quantity -= sell_quantity;
                    
                    if self.position.quantity == 0.0 {
                        self.position.entry_price = 0.0;
                    }
                    
                    let record = TradeRecord {
                        timestamp,
                        signal: signal.clone(),
                        price: fill_price,
                        quantity: sell_quantity,
                        fees,
                        slippage,
                        rebates,
                        pnl,
                        order_type: order_type.clone(),
                    };
                    
                    self.metrics.record_trade(record);
                    self.metrics.update_equity(self.realized_pnl);
                    
                    info!(
                        "SELL: price={:.6}, qty={:.6}, pnl={:.6}, fees={:.6}, rebates={:.6}, slippage={:.6}",
                        fill_price, sell_quantity, pnl, fees, rebates, slippage
                    );
                }
            }
            Signal::Hold => {
                // Do nothing
            }
        }
    }

    fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.position.quantity > 0.0 {
            (current_price - self.position.entry_price) * self.position.quantity
        } else {
            0.0
        }
    }

    fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    fn position(&self) -> Position {
        self.position.clone()
    }

    async fn account_status(&self) -> Result<(), anyhow::Error> {
        // For backtesting, this is not needed
        Ok(())
    }
}

// Add rand dependency for random number generation
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    pub fn random<T>() -> T 
    where
        T: std::ops::Rem<f64, Output = T> + From<f64>,
    {
        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        let hash = hasher.finish();
        let random_f64 = (hash as f64) / (u64::MAX as f64);
        T::from(random_f64)
    }
} 