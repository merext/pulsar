use crate::signal::Signal;
use crate::trader::{Position, Trader};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use toml;
use tracing::warn;
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
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

impl TradingConfig {
    /// # Errors
    ///
    /// Will return `Err` if the config file cannot be read or parsed.
    pub fn load() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Deprecated implicit loader; prefer from_file
        Self::from_file("config/trading_config.toml")
    }

    /// # Errors
    ///
    /// Will return `Err` if the config file cannot be read or parsed.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&config_content)?;
        Ok(config)
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

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMetrics {
    #[must_use]
    pub const fn new() -> Self {
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

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            self.winning_trades as f64 / self.total_trades as f64
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_trade_pnl(&self) -> f64 {
        if self.total_trades == 0 || self.trade_history.is_empty() {
            0.0
        } else {
            self.trade_history.iter().map(|t| t.pnl).sum::<f64>() / self.total_trades as f64
        }
    }

    #[must_use]
    pub fn net_pnl_after_costs(&self) -> f64 {
        if self.trade_history.is_empty() {
            0.0
        } else {
            self.trade_history.iter().map(|t| t.pnl).sum::<f64>()
        }
    }

    #[must_use]
    pub fn gross_pnl(&self) -> f64 {
        if self.trade_history.is_empty() {
            0.0
        } else {
            // Gross PnL excludes costs (fees, slippage, rebates)
            self.trade_history.iter()
                .map(|t| t.pnl + t.fees - t.rebates + t.slippage)
                .sum::<f64>()
        }
    }

    #[must_use]
    pub fn total_costs(&self) -> f64 {
        self.total_fees - self.total_rebates + self.total_slippage
    }

    #[must_use]
    pub fn profit_factor(&self) -> f64 {
        if self.trade_history.is_empty() {
            return 0.0;
        }

        let gross_profit: f64 = self.trade_history.iter()
            .filter(|t| t.pnl > 0.0)
            .map(|t| t.pnl)
            .sum();
        
        let gross_loss: f64 = self.trade_history.iter()
            .filter(|t| t.pnl < 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        
        if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY // All profitable trades
        } else {
            0.0 // No profitable trades
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_win(&self) -> f64 {
        if self.trade_history.is_empty() {
            return 0.0;
        }

        let winning_trades: Vec<&TradeRecord> = self.trade_history.iter()
            .filter(|t| t.pnl > 0.0)
            .collect();
        
        if winning_trades.is_empty() {
            0.0
        } else {
            winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn average_loss(&self) -> f64 {
        if self.trade_history.is_empty() {
            return 0.0;
        }

        let losing_trades: Vec<&TradeRecord> = self.trade_history.iter()
            .filter(|t| t.pnl < 0.0)
            .collect();
        
        if losing_trades.is_empty() {
            0.0
        } else {
            losing_trades.iter().map(|t| t.pnl.abs()).sum::<f64>() / losing_trades.len() as f64
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn sharpe_ratio(&self) -> f64 {
        if self.trade_history.len() < 2 {
            return 0.0;
        }
        
        let returns: Vec<f64> = self.trade_history.iter()
            .map(|t| t.pnl)
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        if variance > 0.0 {
            // Annualized risk-free rate (2% per year, adjusted for trade frequency)
            let risk_free_rate = 0.02 / 252.0; // Assuming daily trading
            let excess_return = mean_return - risk_free_rate;
            excess_return / variance.sqrt()
        } else {
            0.0
        }
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
    /// # Errors
    ///
    /// Will return `Err` if the config file cannot be read or parsed.
    pub fn new_with_config(symbol: &str, config: TradingConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
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

    /// Calculate realistic slippage based on market conditions
    pub fn calculate_slippage(&self, price: f64, quantity: f64, volatility: f64) -> f64 {
        let base_slippage = self.config.slippage.min_slippage;
        let max_slippage = self.config.slippage.max_slippage;
        
        // Size-based slippage (larger trades = more slippage)
        let size_factor = (quantity / 100.0).min(1.0); // Normalize to 100 units
        let size_slippage = base_slippage + (max_slippage - base_slippage) * size_factor;
        
        // Volatility-based slippage (higher volatility = more slippage)
        let volatility_factor = (volatility / 0.01).min(2.0); // Normalize to 1% volatility
        let volatility_slippage = size_slippage * (1.0 + volatility_factor * self.config.slippage.volatility_multiplier);
        
        // Price-based slippage (lower prices = higher relative slippage)
        let price_factor = (0.1 / price).min(1.0);
        let final_slippage = volatility_slippage * price_factor;
        
        // Market impact simulation (larger orders move the market)
        let market_impact = if quantity > 50.0 {
            let impact_factor = (quantity - 50.0) / 50.0; // 0-1 scale
            final_slippage * (1.0 + impact_factor * 0.5) // Up to 50% additional slippage
        } else {
            final_slippage
        };
        
        // Clamp to configured limits
        market_impact.clamp(base_slippage, max_slippage)
    }

    /// Calculate realistic fees based on order type and market conditions
    pub fn calculate_fees(&self, price: f64, quantity: f64, order_type: &OrderType) -> f64 {
        let notional = price * quantity;
        
        // Base fee calculation
        let base_fee_rate = match order_type {
            OrderType::Market => self.config.exchange.taker_fee,
            OrderType::Limit => self.config.exchange.maker_fee,
            OrderType::Maker => self.config.exchange.maker_fee,
            OrderType::Taker => self.config.exchange.taker_fee,
        };
        
        // Safe fee rate clamping
        let safe_fee_rate = base_fee_rate.clamp(0.0, 0.01); // Max 1%
        
        // Volume discount simulation (larger trades get better rates)
        let volume_discount = if notional > 1000.0 {
            0.9 // 10% discount for trades > $1000
        } else if notional > 100.0 {
            0.95 // 5% discount for trades > $100
        } else {
            1.0 // No discount
        };
        
        notional * safe_fee_rate * volume_discount
    }

    /// Calculate realistic rebates based on order type and market conditions
    pub fn calculate_rebates(&self, price: f64, quantity: f64, order_type: &OrderType) -> f64 {
        let notional = price * quantity;
        
        // Only limit orders get rebates (maker orders)
        if matches!(order_type, OrderType::Limit | OrderType::Maker) {
            let base_rebate_rate = self.config.exchange.maker_rebate;
            
            // Safe rebate rate clamping
            let safe_rebate_rate = base_rebate_rate.clamp(0.0, 0.001); // Max 0.1%
            
            // Volume-based rebate scaling
            let volume_multiplier = if notional > 1000.0 {
                1.2 // 20% bonus for large trades
            } else if notional > 100.0 {
                1.1 // 10% bonus for medium trades
            } else {
                1.0 // Standard rate
            };
            
            notional * safe_rebate_rate * volume_multiplier
        } else {
            0.0 // No rebates for market orders
        }
    }

    pub fn should_trade(
        &mut self,
        _signal: &Signal,
        confidence: f64,
        _current_price: f64,
        timestamp: f64,
    ) -> bool {
        // Validate inputs
        if !(0.0..=1.0).contains(&confidence) || timestamp < 0.0 {
            return false;
        }

        // Check circuit breaker
        if self.is_circuit_breaker_active {
            #[allow(clippy::cast_precision_loss)]
            if timestamp - self.circuit_breaker_start
                > self.config.risk_management.cooldown_period as f64
            {
                self.is_circuit_breaker_active = false; // Circuit breaker expired
                return true;
            }
            return false;
        }

        // Check signal strength
        if confidence < self.config.risk_management.min_signal_strength {
            if self.metrics.total_trades % 100 == 0 {
                            tracing::info!(
                confidence = confidence,
                min_signal_strength = self.config.risk_management.min_signal_strength,
                "Trade rejected - confidence below minimum threshold"
            );
            }
            return false;
        }
        
        // Debug logging for successful trade approval
        if self.metrics.total_trades % 100 == 0 {
                        tracing::info!(
                confidence = confidence,
                min_signal_strength = self.config.risk_management.min_signal_strength,
                "Trade approved - confidence meets minimum threshold"
            );
        }

        // Check consecutive losses
        if self.metrics.consecutive_losses >= self.config.risk_management.max_consecutive_losses {
            self.is_circuit_breaker_active = true;
            self.circuit_breaker_start = timestamp;
            warn!("Circuit breaker activated due to consecutive losses");
            return false;
        }

        // Check drawdown - respect backtest-specific settings
        let effective_max_drawdown =
            if let Some(ref backtest_settings) = self.config.backtest_settings {
                if backtest_settings.disable_drawdown_limit {
                    f64::INFINITY // Disable drawdown limit completely
                } else {
                    backtest_settings
                        .max_drawdown_override
                        .unwrap_or(self.config.risk_management.max_drawdown)
                }
            } else {
                self.config.risk_management.max_drawdown
            };

        if self.metrics.max_drawdown > effective_max_drawdown {
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

    #[must_use]
    pub fn determine_order_type(
        &self,
        _signal: &Signal,
        confidence: f64,
        spread: f64,
    ) -> OrderType {
        // Validate inputs
        if !(0.0..=1.0).contains(&confidence) || spread < 0.0 {
            return OrderType::Market; // Default to market order for invalid inputs
        }

        // Use maker orders for high confidence signals or when spread is wide
        if confidence > self.config.risk_management.min_signal_strength
            || spread > self.config.exchange.tick_size * 100.0
        {
            OrderType::Maker
        } else {
            OrderType::Taker
        }
    }

    #[must_use]
    pub fn simulate_fill(
        &self,
        order_type: &OrderType,
        price: f64,
        quantity: f64,
    ) -> (f64, f64, f64) {
        // Validate inputs
        if price <= 0.0 || quantity <= 0.0 {
            return (price, 0.0, 0.0); // Return zero fill for invalid inputs
        }
        let (fill_price, slippage, fill_quantity) = match order_type {
            OrderType::Market => {
                let slippage = self.calculate_slippage(
                    price,
                    quantity,
                    self.config.slippage.market_order_volatility,
                );
                let fill_price = price + slippage;
                (fill_price, slippage, quantity)
            }
            OrderType::Limit => {
                // Limit orders might not fill immediately
                self.config.backtest_settings.as_ref().map_or_else(
                    || {
                        // Default fallback
                        if rand::thread_rng().gen_bool(0.7) {
                            (price, 0.0, quantity)
                        } else {
                            (price, 0.0, quantity * 0.5)
                        }
                    },
                    |backtest_settings| {
                        if rand::thread_rng().gen_bool(backtest_settings.limit_order_fill_rate) {
                            (price, 0.0, quantity)
                        } else {
                            (price, 0.0, quantity * backtest_settings.partial_fill_ratio)
                        }
                    },
                )
            }
            OrderType::Maker => {
                // Maker orders get better prices
                let _rebate = self.calculate_rebates(price, quantity, &OrderType::Limit);
                (price, 0.0, quantity)
            }
            OrderType::Taker => {
                let slippage = self.calculate_slippage(
                    price,
                    quantity,
                    self.config.slippage.taker_order_volatility,
                );
                let fill_price = price + slippage;
                (fill_price, slippage, quantity)
            }
        };

        (fill_price, slippage, fill_quantity)
    }

    #[must_use]
    pub fn calculate_latency(&self) -> f64 {
        // Check if we have backtest settings for latency simulation
        self.config.backtest_settings.as_ref().map_or(0.0, |backtest_settings| {
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
                let latency = if rand::thread_rng().gen_bool(backtest_settings.latency_spike_probability)
                {
                    base_latency * backtest_settings.latency_spike_multiplier
                } else {
                    base_latency
                };

                // Add jitter (±20% variation)
                let jitter = latency * rand::thread_rng().gen_range(-0.2..0.2);
                latency + jitter
            } else {
                0.0 // No latency simulation
            }
        })
    }

    /// Calculate optimal position size based on confidence, risk factors, and exchange constraints
    /// 
    /// # Arguments
    /// * `symbol` - Trading symbol (currently unused but kept for future use)
    /// * `price` - Current market price
    /// * `confidence` - Signal confidence (0.0 to 1.0)
    /// * `trading_size_min` - Minimum trade size
    /// * `trading_size_max` - Maximum trade size
    /// * `trading_size_step` - Step size for rounding
    /// 
    /// # Returns
    /// * `f64` - Calculated position size respecting all constraints
    #[must_use]
    pub fn calculate_position_size(
        &self,
        _symbol: &str,
        price: f64,
        confidence: f64,
        trading_size_min: f64,
        trading_size_max: f64,
        trading_size_step: f64,
    ) -> f64 {
        // Validate inputs
        if price <= 0.0 || !(0.0..=1.0).contains(&confidence) {
            return trading_size_min;
        }

        // Base quantity using linear interpolation between min and max
        let base_quantity = confidence.mul_add(trading_size_max - trading_size_min, trading_size_min);
        
        // Confidence-based scaling
        let confidence_multiplier = if confidence > 0.8 {
            // High confidence gets boost
            1.2
        } else if confidence > 0.6 {
            // Medium confidence gets standard scaling
            1.0
        } else {
            // Low confidence gets reduction
            confidence * 0.8
        };

        // Volatility adjustment (simplified)
        let volatility_factor = if let Some(last_trade) = self.metrics.trade_history.back() {
            let price_change = (last_trade.price - price).abs() / price;
            (1.0 - price_change * 10.0).max(0.5) // Reduce size for high volatility
        } else {
            1.0
        };

        // Drawdown adjustment
        let drawdown_factor = if self.metrics.max_drawdown > 0.05 {
            (1.0 - self.metrics.max_drawdown * 2.0).max(0.3) // Reduce size during drawdowns
        } else {
            1.0
        };

        // Performance adjustment
        let performance_factor = if self.metrics.total_trades > 10 {
            let win_rate = self.metrics.win_rate();
            if win_rate > 0.6 {
                1.1 // Slight boost when winning
            } else if win_rate < 0.3 {
                0.7 // Reduce size when losing
            } else {
                1.0
            }
        } else {
            1.0
        };

        // Calculate final size with all adjustments
        let dynamic_quantity = base_quantity * confidence_multiplier * volatility_factor * drawdown_factor * performance_factor;
        
        // Apply step size rounding
        let step_size = if trading_size_step > 0.0 { trading_size_step } else { self.config.exchange.step_size };
        let rounded_quantity = if step_size > 0.0 {
            (dynamic_quantity / step_size).floor() * step_size
        } else {
            dynamic_quantity
        };

        // Apply minimum notional requirement
        let min_notional = self.config.exchange.min_notional;
        let min_qty = if price > 0.0 { min_notional / price } else { trading_size_min };
        let final_quantity = rounded_quantity.max(min_qty);

        // Final bounds check
        final_quantity.clamp(trading_size_min, trading_size_max)
    }

    /// Update position based on trade execution
    fn update_position(&mut self, signal: Signal, fill_price: f64, quantity: f64) {
        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 {
                    self.position.quantity = quantity;
                    self.position.entry_price = fill_price;
                } else {
                    // Average down
                    let total_quantity = self.position.quantity + quantity;
                    let total_cost = (self.position.entry_price * self.position.quantity) + (fill_price * quantity);
                    self.position.entry_price = total_cost / total_quantity;
                    self.position.quantity = total_quantity;
                }
            }
            Signal::Sell => {
                if self.position.quantity > 0.0 {
                    let sell_quantity = quantity.min(self.position.quantity);
                    self.position.quantity -= sell_quantity;
                    
                    if self.position.quantity == 0.0 {
                        self.position.entry_price = 0.0;
                    }
                }
            }
            Signal::Hold => {}
        }
    }

    /// Update PnL based on current position and market price
    fn update_pnl(&mut self, current_price: f64, _timestamp: f64) {
        if self.position.quantity > 0.0 && self.position.entry_price > 0.0 {
            let unrealized_pnl = (current_price - self.position.entry_price) * self.position.quantity;
            // Note: unrealized_pnl is calculated by method, not stored as field
            // We'll update the metrics instead
            
            // Update equity curve
            self.metrics.update_equity(self.realized_pnl + unrealized_pnl);
        }
    }

    /// Record trade in metrics for performance analysis
    fn record_trade(&mut self, signal: Signal, fill_price: f64, quantity: f64, fees: f64, slippage: f64, rebates: f64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let order_type = if matches!(signal, Signal::Buy) {
            OrderType::Market
        } else {
            OrderType::Limit
        };

        let pnl = match signal {
            Signal::Sell => {
                if self.position.entry_price > 0.0 {
                    let gross_revenue = fill_price * quantity;
                    let net_revenue = gross_revenue - fees + rebates;
                    let cost = self.position.entry_price * quantity;
                    net_revenue - cost
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        if matches!(signal, Signal::Sell) && pnl != 0.0 {
            self.realized_pnl += pnl;
        }

        let record = TradeRecord {
            timestamp,
            signal,
            price: fill_price,
            quantity,
            fees,
            slippage,
            rebates,
            pnl,
            order_type,
        };

        self.metrics.record_trade(record);
    }

    /// Calculate current market volatility based on recent price movements
    fn calculate_current_volatility(&self, _current_price: f64) -> f64 {
        // Since we don't have price_history field, use a reasonable default
        // In a real implementation, this would be calculated from market data
        0.002 // Default 0.2% volatility for DOGEUSDT
    }
}

#[async_trait::async_trait]
impl Trader for TradingEngine {
    fn calculate_trade_size(
        &self,
        symbol: &str,
        price: f64,
        confidence: f64,
        trading_size_min: f64,
        trading_size_max: f64,
        trading_size_step: f64,
    ) -> f64 {
        // Use the consolidated position sizing function
        self.calculate_position_size(symbol, price, confidence, trading_size_min, trading_size_max, trading_size_step)
    }

    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64) {
        // This would be implemented for live trading
        // For now, we'll use on_emulate for backtesting
        self.on_emulate(signal, price, quantity).await;
    }

    /// Enhanced trade execution with realistic latency simulation
    async fn on_emulate(&mut self, signal: Signal, price: f64, quantity: f64) -> Option<(f64, f64, f64, f64, OrderType)> {
        // Simulate realistic order processing latency
        let order_processing_latency = match signal {
            Signal::Buy | Signal::Sell => {
                // Market orders are faster than limit orders
                let base_latency = 0.5; // 0.5ms base
                let random_factor = rand::thread_rng().gen_range(0.0..1.0);
                std::time::Duration::from_micros((base_latency * 1000.0 + random_factor * 500.0) as u64)
            }
            Signal::Hold => std::time::Duration::from_micros(0),
        };

        // Simulate order processing delay
        if !order_processing_latency.is_zero() {
            tokio::time::sleep(order_processing_latency).await;
        }

        // Determine order type based on signal confidence and market conditions
        let order_type = self.determine_order_type(&signal, 0.8, self.config.exchange.tick_size * 10.0);
        
        // Calculate realistic market impact and fill price
        let volatility = self.calculate_current_volatility(price);
        let slippage = self.calculate_slippage(price, quantity, volatility);
        
        // Fill price reflects market impact
        let fill_price = match signal {
            Signal::Buy => price + slippage,  // Buy orders get filled at higher price
            Signal::Sell => price - slippage, // Sell orders get filled at lower price
            Signal::Hold => price,
        };
        
        // Calculate fees and rebates
        let fees = self.calculate_fees(fill_price, quantity, &order_type);
        let rebates = self.calculate_rebates(fill_price, quantity, &order_type);
        
        // Update position and PnL
        self.update_position(signal, fill_price, quantity);
        self.update_pnl(fill_price, std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64());
        
        // Record trade with realistic metrics
        self.record_trade(signal, fill_price, quantity, fees, slippage, rebates);
        
        Some((fill_price, fees, slippage, rebates, order_type))
    }

    fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.position.quantity > 0.0 && current_price > 0.0 && self.position.entry_price > 0.0 {
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


