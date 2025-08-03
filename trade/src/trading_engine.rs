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
    pub risk_management: RiskManagementConfig,
    pub order_management: OrderManagementConfig,
    pub performance_tracking: PerformanceTrackingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub taker_fee: f64,
    pub maker_fee: f64,
    pub maker_rebate: f64,
    pub min_slippage: f64,
    pub max_slippage: f64,
    pub slippage_volatility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManagementConfig {
    pub max_position_size: f64,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
    pub min_signal_strength: f64,
    pub min_edge_over_cost: f64,
    pub max_consecutive_losses: usize,
    pub cooldown_period: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderManagementConfig {
    pub use_limit_orders: bool,
    pub limit_order_timeout: f64,
    pub min_spread_capture: f64,
    pub queue_position_penalty: f64,
    pub partial_fill_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrackingConfig {
    pub track_fill_rate: bool,
    pub track_slippage: bool,
    pub track_fees: bool,
    pub track_rebates: bool,
    pub track_queue_position: bool,
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
        
        let drawdown = (self.peak_equity - new_equity) / self.peak_equity;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
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

    pub fn calculate_slippage(&self, _price: f64, volatility: f64) -> f64 {
        let base_slippage = self.config.exchange.min_slippage;
        let max_slippage = self.config.exchange.max_slippage;
        let volatility_factor = volatility * self.config.exchange.slippage_volatility;
        
        let slippage = base_slippage + (max_slippage - base_slippage) * volatility_factor;
        slippage.min(max_slippage)
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
        if !self.config.order_management.use_limit_orders {
            return OrderType::Market;
        }

        // Use maker orders for high confidence signals or when spread is wide
        if confidence > 0.8 || spread > self.config.order_management.min_spread_capture {
            OrderType::Maker
        } else {
            OrderType::Taker
        }
    }

    pub fn simulate_fill(&self, order_type: &OrderType, price: f64, quantity: f64) -> (f64, f64, f64) {
        let (fill_price, slippage, fill_quantity) = match order_type {
            OrderType::Market => {
                let slippage = self.calculate_slippage(price, 0.1); // Assume some volatility
                let fill_price = price + slippage;
                (fill_price, slippage, quantity)
            }
            OrderType::Limit => {
                // Limit orders might not fill immediately
                if rand::random::<f64>() < 0.7 {
                    (price, 0.0, quantity)
                } else {
                    (price, 0.0, quantity * 0.5) // Partial fill
                }
            }
            OrderType::Maker => {
                // Maker orders get better prices
                let _rebate = self.calculate_rebates(price, quantity);
                (price, 0.0, quantity)
            }
            OrderType::Taker => {
                let slippage = self.calculate_slippage(price, 0.2);
                let fill_price = price + slippage;
                (fill_price, slippage, quantity)
            }
        };

        (fill_price, slippage, fill_quantity)
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

        let order_type = self.determine_order_type(&signal, 0.8, 0.0001); // Assume some spread
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