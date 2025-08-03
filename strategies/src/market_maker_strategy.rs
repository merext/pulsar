//! # HFT Market Maker Strategy
//! 
//! This is a specialized high-frequency trading strategy designed for market making:
//! - Order book analysis and spread capture
//! - Ultra-fast bid/ask placement
//! - Inventory management
//! - Risk controls for market making
//! - Tick-by-tick order book updates
//! 
//! Designed for HFT market making operations with sub-millisecond latency.

use crate::config::{StrategyConfig, DefaultConfig};
use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;
use toml;

#[derive(Clone, Debug)]
pub struct MarketMakerStrategy {
    // Order book state (simplified for speed)
    best_bid: f64,
    best_ask: f64,
    bid_size: f64,
    ask_size: f64,
    spread: f64,
    mid_price: f64,
    

    signal_threshold: f64,
    
    // Current inventory and position
    current_inventory: f64,
    current_position: Position,
    avg_buy_price: f64,
    avg_sell_price: f64,
    

    total_pnl: f64,
    
    // Performance tracking
    trades_made: u32,
    spread_captured: f64,
    
    // Market state
    volatility: f64,
    volume_profile: [f64; 24], // Hourly volume profile
    current_hour: u8,
    
    // Momentum tracking (for trade data optimization)
    last_price: f64,
    price_momentum: f64,
    momentum_threshold: f64,
    
    // Configuration parameters (loaded from config file)
    bid_price_offset: f64,
    ask_price_offset: f64,
    order_book_size_multiplier: f64,
    volatility_alpha: f64,
    volume_profile_alpha: f64,
    volume_profile_beta: f64,
}

impl MarketMakerStrategy {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("market_maker_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });


        let signal_threshold = config.get_or("signal_threshold", DefaultConfig::hft_mm_signal_threshold());



        // Order Book Simulation (for backtesting)
        let bid_price_offset = config.get_or("bid_price_offset", 0.9999);
        let ask_price_offset = config.get_or("ask_price_offset", 1.0001);
        let order_book_size_multiplier = config.get_or("order_book_size_multiplier", 10.0);

        // Position Management


        // Volatility Calculation
        let volatility_alpha = config.get_or("volatility_alpha", 0.1);

        // Volume Profile
        let volume_profile_alpha = config.get_or("volume_profile_alpha", 0.9);
        let volume_profile_beta = config.get_or("volume_profile_beta", 0.1);
        
        // Momentum tracking
        let momentum_threshold = config.get_or("momentum_threshold", 0.0001);

        Self {
            best_bid: 0.0,
            best_ask: 0.0,
            bid_size: 0.0,
            ask_size: 0.0,
            spread: 0.0,
            mid_price: 0.0,

            signal_threshold,
            current_inventory: 0.0,
            current_position: Position {
                symbol: String::new(),
                quantity: 0.0,
                entry_price: 0.0,
            },
            avg_buy_price: 0.0,
            avg_sell_price: 0.0,

            total_pnl: 0.0,
            trades_made: 0,
            spread_captured: 0.0,
            volatility: 0.0,
            volume_profile: [1.0; 24],
            current_hour: 0,
            // Store all config parameters



            bid_price_offset,
            ask_price_offset,
            order_book_size_multiplier,

            volatility_alpha,
            volume_profile_alpha,
            volume_profile_beta,
            last_price: 0.0,
            price_momentum: 0.0,
            momentum_threshold,
        }
    }

    #[inline]
    pub fn update_order_book(&mut self, bid: f64, ask: f64, bid_size: f64, ask_size: f64) {
        self.best_bid = bid;
        self.best_ask = ask;
        self.bid_size = bid_size;
        self.ask_size = ask_size;
        self.spread = ask - bid;
        self.mid_price = (bid + ask) / 2.0;
    }



    #[inline]
    fn update_inventory(&mut self, trade_price: f64, trade_size: f64, is_buy: bool) {
        if is_buy {
            // We sold (someone bought from us)
            self.current_inventory -= trade_size;
            self.avg_sell_price = if self.avg_sell_price == 0.0 {
                trade_price
            } else {
                (self.avg_sell_price + trade_price) / 2.0
            };
        } else {
            // We bought (someone sold to us)
            self.current_inventory += trade_size;
            self.avg_buy_price = if self.avg_buy_price == 0.0 {
                trade_price
            } else {
                (self.avg_buy_price + trade_price) / 2.0
            };
        }
        
        // Update position
        if self.current_inventory > 0.0 {
            self.current_position = Position {
                symbol: "DOGEUSDT".to_string(),
                quantity: self.current_inventory,
                entry_price: self.avg_buy_price,
            };
        } else {
            self.current_position = Position {
                symbol: String::new(),
                quantity: 0.0,
                entry_price: 0.0,
            };
        }
    }

    #[inline]
    fn calculate_pnl(&mut self) {
        // Calculate P&L from spread capture and inventory
        let spread_pnl = self.spread_captured;
        let inventory_pnl = if self.current_inventory > 0.0 {
            (self.mid_price - self.avg_buy_price) * self.current_inventory
        } else if self.current_inventory < 0.0 {
            (self.avg_sell_price - self.mid_price) * self.current_inventory.abs()
        } else {
            0.0
        };
        
        self.total_pnl = spread_pnl + inventory_pnl;
    }

    #[inline]
    fn update_volatility(&mut self, price_change: f64) {
        // Simple exponential volatility calculation
        self.volatility = self.volatility_alpha * price_change.abs() + (1.0 - self.volatility_alpha) * self.volatility;
    }

    #[inline]
    fn update_volume_profile(&mut self, volume: f64) {
        // Update hourly volume profile
        self.volume_profile[self.current_hour as usize] = 
            self.volume_profile_alpha * self.volume_profile[self.current_hour as usize] + self.volume_profile_beta * volume;
    }





    #[inline]
    fn generate_market_making_signal(&self) -> (Signal, f64) {
        // Pure momentum strategy like HFT Ultra Fast (simplified market making)
        
        // Ultra-aggressive momentum approach
        let momentum_factor = if self.price_momentum.abs() > self.momentum_threshold { 2.5 } else { 1.0 };
        
        // Simple momentum-based signal generation (like successful strategies)
        let result = if self.price_momentum > self.momentum_threshold {
            // Any positive momentum - buy (like HFT Ultra Fast)
            let momentum_strength = (self.price_momentum * 3000.0).min(1.0);
            (Signal::Buy, momentum_strength * momentum_factor)
        } else if self.price_momentum < -self.momentum_threshold {
            // Any negative momentum - sell (like HFT Ultra Fast)
            let momentum_strength = (self.price_momentum.abs() * 3000.0).min(1.0);
            (Signal::Sell, momentum_strength * momentum_factor)
        } else {
            // No momentum - hold
            (Signal::Hold, 0.0)
        };
        
        // Apply signal threshold filter
        let (signal, confidence) = result;
        if confidence < self.signal_threshold {
            (Signal::Hold, 0.0)
        } else {
            (signal, confidence)
        }
    }
}

#[async_trait::async_trait]
impl Strategy for MarketMakerStrategy {
    fn get_info(&self) -> String {
        format!(
            "HFT Market Maker Strategy (trades: {}, pnl: {:.4}, inventory: {:.2})",
            self.trades_made,
            self.total_pnl,
            self.current_inventory
        )
    }

    async fn on_trade(&mut self, trade: TradeData) {
        let price = trade.price;
        let volume = trade.qty;
        let is_buyer_maker = trade.is_buyer_maker;
        
        // Update price momentum
        if self.last_price > 0.0 {
            self.price_momentum = (price - self.last_price) / self.last_price;
        }
        
        // For backtesting, simulate order book updates based on trade data
        // In real implementation, you'd have actual order book data
        if self.best_bid == 0.0 || self.best_ask == 0.0 {
            // Initialize order book with first trade
            self.best_bid = price * self.bid_price_offset;
            self.best_ask = price * self.ask_price_offset;
            self.mid_price = price;
        } else {
            // Update order book based on trade direction
            if is_buyer_maker {
                // Buyer was maker, so ask was hit
                self.best_ask = price;
                self.best_bid = price * self.bid_price_offset; // Update bid below
            } else {
                // Seller was maker, so bid was hit
                self.best_bid = price;
                self.best_ask = price * self.ask_price_offset; // Update ask above
            }
            self.mid_price = (self.best_bid + self.best_ask) / 2.0;
            self.spread = self.best_ask - self.best_bid;
        }
        
        // Update bid/ask sizes (simplified)
        self.bid_size = volume * self.order_book_size_multiplier; // Simulate larger order book
        self.ask_size = volume * self.order_book_size_multiplier;
        
        // Update inventory if this is our trade (simplified for backtesting)
        // In real implementation, you'd check if this trade was executed against our orders
        if self.trades_made % 10 == 0 { // Simulate occasional inventory updates
            self.update_inventory(price, volume, is_buyer_maker);
        }
        
        // Update volatility
        if self.mid_price > 0.0 {
            let price_change = (price - self.mid_price) / self.mid_price;
            self.update_volatility(price_change);
        }
        
        // Update volume profile
        self.update_volume_profile(volume);
        
        // Update current hour (simplified)
        self.current_hour = ((trade.time / 3600000) % 24) as u8;
        
        // Calculate P&L
        self.calculate_pnl();
        
        // Update trade count
        self.trades_made += 1;
        
        // Update last price for momentum calculation
        self.last_price = price;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Generate market making signal
        self.generate_market_making_signal()
    }
} 