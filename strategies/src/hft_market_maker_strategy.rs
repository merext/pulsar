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
pub struct HftMarketMakerStrategy {
    // Order book state (simplified for speed)
    best_bid: f64,
    best_ask: f64,
    bid_size: f64,
    ask_size: f64,
    spread: f64,
    mid_price: f64,
    
    // Market making parameters
    spread_multiplier: f64,
    min_spread: f64,
    signal_threshold: f64,
    
    // Current inventory and position
    current_inventory: f64,
    current_position: Position,
    avg_buy_price: f64,
    avg_sell_price: f64,
    
    // Risk management
    max_inventory: f64,
    total_pnl: f64,
    
    // Performance tracking
    trades_made: u32,
    spread_captured: f64,
    
    // Market state
    volatility: f64,
    volume_profile: [f64; 24], // Hourly volume profile
    current_hour: u8,
    
    // Configuration parameters (loaded from config file)
    hedge_threshold: f64,
    _inventory_cost_weight: f64,
    _max_loss_per_trade: f64,
    _max_daily_loss: f64,
    _correlation_limit: f64,
    hedge_confidence: f64,
    bid_ask_confidence: f64,
    bid_only_confidence: f64,
    ask_only_confidence: f64,
    volatility_threshold: f64,
    momentum_confidence: f64,
    default_confidence: f64,
    bid_price_offset: f64,
    ask_price_offset: f64,
    order_book_size_multiplier: f64,
    inventory_ratio_threshold: f64,
    price_competitiveness_bid: f64,
    price_competitiveness_ask: f64,
    volatility_alpha: f64,
    volume_profile_alpha: f64,
    volume_profile_beta: f64,
}

impl HftMarketMakerStrategy {
    pub fn new() -> Self {
        // Load configuration from file
        let config = StrategyConfig::load_strategy_config("hft_market_maker_strategy")
            .unwrap_or_else(|_| {
                // Use defaults if config file not found
                StrategyConfig { config: toml::Value::Table(toml::map::Map::new()) }
            });

        // Order Book Parameters
        let spread_multiplier = config.get_or("spread_multiplier", 1.5);
        let min_spread = config.get_or("min_spread", 0.0001);
        let _max_position = config.get_or("max_position", 1000.0);
        let _inventory_target = config.get_or("inventory_target", 0.0);

        // Inventory Management
        let max_inventory = config.get_or("max_inventory", 5000.0);
        let hedge_threshold = config.get_or("hedge_threshold", 0.8);
        let _inventory_cost_weight = config.get_or("inventory_cost_weight", 0.1);

        // Risk Controls
        let _max_loss_per_trade = config.get_or("max_loss_per_trade", 0.001);
        let _max_daily_loss = config.get_or("max_daily_loss", 0.01);
        let _correlation_limit = config.get_or("correlation_limit", 0.7);
        let signal_threshold = config.get_or("signal_threshold", DefaultConfig::hft_mm_signal_threshold());

        // Signal Generation Parameters
        let hedge_confidence = config.get_or("hedge_confidence", 0.8);
        let bid_ask_confidence = config.get_or("bid_ask_confidence", 0.5);
        let bid_only_confidence = config.get_or("bid_only_confidence", 0.6);
        let ask_only_confidence = config.get_or("ask_only_confidence", 0.6);
        let volatility_threshold = config.get_or("volatility_threshold", 0.001);
        let momentum_confidence = config.get_or("momentum_confidence", 0.3);
        let default_confidence = config.get_or("default_confidence", 0.2);

        // Order Book Simulation (for backtesting)
        let bid_price_offset = config.get_or("bid_price_offset", 0.9999);
        let ask_price_offset = config.get_or("ask_price_offset", 1.0001);
        let order_book_size_multiplier = config.get_or("order_book_size_multiplier", 10.0);

        // Position Management
        let inventory_ratio_threshold = config.get_or("inventory_ratio_threshold", 0.8);
        let price_competitiveness_bid = config.get_or("price_competitiveness_bid", 0.9995);
        let price_competitiveness_ask = config.get_or("price_competitiveness_ask", 1.0005);

        // Volatility Calculation
        let volatility_alpha = config.get_or("volatility_alpha", 0.1);

        // Volume Profile
        let volume_profile_alpha = config.get_or("volume_profile_alpha", 0.9);
        let volume_profile_beta = config.get_or("volume_profile_beta", 0.1);

        Self {
            best_bid: 0.0,
            best_ask: 0.0,
            bid_size: 0.0,
            ask_size: 0.0,
            spread: 0.0,
            mid_price: 0.0,
            spread_multiplier,
            min_spread,
            signal_threshold,
            current_inventory: 0.0,
            current_position: Position {
                symbol: String::new(),
                quantity: 0.0,
                entry_price: 0.0,
            },
            avg_buy_price: 0.0,
            avg_sell_price: 0.0,
            max_inventory,
            total_pnl: 0.0,
            trades_made: 0,
            spread_captured: 0.0,
            volatility: 0.0,
            volume_profile: [1.0; 24],
            current_hour: 0,
            // Store all config parameters
            hedge_threshold,
            _inventory_cost_weight,
            _max_loss_per_trade,
            _max_daily_loss,
            _correlation_limit,
            hedge_confidence,
            bid_ask_confidence,
            bid_only_confidence,
            ask_only_confidence,
            volatility_threshold,
            momentum_confidence,
            default_confidence,
            bid_price_offset,
            ask_price_offset,
            order_book_size_multiplier,
            inventory_ratio_threshold,
            price_competitiveness_bid,
            price_competitiveness_ask,
            volatility_alpha,
            volume_profile_alpha,
            volume_profile_beta,
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
    fn calculate_optimal_spread(&self) -> f64 {
        // Dynamic spread calculation based on volatility and inventory
        let base_spread = self.spread.max(self.min_spread);
        let volatility_adjustment = self.volatility * 2.0;
        let inventory_adjustment = (self.current_inventory / self.max_inventory).abs() * 0.5;
        
        base_spread * self.spread_multiplier + volatility_adjustment + inventory_adjustment
    }

    #[inline]
    fn calculate_bid_ask_prices(&self) -> (f64, f64) {
        let optimal_spread = self.calculate_optimal_spread();
        let half_spread = optimal_spread / 2.0;
        
        let bid_price = self.mid_price - half_spread;
        let ask_price = self.mid_price + half_spread;
        
        (bid_price, ask_price)
    }

    #[inline]
    fn should_place_bid(&self) -> bool {
        // Check if we should place a bid
        let (our_bid, _) = self.calculate_bid_ask_prices();
        
        // For backtesting, be more aggressive with bid placement
        let price_competitive = our_bid > self.best_bid * self.price_competitiveness_bid;
        let inventory_ok = self.current_inventory < self.max_inventory * self.inventory_ratio_threshold;
        let spread_wide_enough = self.spread > self.min_spread;
        
        price_competitive && inventory_ok && spread_wide_enough
    }

    #[inline]
    fn should_place_ask(&self) -> bool {
        // Check if we should place an ask
        let (_, our_ask) = self.calculate_bid_ask_prices();
        
        // For backtesting, be more aggressive with ask placement
        let price_competitive = our_ask < self.best_ask * self.price_competitiveness_ask;
        let inventory_ok = self.current_inventory > -self.max_inventory * self.inventory_ratio_threshold;
        let spread_wide_enough = self.spread > self.min_spread;
        
        price_competitive && inventory_ok && spread_wide_enough
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
    fn should_hedge(&self) -> bool {
        // Check if we need to hedge our inventory
        let inventory_ratio = self.current_inventory.abs() / self.max_inventory;
        inventory_ratio > self.hedge_threshold // Hedge when inventory > threshold
    }

    #[inline]
    fn generate_market_making_signal(&self) -> (Signal, f64) {
        // Market making signal generation
        
        // Check if we should place orders
        let place_bid = self.should_place_bid();
        let place_ask = self.should_place_ask();
        
        // Check if we need to hedge
        let need_hedge = self.should_hedge();
        
        // Generate signal based on market making logic
        let result = if need_hedge {
            // Hedge signal
            if self.current_inventory > 0.0 {
                (Signal::Sell, self.hedge_confidence) // Sell to reduce long inventory
            } else {
                (Signal::Buy, self.hedge_confidence)  // Buy to reduce short inventory
            }
        } else if place_bid && place_ask {
            // Both sides of the book - alternate between buy and sell for backtesting
            if self.trades_made % 2 == 0 {
                (Signal::Buy, self.bid_ask_confidence) // Buy on even trades
            } else {
                (Signal::Sell, self.bid_ask_confidence) // Sell on odd trades
            }
        } else if place_bid {
            // Only bid side
            (Signal::Buy, self.bid_only_confidence) // Stronger buy signal for backtesting
        } else if place_ask {
            // Only ask side
            (Signal::Sell, self.ask_only_confidence) // Stronger sell signal for backtesting
        } else {
            // No orders - try to generate signals based on price movement
            if self.volatility > self.volatility_threshold {
                // High volatility - try to capture momentum
                if self.current_inventory < 0.0 {
                    (Signal::Buy, self.momentum_confidence) // Buy to cover short position
                } else {
                    (Signal::Sell, self.momentum_confidence) // Sell to reduce long position
                }
            } else {
                // Default to buy signals when no other conditions are met
                (Signal::Buy, self.default_confidence)
            }
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
impl Strategy for HftMarketMakerStrategy {
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