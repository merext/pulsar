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

use trade::models::TradeData;
use trade::trader::Position;
use crate::strategy::Strategy;
use trade::signal::Signal;

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
    

}

impl HftMarketMakerStrategy {
    pub fn new(signal_threshold: f64) -> Self {
        Self {
            best_bid: 0.0,
            best_ask: 0.0,
            bid_size: 0.0,
            ask_size: 0.0,
            spread: 0.0,
            mid_price: 0.0,
            spread_multiplier: 1.5,
            min_spread: 0.0001, // 0.01% minimum spread
            signal_threshold, // Use the parameter
            current_inventory: 0.0,
            current_position: Position {
                symbol: String::new(),
                quantity: 0.0,
                entry_price: 0.0,
            },
            avg_buy_price: 0.0,
            avg_sell_price: 0.0,
            max_inventory: 5000.0,
            total_pnl: 0.0,
            trades_made: 0,
            spread_captured: 0.0,
            volatility: 0.0,
            volume_profile: [1.0; 24],
            current_hour: 0,

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
        
        // Place bid if our price is competitive and we have room for inventory
        our_bid > self.best_bid && 
        self.current_inventory < self.max_inventory &&
        self.current_inventory > -self.max_inventory
    }

    #[inline]
    fn should_place_ask(&self) -> bool {
        // Check if we should place an ask
        let (_, our_ask) = self.calculate_bid_ask_prices();
        
        // Place ask if our price is competitive and we have room for inventory
        our_ask < self.best_ask && 
        self.current_inventory < self.max_inventory &&
        self.current_inventory > -self.max_inventory
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
        let alpha = 0.1;
        self.volatility = alpha * price_change.abs() + (1.0 - alpha) * self.volatility;
    }

    #[inline]
    fn update_volume_profile(&mut self, volume: f64) {
        // Update hourly volume profile
        self.volume_profile[self.current_hour as usize] = 
            0.9 * self.volume_profile[self.current_hour as usize] + 0.1 * volume;
    }



    #[inline]
    fn should_hedge(&self) -> bool {
        // Check if we need to hedge our inventory
        let inventory_ratio = self.current_inventory.abs() / self.max_inventory;
        inventory_ratio > 0.8 // Hedge when inventory > 80% of max
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
                (Signal::Sell, 0.8) // Sell to reduce long inventory
            } else {
                (Signal::Buy, 0.8)  // Buy to reduce short inventory
            }
        } else if place_bid && place_ask {
            // Both sides of the book
            (Signal::Hold, 0.0) // No directional signal, just market making
        } else if place_bid {
            // Only bid side
            (Signal::Buy, 0.3) // Slight buy bias
        } else if place_ask {
            // Only ask side
            (Signal::Sell, 0.3) // Slight sell bias
        } else {
            // No orders
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
        
        // Update inventory if this is our trade
        // In real implementation, you'd check if this trade was executed against our orders
        self.update_inventory(price, volume, is_buyer_maker);
        
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