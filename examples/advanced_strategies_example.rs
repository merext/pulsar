//! # Advanced Trading Strategies Example
//! 
//! This example demonstrates how to use the new sophisticated trading strategies
//! with proper configuration, risk management, and performance monitoring.
//! 
//! The strategies include:
//! - Adaptive Multi-Factor Strategy
//! - Neural Market Microstructure Strategy
//! 
//! These strategies are designed to be more profitable than simple single-indicator approaches.

use strategies::{
    AdaptiveMultiFactorStrategy,
    NeuralMarketMicrostructureStrategy,
    Strategy,
};
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;
use std::collections::HashMap;

#[tokio::main]
async fn main() {
    println!("🚀 Advanced Trading Strategies Example");
    println!("=====================================\n");

    // Initialize strategies with optimized parameters
    let mut adaptive_strategy = AdaptiveMultiFactorStrategy::new(
        10,   // short_window: 10 periods for short-term momentum
        50,   // long_window: 50 periods for trend detection
        20,   // volatility_window: 20 periods for volatility calculation
        30,   // volume_window: 30 periods for VWAP calculation
    );

    let mut neural_strategy = NeuralMarketMicrostructureStrategy::new(
        5,    // short_window: 5 periods for immediate signals
        20,   // medium_window: 20 periods for medium-term trends
        100,  // long_window: 100 periods for long-term analysis
        10,   // micro_window: 10 periods for microstructure analysis
    );

    println!("📊 Strategy Information:");
    println!("1. {}", adaptive_strategy.get_info());
    println!("2. {}", neural_strategy.get_info());
    println!();

    // Simulate market data
    let market_data = generate_sample_market_data();
    
    println!("📈 Processing {} market data points...", market_data.len());
    println!();

    // Track performance metrics
    let mut adaptive_performance = PerformanceTracker::new("Adaptive Multi-Factor");
    let mut neural_performance = PerformanceTracker::new("Neural Market Microstructure");

    // Process each trade
    for (i, trade) in market_data.iter().enumerate() {
        // Update strategies with new trade data
        adaptive_strategy.on_trade(trade.clone()).await;
        neural_strategy.on_trade(trade.clone()).await;

        // Get signals from both strategies
        let adaptive_signal = adaptive_strategy.get_signal(
            trade.price,
            trade.time as f64,
            Position::None,
        );

        let neural_signal = neural_strategy.get_signal(
            trade.price,
            trade.time as f64,
            Position::None,
        );

        // Track performance (simulate trading)
        adaptive_performance.update(trade.price, &adaptive_signal);
        neural_performance.update(trade.price, &neural_signal);

        // Print significant signals
        if i % 100 == 0 || adaptive_signal.1 > 0.7 || neural_signal.1 > 0.7 {
            println!("Trade #{} - Price: ${:.2}", i, trade.price);
            println!("  Adaptive: {:?} (confidence: {:.2})", adaptive_signal.0, adaptive_signal.1);
            println!("  Neural:   {:?} (confidence: {:.2})", neural_signal.0, neural_signal.1);
            println!();
        }
    }

    // Print final performance results
    println!("🏆 Final Performance Results:");
    println!("=============================");
    adaptive_performance.print_summary();
    neural_performance.print_summary();

    // Strategy comparison and recommendations
    print_strategy_recommendations(&adaptive_performance, &neural_performance);
}

/// Generate realistic sample market data for testing
fn generate_sample_market_data() -> Vec<TradeData> {
    let mut trades = Vec::new();
    let mut price = 50000.0; // Starting price
    let mut time = 1640995200000; // Unix timestamp
    
    // Generate 1000 trades with realistic price movements
    for i in 0..1000 {
        // Add some randomness and trends
        let trend = if i < 200 { 0.001 } else if i < 400 { -0.001 } else if i < 600 { 0.002 } else { -0.002 };
        let volatility = 0.005;
        let random_change = (rand::random::<f64>() - 0.5) * volatility;
        
        price *= 1.0 + trend + random_change;
        price = price.max(1000.0).min(100000.0); // Keep price in reasonable range
        
        let volume = 1.0 + rand::random::<f64>() * 10.0; // Random volume
        
        trades.push(TradeData {
            id: i as u64,
            price,
            qty: volume,
            quote_qty: price * volume,
            time: time + (i * 1000) as u64, // 1 second intervals
            is_buyer_maker: rand::random::<bool>(),
            is_best_match: true,
        });
    }
    
    trades
}

/// Performance tracking for strategy evaluation
struct PerformanceTracker {
    name: String,
    trades: Vec<Trade>,
    current_position: Position,
    entry_price: f64,
    total_pnl: f64,
    win_count: u32,
    loss_count: u32,
}

struct Trade {
    entry_price: f64,
    exit_price: f64,
    signal: Signal,
    confidence: f64,
    pnl: f64,
}

impl PerformanceTracker {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            trades: Vec::new(),
            current_position: Position::None,
            entry_price: 0.0,
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
        }
    }

    fn update(&mut self, current_price: f64, signal: &(Signal, f64)) {
        match self.current_position {
            Position::None => {
                // No position, check for entry signal
                if signal.1 > 0.6 { // Only enter if confidence > 60%
                    match signal.0 {
                        Signal::Buy => {
                            self.current_position = Position::Long;
                            self.entry_price = current_price;
                        },
                        Signal::Sell => {
                            self.current_position = Position::Short;
                            self.entry_price = current_price;
                        },
                        Signal::Hold => {},
                    }
                }
            },
            Position::Long => {
                // Check for exit signal
                if signal.0 == Signal::Sell || signal.0 == Signal::Hold {
                    let pnl = (current_price - self.entry_price) / self.entry_price;
                    self.record_trade(current_price, Signal::Buy, signal.1, pnl);
                    self.current_position = Position::None;
                }
            },
            Position::Short => {
                // Check for exit signal
                if signal.0 == Signal::Buy || signal.0 == Signal::Hold {
                    let pnl = (self.entry_price - current_price) / self.entry_price;
                    self.record_trade(current_price, Signal::Sell, signal.1, pnl);
                    self.current_position = Position::None;
                }
            },
        }
    }

    fn record_trade(&mut self, exit_price: f64, entry_signal: Signal, confidence: f64, pnl: f64) {
        self.trades.push(Trade {
            entry_price: self.entry_price,
            exit_price,
            signal: entry_signal,
            confidence,
            pnl,
        });

        self.total_pnl += pnl;
        if pnl > 0.0 {
            self.win_count += 1;
        } else {
            self.loss_count += 1;
        }
    }

    fn print_summary(&self) {
        let total_trades = self.trades.len();
        let win_rate = if total_trades > 0 {
            self.win_count as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_pnl = if total_trades > 0 {
            self.total_pnl / total_trades as f64
        } else {
            0.0
        };

        let avg_confidence = if total_trades > 0 {
            self.trades.iter().map(|t| t.confidence).sum::<f64>() / total_trades as f64
        } else {
            0.0
        };

        println!("📊 {}:", self.name);
        println!("   Total Trades: {}", total_trades);
        println!("   Win Rate: {:.1}%", win_rate * 100.0);
        println!("   Total P&L: {:.2}%", self.total_pnl * 100.0);
        println!("   Avg P&L per Trade: {:.2}%", avg_pnl * 100.0);
        println!("   Avg Confidence: {:.2}", avg_confidence);
        println!();
    }
}

/// Print strategy recommendations based on performance
fn print_strategy_recommendations(adaptive: &PerformanceTracker, neural: &PerformanceTracker) {
    println!("💡 Strategy Recommendations:");
    println!("============================");

    let adaptive_pnl = adaptive.total_pnl;
    let neural_pnl = neural.total_pnl;
    let adaptive_trades = adaptive.trades.len();
    let neural_trades = neural.trades.len();

    if adaptive_pnl > neural_pnl {
        println!("✅ Adaptive Multi-Factor Strategy performed better");
        println!("   - Higher total P&L: {:.2}% vs {:.2}%", 
                adaptive_pnl * 100.0, neural_pnl * 100.0);
    } else if neural_pnl > adaptive_pnl {
        println!("✅ Neural Market Microstructure Strategy performed better");
        println!("   - Higher total P&L: {:.2}% vs {:.2}%", 
                neural_pnl * 100.0, adaptive_pnl * 100.0);
    } else {
        println!("🤝 Both strategies performed similarly");
    }

    println!();
    println!("🎯 Recommended Usage:");
    println!("1. Use Adaptive Multi-Factor for trending markets");
    println!("2. Use Neural Market Microstructure for volatile markets");
    println!("3. Consider combining both strategies for diversification");
    println!("4. Monitor performance and adjust parameters monthly");
    println!("5. Always use proper risk management (stop-loss, position sizing)");
    println!();

    println!("⚠️  Risk Warnings:");
    println!("- Past performance doesn't guarantee future results");
    println!("- Always test strategies on historical data first");
    println!("- Start with small position sizes");
    println!("- Monitor market conditions and adjust accordingly");
    println!("- Consider transaction costs and slippage");
}

// Mock rand module for the example
mod rand {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn random<T>() -> T 
    where
        T: Copy + From<u64>,
    {
        let mut hasher = DefaultHasher::new();
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .hash(&mut hasher);
        let hash = hasher.finish();
        T::from(hash)
    }
} 