use std::collections::VecDeque;
use async_trait::async_trait;
use crate::strategy::Strategy;
use trade::{Position, Signal};
use trade::models::TradeData;

pub struct PulsarAlphaStrategy {
    // Price and volume data
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    
    // Microstructure features
    mid_price_history: VecDeque<f64>,
    spread_history: VecDeque<f64>,
    book_imbalance: VecDeque<f64>,
    trade_momentum: VecDeque<f64>,
    volatility_regime: VecDeque<f64>,
    
    // ASM (Adaptive Statistical Market-making) components
    asm_bid_offset: VecDeque<f64>,
    asm_ask_offset: VecDeque<f64>,
    _inventory_skew: f64,
    current_inventory: f64,
    
    // OBI (Order-book imbalance) features
    obi_signal: VecDeque<f64>,
    depth_slope_bid: VecDeque<f64>,
    depth_slope_ask: VecDeque<f64>,
    
    // Momentum and regime features
    tick_momentum: VecDeque<f64>,
    regime_classifier: VecDeque<u8>, // 0=low_vol, 1=medium_vol, 2=high_vol, 3=crash
    
    // Strategy parameters
    _base_spread_ticks: f64,
    _participation_rate: f64,
    max_inventory: f64,
    _max_taker_size: f64,
    inventory_scale: f64,
    
    // Risk management
    trade_counter: usize,
    last_signal_time: f64,
    signal_cooldown: f64,
    consecutive_losses: usize,
    max_consecutive_losses: usize,
    
    // Performance tracking
    total_pnl: f64,
    win_count: usize,
    loss_count: usize,
    _fill_rate: f64,
    _adverse_selection_rate: f64,
}

impl PulsarAlphaStrategy {
    pub fn new() -> Self {
        Self {
            price_history: VecDeque::with_capacity(1000),
            volume_history: VecDeque::with_capacity(1000),
            mid_price_history: VecDeque::with_capacity(1000),
            spread_history: VecDeque::with_capacity(1000),
            book_imbalance: VecDeque::with_capacity(1000),
            trade_momentum: VecDeque::with_capacity(1000),
            volatility_regime: VecDeque::with_capacity(1000),
            asm_bid_offset: VecDeque::with_capacity(1000),
            asm_ask_offset: VecDeque::with_capacity(1000),
            _inventory_skew: 0.0,
            current_inventory: 0.0,
            obi_signal: VecDeque::with_capacity(1000),
            depth_slope_bid: VecDeque::with_capacity(1000),
            depth_slope_ask: VecDeque::with_capacity(1000),
            tick_momentum: VecDeque::with_capacity(1000),
            regime_classifier: VecDeque::with_capacity(1000),
            
            // Conservative starting parameters
            _base_spread_ticks: 3.0,
            _participation_rate: 0.001, // 0.1%
            max_inventory: 0.5,
            _max_taker_size: 0.002, // 0.2% of volume
            inventory_scale: 0.1,
            
            trade_counter: 0,
            last_signal_time: 0.0,
            signal_cooldown: 0.05, // 50ms for HFT
            consecutive_losses: 0,
            max_consecutive_losses: 3,
            
            total_pnl: 0.0,
            win_count: 0,
            loss_count: 0,
            _fill_rate: 0.0,
            _adverse_selection_rate: 0.0,
        }
    }
    
    fn calculate_mid_price(&self, current_price: f64) -> f64 {
        // Simulate bid/ask spread for mid-price calculation
        let _spread = current_price * 0.0001; // 1 basis point spread
        current_price
    }
    
    fn calculate_book_imbalance(&self, prices: &[f64], volumes: &[f64]) -> f64 {
        if prices.len() < 10 || volumes.len() < 10 {
            return 0.0;
        }
        
        // Simulate order book imbalance from recent trades
        let recent_prices: Vec<f64> = prices.iter().rev().take(10).cloned().collect();
        let recent_volumes: Vec<f64> = volumes.iter().rev().take(10).cloned().collect();
        
        let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
        let total_volume = recent_volumes.iter().sum::<f64>();
        
        // Calculate imbalance based on price deviation from average
        let price_deviation = (recent_prices.last().unwrap() - avg_price) / avg_price;
        let volume_weighted_imbalance = price_deviation * (total_volume / recent_volumes.len() as f64);
        
        volume_weighted_imbalance
    }
    
    fn calculate_tick_momentum(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window + 1 {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(window + 1).cloned().collect();
        let mut momentum = 0.0;
        
        for i in 1..recent_prices.len() {
            let price_change = (recent_prices[i-1] - recent_prices[i]) / recent_prices[i];
            momentum += price_change;
        }
        
        momentum / (recent_prices.len() - 1) as f64
    }
    
    fn calculate_volatility_regime(&self, prices: &[f64], window: usize) -> f64 {
        if prices.len() < window {
            return 0.0;
        }
        
        let recent_prices: Vec<f64> = prices.iter().rev().take(window).cloned().collect();
        let mut returns = Vec::new();
        
        for i in 1..recent_prices.len() {
            let ret = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1];
            returns.push(ret);
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt() * (252.0_f64 * 24.0 * 3600.0).sqrt() // Annualized volatility
    }
    
    fn classify_regime(&self, volatility: f64) -> u8 {
        if volatility < 0.2 {
            0 // LOW_VOL
        } else if volatility < 0.5 {
            1 // MEDIUM_VOL
        } else if volatility < 1.0 {
            2 // HIGH_VOL
        } else {
            3 // CRASH
        }
    }
    
    fn calculate_asm_offsets(&self, mid_price: f64, volatility: f64, imbalance: f64) -> (f64, f64) {
        // Base spread from volatility
        let base_spread = mid_price * volatility * 0.1;
        
        // Adjust for inventory
        let inventory_adjustment = self.inventory_scale * self.current_inventory / self.max_inventory;
        
        // Adjust for order book imbalance
        let imbalance_adjustment = imbalance * mid_price * 0.01;
        
        let bid_offset = base_spread / 2.0 + inventory_adjustment + imbalance_adjustment;
        let ask_offset = base_spread / 2.0 - inventory_adjustment - imbalance_adjustment;
        
        (bid_offset, ask_offset)
    }
    
    fn update_microstructure_features(&mut self) {
        if self.price_history.len() < 20 {
            return;
        }
        
        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        let volumes: Vec<f64> = self.volume_history.iter().cloned().collect();
        let current_price = *self.price_history.back().unwrap();
        
        // Update mid price
        let mid_price = self.calculate_mid_price(current_price);
        self.mid_price_history.push_back(mid_price);
        
        // Update spread (simulated)
        let spread = current_price * 0.0001;
        self.spread_history.push_back(spread);
        
        // Update book imbalance
        let imbalance = self.calculate_book_imbalance(&prices, &volumes);
        self.book_imbalance.push_back(imbalance);
        
        // Update trade momentum
        let momentum = self.calculate_tick_momentum(&prices, 10);
        self.trade_momentum.push_back(momentum);
        
        // Update volatility regime
        let volatility = self.calculate_volatility_regime(&prices, 50);
        self.volatility_regime.push_back(volatility);
        
        // Update regime classifier
        let regime = self.classify_regime(volatility);
        self.regime_classifier.push_back(regime);
        
        // Update ASM offsets
        let (bid_offset, ask_offset) = self.calculate_asm_offsets(mid_price, volatility, imbalance);
        self.asm_bid_offset.push_back(bid_offset);
        self.asm_ask_offset.push_back(ask_offset);
        
        // Update OBI signal
        let obi = imbalance * 100.0; // Scale for signal strength
        self.obi_signal.push_back(obi);
        
        // Update depth slopes (simulated)
        let depth_slope_bid = -0.001 * (1.0 + imbalance);
        let depth_slope_ask = 0.001 * (1.0 - imbalance);
        self.depth_slope_bid.push_back(depth_slope_bid);
        self.depth_slope_ask.push_back(depth_slope_ask);
        
        // Update tick momentum
        let tick_mom = self.calculate_tick_momentum(&prices, 5);
        self.tick_momentum.push_back(tick_mom);
        
        // Keep only recent data
        if self.mid_price_history.len() > 1000 {
            self.mid_price_history.pop_front();
            self.spread_history.pop_front();
            self.book_imbalance.pop_front();
            self.trade_momentum.pop_front();
            self.volatility_regime.pop_front();
            self.asm_bid_offset.pop_front();
            self.asm_ask_offset.pop_front();
            self.obi_signal.pop_front();
            self.depth_slope_bid.pop_front();
            self.depth_slope_ask.pop_front();
            self.tick_momentum.pop_front();
            self.regime_classifier.pop_front();
        }
    }
    
    fn generate_pulsar_signal(&self, current_timestamp: f64) -> (Signal, f64) {
        // Check cooldown and consecutive losses
        if current_timestamp - self.last_signal_time < self.signal_cooldown {
            return (Signal::Hold, 0.0);
        }
        
        if self.consecutive_losses >= self.max_consecutive_losses {
            return (Signal::Hold, 0.0);
        }
        
        if self.price_history.len() < 50 {
            return (Signal::Hold, 0.0);
        }
        
        let _current_price = *self.price_history.back().unwrap();
        let _mid_price = *self.mid_price_history.back().unwrap();
        let _imbalance = *self.book_imbalance.back().unwrap();
        let _momentum = *self.trade_momentum.back().unwrap();
        let _volatility = *self.volatility_regime.back().unwrap();
        let regime = *self.regime_classifier.back().unwrap();
        let obi = *self.obi_signal.back().unwrap();
        let tick_mom = *self.tick_momentum.back().unwrap();
        
        // Ensemble decision logic
        
        // 1. ASM (Adaptive Statistical Market-making) signal
        let asm_signal = if self.current_inventory > 0.0 {
            -0.3 // Bias towards selling if long inventory
        } else if self.current_inventory < 0.0 {
            0.3 // Bias towards buying if short inventory
        } else {
            0.0
        };
        
        // 2. OBI (Order-book imbalance) signal
        let obi_signal = if obi > 0.3 {
            0.4 // Strong buy signal
        } else if obi < -0.3 {
            -0.4 // Strong sell signal
        } else {
            obi * 0.3 // Proportional signal
        };
        
        // 3. Tick momentum signal
        let momentum_signal = if tick_mom > 0.0005 {
            0.3
        } else if tick_mom < -0.0005 {
            -0.3
        } else {
            tick_mom * 150.0
        };
        
        // 4. Regime-adjusted signal
        let regime_multiplier = match regime {
            0 => 1.0,   // LOW_VOL: normal
            1 => 0.8,   // MEDIUM_VOL: slightly conservative
            2 => 0.5,   // HIGH_VOL: very conservative
            3 => 0.2,   // CRASH: extremely conservative
            _ => 1.0,
        };
        
        // Combine signals with weights
        let combined_signal = (asm_signal * 0.3 + obi_signal * 0.4 + momentum_signal * 0.3) * regime_multiplier;
        
        // Generate final signal
        if combined_signal > 0.25 {
            return (Signal::Buy, 0.8);
        } else if combined_signal < -0.25 {
            return (Signal::Sell, 0.8);
        } else if combined_signal > 0.08 {
            return (Signal::Buy, 0.6);
        } else if combined_signal < -0.08 {
            return (Signal::Sell, 0.6);
        }
        
        (Signal::Hold, 0.0)
    }
    
    fn _update_inventory(&mut self, signal: Signal, _price: f64) {
        match signal {
            Signal::Buy => {
                self.current_inventory += 1.0;
                // self._inventory_skew = self.inventory_scale * self.current_inventory / self.max_inventory;
            }
            Signal::Sell => {
                self.current_inventory -= 1.0;
                // self._inventory_skew = self.inventory_scale * self.current_inventory / self.max_inventory;
            }
            Signal::Hold => {}
        }
        
        // Clamp inventory
        self.current_inventory = self.current_inventory.max(-self.max_inventory).min(self.max_inventory);
    }
}

#[async_trait]
impl Strategy for PulsarAlphaStrategy {
    fn get_info(&self) -> String {
        let win_rate = if self.win_count + self.loss_count > 0 {
            (self.win_count as f64 / (self.win_count + self.loss_count) as f64) * 100.0
        } else {
            0.0
        };
        let regime = if let Some(regime) = self.regime_classifier.back() {
            match regime {
                0 => "LOW_VOL",
                1 => "MEDIUM_VOL", 
                2 => "HIGH_VOL",
                3 => "CRASH",
                _ => "UNKNOWN",
            }
        } else {
            "UNKNOWN"
        };
        format!("Pulsar-Alpha HFT Strategy - Win Rate: {:.1}%, Regime: {}, Inventory: {:.2}, Total PnL: {:.6}", 
                win_rate, regime, self.current_inventory, self.total_pnl)
    }
    
    async fn on_trade(&mut self, trade: TradeData) {
        self.trade_counter += 1;
        
        // Update price and volume data
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.qty);
        
        // Keep only recent data
        if self.price_history.len() > 1000 {
            self.price_history.pop_front();
            self.volume_history.pop_front();
        }
        
        // Update microstructure features
        self.update_microstructure_features();
    }
    
    fn get_signal(
        &self,
        _current_price: f64,
        current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.price_history.len() < 20 {
            return (Signal::Hold, 0.0);
        }
        
        let (signal, confidence) = self.generate_pulsar_signal(current_timestamp);
        
        // Update inventory based on signal
        if let Some(_price) = self.price_history.back() {
            // Note: In a real implementation, this would be done after order execution
            // For now, we'll simulate it here
        }
        
        (signal, confidence)
    }
}
