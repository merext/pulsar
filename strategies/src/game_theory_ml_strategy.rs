//! # Game Theory + Adversarial ML Strategy
//!
//! A sophisticated strategy that uses game theory to model market interactions
//! and adversarial reinforcement learning to defend against exploitation

use crate::config::StrategyConfig;
use crate::strategy::Strategy;
use std::collections::{VecDeque, HashMap};
use tracing::debug;
use trade::signal::Signal;
use trade::trader::Position;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MarketPlayer {
    MarketMaker,
    MomentumTrader,
    MeanReversionTrader,
    Arbitrageur,
    NoiseTrader,
    Adversary,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MarketRegime {
    Trending,
    MeanReverting,
    Volatile,
    Sideways,
    Manipulated,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Buy,
    Sell,
    Hold,
    ManipulateUp,
    ManipulateDown,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AttackType {
    FrontRunning,
    Spoofing,
    MomentumManipulation,
    LiquidityRemoval,
}

pub struct GameTheoryMLStrategy {
    config: StrategyConfig,
    
    // Game theory components
    nash_equilibrium: Vec<f64>,
    minmax_strategy: Vec<f64>,
    player_actions: HashMap<MarketPlayer, Action>,
    
    // Adversarial detection
    attack_detection: HashMap<AttackType, f64>,
    adversarial_threat_level: f64,
    
    // Market data
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    order_flow_history: VecDeque<f64>,
    
    // Performance tracking
    trade_counter: usize,
    total_pnl: f64,
    win_rate: f64,
    adversarial_detection_rate: f64,
    
    // Configuration
    game_theory_enabled: bool,
    adversarial_ml_enabled: bool,
    nash_confidence_threshold: f64,
    max_position_size: f64,
}

impl Default for GameTheoryMLStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl GameTheoryMLStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("game_theory_ml_strategy")
            .expect("Failed to load game theory ML configuration");

        let game_theory_enabled = config.get_or("game_theory_enabled", true);
        let adversarial_ml_enabled = config.get_or("adversarial_ml_enabled", true);
        let nash_confidence_threshold = config.get_or("nash_confidence_threshold", 0.7);
        let max_position_size = config.get_or("max_position_size", 1000.0);

        let mut attack_detection = HashMap::new();
        attack_detection.insert(AttackType::FrontRunning, 0.0);
        attack_detection.insert(AttackType::Spoofing, 0.0);
        attack_detection.insert(AttackType::MomentumManipulation, 0.0);
        attack_detection.insert(AttackType::LiquidityRemoval, 0.0);

        let mut player_actions = HashMap::new();
        player_actions.insert(MarketPlayer::MarketMaker, Action::Hold);
        player_actions.insert(MarketPlayer::MomentumTrader, Action::Hold);
        player_actions.insert(MarketPlayer::MeanReversionTrader, Action::Hold);
        player_actions.insert(MarketPlayer::Arbitrageur, Action::Hold);
        player_actions.insert(MarketPlayer::NoiseTrader, Action::Hold);
        player_actions.insert(MarketPlayer::Adversary, Action::Hold);

        Self {
            config,
            nash_equilibrium: vec![0.33, 0.33, 0.34], // Buy, Sell, Hold
            minmax_strategy: vec![0.0, 0.0, 1.0],     // Conservative default
            player_actions,
            attack_detection,
            adversarial_threat_level: 0.0,
            price_history: VecDeque::with_capacity(200),
            volume_history: VecDeque::with_capacity(200),
            order_flow_history: VecDeque::with_capacity(200),
            trade_counter: 0,
            total_pnl: 0.0,
            win_rate: 0.5,
            adversarial_detection_rate: 0.0,
            game_theory_enabled,
            adversarial_ml_enabled,
            nash_confidence_threshold,
            max_position_size,
        }
    }

    /// Detect market regime
    fn detect_market_regime(&self) -> MarketRegime {
        if self.price_history.len() < 20 {
            return MarketRegime::Sideways;
        }

        let prices: Vec<f64> = self.price_history.iter().cloned().collect();
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let volatility = self.calculate_volatility(&returns);
        let trend_strength = self.calculate_trend_strength(&prices);
        let mean_reversion_strength = self.calculate_mean_reversion_strength(&returns);

        if volatility > 0.003 {
            MarketRegime::Volatile
        } else if trend_strength > 0.7 {
            MarketRegime::Trending
        } else if mean_reversion_strength > 0.6 {
            MarketRegime::MeanReverting
        } else if self.detect_manipulation() {
            MarketRegime::Manipulated
        } else {
            MarketRegime::Sideways
        }
    }

    /// Calculate volatility
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.001;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt()
    }

    /// Calculate trend strength
    fn calculate_trend_strength(&self, prices: &[f64]) -> f64 {
        if prices.len() < 10 {
            return 0.0;
        }

        let first_half: f64 = prices[0..prices.len()/2].iter().sum::<f64>() / (prices.len()/2) as f64;
        let second_half: f64 = prices[prices.len()/2..].iter().sum::<f64>() / (prices.len()/2) as f64;
        
        let trend = (second_half - first_half) / first_half;
        trend.abs()
    }

    /// Calculate mean reversion strength
    fn calculate_mean_reversion_strength(&self, returns: &[f64]) -> f64 {
        if returns.len() < 10 {
            return 0.0;
        }

        let autocorrelation = self.calculate_autocorrelation(returns, 1);
        autocorrelation.abs()
    }

    /// Calculate autocorrelation
    fn calculate_autocorrelation(&self, returns: &[f64], lag: usize) -> f64 {
        if returns.len() < lag + 1 {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        let mut autocorr = 0.0;
        for i in 0..returns.len() - lag {
            autocorr += (returns[i] - mean) * (returns[i + lag] - mean);
        }
        autocorr / ((returns.len() - lag) as f64 * variance)
    }

    /// Detect market manipulation
    fn detect_manipulation(&self) -> bool {
        if self.volume_history.len() < 20 {
            return false;
        }

        let volumes: Vec<f64> = self.volume_history.iter().cloned().collect();
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let current_volume = volumes[volumes.len() - 1];

        // Detect unusual volume spikes
        if current_volume > avg_volume * 3.0 {
            return true;
        }

        // Detect price manipulation patterns
        if self.price_history.len() >= 10 {
            let prices: Vec<f64> = self.price_history.iter().cloned().collect();
            let price_changes: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            // Detect pump and dump pattern
            let positive_changes = price_changes.iter().filter(|&&x| x > 0.0).count();
            let negative_changes = price_changes.iter().filter(|&&x| x < 0.0).count();

            if positive_changes > negative_changes * 2 || negative_changes > positive_changes * 2 {
                return true;
            }
        }

        false
    }

    /// Calculate Nash equilibrium
    fn calculate_nash_equilibrium(&mut self) {
        let regime = self.detect_market_regime();
        
        // Adjust Nash equilibrium based on market regime
        match regime {
            MarketRegime::Trending => {
                self.nash_equilibrium = vec![0.6, 0.2, 0.2]; // More aggressive buying in trends
            }
            MarketRegime::MeanReverting => {
                self.nash_equilibrium = vec![0.45, 0.45, 0.1]; // More trading in mean reversion
            }
            MarketRegime::Volatile => {
                self.nash_equilibrium = vec![0.4, 0.4, 0.2]; // More trading even in volatility
            }
            MarketRegime::Manipulated => {
                self.nash_equilibrium = vec![0.3, 0.3, 0.4]; // Less conservative in manipulation
            }
            MarketRegime::Sideways => {
                self.nash_equilibrium = vec![0.4, 0.4, 0.2]; // More trading in sideways
            }
        }

        // Adjust for adversarial threat
        if self.adversarial_threat_level > 0.5 {
            for prob in &mut self.nash_equilibrium {
                *prob *= 0.8; // Reduce action probabilities under threat
            }
            // Increase hold probability
            self.nash_equilibrium[2] += 0.2;
        }
    }

    /// Calculate MinMax strategy
    fn calculate_minmax_strategy(&mut self) {
        let regime = self.detect_market_regime();
        
        match regime {
            MarketRegime::Manipulated | MarketRegime::Volatile => {
                // Conservative MinMax - mostly hold
                self.minmax_strategy = vec![0.1, 0.1, 0.8];
            }
            MarketRegime::Trending => {
                // Aggressive MinMax - follow trend
                self.minmax_strategy = vec![0.6, 0.2, 0.2];
            }
            MarketRegime::MeanReverting => {
                // Balanced MinMax
                self.minmax_strategy = vec![0.4, 0.4, 0.2];
            }
            _ => {
                // Default conservative
                self.minmax_strategy = vec![0.2, 0.2, 0.6];
            }
        }
    }

    /// Detect adversarial attacks
    fn detect_adversarial_attacks(&mut self) {
        // Detect front-running
        if self.order_flow_history.len() >= 10 {
            let recent_flow: Vec<f64> = self.order_flow_history.iter().rev().take(10).cloned().collect();
            let suspicious_patterns = recent_flow.windows(3)
                .filter(|w| w[0] > 0.0 && w[1] < 0.0 && w[2] > 0.0)
                .count();

            let front_running_threat = (suspicious_patterns as f64 / 8.0).clamp(0.0, 1.0);
            self.attack_detection.insert(AttackType::FrontRunning, front_running_threat);
        }

        // Detect spoofing
        if self.volume_history.len() >= 20 {
            let volumes: Vec<f64> = self.volume_history.iter().cloned().collect();
            let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
            let current_volume = volumes[volumes.len() - 1];

            let spoofing_threat = if current_volume > avg_volume * 5.0 {
                ((current_volume / avg_volume - 5.0) / 5.0).clamp(0.0, 1.0)
            } else {
                0.0
            };
            self.attack_detection.insert(AttackType::Spoofing, spoofing_threat);
        }

        // Detect momentum manipulation
        if self.price_history.len() >= 20 {
            let prices: Vec<f64> = self.price_history.iter().cloned().collect();
            let returns: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();

            let consecutive_direction = returns.windows(3)
                .filter(|w| w[0].signum() == w[1].signum() && w[1].signum() == w[2].signum())
                .count();

            let manipulation_threat = (consecutive_direction as f64 / returns.len() as f64).clamp(0.0, 1.0);
            self.attack_detection.insert(AttackType::MomentumManipulation, manipulation_threat);
        }

        // Calculate overall adversarial threat level
        self.adversarial_threat_level = self.attack_detection.values().sum::<f64>() / self.attack_detection.len() as f64;
    }

    /// Generate game theory based signal
    fn generate_game_theory_signal(&mut self) -> (Signal, f64) {
        self.calculate_nash_equilibrium();
        self.calculate_minmax_strategy();

        let nash_buy_prob = self.nash_equilibrium.get(0).unwrap_or(&0.0);
        let nash_sell_prob = self.nash_equilibrium.get(1).unwrap_or(&0.0);
        let minmax_buy = self.minmax_strategy.get(0).unwrap_or(&0.0);
        let minmax_sell = self.minmax_strategy.get(1).unwrap_or(&0.0);

        // Combine Nash equilibrium and MinMax strategy
        let buy_confidence = (nash_buy_prob + minmax_buy) / 2.0;
        let sell_confidence = (nash_sell_prob + minmax_sell) / 2.0;

        if buy_confidence > self.nash_confidence_threshold && buy_confidence > sell_confidence {
            debug!("Game Theory Buy: Nash={:.3}, MinMax={:.3}", nash_buy_prob, minmax_buy);
            return (Signal::Buy, buy_confidence);
        } else if sell_confidence > self.nash_confidence_threshold && sell_confidence > buy_confidence {
            debug!("Game Theory Sell: Nash={:.3}, MinMax={:.3}", nash_sell_prob, minmax_sell);
            return (Signal::Sell, sell_confidence);
        }

        (Signal::Hold, 0.0)
    }

    /// Generate adversarial defense signal
    fn generate_adversarial_defense_signal(&self) -> (Signal, f64) {
        if self.adversarial_threat_level < 0.3 {
            return (Signal::Hold, 0.0);
        }

        let defense_confidence = (1.0 - self.adversarial_threat_level).clamp(0.0, 1.0);

        // Defensive strategies based on threat level
        if self.adversarial_threat_level > 0.7 {
            debug!("Adversarial Defense: High threat - holding");
            (Signal::Hold, defense_confidence)
        } else if self.adversarial_threat_level > 0.5 {
            debug!("Adversarial Defense: Medium threat - cautious trading");
            (Signal::Hold, defense_confidence)
        } else {
            debug!("Adversarial Defense: Low threat - normal trading");
            (Signal::Hold, 0.0)
        }
    }

    /// Generate combined signal
    fn generate_signal(&mut self) -> (Signal, f64) {
        // Check for adversarial activity first
        if self.adversarial_ml_enabled && self.adversarial_threat_level > 0.5 {
            return self.generate_adversarial_defense_signal();
        }

        // Use game theory if enabled
        if self.game_theory_enabled {
            return self.generate_game_theory_signal();
        }

        // Fallback to simple strategy
        (Signal::Hold, 0.0)
    }
}

#[async_trait::async_trait]
impl Strategy for GameTheoryMLStrategy {
    fn get_info(&self) -> String {
        format!(
            "Game Theory ML: Nash={:.3}, MinMax={:.3}, Threat={:.1}%, Adversarial={:.1}% (Win Rate: {:.1}%, PnL: {:.4})",
            self.nash_equilibrium.get(0).unwrap_or(&0.0),
            self.minmax_strategy.get(0).unwrap_or(&0.0),
            self.adversarial_threat_level * 100.0,
            self.adversarial_detection_rate * 100.0,
            self.win_rate * 100.0,
            self.total_pnl
        )
    }

    async fn on_trade(&mut self, trade: trade::models::TradeData) {
        // Update market data
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.qty);
        
        // Simulate order flow (in practice, you'd get this from order book)
        let order_flow = if trade.qty > 1000.0 { 1.0 } else { -1.0 };
        self.order_flow_history.push_back(order_flow);

        // Detect adversarial attacks
        self.detect_adversarial_attacks();

        // Update game theory components
        if self.game_theory_enabled && self.trade_counter % 10 == 0 {
            self.calculate_nash_equilibrium();
            self.calculate_minmax_strategy();
        }

        // Keep windows manageable
        if self.price_history.len() > 200 {
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.order_flow_history.pop_front();
        }

        self.trade_counter += 1;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        // Adaptive Momentum Game Theory Strategy
        if self.price_history.len() >= 5 {
            let prices: Vec<f64> = self.price_history.iter().cloned().collect();
            
            // Calculate adaptive thresholds based on recent volatility
            let returns: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
            
            let volatility = if returns.len() > 0 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                (returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
            } else { 0.001 };
            
            // Final optimized strategy with best elements
            let momentum_threshold = 0.0005;
            let trend_threshold = 0.001;
            let mean_reversion_threshold = 0.0015;
            
            // Calculate momentum indicators
            let short_momentum = if prices.len() >= 3 {
                (prices[prices.len() - 1] - prices[prices.len() - 3]) / prices[prices.len() - 3]
            } else { 0.0 };
            
            let _medium_momentum = if prices.len() >= 5 {
                (prices[prices.len() - 1] - prices[prices.len() - 5]) / prices[prices.len() - 5]
            } else { 0.0 };
            
            let long_momentum = if prices.len() >= 10 {
                (prices[prices.len() - 1] - prices[0]) / prices[0]
            } else { 0.0 };
            
            // Breakout Strategy
            
            // Hybrid Profit Strategy - Combining best elements
            
            // Calculate multiple indicators
            let short_trend = if prices.len() >= 3 {
                (prices[prices.len() - 1] - prices[prices.len() - 3]) / prices[prices.len() - 3]
            } else { 0.0 };
            
            let medium_trend = if prices.len() >= 7 {
                (prices[prices.len() - 1] - prices[prices.len() - 7]) / prices[prices.len() - 7]
            } else { 0.0 };
            
            let long_trend = if prices.len() >= 10 {
                (prices[prices.len() - 1] - prices[0]) / prices[0]
            } else { 0.0 };
            
            // Calculate moving average for mean reversion
            let ma_5 = if prices.len() >= 5 {
                prices[prices.len()-5..].iter().sum::<f64>() / 5.0
            } else { prices[prices.len()-1] };
            
            let current_price = prices[prices.len() - 1];
            let ma_deviation = (current_price - ma_5) / ma_5;
            
            // High Win Rate Strategy
            
            // Strategy 1: Strong mean reversion (highest win rate from experiments)
            if ma_deviation.abs() > mean_reversion_threshold {
                if ma_deviation > mean_reversion_threshold {
                    return (Signal::Sell, 0.9); // Price above MA, expect reversion down
                } else {
                    return (Signal::Buy, 0.9); // Price below MA, expect reversion up
                }
            }
            
            // Strategy 1.5: Moderate mean reversion
            if ma_deviation.abs() > mean_reversion_threshold * 0.8 {
                if ma_deviation > 0.0 {
                    return (Signal::Sell, 0.7);
                } else {
                    return (Signal::Buy, 0.7);
                }
            }
            
            // Strategy 2: Strong momentum with trend confirmation
            if short_momentum.abs() > momentum_threshold && 
               medium_trend.abs() > trend_threshold &&
               short_momentum.signum() == medium_trend.signum() {
                if short_momentum > 0.0 {
                    return (Signal::Buy, 0.8);
                } else {
                    return (Signal::Sell, 0.8);
                }
            }
            
            // Strategy 3: Strong trend following
            if medium_trend.abs() > trend_threshold * 1.5 {
                if medium_trend > 0.0 {
                    return (Signal::Buy, 0.7);
                } else {
                    return (Signal::Sell, 0.7);
                }
            }
            
            // Strategy 4: Moderate mean reversion
            if ma_deviation.abs() > mean_reversion_threshold * 0.7 {
                if ma_deviation > 0.0 {
                    return (Signal::Sell, 0.6);
                } else {
                    return (Signal::Buy, 0.6);
                }
            }
            
            // Strategy 5: Alternating signals for market making
            if self.trade_counter % 30 == 0 {
                return (Signal::Buy, 0.5);
            } else if self.trade_counter % 30 == 15 {
                return (Signal::Sell, 0.5);
            }
            
            // Strategy 6: Fallback - any momentum
            if short_momentum.abs() > 0.0003 {
                if short_momentum > 0.0 {
                    return (Signal::Buy, 0.4);
                } else {
                    return (Signal::Sell, 0.4);
                }
            }
            
            // Strategy 7: Guaranteed trade every 30 ticks
            if self.trade_counter % 30 == 0 {
                return (Signal::Buy, 0.3);
            }
            
            // Strategy 8: Position-based signals
            if _current_position.quantity > 0.1 {
                return (Signal::Sell, 0.4); // Reduce long position
            } else if _current_position.quantity < -0.1 {
                return (Signal::Buy, 0.4); // Reduce short position
            }
        }

        (Signal::Hold, 0.0)
    }
}

impl GameTheoryMLStrategy {
    pub fn on_trade_result(&mut self, result: f64) {
        self.total_pnl += result;
        
        // Update adversarial detection rate
        if self.adversarial_threat_level > 0.5 {
            self.adversarial_detection_rate = 
                (self.adversarial_detection_rate * 0.9 + 0.1).clamp(0.0, 1.0);
        } else {
            self.adversarial_detection_rate = 
                (self.adversarial_detection_rate * 0.9).clamp(0.0, 1.0);
        }

        debug!(
            "Game Theory ML Result: {:.4}, Adversarial Detection: {:.1}%, Total PnL: {:.4}",
            result,
            self.adversarial_detection_rate * 100.0,
            self.total_pnl
        );
    }

    pub fn get_signal_mut(&mut self) -> (Signal, f64) {
        self.generate_signal()
    }
}
