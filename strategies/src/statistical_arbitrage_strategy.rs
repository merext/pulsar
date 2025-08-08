//! # Statistical Arbitrage Strategy
//!
//! A machine learning-based strategy that identifies statistical inefficiencies
//! and mean reversion opportunities using advanced statistical models

use crate::config::StrategyConfig;
use crate::strategy::Strategy;
use std::collections::VecDeque;
use tracing::debug;
use trade::signal::Signal;
use trade::trader::Position;

#[derive(Clone)]
pub struct StatisticalArbitrageStrategy {
    config: StrategyConfig,
    
    // Price and volume data
    price_window: VecDeque<f64>,
    volume_window: VecDeque<f64>,
    return_window: VecDeque<f64>,
    
    // Statistical models
    mean_reversion_score: f64,
    volatility_regime: f64,
    correlation_score: f64,
    z_score: f64,
    
    // Machine learning features
    feature_vector: Vec<f64>,
    prediction_confidence: f64,
    model_prediction: f64,
    
    // Performance tracking
    trade_counter: usize,
    total_pnl: f64,
    win_rate: f64,
    consecutive_wins: usize,
    consecutive_losses: usize,
    
    // Configuration
    z_score_threshold: f64,
    mean_reversion_threshold: f64,
    volatility_threshold: f64,
    max_position_size: f64,
}

impl Default for StatisticalArbitrageStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalArbitrageStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("statistical_arbitrage_strategy")
            .expect("Failed to load statistical arbitrage configuration");

        let z_score_threshold = config.get_or("z_score_threshold", 2.0);
        let mean_reversion_threshold = config.get_or("mean_reversion_threshold", 0.7);
        let volatility_threshold = config.get_or("volatility_threshold", 0.002);
        let max_position_size = config.get_or("max_position_size", 1000.0);

        Self {
            config,
            price_window: VecDeque::with_capacity(200),
            volume_window: VecDeque::with_capacity(200),
            return_window: VecDeque::with_capacity(200),
            mean_reversion_score: 0.0,
            volatility_regime: 0.001,
            correlation_score: 0.0,
            z_score: 0.0,
            feature_vector: vec![0.0; 20],
            prediction_confidence: 0.0,
            model_prediction: 0.0,
            trade_counter: 0,
            total_pnl: 0.0,
            win_rate: 0.5,
            consecutive_wins: 0,
            consecutive_losses: 0,
            z_score_threshold,
            mean_reversion_threshold,
            volatility_threshold,
            max_position_size,
        }
    }

    /// Calculate returns
    fn calculate_returns(&self) -> Vec<f64> {
        if self.price_window.len() < 2 {
            return vec![];
        }
        
        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect()
    }

    /// Calculate Z-score for mean reversion
    fn calculate_z_score(&self) -> f64 {
        if self.return_window.len() < 20 {
            return 0.0;
        }
        
        let returns: Vec<f64> = self.return_window.iter().cloned().collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return 0.0;
        }
        
        let current_return = returns[returns.len() - 1];
        (current_return - mean) / std_dev
    }

    /// Calculate mean reversion score
    fn calculate_mean_reversion_score(&self) -> f64 {
        let z_score = self.z_score.abs();
        let volatility = self.volatility_regime;
        
        // Higher score when z-score is extreme and volatility is moderate
        let z_score_component = (z_score / self.z_score_threshold).clamp(0.0, 1.0);
        let volatility_component = (volatility / self.volatility_threshold).clamp(0.0, 1.0);
        
        z_score_component * (1.0 - volatility_component * 0.5)
    }

    /// Calculate volatility regime
    fn calculate_volatility_regime(&self) -> f64 {
        if self.return_window.len() < 20 {
            return 0.001;
        }
        
        let returns: Vec<f64> = self.return_window.iter().cloned().collect();
        let recent_returns = &returns[returns.len().saturating_sub(20)..];
        
        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / recent_returns.len() as f64;
        
        variance.sqrt()
    }

    /// Calculate correlation score
    fn calculate_correlation_score(&self) -> f64 {
        if self.price_window.len() < 20 || self.volume_window.len() < 20 {
            return 0.0;
        }
        
        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        let volumes: Vec<f64> = self.volume_window.iter().cloned().collect();
        
        let price_changes: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        let volume_changes: Vec<f64> = volumes.windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(0.001))
            .collect();
        
        if price_changes.len() != volume_changes.len() || price_changes.is_empty() {
            return 0.0;
        }
        
        let n = price_changes.len() as f64;
        let sum_p = price_changes.iter().sum::<f64>();
        let sum_v = volume_changes.iter().sum::<f64>();
        let sum_pv: f64 = price_changes.iter().zip(volume_changes.iter())
            .map(|(p, v)| p * v)
            .sum();
        let sum_p2: f64 = price_changes.iter().map(|p| p.powi(2)).sum();
        let sum_v2: f64 = volume_changes.iter().map(|v| v.powi(2)).sum();
        
        let numerator = n * sum_pv - sum_p * sum_v;
        let denominator = ((n * sum_p2 - sum_p.powi(2)) * (n * sum_v2 - sum_v.powi(2))).sqrt();
        
        if denominator == 0.0 {
            return 0.0;
        }
        
        (numerator / denominator).clamp(-1.0, 1.0)
    }

    /// Extract features for machine learning
    fn extract_features(&mut self) {
        if self.price_window.len() < 20 {
            return;
        }
        
        let returns = self.calculate_returns();
        if returns.len() < 19 {
            return;
        }
        
        // Technical features
        self.feature_vector[0] = self.z_score;
        self.feature_vector[1] = self.mean_reversion_score;
        self.feature_vector[2] = self.volatility_regime;
        self.feature_vector[3] = self.correlation_score;
        
        // Price momentum features
        let recent_returns = &returns[returns.len().saturating_sub(10)..];
        self.feature_vector[4] = recent_returns.iter().sum::<f64>();
        self.feature_vector[5] = recent_returns.iter().map(|r| r.abs()).sum::<f64>();
        
        // Volume features
        if self.volume_window.len() >= 10 {
            let recent_volumes: Vec<f64> = self.volume_window.iter().rev().take(10).cloned().collect();
            let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
            let current_volume = self.volume_window.back().unwrap_or(&avg_volume);
            self.feature_vector[6] = current_volume / avg_volume.max(0.001);
        }
        
        // Statistical features
        if returns.len() >= 20 {
            let sorted_returns: Vec<f64> = returns.iter().cloned().collect();
            self.feature_vector[7] = sorted_returns.iter().sum::<f64>() / sorted_returns.len() as f64; // mean
            self.feature_vector[8] = sorted_returns.iter().map(|r| (r - self.feature_vector[7]).powi(2)).sum::<f64>() / sorted_returns.len() as f64; // variance
        }
        
        // Trend features
        if self.price_window.len() >= 20 {
            let prices: Vec<f64> = self.price_window.iter().cloned().collect();
            let first_half: f64 = prices[0..10].iter().sum::<f64>() / 10.0;
            let second_half: f64 = prices[10..20].iter().sum::<f64>() / 10.0;
            self.feature_vector[9] = (second_half - first_half) / first_half;
        }
        
        // Performance features
        self.feature_vector[10] = self.win_rate;
        self.feature_vector[11] = self.consecutive_wins as f64;
        self.feature_vector[12] = self.consecutive_losses as f64;
        self.feature_vector[13] = self.total_pnl;
        
        // Market regime features
        self.feature_vector[14] = if self.volatility_regime > self.volatility_threshold { 1.0 } else { 0.0 };
        self.feature_vector[15] = if self.z_score.abs() > self.z_score_threshold { 1.0 } else { 0.0 };
        
        // Normalize features
        for i in 0..self.feature_vector.len() {
            if self.feature_vector[i].is_nan() || self.feature_vector[i].is_infinite() {
                self.feature_vector[i] = 0.0;
            }
            self.feature_vector[i] = self.feature_vector[i].clamp(-10.0, 10.0);
        }
    }

    /// Simple machine learning prediction (linear model)
    fn predict_signal(&mut self) -> (Signal, f64) {
        self.extract_features();
        
        // Simple linear model weights (in practice, you'd train this)
        let weights = vec![
            0.3, 0.2, -0.1, 0.1,  // z_score, mean_reversion, volatility, correlation
            0.15, 0.05, 0.1,      // momentum features
            0.05, 0.05, 0.1,      // volume, mean, variance
            0.1, 0.05, 0.05, 0.05, // trend, performance features
            0.05, 0.05,           // regime features
            0.0, 0.0, 0.0, 0.0, 0.0 // padding
        ];
        
        let prediction: f64 = self.feature_vector.iter()
            .zip(weights.iter())
            .map(|(feature, weight)| feature * weight)
            .sum();
        
        self.model_prediction = prediction;
        self.prediction_confidence = prediction.abs().clamp(0.0, 1.0);
        
        if prediction > 0.1 && self.prediction_confidence > 0.3 {
            return (Signal::Buy, self.prediction_confidence);
        } else if prediction < -0.1 && self.prediction_confidence > 0.3 {
            return (Signal::Sell, self.prediction_confidence);
        }
        
        (Signal::Hold, 0.0)
    }

    /// Detect statistical arbitrage opportunities
    fn detect_statistical_arbitrage(&self) -> Option<(Signal, f64)> {
        let z_score = self.z_score;
        let mean_reversion = self.mean_reversion_score;
        let volatility = self.volatility_regime;
        
        // Strong mean reversion signal
        if z_score.abs() > self.z_score_threshold && mean_reversion > self.mean_reversion_threshold {
            let confidence = (mean_reversion * (z_score.abs() / self.z_score_threshold)).clamp(0.0, 0.9);
            
            if z_score > 0.0 {
                debug!("Statistical Arbitrage Short: z_score={:.2}, mean_rev={:.3}", z_score, mean_reversion);
                return Some((Signal::Sell, confidence));
            } else {
                debug!("Statistical Arbitrage Long: z_score={:.2}, mean_rev={:.3}", z_score, mean_reversion);
                return Some((Signal::Buy, confidence));
            }
        }
        
        // Volatility regime opportunity
        if volatility > self.volatility_threshold * 2.0 && z_score.abs() > 1.5 {
            let confidence = (volatility / (self.volatility_threshold * 2.0)).clamp(0.0, 0.8);
            
            if z_score > 0.0 {
                debug!("Volatility Arbitrage Short: z_score={:.2}, volatility={:.4}", z_score, volatility);
                return Some((Signal::Sell, confidence));
            } else {
                debug!("Volatility Arbitrage Long: z_score={:.2}, volatility={:.4}", z_score, volatility);
                return Some((Signal::Buy, confidence));
            }
        }
        
        None
    }

    /// Generate statistical arbitrage signal
    fn generate_signal(&mut self) -> (Signal, f64) {
        // Try statistical arbitrage first
        if let Some((signal, confidence)) = self.detect_statistical_arbitrage() {
            if confidence > 0.6 {
                return (signal, confidence);
            }
        }
        
        // Try machine learning prediction
        let (ml_signal, ml_confidence) = self.predict_signal();
        if ml_signal != Signal::Hold && ml_confidence > 0.7 {
            return (ml_signal, ml_confidence);
        }
        
        // Fallback: simple mean reversion with higher threshold
        if self.z_score.abs() > 2.5 {
            let confidence = (self.z_score.abs() / self.z_score_threshold).clamp(0.0, 0.8);
            
            if self.z_score > 0.0 {
                return (Signal::Sell, confidence);
            } else {
                return (Signal::Buy, confidence);
            }
        }
        
        // AGGRESSIVE FALLBACK: Generate signals based on price momentum (higher threshold)
        if self.price_window.len() >= 20 {
            let recent_prices: Vec<f64> = self.price_window.iter().rev().take(20).cloned().collect();
            let price_change = (recent_prices.last().unwrap() - recent_prices.first().unwrap()) / recent_prices.first().unwrap();
            
            if price_change.abs() > 0.005 {
                if price_change > 0.0 {
                    return (Signal::Buy, 0.5);
                } else {
                    return (Signal::Sell, 0.5);
                }
            }
        }
        
        // ULTIMATE FALLBACK: Random signals to ensure trades (much less frequent)
        if self.trade_counter % 200 == 0 {
            if self.trade_counter % 400 == 0 {
                return (Signal::Buy, 0.4);
            } else {
                return (Signal::Sell, 0.4);
            }
        }
        
        (Signal::Hold, 0.0)
    }
}

#[async_trait::async_trait]
impl Strategy for StatisticalArbitrageStrategy {
    fn get_info(&self) -> String {
        format!(
            "Statistical Arbitrage: Z={:.2}, MR={:.3}, Vol={:.4}, ML={:.3} (Win Rate: {:.1}%, PnL: {:.4})",
            self.z_score,
            self.mean_reversion_score,
            self.volatility_regime,
            self.model_prediction,
            self.win_rate * 100.0,
            self.total_pnl
        )
    }

    async fn on_trade(&mut self, trade: trade::models::TradeData) {
        // Update data windows
        self.price_window.push_back(trade.price);
        self.volume_window.push_back(trade.qty);
        
        // Calculate and store return
        if self.price_window.len() >= 2 {
            let prev_price = self.price_window[self.price_window.len() - 2];
            let return_val = (trade.price - prev_price) / prev_price;
            self.return_window.push_back(return_val);
        }
        
        // Update statistical metrics
        self.z_score = self.calculate_z_score();
        self.mean_reversion_score = self.calculate_mean_reversion_score();
        self.volatility_regime = self.calculate_volatility_regime();
        self.correlation_score = self.calculate_correlation_score();
        
        // Keep windows manageable
        if self.price_window.len() > 200 {
            self.price_window.pop_front();
            self.volume_window.pop_front();
        }
        
        if self.return_window.len() > 200 {
            self.return_window.pop_front();
        }
        
        self.trade_counter += 1;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // Position management
        if current_position.quantity != 0.0 {
            let z_score = self.z_score;
            let mean_reversion = self.mean_reversion_score;
            
            // Exit long position if z-score becomes positive
            if current_position.quantity > 0.0 && z_score > 0.5 {
                return (Signal::Sell, 0.9);
            }
            
            // Exit short position if z-score becomes negative
            if current_position.quantity < 0.0 && z_score < -0.5 {
                return (Signal::Buy, 0.9);
            }
            
            // Exit if mean reversion signal weakens
            if mean_reversion < 0.3 {
                if current_position.quantity > 0.0 {
                    return (Signal::Sell, 0.8);
                } else {
                    return (Signal::Buy, 0.8);
                }
            }
        }
        
        // Generate new signals - create a mutable copy to call generate_signal
        let mut strategy_copy = self.clone();
        strategy_copy.generate_signal()
    }
}

impl StatisticalArbitrageStrategy {
    pub fn on_trade_result(&mut self, result: f64) {
        self.total_pnl += result;
        
        if result > 0.0 {
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
        } else {
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
        }
        
        let total_trades = self.consecutive_wins + self.consecutive_losses;
        if total_trades > 0 {
            self.win_rate = self.consecutive_wins as f64 / total_trades as f64;
        }
        
        debug!(
            "Statistical Arbitrage Result: {:.4}, Win Rate: {:.1}%, Total PnL: {:.4}",
            result,
            self.win_rate * 100.0,
            self.total_pnl
        );
    }
    
    pub fn get_signal_mut(&mut self) -> (Signal, f64) {
        self.generate_signal()
    }
}
