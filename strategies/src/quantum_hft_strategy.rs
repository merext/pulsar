//! # Fractal-Based HFT Strategy
//! 
//! A sophisticated trading strategy based on fractal geometry and market structure analysis

use crate::config::StrategyConfig;

use crate::strategy::Strategy;
use std::collections::VecDeque;
use tracing::debug;
use trade::signal::Signal;
use trade::trader::Position;

#[derive(Debug, Clone)]
struct FeatureVector {
    price_features: Vec<f64>,
    volume_features: Vec<f64>,
    technical_features: Vec<f64>,
    statistical_features: Vec<f64>,
    temporal_features: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PredictionModel {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<f64>,
}

#[derive(Debug, Clone)]
struct StatisticalModel {
    mean: f64,
    variance: f64,
    skewness: f64,
    kurtosis: f64,
    autocorrelation: Vec<f64>,
    volatility_regime: VolatilityRegime,
}

#[derive(Debug, Clone, PartialEq)]
enum VolatilityRegime {
    Low,
    Medium,
    High,
    Extreme,
}

#[derive(Debug, Clone)]
struct MarketMicrostructure {
    bid_ask_spread: f64,
    order_flow_imbalance: f64,
    volume_profile: Vec<f64>,
    price_impact: f64,
    market_depth: f64,
}

#[derive(Debug, Clone)]
struct TimeSeriesModel {
    ar_coefficients: Vec<f64>,
    ma_coefficients: Vec<f64>,
    seasonal_patterns: Vec<f64>,
    trend_component: f64,
    residual_variance: f64,
}

pub struct QuantumHftStrategy {
    config: StrategyConfig,
    
    // Data windows
    price_window: VecDeque<f64>,
    volume_window: VecDeque<f64>,
    timestamp_window: VecDeque<f64>,
    
    // Feature engineering
    feature_window: VecDeque<FeatureVector>,
    feature_dimension: usize,
    
    // ML Models
    price_prediction_model: PredictionModel,
    volatility_model: StatisticalModel,
    microstructure_model: MarketMicrostructure,
    time_series_model: TimeSeriesModel,
    
    // Ensemble weights
    model_weights: Vec<f64>,
    
    // Prediction tracking
    predictions: VecDeque<f64>,
    actual_movements: VecDeque<f64>,
    prediction_accuracy: f64,
    
    // Advanced statistics
    rolling_mean: VecDeque<f64>,
    rolling_std: VecDeque<f64>,
    rolling_skew: VecDeque<f64>,
    rolling_kurt: VecDeque<f64>,
    
    // Market regime detection
    current_regime: VolatilityRegime,
    regime_confidence: f64,
    
    // Performance tracking
    win_rate: f64,
    consecutive_wins: usize,
    consecutive_losses: usize,
    trade_counter: usize,
    
    // Risk management
    position_size_multiplier: f64,
    max_position_size: f64,
    stop_loss_threshold: f64,
    take_profit_threshold: f64,
}

impl QuantumHftStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("quantum_hft_strategy")
            .expect("Failed to load quantum_hft_strategy configuration");
        
        let feature_dimension = config.get_or("feature_dimension", 50);
        let learning_rate = config.get_or("learning_rate", 0.001);
        let momentum = config.get_or("momentum", 0.9);
        let max_position_size = config.get_or("max_position_size", 1000.0);
        let stop_loss_threshold = config.get_or("stop_loss_threshold", 0.02);
        let take_profit_threshold = config.get_or("take_profit_threshold", 0.03);
        
        Self {
            config,
            price_window: VecDeque::with_capacity(1000),
            volume_window: VecDeque::with_capacity(1000),
            timestamp_window: VecDeque::with_capacity(1000),
            feature_window: VecDeque::with_capacity(100),
            feature_dimension,
            price_prediction_model: PredictionModel {
                weights: vec![0.0; feature_dimension],
                bias: 0.0,
                learning_rate,
                momentum,
                velocity: vec![0.0; feature_dimension],
            },
            volatility_model: StatisticalModel {
                mean: 0.0,
                variance: 1.0,
                skewness: 0.0,
                kurtosis: 3.0,
                autocorrelation: vec![0.0; 20],
                volatility_regime: VolatilityRegime::Medium,
            },
            microstructure_model: MarketMicrostructure {
                bid_ask_spread: 0.0001,
                order_flow_imbalance: 0.0,
                volume_profile: vec![0.0; 10],
                price_impact: 0.0,
                market_depth: 1000.0,
            },
            time_series_model: TimeSeriesModel {
                ar_coefficients: vec![0.0; 5],
                ma_coefficients: vec![0.0; 5],
                seasonal_patterns: vec![0.0; 24],
                trend_component: 0.0,
                residual_variance: 1.0,
            },
            model_weights: vec![0.25, 0.25, 0.25, 0.25], // Equal weights initially
            predictions: VecDeque::with_capacity(100),
            actual_movements: VecDeque::with_capacity(100),
            prediction_accuracy: 0.5,
            rolling_mean: VecDeque::with_capacity(100),
            rolling_std: VecDeque::with_capacity(100),
            rolling_skew: VecDeque::with_capacity(100),
            rolling_kurt: VecDeque::with_capacity(100),
            current_regime: VolatilityRegime::Medium,
            regime_confidence: 0.5,
            win_rate: 0.5,
            consecutive_wins: 0,
            consecutive_losses: 0,
            trade_counter: 0,
            position_size_multiplier: 1.0,
            max_position_size,
            stop_loss_threshold,
            take_profit_threshold,
        }
    }

    /// Extract comprehensive features from market data
    fn extract_features(&mut self) -> FeatureVector {
        if self.price_window.len() < 50 {
            return FeatureVector {
                price_features: vec![0.0; 20],
                volume_features: vec![0.0; 10],
                technical_features: vec![0.0; 10],
                statistical_features: vec![0.0; 5],
                temporal_features: vec![0.0; 5],
            };
        }

        let prices: Vec<f64> = self.price_window.iter().cloned().collect();
        let volumes: Vec<f64> = self.volume_window.iter().cloned().collect();
        
        // Price features (20 features)
        let price_features = self.extract_price_features(&prices);
        
        // Volume features (10 features)
        let volume_features = self.extract_volume_features(&volumes);
        
        // Technical indicators (10 features)
        let technical_features = self.extract_technical_features(&prices);
        
        // Statistical features (5 features)
        let statistical_features = self.extract_statistical_features(&prices);
        
        // Temporal features (5 features)
        let temporal_features = self.extract_temporal_features();
        
        FeatureVector {
            price_features,
            volume_features,
            technical_features,
            statistical_features,
            temporal_features,
        }
    }

    /// Extract price-based features
    fn extract_price_features(&self, prices: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Returns at different timeframes
        for window in [1, 2, 5, 10, 20] {
            if prices.len() > window {
                let current = prices[prices.len() - 1];
                let past = prices[prices.len() - 1 - window];
                features.push((current - past) / past);
            } else {
                features.push(0.0);
            }
        }
        
        // Price momentum
        for window in [5, 10, 20, 50] {
            if prices.len() > 2 * window {
                let recent = prices[prices.len() - window..].iter().sum::<f64>() / window as f64;
                let older = prices[prices.len() - 2 * window..prices.len() - window].iter().sum::<f64>() / window as f64;
                features.push((recent - older) / older.max(0.0001));
            } else {
                features.push(0.0);
            }
        }
        
        // Price volatility
        for window in [10, 20, 50] {
            if prices.len() > window {
                let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                let recent_returns = &returns[returns.len().saturating_sub(window)..];
                let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
                let variance = recent_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / recent_returns.len() as f64;
                features.push(variance.sqrt());
            } else {
                features.push(0.0);
            }
        }
        
        // Price levels and support/resistance
        let current_price = prices[prices.len() - 1];
        let high = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = high - low;
        if range > 0.0 {
            features.push((current_price - low) / range); // Normalized position
        } else {
            features.push(0.5); // Default to middle if no range
        }
        features.push(current_price / high.max(0.0001)); // Distance from high
        features.push(current_price / low.max(0.0001)); // Distance from low
        
        // Fill remaining slots
        while features.len() < 20 {
            features.push(0.0);
        }
        
        features
    }

    /// Extract volume-based features
    fn extract_volume_features(&self, volumes: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Volume momentum
        for window in [5, 10, 20] {
            if volumes.len() > 2 * window {
                let recent = volumes[volumes.len() - window..].iter().sum::<f64>() / window as f64;
                let older = volumes[volumes.len() - 2 * window..volumes.len() - window].iter().sum::<f64>() / window as f64;
                features.push((recent - older) / older.max(1.0));
            } else {
                features.push(0.0);
            }
        }
        
        // Volume volatility
        for window in [10, 20] {
            if volumes.len() > window {
                let recent_volumes = &volumes[volumes.len() - window..];
                let mean = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
                let variance = recent_volumes.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent_volumes.len() as f64;
                features.push(variance.sqrt() / mean.max(1.0));
            } else {
                features.push(0.0);
            }
        }
        
        // Volume profile
        let current_volume = volumes[volumes.len() - 1];
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        features.push(current_volume / avg_volume.max(1.0));
        
        // Volume trend
        if volumes.len() > 20 {
            let recent_avg = volumes[volumes.len() - 10..].iter().sum::<f64>() / 10.0;
            let older_avg = volumes[volumes.len() - 20..volumes.len() - 10].iter().sum::<f64>() / 10.0;
            features.push((recent_avg - older_avg) / older_avg.max(1.0));
        } else {
            features.push(0.0);
        }
        
        // Fill remaining slots
        while features.len() < 10 {
            features.push(0.0);
        }
        
        features
    }

    /// Extract technical indicator features
    fn extract_technical_features(&self, prices: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();
        
        // RSI
        if prices.len() > 14 {
            let rsi = self.calculate_rsi(prices, 14);
            features.push(rsi);
        } else {
            features.push(50.0);
        }
        
        // MACD
        if prices.len() > 26 {
            let (macd, signal) = self.calculate_macd(prices);
            features.push(macd);
            features.push(signal);
        } else {
            features.push(0.0);
            features.push(0.0);
        }
        
        // Bollinger Bands
        if prices.len() > 20 {
            let (upper, middle, lower) = self.calculate_bollinger_bands(prices, 20);
            let current_price = prices[prices.len() - 1];
            features.push((current_price - lower) / (upper - lower)); // BB position
            features.push((current_price - middle) / middle); // BB deviation
        } else {
            features.push(0.5);
            features.push(0.0);
        }
        
        // Moving averages
        for period in [5, 10, 20] {
            if prices.len() > period {
                let ma = prices[prices.len() - period..].iter().sum::<f64>() / period as f64;
                let current_price = prices[prices.len() - 1];
                features.push((current_price - ma) / ma);
            } else {
                features.push(0.0);
            }
        }
        
        // Fill remaining slots
        while features.len() < 10 {
            features.push(0.0);
        }
        
        features
    }

    /// Extract statistical features
    fn extract_statistical_features(&self, prices: &[f64]) -> Vec<f64> {
        let mut features = Vec::new();
        
        if prices.len() > 50 {
            let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
            
            // Mean return
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            features.push(mean);
            
            // Standard deviation
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();
            features.push(std_dev);
            
            // Skewness
            let skewness = returns.iter().map(|r| ((r - mean) / std_dev).powi(3)).sum::<f64>() / returns.len() as f64;
            features.push(skewness);
            
            // Kurtosis
            let kurtosis = returns.iter().map(|r| ((r - mean) / std_dev).powi(4)).sum::<f64>() / returns.len() as f64;
            features.push(kurtosis);
            
            // Autocorrelation
            let autocorr = self.calculate_autocorrelation(&returns, 1);
            features.push(autocorr);
        } else {
            features.extend(vec![0.0; 5]);
        }
        
        features
    }

    /// Extract temporal features
    fn extract_temporal_features(&self) -> Vec<f64> {
        let mut features = Vec::new();
        
        if let Some(current_time) = self.timestamp_window.back() {
            // Time of day (normalized to 0-1)
            let seconds_since_midnight = (current_time % 86400.0) / 86400.0;
            features.push(seconds_since_midnight);
            
            // Day of week (normalized to 0-1)
            let day_of_week = ((current_time / 86400.0) as i64 % 7) as f64 / 7.0;
            features.push(day_of_week);
            
            // Time since last trade
            if self.timestamp_window.len() > 1 {
                let time_diff = current_time - self.timestamp_window[self.timestamp_window.len() - 2];
                features.push(time_diff.min(3600.0) / 3600.0); // Normalized to 1 hour
            } else {
                features.push(0.0);
            }
            
            // Trade frequency
            let recent_trades = self.timestamp_window.len().min(100);
            features.push(recent_trades as f64 / 100.0);
            
            // Market session (assuming 24/7 for crypto)
            features.push(1.0);
        } else {
            features.extend(vec![0.0; 5]);
        }
        
        features
    }

    /// Calculate RSI
    fn calculate_rsi(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period + 1 {
            return 50.0;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in prices.len() - period..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate MACD
    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64) {
        if prices.len() < 26 {
            return (0.0, 0.0);
        }
        
        let ema12 = self.calculate_ema(prices, 12);
        let ema26 = self.calculate_ema(prices, 26);
        let macd = ema12 - ema26;
        
        // Simple signal line (9-period EMA of MACD)
        let signal = macd * 0.2 + 0.8 * (self.predictions.back().unwrap_or(&0.0));
        
        (macd, signal)
    }

    /// Calculate EMA
    fn calculate_ema(&self, prices: &[f64], period: usize) -> f64 {
        if prices.len() < period {
            return prices[prices.len() - 1];
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema = prices[prices.len() - period..].iter().sum::<f64>() / period as f64;
        
        for i in prices.len() - period..prices.len() {
            ema = (prices[i] * multiplier) + (ema * (1.0 - multiplier));
        }
        
        ema
    }

    /// Calculate Bollinger Bands
    fn calculate_bollinger_bands(&self, prices: &[f64], period: usize) -> (f64, f64, f64) {
        if prices.len() < period {
            let price = prices[prices.len() - 1];
            return (price, price, price);
        }
        
        let recent_prices = &prices[prices.len() - period..];
        let mean = recent_prices.iter().sum::<f64>() / period as f64;
        let variance = recent_prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();
        
        (mean + 2.0 * std_dev, mean, mean - 2.0 * std_dev)
    }

    /// Calculate autocorrelation
    fn calculate_autocorrelation(&self, returns: &[f64], lag: usize) -> f64 {
        if returns.len() < lag + 1 {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        
        if variance == 0.0 {
            return 0.0;
        }
        
        let mut autocorr = 0.0;
        for i in lag..returns.len() {
            autocorr += (returns[i] - mean) * (returns[i - lag] - mean);
        }
        
        autocorr / ((returns.len() - lag) as f64 * variance)
    }

    /// Predict price movement using ensemble of models
    fn predict_price_movement(&mut self, features: &FeatureVector) -> f64 {
        let mut all_features = Vec::new();
        all_features.extend(&features.price_features);
        all_features.extend(&features.volume_features);
        all_features.extend(&features.technical_features);
        all_features.extend(&features.statistical_features);
        all_features.extend(&features.temporal_features);
        
        // Ensure feature vector matches model dimension
        while all_features.len() < self.feature_dimension {
            all_features.push(0.0);
        }
        all_features.truncate(self.feature_dimension);
        
        // Get predictions from different models
        let ml_prediction = self.ml_predict(&all_features);
        let statistical_prediction = self.statistical_predict();
        let microstructure_prediction = self.microstructure_predict();
        let time_series_prediction = self.time_series_predict();
        
        // Ensemble prediction
        let ensemble_prediction = 
            ml_prediction * self.model_weights[0] +
            statistical_prediction * self.model_weights[1] +
            microstructure_prediction * self.model_weights[2] +
            time_series_prediction * self.model_weights[3];
        
        // Update prediction tracking
        self.predictions.push_back(ensemble_prediction);
        if self.predictions.len() > 100 {
            self.predictions.pop_front();
        }
        
        ensemble_prediction
    }

    /// ML model prediction
    fn ml_predict(&mut self, features: &[f64]) -> f64 {
        let mut prediction = self.price_prediction_model.bias;
        
        for (i, &feature) in features.iter().enumerate() {
            if i < self.price_prediction_model.weights.len() {
                prediction += feature * self.price_prediction_model.weights[i];
            }
        }
        
        // Apply activation function (tanh for bounded output)
        prediction.tanh()
    }

    /// Statistical model prediction
    fn statistical_predict(&self) -> f64 {
        // Based on current statistical regime
        match self.volatility_model.volatility_regime {
            VolatilityRegime::Low => 0.1,
            VolatilityRegime::Medium => 0.0,
            VolatilityRegime::High => -0.1,
            VolatilityRegime::Extreme => -0.2,
        }
    }

    /// Microstructure model prediction
    fn microstructure_predict(&self) -> f64 {
        // Predict based on order flow imbalance
        self.microstructure_model.order_flow_imbalance * 0.5
    }

    /// Time series model prediction
    fn time_series_predict(&self) -> f64 {
        // Simple AR(1) prediction
        if let Some(last_prediction) = self.predictions.back() {
            *last_prediction * 0.8
        } else {
            0.0
        }
    }

    /// Update models with new data
    fn update_models(&mut self, actual_movement: f64) {
        // Update ML model weights
        if let Some(prediction) = self.predictions.back() {
            let error = actual_movement - prediction;
            let learning_rate = self.price_prediction_model.learning_rate;
            
            // Update bias
            self.price_prediction_model.bias += learning_rate * error;
            
            // Update weights with momentum
            if let Some(features) = self.feature_window.back() {
                let mut all_features: Vec<f64> = Vec::new();
                all_features.extend(&features.price_features);
                all_features.extend(&features.volume_features);
                all_features.extend(&features.technical_features);
                all_features.extend(&features.statistical_features);
                all_features.extend(&features.temporal_features);
                
                for (i, &feature) in all_features.iter().enumerate() {
                    if i < self.price_prediction_model.weights.len() {
                        let gradient = error * feature;
                        self.price_prediction_model.velocity[i] = 
                            self.price_prediction_model.momentum * self.price_prediction_model.velocity[i] + 
                            learning_rate * gradient;
                        self.price_prediction_model.weights[i] += self.price_prediction_model.velocity[i];
                    }
                }
            }
        }
        
        // Update prediction accuracy
        self.actual_movements.push_back(actual_movement);
        if self.actual_movements.len() > 100 {
            self.actual_movements.pop_front();
        }
        
        if self.predictions.len() == self.actual_movements.len() && self.predictions.len() > 10 {
            let mut correct = 0;
            for (pred, actual) in self.predictions.iter().zip(self.actual_movements.iter()) {
                if (*pred > 0.0 && *actual > 0.0) || (*pred < 0.0 && *actual < 0.0) {
                    correct += 1;
                }
            }
            self.prediction_accuracy = correct as f64 / self.predictions.len() as f64;
        }
        
        // Update ensemble weights based on performance
        self.update_ensemble_weights();
    }

    /// Update ensemble weights based on model performance
    fn update_ensemble_weights(&mut self) {
        // Simple adaptive weighting based on prediction accuracy
        let accuracy = self.prediction_accuracy;
        if accuracy > 0.6 {
            // Increase ML model weight if performing well
            self.model_weights[0] = (self.model_weights[0] * 1.1).min(0.5);
            // Normalize other weights
            let remaining_weight = 1.0 - self.model_weights[0];
            let other_weights = remaining_weight / 3.0;
            self.model_weights[1] = other_weights;
            self.model_weights[2] = other_weights;
            self.model_weights[3] = other_weights;
        }
    }

    /// Detect market regime
    fn detect_market_regime(&mut self) {
        if self.rolling_std.len() < 20 {
            return;
        }
        
        let current_volatility = self.rolling_std[self.rolling_std.len() - 1];
        let avg_volatility = self.rolling_std.iter().sum::<f64>() / self.rolling_std.len() as f64;
        
        self.current_regime = if current_volatility < avg_volatility * 0.5 {
            VolatilityRegime::Low
        } else if current_volatility < avg_volatility * 1.5 {
            VolatilityRegime::Medium
        } else if current_volatility < avg_volatility * 3.0 {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        };
        
        self.regime_confidence = (current_volatility / avg_volatility).min(1.0);
    }

    /// Calculate position size based on prediction confidence
    fn calculate_position_size(&self, prediction: f64) -> f64 {
        let confidence = prediction.abs();
        let base_size = self.max_position_size * 0.1;
        
        // Scale position size by prediction confidence and regime
        let regime_multiplier = match self.current_regime {
            VolatilityRegime::Low => 1.5,
            VolatilityRegime::Medium => 1.0,
            VolatilityRegime::High => 0.7,
            VolatilityRegime::Extreme => 0.3,
        };
        
        let size = base_size * confidence * regime_multiplier * self.position_size_multiplier;
        size.min(self.max_position_size)
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, trade_result: f64) {
        if trade_result > 0.0 {
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
        } else {
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
        }
        
        // Update win rate
        let total_trades = self.consecutive_wins + self.consecutive_losses;
        if total_trades > 0 {
            self.win_rate = self.consecutive_wins as f64 / total_trades as f64;
        }
        
        // Adjust position size based on performance
        if self.consecutive_wins >= 3 {
            self.position_size_multiplier = (self.position_size_multiplier * 1.1).min(2.0);
        } else if self.consecutive_losses >= 3 {
            self.position_size_multiplier = (self.position_size_multiplier * 0.9).max(0.5);
        }
    }
}

#[async_trait::async_trait]
impl Strategy for QuantumHftStrategy {
    fn get_info(&self) -> String {
        format!("Quantum HFT Strategy - Deep Statistics & ML (Accuracy: {:.2}%, Regime: {:?})", 
                self.prediction_accuracy * 100.0, self.current_regime)
    }

    async fn on_trade(&mut self, trade: trade::models::TradeData) {
        // Update data windows
        self.price_window.push_back(trade.price);
        self.volume_window.push_back(trade.qty);
        self.timestamp_window.push_back(trade.time as f64);
        
        // Keep windows at capacity
        if self.price_window.len() > self.price_window.capacity() {
            self.price_window.pop_front();
        }
        if self.volume_window.len() > self.volume_window.capacity() {
            self.volume_window.pop_front();
        }
        if self.timestamp_window.len() > self.timestamp_window.capacity() {
            self.timestamp_window.pop_front();
        }
        
        // Update rolling statistics
        if self.price_window.len() >= 20 {
            let recent_prices: Vec<f64> = self.price_window.iter().rev().take(20).cloned().collect();
            let mean = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
            let variance = recent_prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / recent_prices.len() as f64;
            let std_dev = variance.sqrt();
            
            self.rolling_mean.push_back(mean);
            self.rolling_std.push_back(std_dev);
            
            // Calculate skewness and kurtosis
            let skewness = recent_prices.iter().map(|p| ((p - mean) / std_dev).powi(3)).sum::<f64>() / recent_prices.len() as f64;
            let kurtosis = recent_prices.iter().map(|p| ((p - mean) / std_dev).powi(4)).sum::<f64>() / recent_prices.len() as f64;
            
            self.rolling_skew.push_back(skewness);
            self.rolling_kurt.push_back(kurtosis);
            
            // Keep rolling windows at capacity
            if self.rolling_mean.len() > 100 {
                self.rolling_mean.pop_front();
                self.rolling_std.pop_front();
                self.rolling_skew.pop_front();
                self.rolling_kurt.pop_front();
            }
        }
        
        // Extract features and predict
        if self.price_window.len() >= 50 {
            let features = self.extract_features();
            self.feature_window.push_back(features.clone());
            if self.feature_window.len() > 100 {
                self.feature_window.pop_front();
            }
            
            // Predict price movement
            let prediction = self.predict_price_movement(&features);
            
            // Update models with actual movement (if we have previous prediction)
            if let Some(prev_price) = self.price_window.get(self.price_window.len() - 2) {
                let current_price = trade.price;
                let actual_movement = (current_price - prev_price) / prev_price;
                self.update_models(actual_movement);
            }
            
            // Detect market regime
            self.detect_market_regime();
            
            debug!("Prediction: {:.4}, Accuracy: {:.2}%, Regime: {:?}", 
                   prediction, self.prediction_accuracy * 100.0, self.current_regime);
        } else {
            debug!("Not enough data yet: {}/50", self.price_window.len());
        }
        
        self.trade_counter += 1;
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        if self.price_window.len() < 50 {
            return (Signal::Hold, 0.0);
        }
        
        // Get latest prediction
        let prediction = self.predictions.back().unwrap_or(&0.0);
        let confidence = prediction.abs().min(1.0);
        
        debug!("Signal Check: prediction={:.4}, confidence={:.2}, position={:.2}", 
               prediction, confidence, current_position.quantity);
        
        // Position management based on predictions
        if current_position.quantity > 0.0 {
            // Long position - check for exit - MORE AGGRESSIVE
            if *prediction < -0.005 || self.consecutive_losses >= 3 {
                debug!("ML Long Exit: prediction={:.4}, confidence={:.2}", prediction, confidence);
                return (Signal::Sell, confidence);
            }
        } else if current_position.quantity < 0.0 {
            // Short position - check for exit - MORE AGGRESSIVE
            if *prediction > 0.005 || self.consecutive_losses >= 3 {
                debug!("ML Short Exit: prediction={:.4}, confidence={:.2}", prediction, confidence);
                return (Signal::Buy, confidence);
            }
        } else {
            // No position - look for entry based on prediction - MORE AGGRESSIVE
            if *prediction > 0.005 && confidence > 0.1 {
                debug!("ML Long Entry: prediction={:.4}, confidence={:.2}", prediction, confidence);
                return (Signal::Buy, confidence);
            } else if *prediction < -0.005 && confidence > 0.1 {
                debug!("ML Short Entry: prediction={:.4}, confidence={:.2}", prediction, confidence);
                return (Signal::Sell, confidence);
            }
            
            // Fallback: generate signals based on simple momentum if no ML prediction
            if self.trade_counter % 100 == 0 && self.price_window.len() >= 20 {
                let recent_prices: Vec<f64> = self.price_window.iter().rev().take(20).cloned().collect();
                let momentum = (recent_prices[0] - recent_prices[19]) / recent_prices[19];
                if momentum.abs() > 0.001 {
                    debug!("ML Fallback Signal: momentum={:.4}", momentum);
                    if momentum > 0.0 {
                        return (Signal::Buy, 0.3);
                    } else {
                        return (Signal::Sell, 0.3);
                    }
                }
            }
        }
        
        (Signal::Hold, 0.0)
    }
}

impl QuantumHftStrategy {
    pub fn on_trade_result(&mut self, result: f64) {
        self.update_performance_metrics(result);
    }
}
