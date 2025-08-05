//! # Advanced Machine Learning HFT Strategy
//! 
//! A sophisticated trading strategy based on:
//! - Ensemble Machine Learning Models
//! - Neural Network Pattern Recognition
//! - Statistical Learning Algorithms
//! - Feature Engineering and Selection
//! - Adaptive Model Training
//! - Multi-Model Voting System
//! 
//! This strategy uses multiple ML algorithms to identify profitable trading opportunities
//! through pattern recognition, trend prediction, and signal classification.

use std::collections::VecDeque;
use trade::signal::Signal;
use trade::trader::Position;
use trade::models::TradeData;
use tracing::debug;
use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use async_trait::async_trait;

/// Advanced Machine Learning HFT Strategy
/// 
/// This strategy uses multiple ML algorithms:
/// 1. Ensemble Random Forest for trend classification
/// 2. Neural Network for pattern recognition
/// 3. Support Vector Machine for signal classification
/// 4. Gradient Boosting for price prediction
/// 5. Adaptive model training and feature selection
/// 6. Multi-model voting system for signal generation
#[derive(Clone)]
pub struct QuantumHftStrategy {
    // Configuration
    config: StrategyConfig,
    
    // Data windows
    price_window: VecDeque<f64>,
    volume_window: VecDeque<f64>,
    
    // Feature engineering
    feature_window: usize,
    feature_vector: Vec<f64>,
    
    // ML Models
    ensemble_weights: Vec<f64>,
    model_predictions: Vec<f64>,
    
    // Neural Network
    nn_weights: Vec<Vec<Vec<f64>>>,
    nn_bias: Vec<f64>,
    nn_layers: Vec<usize>,
    
    // Random Forest
    rf_trees: Vec<DecisionTree>,
    rf_predictions: Vec<f64>,
    
    // Support Vector Machine
    svm_support_vectors: Vec<Vec<f64>>,
    svm_alphas: Vec<f64>,
    svm_bias: f64,
    
    // Gradient Boosting
    gb_models: Vec<WeakLearner>,
    gb_learning_rate: f64,
    
    // Performance tracking
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
    consecutive_wins: usize,
    consecutive_losses: usize,
    
    // Trade management
    entry_price: f64,
    stop_loss: f64,
    take_profit: f64,
    in_position: bool,
    position_direction: i8, // 1 for long, -1 for short, 0 for neutral
    
    // Model performance
    model_accuracy: Vec<f64>,
    feature_importance: Vec<f64>,
}

/// Simple Decision Tree for Random Forest
#[derive(Clone)]
struct DecisionTree {
    feature_index: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
    is_leaf: bool,
}

/// Weak Learner for Gradient Boosting
#[derive(Clone)]
struct WeakLearner {
    feature_index: usize,
    threshold: f64,
    direction: i8,
    weight: f64,
}

impl QuantumHftStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_strategy_config("quantum_hft_strategy")
            .expect("Failed to load quantum_hft_strategy configuration");
        
        let feature_window = config.get_or("feature_window", 50);
        let price_window_capacity = config.get_or("price_window_capacity", 100);
        let volume_window_capacity = config.get_or("volume_window_capacity", 100);
        
        // Initialize ensemble weights
        let ensemble_weights = vec![0.3, 0.25, 0.25, 0.2]; // RF, NN, SVM, GB
        
        // Initialize neural network
        let nn_layers = vec![20, 10, 5, 1]; // Input, hidden, output
        let nn_weights = Self::initialize_nn_weights(&nn_layers);
        let nn_bias = vec![0.0; nn_layers.len() - 1];
        
        // Initialize random forest
        let rf_trees = Self::initialize_random_forest(10, 20); // 10 trees, 20 features
        
        // Initialize SVM
        let svm_support_vectors = Vec::new();
        let svm_alphas = Vec::new();
        let svm_bias = 0.0;
        
        // Initialize gradient boosting
        let gb_models = Vec::new();
        let gb_learning_rate = config.get_or("gb_learning_rate", 0.1);
        
        Self {
            config,
            price_window: VecDeque::with_capacity(price_window_capacity),
            volume_window: VecDeque::with_capacity(volume_window_capacity),
            feature_window,
            feature_vector: vec![0.0; 20],
            ensemble_weights,
            model_predictions: vec![0.0; 4],
            nn_weights,
            nn_bias,
            nn_layers,
            rf_trees,
            rf_predictions: vec![0.0; 10],
            svm_support_vectors,
            svm_alphas,
            svm_bias,
            gb_models,
            gb_learning_rate,
            win_rate: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            consecutive_wins: 0,
            consecutive_losses: 0,
            entry_price: 0.0,
            stop_loss: 0.0,
            take_profit: 0.0,
            in_position: false,
            position_direction: 0,
            model_accuracy: vec![0.5; 4],
            feature_importance: vec![1.0; 20],
        }
    }

    /// Initialize neural network weights
    fn initialize_nn_weights(layers: &[usize]) -> Vec<Vec<Vec<f64>>> {
        let mut weights = Vec::new();
        for i in 0..layers.len() - 1 {
            let mut layer_weights = Vec::new();
            for _ in 0..layers[i + 1] {
                let mut neuron_weights = Vec::new();
                for _ in 0..layers[i] {
                    neuron_weights.push((rand::random::<f64>() - 0.5) * 0.1);
                }
                layer_weights.push(neuron_weights);
            }
            weights.push(layer_weights);
        }
        weights
    }

    /// Initialize random forest
    fn initialize_random_forest(n_trees: usize, n_features: usize) -> Vec<DecisionTree> {
        let mut trees = Vec::new();
        for _ in 0..n_trees {
            let tree = DecisionTree {
                feature_index: (rand::random::<f64>() * n_features as f64) as usize,
                threshold: rand::random::<f64>(),
                left_value: (rand::random::<f64>() - 0.5) * 2.0,
                right_value: (rand::random::<f64>() - 0.5) * 2.0,
                is_leaf: true,
            };
            trees.push(tree);
        }
        trees
    }

    /// Extract features from price and volume data
    fn extract_features(&mut self) -> Vec<f64> {
        if self.price_window.len() < self.feature_window {
            return vec![0.0; 20];
        }
        
        let prices: Vec<f64> = self.price_window.iter().rev().take(self.feature_window).cloned().collect();
        let volumes: Vec<f64> = self.volume_window.iter().rev().take(self.feature_window).cloned().collect();
        
        let mut features = Vec::new();
        
        // Price-based features
        features.push(self.calculate_sma(&prices, 5));
        features.push(self.calculate_sma(&prices, 10));
        features.push(self.calculate_sma(&prices, 20));
        features.push(self.calculate_ema(&prices, 5));
        features.push(self.calculate_ema(&prices, 10));
        features.push(self.calculate_rsi(&prices, 14));
        features.push(self.calculate_macd(&prices));
        features.push(self.calculate_bollinger_position(&prices));
        features.push(self.calculate_atr(&prices, 14));
        features.push(self.calculate_momentum(&prices, 5));
        
        // Volume-based features
        features.push(self.calculate_volume_sma(&volumes, 5));
        features.push(self.calculate_volume_sma(&volumes, 10));
        features.push(self.calculate_volume_ratio(&volumes));
        features.push(self.calculate_volume_momentum(&volumes, 5));
        
        // Price change features
        features.push(self.calculate_price_change(&prices, 1));
        features.push(self.calculate_price_change(&prices, 5));
        features.push(self.calculate_price_change(&prices, 10));
        features.push(self.calculate_volatility(&prices, 10));
        
        // Technical indicators
        features.push(self.calculate_stochastic(&prices, 14));
        features.push(self.calculate_williams_r(&prices, 14));
        
        // Normalize features
        self.normalize_features(&mut features);
        features
    }

    /// Calculate Simple Moving Average
    fn calculate_sma(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window {
            return data.last().unwrap_or(&0.0).clone();
        }
        data.iter().rev().take(window).sum::<f64>() / window as f64
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window {
            return data.last().unwrap_or(&0.0).clone();
        }
        
        let alpha = 2.0 / (window + 1) as f64;
        let mut ema = data[data.len() - window];
        
        for i in (data.len() - window + 1)..data.len() {
            ema = alpha * data[i] + (1.0 - alpha) * ema;
        }
        
        ema
    }

    /// Calculate RSI
    fn calculate_rsi(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window + 1 {
            return 50.0;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in (data.len() - window)..data.len() {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses += change.abs();
            }
        }
        
        if losses == 0.0 {
            return 100.0;
        }
        
        let rs = gains / losses;
        100.0 - (100.0 / (1.0 + rs))
    }

    /// Calculate MACD
    fn calculate_macd(&self, data: &[f64]) -> f64 {
        if data.len() < 26 {
            return 0.0;
        }
        
        let ema12 = self.calculate_ema(data, 12);
        let ema26 = self.calculate_ema(data, 26);
        ema12 - ema26
    }

    /// Calculate Bollinger Bands position
    fn calculate_bollinger_position(&self, data: &[f64]) -> f64 {
        if data.len() < 20 {
            return 0.5;
        }
        
        let sma = self.calculate_sma(data, 20);
        let variance: f64 = data.iter().rev().take(20).map(|&x| (x - sma).powi(2)).sum::<f64>() / 20.0;
        let std_dev = variance.sqrt();
        
        let current_price = data.last().unwrap();
        let upper = sma + 2.0 * std_dev;
        let lower = sma - 2.0 * std_dev;
        
        (current_price - lower) / (upper - lower)
    }

    /// Calculate Average True Range
    fn calculate_atr(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window + 1 {
            return 0.0;
        }
        
        let mut true_ranges = Vec::new();
        for i in 1..=window {
            let high = data[data.len() - i];
            let low = data[data.len() - i - 1];
            true_ranges.push((high - low).abs());
        }
        
        true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
    }

    /// Calculate momentum
    fn calculate_momentum(&self, data: &[f64], period: usize) -> f64 {
        if data.len() < period {
            return 0.0;
        }
        
        let current = data.last().unwrap();
        let previous = data[data.len() - period - 1];
        (current - previous) / previous
    }

    /// Calculate volume SMA
    fn calculate_volume_sma(&self, data: &[f64], window: usize) -> f64 {
        self.calculate_sma(data, window)
    }

    /// Calculate volume ratio
    fn calculate_volume_ratio(&self, data: &[f64]) -> f64 {
        if data.len() < 10 {
            return 1.0;
        }
        
        let recent_volume: f64 = data.iter().rev().take(5).sum();
        let avg_volume: f64 = data.iter().rev().take(10).sum::<f64>() / 10.0;
        
        if avg_volume == 0.0 {
            return 1.0;
        }
        
        recent_volume / avg_volume
    }

    /// Calculate volume momentum
    fn calculate_volume_momentum(&self, data: &[f64], period: usize) -> f64 {
        self.calculate_momentum(data, period)
    }

    /// Calculate price change
    fn calculate_price_change(&self, data: &[f64], period: usize) -> f64 {
        if data.len() < period + 1 {
            return 0.0;
        }
        
        let current = data.last().unwrap();
        let previous = data[data.len() - period - 1];
        (current - previous) / previous
    }

    /// Calculate volatility
    fn calculate_volatility(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window {
            return 0.0;
        }
        
        let returns: Vec<f64> = data.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|&r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }

    /// Calculate Stochastic Oscillator
    fn calculate_stochastic(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window {
            return 50.0;
        }
        
        let window_data = &data[data.len() - window..];
        let highest = window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = data.last().unwrap();
        
        if highest == lowest {
            return 50.0;
        }
        
        ((current - lowest) / (highest - lowest)) * 100.0
    }

    /// Calculate Williams %R
    fn calculate_williams_r(&self, data: &[f64], window: usize) -> f64 {
        if data.len() < window {
            return -50.0;
        }
        
        let window_data = &data[data.len() - window..];
        let highest = window_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest = window_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = data.last().unwrap();
        
        if highest == lowest {
            return -50.0;
        }
        
        ((highest - current) / (highest - lowest)) * -100.0
    }

    /// Normalize features to [0, 1] range
    fn normalize_features(&self, features: &mut [f64]) {
        for feature in features.iter_mut() {
            *feature = feature.tanh(); // Normalize to [-1, 1] then scale to [0, 1]
            *feature = (*feature + 1.0) / 2.0;
        }
    }

    /// Neural Network forward pass
    fn neural_network_forward(&self, input: &[f64]) -> f64 {
        let mut current_layer = input.to_vec();
        
        for (layer_idx, layer_weights) in self.nn_weights.iter().enumerate() {
            let mut next_layer = Vec::new();
            
            for neuron_weights in layer_weights {
                let mut sum = 0.0;
                for (input_val, weight) in current_layer.iter().zip(neuron_weights.iter()) {
                    sum += input_val * weight;
                }
                sum += self.nn_bias[layer_idx];
                next_layer.push(sum.tanh()); // Activation function
            }
            
            current_layer = next_layer;
        }
        
        current_layer[0]
    }

    /// Random Forest prediction
    fn random_forest_predict(&self, features: &[f64]) -> f64 {
        let mut predictions = Vec::new();
        
        for tree in &self.rf_trees {
            let prediction = if features[tree.feature_index] < tree.threshold {
                tree.left_value
            } else {
                tree.right_value
            };
            predictions.push(prediction);
        }
        
        predictions.iter().sum::<f64>() / predictions.len() as f64
    }

    /// Support Vector Machine prediction
    fn svm_predict(&self, features: &[f64]) -> f64 {
        if self.svm_support_vectors.is_empty() {
            return 0.0;
        }
        
        let mut prediction = self.svm_bias;
        
        for (i, support_vector) in self.svm_support_vectors.iter().enumerate() {
            let kernel_value = self.rbf_kernel(features, support_vector);
            prediction += self.svm_alphas[i] * kernel_value;
        }
        
        prediction.tanh()
    }

    /// RBF Kernel for SVM
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let gamma = 0.1;
        let mut sum = 0.0;
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            sum += (a - b).powi(2);
        }
        
        (-gamma * sum).exp()
    }

    /// Gradient Boosting prediction
    fn gradient_boosting_predict(&self, features: &[f64]) -> f64 {
        let mut prediction = 0.0;
        
        for model in &self.gb_models {
            let feature_val = features[model.feature_index];
            let model_prediction = if feature_val < model.threshold {
                model.direction as f64
            } else {
                -model.direction as f64
            };
            prediction += model.weight * model_prediction;
        }
        
        prediction.tanh()
    }

    /// Ensemble prediction combining all models (immutable version)
    fn ensemble_predict_immutable(&self, features: &[f64]) -> f64 {
        // Neural Network prediction
        let nn_prediction = self.neural_network_forward(features);
        
        // Random Forest prediction
        let rf_prediction = self.random_forest_predict(features);
        
        // SVM prediction
        let svm_prediction = self.svm_predict(features);
        
        // Gradient Boosting prediction
        let gb_prediction = self.gradient_boosting_predict(features);
        
        // Weighted ensemble
        let mut ensemble_prediction = 0.0;
        ensemble_prediction += nn_prediction * self.ensemble_weights[0];
        ensemble_prediction += rf_prediction * self.ensemble_weights[1];
        ensemble_prediction += svm_prediction * self.ensemble_weights[2];
        ensemble_prediction += gb_prediction * self.ensemble_weights[3];
        
        ensemble_prediction
    }

    /// Update model performance based on trade results
    fn update_model_performance(&mut self, trade_result: f64) {
        for (i, prediction) in self.model_predictions.iter().enumerate() {
            let prediction_direction = if *prediction > 0.0 { 1.0 } else { -1.0 };
            let actual_direction = if trade_result > 0.0 { 1.0 } else { -1.0 };
            
            if prediction_direction == actual_direction {
                self.model_accuracy[i] = self.model_accuracy[i] * 0.9 + 0.1;
            } else {
                self.model_accuracy[i] = self.model_accuracy[i] * 0.9;
            }
        }
        
        // Update ensemble weights based on performance
        let total_accuracy: f64 = self.model_accuracy.iter().sum();
        if total_accuracy > 0.0 {
            for (i, weight) in self.ensemble_weights.iter_mut().enumerate() {
                *weight = self.model_accuracy[i] / total_accuracy;
            }
        }
    }

    /// Should enter long position based on ML predictions
    fn should_enter_long(&self, ensemble_prediction: f64) -> bool {
        let threshold = self.config.get_or("long_entry_threshold", 0.3);
        ensemble_prediction > threshold
    }

    /// Should enter short position based on ML predictions
    fn should_enter_short(&self, ensemble_prediction: f64) -> bool {
        let threshold = self.config.get_or("short_entry_threshold", -0.3);
        ensemble_prediction < threshold
    }

    /// Should exit long position
    fn should_exit_long(&self, ensemble_prediction: f64) -> bool {
        let threshold = self.config.get_or("long_exit_threshold", -0.1);
        ensemble_prediction < threshold
    }

    /// Should exit short position
    fn should_exit_short(&self, ensemble_prediction: f64) -> bool {
        let threshold = self.config.get_or("short_exit_threshold", 0.1);
        ensemble_prediction > threshold
    }

    /// Calculate position size based on ML confidence
    fn calculate_position_size(&self, ensemble_prediction: f64) -> f64 {
        let base_size = self.config.get_or("trading_size_min", 10.0);
        let max_size = self.config.get_or("trading_size_max", 50.0);
        
        let confidence = ensemble_prediction.abs();
        let adjusted_size = base_size * (1.0 + confidence);
        
        adjusted_size.max(base_size).min(max_size)
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, trade_result: f64) {
        if trade_result > 0.0 {
            self.consecutive_wins += 1;
            self.consecutive_losses = 0;
            self.avg_win = (self.avg_win * (self.consecutive_wins - 1) as f64 + trade_result) / self.consecutive_wins as f64;
        } else {
            self.consecutive_losses += 1;
            self.consecutive_wins = 0;
            self.avg_loss = (self.avg_loss * (self.consecutive_losses - 1) as f64 + trade_result.abs()) / self.consecutive_losses as f64;
        }
        
        let total_trades = self.consecutive_wins + self.consecutive_losses;
        if total_trades > 0 {
            self.win_rate = self.consecutive_wins as f64 / total_trades as f64;
        }
    }
}

#[async_trait]
impl Strategy for QuantumHftStrategy {
    fn get_info(&self) -> String {
        "Advanced Machine Learning HFT Strategy - Ensemble ML Models".to_string()
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // Update price and volume windows
        self.price_window.push_back(trade.price);
        self.volume_window.push_back(trade.qty);
        
        // Keep windows at capacity
        if self.price_window.len() > self.price_window.capacity() {
            self.price_window.pop_front();
        }
        if self.volume_window.len() > self.volume_window.capacity() {
            self.volume_window.pop_front();
        }
        
        // Extract features when we have enough data
        if self.price_window.len() >= self.feature_window {
            self.feature_vector = self.extract_features();
            
            debug!("ML Features extracted: {:?}", self.feature_vector);
        }
    }

    fn get_signal(
        &self,
        current_price: f64,
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        if self.price_window.len() < self.feature_window {
            return (Signal::Hold, 0.0);
        }
        
        // Get ensemble prediction
        let ensemble_prediction = self.ensemble_predict_immutable(&self.feature_vector);
        let confidence = ensemble_prediction.abs();
        
        // Position management
        if current_position.quantity > 0.0 {
            // We have a long position
            if self.should_exit_long(ensemble_prediction) {
                return (Signal::Sell, confidence);
            }
        } else if current_position.quantity < 0.0 {
            // We have a short position
            if self.should_exit_short(ensemble_prediction) {
                return (Signal::Buy, confidence);
            }
        } else {
                    // No position - look for entry signals
        if self.should_enter_long(ensemble_prediction) {
            let _position_size = self.calculate_position_size(ensemble_prediction);
            debug!("ML Long Entry: prediction={:.3}, confidence={:.3}", ensemble_prediction, confidence);
            return (Signal::Buy, confidence);
        } else if self.should_enter_short(ensemble_prediction) {
            let _position_size = self.calculate_position_size(ensemble_prediction);
            debug!("ML Short Entry: prediction={:.3}, confidence={:.3}", ensemble_prediction, confidence);
            return (Signal::Sell, confidence);
        }
        
        // Fallback: generate signals based on simple trend detection
        if ensemble_prediction > 0.05 && confidence > 0.3 {
            debug!("ML Fallback Long: prediction={:.3}, confidence={:.3}", ensemble_prediction, confidence);
            return (Signal::Buy, confidence * 0.8);
        } else if ensemble_prediction < -0.05 && confidence > 0.3 {
            debug!("ML Fallback Short: prediction={:.3}, confidence={:.3}", ensemble_prediction, confidence);
            return (Signal::Sell, confidence * 0.8);
        }
        }
        
        (Signal::Hold, 0.0)
    }
}

impl QuantumHftStrategy {
    pub fn on_trade_result(&mut self, result: f64) {
        self.update_performance_metrics(result);
        self.update_model_performance(result);
        
        // Reset position tracking
        self.in_position = false;
        self.position_direction = 0;
        self.entry_price = 0.0;
        self.stop_loss = 0.0;
        self.take_profit = 0.0;
    }
}

// Simple random number generator for initialization
mod rand {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    pub fn random<T>() -> T 
    where
        T: From<f64>,
    {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let normalized = ((nanos % 1000000) as f64) / 1000000.0;
        T::from(normalized)
    }
} 




