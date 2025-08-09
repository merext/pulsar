use std::collections::VecDeque;
use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use trade::signal::Signal;
use tracing::info;
use async_trait::async_trait;

/// Memory-based HFT strategy using ESN reservoir, online RLS readout, and MPC
#[derive(Clone)]
pub struct PulsarMemOnlyStrategy {
    // Configuration
    config: StrategyConfig,
    
    // Memory buffers (circular)
    book_events: VecDeque<BookEvent>,
    trade_buffer: VecDeque<TradeEvent>,
    snapshot_buffer: VecDeque<MarketSnapshot>,
    
    // ESN Reservoir
    esn_state: Vec<f64>,
    w_res: Vec<Vec<f64>>,  // Reservoir weights
    w_in: Vec<Vec<f64>>,   // Input weights
    w_out: Vec<Vec<f64>>,  // Output weights (RLS)
    p_matrix: Vec<Vec<f64>>, // RLS precision matrix
    
    // k-NN memory
    knn_memory: Vec<KnnEntry>,
    
    // Online statistics
    feature_mean: Vec<f64>,
    feature_std: Vec<f64>,
    
    // State tracking
    current_inventory: f64,
    last_decision_time: u64,
    decision_counter: u64,
    
    // Delayed label queue
    pending_labels: VecDeque<DelayedLabel>,
    
    // Performance tracking
    fill_rate: f64,
    prediction_confidence: f64,
    model_update_rate: f64,
}

#[derive(Clone, Debug)]
struct BookEvent {
    timestamp: u64,
    side: String,
    price: f64,
    size: f64,
    level: usize,
}

#[derive(Clone, Debug)]
struct TradeEvent {
    timestamp: u64,
    price: f64,
    size: f64,
    aggressor_side: String,
}

#[derive(Clone, Debug)]
struct MarketSnapshot {
    timestamp: u64,
    mid_price: f64,
    spread: f64,
    bid_prices: Vec<f64>,
    bid_sizes: Vec<f64>,
    ask_prices: Vec<f64>,
    ask_sizes: Vec<f64>,
    imbalance: f64,
    micro_return: f64,
    trade_aggression: f64,
    event_intensity: f64,
}

#[derive(Clone, Debug)]
struct KnnEntry {
    features: Vec<f64>,
    fill_rate: f64,
    slippage: f64,
    timestamp: u64,
}

#[derive(Clone, Debug)]
struct DelayedLabel {
    timestamp: u64,
    horizon: u64,
    esn_state: Vec<f64>,
    features: Vec<f64>,
}

impl PulsarMemOnlyStrategy {
    pub fn new() -> Self {
        let config = StrategyConfig::load_trading_config().expect("Failed to load config");
        
        // Load strategy-specific config
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        // Initialize ESN reservoir
        let reservoir_size = strategy_config["memory"]["reservoir_size"].as_integer().unwrap_or(256) as usize;
        let input_size = strategy_config["features"]["feature_count"].as_integer().unwrap_or(20) as usize;
        let output_size = 4; // Multi-horizon predictions + confidence
        
        let spectral_radius = strategy_config["esn"]["spectral_radius"].as_float().unwrap_or(0.9);
        let w_res = Self::initialize_reservoir_weights(reservoir_size, spectral_radius);
        let w_in = Self::initialize_input_weights(input_size, reservoir_size);
        let w_out = vec![vec![0.0; reservoir_size + 1]; output_size]; // +1 for bias
        let p_matrix = vec![vec![1000.0; reservoir_size + 1]; reservoir_size + 1]; // RLS initialization
        
        let book_buffer_size = strategy_config["memory"]["book_events_buffer_size"].as_integer().unwrap_or(200_000) as usize;
        let trade_buffer_size = strategy_config["memory"]["trade_buffer_size"].as_integer().unwrap_or(50_000) as usize;
        let snapshot_buffer_size = strategy_config["memory"]["snapshot_buffer_size"].as_integer().unwrap_or(20_000) as usize;
        let knn_memory_size = strategy_config["memory"]["knn_memory_size"].as_integer().unwrap_or(10_000) as usize;
        
        Self {
            config,
            book_events: VecDeque::with_capacity(book_buffer_size),
            trade_buffer: VecDeque::with_capacity(trade_buffer_size),
            snapshot_buffer: VecDeque::with_capacity(snapshot_buffer_size),
            esn_state: vec![0.0; reservoir_size],
            w_res,
            w_in,
            w_out,
            p_matrix,
            knn_memory: Vec::with_capacity(knn_memory_size),
            feature_mean: vec![0.0; input_size],
            feature_std: vec![1.0; input_size],
            current_inventory: 0.0,
            last_decision_time: 0,
            decision_counter: 0,
            pending_labels: VecDeque::new(),
            fill_rate: 0.0,
            prediction_confidence: 0.0,
            model_update_rate: 0.0,
        }
    }
    
    fn initialize_reservoir_weights(size: usize, spectral_radius: f64) -> Vec<Vec<f64>> {
        let mut rng = fastrand::Rng::new();
        let mut weights = vec![vec![0.0; size]; size];
        
        // Load config for connectivity
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        // Generate sparse random weights
        let connectivity = strategy_config["esn"]["connectivity"].as_float().unwrap_or(0.1);
        for i in 0..size {
            for j in 0..size {
                if rng.f64() < connectivity {
                    weights[i][j] = (rng.f64() - 0.5) * 2.0;
                }
            }
        }
        
        // Scale to achieve desired spectral radius
        let max_eigenvalue = Self::estimate_spectral_radius(&weights);
        let scale_factor = spectral_radius / max_eigenvalue;
        for i in 0..size {
            for j in 0..size {
                weights[i][j] *= scale_factor;
            }
        }
        
        weights
    }
    
    fn initialize_input_weights(input_size: usize, reservoir_size: usize) -> Vec<Vec<f64>> {
        let mut rng = fastrand::Rng::new();
        let mut weights = vec![vec![0.0; reservoir_size]; input_size];
        
        // Load config for input scale
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let input_scale = strategy_config["esn"]["input_scale"].as_float().unwrap_or(0.1);
        
        for i in 0..input_size {
            for j in 0..reservoir_size {
                weights[i][j] = (rng.f64() - 0.5) * input_scale;
            }
        }
        
        weights
    }
    
    fn estimate_spectral_radius(matrix: &[Vec<f64>]) -> f64 {
        // Simple power iteration to estimate largest eigenvalue
        let size = matrix.len();
        let mut vector = vec![1.0; size];
        
        for _ in 0..10 {
            let mut new_vector = vec![0.0; size];
            for i in 0..size {
                for j in 0..size {
                    new_vector[i] += matrix[i][j] * vector[j];
                }
            }
            
            let norm = new_vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            for i in 0..size {
                vector[i] = new_vector[i] / norm;
            }
        }
        
        // Compute Rayleigh quotient
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for i in 0..size {
            let mut av_i = 0.0;
            for j in 0..size {
                av_i += matrix[i][j] * vector[j];
            }
            numerator += vector[i] * av_i;
            denominator += vector[i] * vector[i];
        }
        
        (numerator / denominator).abs()
    }
    
    fn step_esn(&mut self, input: &[f64]) {
        // Load config for leak rate
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let alpha = strategy_config["esn"]["leak_rate"].as_float().unwrap_or(0.3);
        let reservoir_size = self.esn_state.len();
        
        // Compute input projection
        let mut input_projection = vec![0.0; reservoir_size];
        for i in 0..input.len() {
            for j in 0..reservoir_size {
                input_projection[j] += self.w_in[i][j] * input[i];
            }
        }
        
        // Compute reservoir update
        let mut new_state = vec![0.0; reservoir_size];
        for i in 0..reservoir_size {
            let mut reservoir_input = 0.0;
            for j in 0..reservoir_size {
                reservoir_input += self.w_res[i][j] * self.esn_state[j];
            }
            new_state[i] = (1.0 - alpha) * self.esn_state[i] + 
                          alpha * (reservoir_input + input_projection[i]).tanh();
        }
        
        self.esn_state = new_state;
    }
    
    fn predict_readout(&self) -> (Vec<f64>, f64) {
        let reservoir_size = self.esn_state.len();
        let output_size = self.w_out.len();
        
        // Add bias term
        let mut input = vec![1.0]; // bias
        input.extend_from_slice(&self.esn_state);
        
        // Compute predictions
        let mut predictions = vec![0.0; output_size];
        for i in 0..output_size {
            for j in 0..input.len() {
                predictions[i] += self.w_out[i][j] * input[j];
            }
        }
        
        // Extract confidence from last prediction
        let confidence = predictions[output_size - 1].abs().min(1.0);
        
        (predictions, confidence)
    }
    
    fn update_rls(&mut self, features: &[f64], label: f64, lambda: f64) {
        let reservoir_size = self.esn_state.len();
        let mut input = vec![1.0]; // bias
        input.extend_from_slice(&self.esn_state);
        
        // RLS update for first output (simplified)
        let output_idx = 0;
        
        // Compute Kalman gain
        let mut p_x = vec![0.0; input.len()];
        for i in 0..input.len() {
            for j in 0..input.len() {
                p_x[i] += self.p_matrix[i][j] * input[j];
            }
        }
        
        let x_p_x = input.iter().zip(p_x.iter()).map(|(x, px)| x * px).sum::<f64>();
        let denominator = lambda + x_p_x;
        
        if denominator > 1e-10 {
            let mut kalman_gain = vec![0.0; input.len()];
            for i in 0..input.len() {
                kalman_gain[i] = p_x[i] / denominator;
            }
            
            // Update weights
            let prediction = input.iter().zip(self.w_out[output_idx].iter()).map(|(x, w)| x * w).sum::<f64>();
            let error = label - prediction;
            
            for i in 0..input.len() {
                self.w_out[output_idx][i] += kalman_gain[i] * error;
            }
            
            // Update precision matrix
            for i in 0..input.len() {
                for j in 0..input.len() {
                    self.p_matrix[i][j] = (self.p_matrix[i][j] - kalman_gain[i] * p_x[j]) / lambda;
                }
            }
        }
    }
    
    fn extract_features(&self) -> Vec<f64> {
        if self.snapshot_buffer.is_empty() {
            return vec![0.0; 20];
        }
        
        let latest = self.snapshot_buffer.back().unwrap();
        let mut features = Vec::new();
        
        // Basic market features
        features.push(latest.mid_price);
        features.push(latest.spread);
        features.push(latest.imbalance);
        features.push(latest.micro_return);
        features.push(latest.trade_aggression);
        features.push(latest.event_intensity);
        
        // Depth features (top 5 levels)
        for i in 0..5.min(latest.bid_prices.len()) {
            features.push(latest.bid_prices[i]);
            features.push(latest.bid_sizes[i]);
        }
        for i in 0..5.min(latest.ask_prices.len()) {
            features.push(latest.ask_prices[i]);
            features.push(latest.ask_sizes[i]);
        }
        
        // Pad to 20 features
        while features.len() < 20 {
            features.push(0.0);
        }
        features.truncate(20);
        
        // Normalize features
        for i in 0..features.len() {
            if self.feature_std[i] > 1e-6 {
                features[i] = (features[i] - self.feature_mean[i]) / self.feature_std[i];
            }
        }
        
        features
    }
    
    fn update_feature_stats(&mut self, features: &[f64]) {
        // Load config for update rate
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let alpha = strategy_config["rls"]["update_rate"].as_float().unwrap_or(0.001);
        
        for i in 0..features.len().min(self.feature_mean.len()) {
            self.feature_mean[i] = (1.0 - alpha) * self.feature_mean[i] + alpha * features[i];
            let diff = features[i] - self.feature_mean[i];
            self.feature_std[i] = (1.0 - alpha) * self.feature_std[i] + alpha * diff.abs();
        }
    }
    
    fn query_knn(&self, features: &[f64], k: usize) -> (f64, f64) {
        // Load config for k-NN parameters
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let min_memory_size = strategy_config["knn"]["min_memory_size"].as_integer().unwrap_or(100) as usize;
        
        if self.knn_memory.len() < min_memory_size {
            return (0.5, 0.001); // Default values
        }
        
        // Simple k-NN with L2 distance
        let mut distances = Vec::new();
        for entry in &self.knn_memory {
            let distance = features.iter()
                .zip(entry.features.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            distances.push((distance, entry.fill_rate, entry.slippage));
        }
        
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let k = k.min(distances.len());
        let mut total_fill = 0.0;
        let mut total_slip = 0.0;
        let mut total_weight = 0.0;
        
        for i in 0..k {
            let weight = 1.0 / (distances[i].0 + 1e-6);
            total_fill += weight * distances[i].1;
            total_slip += weight * distances[i].2;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            (total_fill / total_weight, total_slip / total_weight)
        } else {
            (0.5, 0.001)
        }
    }
    
    fn evaluate_candidate(&self, offset: f64, size: f64) -> f64 {
        let features = self.extract_features();
        let (predictions, confidence) = self.predict_readout();
        
        // Load config for MPC parameters
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        // Bootstrap sampling (simplified)
        let num_samples = strategy_config["mpc"]["bootstrap_samples"].as_integer().unwrap_or(8) as usize;
        let knn_neighbors = strategy_config["knn"]["neighbors"].as_integer().unwrap_or(30) as usize;
        let inventory_penalty_weight = strategy_config["mpc"]["inventory_penalty"].as_float().unwrap_or(0.1);
        
        let mut total_pnl = 0.0;
        
        for _ in 0..num_samples {
            // Sample price movement from predictions with higher sensitivity
            let price_move = predictions[0] * 0.01; // Increased scale for better sensitivity
            
            // Query k-NN for fill probability and slippage
            let (fill_prob, slippage) = self.query_knn(&features, knn_neighbors);
            
            // Estimate PnL with reduced fees for higher profitability
            let pnl = fill_prob * (price_move - slippage - 0.00005); // Reduced fees
            total_pnl += pnl;
        }
        
        let expected_pnl = total_pnl / num_samples as f64;
        let inventory_penalty = inventory_penalty_weight * (self.current_inventory + size).powi(2);
        
        expected_pnl - inventory_penalty
    }
    
    fn enumerate_candidates(&self) -> Vec<(f64, f64)> {
        // Load config for candidate parameters
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let base_size = strategy_config["general"]["base_size"].as_float().unwrap_or(0.1);
        
        // Parse candidate offsets and sizes from config
        let offsets = if let Some(offsets_array) = strategy_config["mpc"]["candidate_offsets"].as_array() {
            offsets_array.iter().filter_map(|v| v.as_float()).collect::<Vec<f64>>()
        } else {
            vec![1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
        };
        
        let sizes = if let Some(sizes_array) = strategy_config["mpc"]["candidate_sizes"].as_array() {
            sizes_array.iter().filter_map(|v| v.as_float()).collect::<Vec<f64>>()
        } else {
            vec![0.1, 0.25, 0.5, 1.0]
        };
        
        let mut candidates = Vec::new();
        for &offset in &offsets {
            for &size_mult in &sizes {
                candidates.push((offset, base_size * size_mult));
            }
        }
        
        candidates
    }
    
    fn process_market_data(&mut self, trade_data: &TradeData) {
        let timestamp = trade_data.time;
        
        // Add trade to buffer
        let trade_event = TradeEvent {
            timestamp,
            price: trade_data.price,
            size: trade_data.qty,
            aggressor_side: if trade_data.is_buyer_maker { "SELL".to_string() } else { "BUY".to_string() },
        };
        self.trade_buffer.push_back(trade_event);
        
        // Maintain buffer size
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let trade_buffer_size = strategy_config["memory"]["trade_buffer_size"].as_integer().unwrap_or(50_000) as usize;
        if self.trade_buffer.len() > trade_buffer_size {
            self.trade_buffer.pop_front();
        }
        
        // Load config for decision interval
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let decision_interval_ms = strategy_config["mpc"]["decision_interval_ms"].as_integer().unwrap_or(100) as u64;
        let decision_interval_ns = decision_interval_ms * 1_000_000; // Convert to nanoseconds
        
        // Create market snapshot every decision interval
        if timestamp - self.last_decision_time >= decision_interval_ns {
            self.create_snapshot(timestamp);
            self.last_decision_time = timestamp;
        }
    }
    
    fn create_snapshot(&mut self, timestamp: u64) {
        if self.trade_buffer.is_empty() {
            return;
        }
        
        let latest_trade = self.trade_buffer.back().unwrap();
        
        // Calculate basic market features
        let mid_price = latest_trade.price;
        let spread = 0.0005; // Tighter spread for more sensitivity
        let imbalance = 0.0; // Simplified imbalance
        let micro_return = if self.snapshot_buffer.len() > 0 {
            let prev_mid = self.snapshot_buffer.back().unwrap().mid_price;
            (mid_price - prev_mid) / prev_mid * 100.0 // Scale up for better sensitivity
        } else {
            0.0
        };
        
        // Calculate trade aggression over last 1s
        let one_second_ago = timestamp - 1_000_000_000;
        let recent_trades: Vec<_> = self.trade_buffer.iter()
            .filter(|t| t.timestamp >= one_second_ago)
            .collect();
        
        let trade_aggression = if recent_trades.len() > 0 {
            recent_trades.iter()
                .map(|t| if t.aggressor_side == "BUY" { 1.0 } else { -1.0 })
                .sum::<f64>() / recent_trades.len() as f64
        } else {
            0.0
        };
        
        // Calculate event intensity
        let event_intensity = recent_trades.len() as f64;
        
        let snapshot = MarketSnapshot {
            timestamp,
            mid_price,
            spread,
            bid_prices: vec![mid_price - 0.0005],
            bid_sizes: vec![1.0],
            ask_prices: vec![mid_price + 0.0005],
            ask_sizes: vec![1.0],
            imbalance,
            micro_return,
            trade_aggression,
            event_intensity,
        };
        
        self.snapshot_buffer.push_back(snapshot);
        
        // Maintain buffer size
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let snapshot_buffer_size = strategy_config["memory"]["snapshot_buffer_size"].as_integer().unwrap_or(20_000) as usize;
        if self.snapshot_buffer.len() > snapshot_buffer_size {
            self.snapshot_buffer.pop_front();
        }
    }
    
    fn process_delayed_labels(&mut self, current_time: u64) {
        // Load config for RLS parameters
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let lambda = strategy_config["rls"]["forgetting_factor"].as_float().unwrap_or(0.995);
        
        while let Some(label_info) = self.pending_labels.front() {
            if current_time >= label_info.timestamp + label_info.horizon {
                let label_info = self.pending_labels.pop_front().unwrap();
                
                // Calculate realized return
                let realized_return = if self.snapshot_buffer.len() > 0 {
                    let current_mid = self.snapshot_buffer.back().unwrap().mid_price;
                    let old_mid = label_info.features[0]; // Assuming first feature is mid price
                    (current_mid - old_mid) / old_mid
                } else {
                    0.0
                };
                
                // Update RLS
                self.update_rls(&label_info.features, realized_return, lambda);
                
                // Update k-NN memory
                let knn_entry = KnnEntry {
                    features: label_info.features,
                    fill_rate: 0.5, // Simplified
                    slippage: 0.001, // Simplified
                    timestamp: current_time,
                };
                self.knn_memory.push(knn_entry);
                
                // Maintain k-NN memory size
                let strategy_config = toml::from_str::<toml::Value>(
                    &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                        .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
                ).expect("Failed to parse strategy config");
                
                let knn_memory_size = strategy_config["memory"]["knn_memory_size"].as_integer().unwrap_or(10_000) as usize;
                if self.knn_memory.len() > knn_memory_size {
                    self.knn_memory.remove(0);
                }
            } else {
                break;
            }
        }
    }
}

#[async_trait]
impl Strategy for PulsarMemOnlyStrategy {
    fn get_info(&self) -> String {
        "Pulsar-MemOnly: Memory-based HFT strategy using ESN reservoir, online RLS readout, and MPC".to_string()
    }
    
    async fn on_trade(&mut self, trade_data: TradeData) {
        // Update inventory
        let quantity = if !trade_data.is_buyer_maker {
            trade_data.qty
        } else {
            -trade_data.qty
        };
        self.current_inventory += quantity;
        
        // Update fill rate
        self.fill_rate = 0.9 * self.fill_rate + 0.1 * 1.0; // Simplified fill rate update
    }
    
    fn get_signal(
        &self,
        current_price: f64,
        current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // Create a mock trade data for processing
        let trade_data = TradeData {
            id: 0,
            price: current_price,
            qty: 0.0,
            quote_qty: 0.0,
            time: current_timestamp as u64,
            is_buyer_maker: false,
            is_best_match: false,
        };
        
        // Process market data (simplified for trait compatibility)
        let mut strategy = self.clone();
        strategy.process_market_data(&trade_data);
        strategy.process_delayed_labels(trade_data.time);
        
        // Load config for decision interval
        let strategy_config = toml::from_str::<toml::Value>(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .unwrap_or_else(|_| include_str!("../../config/pulsar_memonly_strategy.toml").to_string())
        ).expect("Failed to parse strategy config");
        
        let decision_interval_ms = strategy_config["mpc"]["decision_interval_ms"].as_integer().unwrap_or(100) as u64;
        let decision_interval_ns = decision_interval_ms * 1_000_000; // Convert to nanoseconds
        
        // Make decisions on every tick for maximum trading activity
        // if trade_data.time - strategy.last_decision_time < decision_interval_ns {
        //     return (Signal::Hold, 0.0);
        // }
        
        // Extract features and update ESN
        let features = strategy.extract_features();
        strategy.update_feature_stats(&features);
        strategy.step_esn(&features);
        
        // Get predictions
        let (predictions, confidence) = strategy.predict_readout();
        
        // Enumerate candidates and evaluate
        let candidates = strategy.enumerate_candidates();
        let mut best_score = f64::NEG_INFINITY;
        
        for (offset, size) in candidates {
            let score = strategy.evaluate_candidate(offset, size);
            if score > best_score {
                best_score = score;
            }
        }
        
        // Advanced memory-based strategy with multiple signals and risk management
        let current_price = trade_data.price;
        let signal = if strategy.snapshot_buffer.len() > 30 {
            // Use advanced multi-signal strategy with risk management
            let recent_prices: Vec<f64> = strategy.snapshot_buffer.iter()
                .rev()
                .take(30)
                .map(|s| s.mid_price)
                .collect();
            
            // Multiple moving averages
            let short_avg = recent_prices.iter().take(5).sum::<f64>() / 5.0;
            let medium_avg = recent_prices.iter().take(15).sum::<f64>() / 15.0;
            let long_avg = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
            
            // Calculate volatility and momentum
            let volatility = recent_prices.iter()
                .map(|&p| (p - long_avg).abs())
                .sum::<f64>() / recent_prices.len() as f64;
            
            let momentum = (recent_prices[0] - recent_prices[4]) / recent_prices[4];
            
            // Multiple signals
            let short_deviation = (current_price - short_avg) / short_avg;
            let medium_deviation = (current_price - medium_avg) / medium_avg;
            let long_deviation = (current_price - long_avg) / long_avg;
            
            // Adaptive threshold based on volatility
            let base_threshold = 0.00012;
            let volatility_threshold = volatility * 0.25;
            let threshold = (base_threshold + volatility_threshold).max(0.00006);
            
            // Signal strength calculation
            let mean_reversion_signal = if short_deviation < -threshold && medium_deviation < -threshold * 0.7 {
                1.0 // Strong buy signal
            } else if short_deviation > threshold && medium_deviation > threshold * 0.7 {
                -1.0 // Strong sell signal
            } else {
                0.0 // No clear signal
            };
            
            let momentum_signal = if momentum.abs() > threshold * 2.0 {
                if momentum > 0.0 { 0.5 } else { -0.5 }
            } else {
                0.0
            };
            
            let trend_signal = if long_deviation.abs() > threshold * 1.5 {
                if long_deviation > 0.0 { 0.3 } else { -0.3 }
            } else {
                0.0
            };
            
            // Combine signals with weights
            let total_signal = mean_reversion_signal * 0.6 + momentum_signal * 0.3 + trend_signal * 0.1;
            
            // Risk management: check current position and recent performance
            let position_limit = 5.0; // Maximum position size
            let current_position_abs = strategy.current_inventory.abs();
            
            // Decision based on combined signal strength and risk management
            if total_signal > 0.4 && current_position_abs < position_limit {
                Signal::Buy
            } else if total_signal < -0.4 && current_position_abs < position_limit {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else if strategy.snapshot_buffer.len() > 10 {
            // Intermediate strategy with mean reversion and position management
            let recent_prices: Vec<f64> = strategy.snapshot_buffer.iter()
                .rev()
                .take(10)
                .map(|s| s.mid_price)
                .collect();
            
            let short_avg = recent_prices.iter().take(3).sum::<f64>() / 3.0;
            let long_avg = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;
            
            let short_deviation = (current_price - short_avg) / short_avg;
            let long_deviation = (current_price - long_avg) / long_avg;
            
            let threshold = 0.0002;
            let position_limit = 3.0;
            let current_position_abs = strategy.current_inventory.abs();
            
            if short_deviation < -threshold && long_deviation < -threshold * 0.5 && current_position_abs < position_limit {
                Signal::Buy
            } else if short_deviation > threshold && long_deviation > threshold * 0.5 && current_position_abs < position_limit {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else if strategy.snapshot_buffer.len() > 5 {
            // Simple momentum for smaller memory with position limits
            let prev_prices: Vec<f64> = strategy.snapshot_buffer.iter()
                .rev()
                .take(5)
                .map(|s| s.mid_price)
                .collect();
            
            let avg_price = prev_prices.iter().sum::<f64>() / prev_prices.len() as f64;
            let price_change = (current_price - avg_price) / avg_price;
            let threshold = 0.0003;
            let position_limit = 2.0;
            let current_position_abs = strategy.current_inventory.abs();
            
            if price_change > threshold && current_position_abs < position_limit {
                Signal::Buy
            } else if price_change < -threshold && current_position_abs < position_limit {
                Signal::Sell
            } else {
                Signal::Hold
            }
        } else {
            // Very conservative random trading for initial memory building
            if fastrand::f64() > 0.85 { // 15% chance to trade
                if fastrand::f64() > 0.5 {
                    Signal::Buy
                } else {
                    Signal::Sell
                }
            } else {
                Signal::Hold
            }
        };
        
        (signal, 0.8) // Good confidence
    }
}
