//! # Strategy Factory
//!
//! A factory for creating and managing different trading strategies

use crate::strategy::Strategy;
use crate::{
    QuantumHftStrategy,
    MicrostructureArbitrageStrategy,
    StatisticalArbitrageStrategy,
    MultiTimeframeMomentumStrategy,
    GameTheoryMLStrategy,
    AdaptiveRegimeStrategy,
    SentimentAnalysisStrategy,
    NeuralNetworkStrategy,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrategyType {
    QuantumHft,
    MicrostructureArbitrage,
    StatisticalArbitrage,
    MultiTimeframeMomentum,
    GameTheoryML,
    AdaptiveRegime,
    SentimentAnalysis,
    NeuralNetwork,
}

impl StrategyType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "quantum_hft" | "quantum" | "hft" => Some(StrategyType::QuantumHft),
            "microstructure" | "microstructure_arbitrage" => Some(StrategyType::MicrostructureArbitrage),
            "statistical" | "statistical_arbitrage" => Some(StrategyType::StatisticalArbitrage),
            "multi_timeframe" | "multitf" | "momentum" => Some(StrategyType::MultiTimeframeMomentum),
            "game_theory" | "gametheory" | "adversarial" => Some(StrategyType::GameTheoryML),
            "adaptive_regime" | "regime" | "adaptive" => Some(StrategyType::AdaptiveRegime),
            "sentiment" | "sentiment_analysis" => Some(StrategyType::SentimentAnalysis),
            "neural" | "neural_network" | "nn" => Some(StrategyType::NeuralNetwork),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            StrategyType::QuantumHft => "quantum_hft".to_string(),
            StrategyType::MicrostructureArbitrage => "microstructure_arbitrage".to_string(),
            StrategyType::StatisticalArbitrage => "statistical_arbitrage".to_string(),
            StrategyType::MultiTimeframeMomentum => "multi_timeframe_momentum".to_string(),
            StrategyType::GameTheoryML => "game_theory_ml".to_string(),
            StrategyType::AdaptiveRegime => "adaptive_regime".to_string(),
            StrategyType::SentimentAnalysis => "sentiment_analysis".to_string(),
            StrategyType::NeuralNetwork => "neural_network".to_string(),
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            StrategyType::QuantumHft => "Quantum HFT Strategy - Multi-strategy approach with RSI, momentum, and mean reversion",
            StrategyType::MicrostructureArbitrage => "Market Microstructure Arbitrage - Exploits order book inefficiencies and spread opportunities",
            StrategyType::StatisticalArbitrage => "Statistical Arbitrage - Machine learning-based mean reversion and statistical inefficiencies",
            StrategyType::MultiTimeframeMomentum => "Multi-Timeframe Momentum - Analyzes momentum across different time horizons",
            StrategyType::GameTheoryML => "Game Theory + Adversarial ML - Models market interactions and defends against exploitation",
            StrategyType::AdaptiveRegime => "Adaptive Regime Strategy - Detects market conditions and switches between different trading approaches",
            StrategyType::SentimentAnalysis => "Sentiment Analysis Strategy - Uses social media sentiment and news analysis for trading decisions",
            StrategyType::NeuralNetwork => "Neural Network Strategy - Uses a simple feedforward neural network for price prediction",
        }
    }
}

pub struct StrategyFactory;

impl StrategyFactory {
    /// Create a new strategy instance based on the strategy type
    pub fn create_strategy(strategy_type: StrategyType) -> Box<dyn Strategy> {
        match strategy_type {
            StrategyType::QuantumHft => Box::new(QuantumHftStrategy::new()),
            StrategyType::MicrostructureArbitrage => Box::new(MicrostructureArbitrageStrategy::new()),
            StrategyType::StatisticalArbitrage => Box::new(StatisticalArbitrageStrategy::new()),
            StrategyType::MultiTimeframeMomentum => Box::new(MultiTimeframeMomentumStrategy::new()),
            StrategyType::GameTheoryML => Box::new(GameTheoryMLStrategy::new()),
            StrategyType::AdaptiveRegime => Box::new(AdaptiveRegimeStrategy::new()),
            StrategyType::SentimentAnalysis => Box::new(SentimentAnalysisStrategy::new()),
            StrategyType::NeuralNetwork => Box::new(NeuralNetworkStrategy::new()),
        }
    }

    /// List all available strategies
    pub fn list_strategies() -> Vec<(StrategyType, &'static str)> {
        vec![
            (StrategyType::AdaptiveRegime, "Adaptive Regime Strategy"),
            (StrategyType::SentimentAnalysis, "Sentiment Analysis Strategy"),
            (StrategyType::NeuralNetwork, "Neural Network Strategy"),
            (StrategyType::GameTheoryML, "Game Theory + Adversarial ML"),
            (StrategyType::MicrostructureArbitrage, "Market Microstructure Arbitrage"),
            (StrategyType::StatisticalArbitrage, "Statistical Arbitrage"),
            (StrategyType::MultiTimeframeMomentum, "Multi-Timeframe Momentum"),
            (StrategyType::QuantumHft, "Quantum HFT Strategy"),
        ]
    }

    /// Get strategy recommendations based on market conditions
    pub fn get_recommendations() -> Vec<(StrategyType, &'static str, f64)> {
        vec![
            (StrategyType::AdaptiveRegime, "Best for dynamic markets with changing conditions", 0.95),
            (StrategyType::SentimentAnalysis, "Best for markets influenced by news and social media", 0.9),
            (StrategyType::NeuralNetwork, "Best for markets with complex patterns and relationships", 0.85),
            (StrategyType::GameTheoryML, "Best for competitive markets with adversarial traders", 0.8),
            (StrategyType::MicrostructureArbitrage, "Best for high-frequency markets with tight spreads", 0.75),
            (StrategyType::StatisticalArbitrage, "Best for mean-reverting markets with clear patterns", 0.7),
            (StrategyType::MultiTimeframeMomentum, "Best for trending markets with strong momentum", 0.65),
            (StrategyType::QuantumHft, "General purpose strategy for various market conditions", 0.6),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_type_parsing() {
        assert_eq!(StrategyType::from_str("quantum_hft"), Some(StrategyType::QuantumHft));
        assert_eq!(StrategyType::from_str("microstructure"), Some(StrategyType::MicrostructureArbitrage));
        assert_eq!(StrategyType::from_str("statistical"), Some(StrategyType::StatisticalArbitrage));
        assert_eq!(StrategyType::from_str("multi_timeframe"), Some(StrategyType::MultiTimeframeMomentum));
        assert_eq!(StrategyType::from_str("game_theory"), Some(StrategyType::GameTheoryML));
        assert_eq!(StrategyType::from_str("adaptive_regime"), Some(StrategyType::AdaptiveRegime));
        assert_eq!(StrategyType::from_str("sentiment_analysis"), Some(StrategyType::SentimentAnalysis));
        assert_eq!(StrategyType::from_str("neural_network"), Some(StrategyType::NeuralNetwork));
        assert_eq!(StrategyType::from_str("unknown"), None);
    }

    #[test]
    fn test_strategy_creation() {
        let strategies = vec![
            StrategyType::QuantumHft,
            StrategyType::MicrostructureArbitrage,
            StrategyType::StatisticalArbitrage,
            StrategyType::MultiTimeframeMomentum,
            StrategyType::GameTheoryML,
            StrategyType::AdaptiveRegime,
            StrategyType::SentimentAnalysis,
            StrategyType::NeuralNetwork,
        ];

        for strategy_type in strategies {
            let strategy = StrategyFactory::create_strategy(strategy_type);
            assert!(!strategy.get_info().is_empty());
        }
    }
}
