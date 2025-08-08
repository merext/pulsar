//! # Strategy Factory
//!
//! A factory for creating and managing different trading strategies

use crate::strategy::Strategy;
use crate::SentimentAnalysisStrategy;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrategyType {
    SentimentAnalysis,
}

impl StrategyType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sentiment" | "sentiment_analysis" => Some(StrategyType::SentimentAnalysis),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            StrategyType::SentimentAnalysis => "sentiment_analysis".to_string(),
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            StrategyType::SentimentAnalysis => "Sentiment Analysis Strategy - Uses social media sentiment and news analysis for trading decisions",
        }
    }
}

pub struct StrategyFactory;

impl StrategyFactory {
    /// Create a new strategy instance based on the strategy type
    pub fn create_strategy(strategy_type: StrategyType) -> Box<dyn Strategy> {
        match strategy_type {
            StrategyType::SentimentAnalysis => Box::new(SentimentAnalysisStrategy::new()),
        }
    }

    /// List all available strategies
    pub fn list_strategies() -> Vec<(StrategyType, &'static str)> {
        vec![
            (StrategyType::SentimentAnalysis, "Sentiment Analysis Strategy"),
        ]
    }

    /// Get strategy recommendations based on market conditions
    pub fn get_recommendations() -> Vec<(StrategyType, &'static str, f64)> {
        vec![
            (StrategyType::SentimentAnalysis, "Best for markets influenced by news and social media", 0.95),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_type_parsing() {
        assert_eq!(StrategyType::from_str("sentiment"), Some(StrategyType::SentimentAnalysis));
        assert_eq!(StrategyType::from_str("sentiment_analysis"), Some(StrategyType::SentimentAnalysis));
        assert_eq!(StrategyType::from_str("invalid"), None);
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = StrategyFactory::create_strategy(StrategyType::SentimentAnalysis);
        assert_eq!(strategy.get_info(), "Sentiment Analysis Strategy");
    }
}
