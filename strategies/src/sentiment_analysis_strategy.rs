use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use trade::signal::Signal;
use std::collections::VecDeque;
use std::collections::HashMap;
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct SentimentData {
    pub social_sentiment: f64,  // -1.0 to 1.0
    pub news_sentiment: f64,    // -1.0 to 1.0
    pub volume_sentiment: f64,  // -1.0 to 1.0
    pub price_sentiment: f64,   // -1.0 to 1.0
    pub overall_sentiment: f64, // -1.0 to 1.0
    pub confidence: f64,        // 0.0 to 1.0
}

#[derive(Debug, Clone)]
pub struct SentimentMetrics {
    pub short_term_sentiment: f64,
    pub medium_term_sentiment: f64,
    pub long_term_sentiment: f64,
    pub sentiment_momentum: f64,
    pub sentiment_volatility: f64,
    pub sentiment_trend: f64,
}

pub struct SentimentAnalysisStrategy {
    config: StrategyConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    sentiment_history: VecDeque<SentimentData>,
    sentiment_metrics: SentimentMetrics,
    sentiment_weights: HashMap<String, f64>,
    sentiment_thresholds: HashMap<String, f64>,
    short_window: usize,
    medium_window: usize,
    long_window: usize,
    max_position_size: f64,
    sentiment_decay_factor: f64,
}

impl SentimentAnalysisStrategy {
    pub fn new() -> Self {
        let mut sentiment_weights = HashMap::new();
        sentiment_weights.insert("social".to_string(), 0.3);
        sentiment_weights.insert("news".to_string(), 0.25);
        sentiment_weights.insert("volume".to_string(), 0.25);
        sentiment_weights.insert("price".to_string(), 0.2);

        let mut sentiment_thresholds = HashMap::new();
        sentiment_thresholds.insert("buy_threshold".to_string(), 0.3);
        sentiment_thresholds.insert("sell_threshold".to_string(), -0.3);
        sentiment_thresholds.insert("strong_buy_threshold".to_string(), 0.6);
        sentiment_thresholds.insert("strong_sell_threshold".to_string(), -0.6);

        Self {
            config: StrategyConfig::load_trading_config().expect("Failed to load trading configuration"),
            price_history: VecDeque::with_capacity(200),
            volume_history: VecDeque::with_capacity(200),
            sentiment_history: VecDeque::with_capacity(100),
            sentiment_metrics: SentimentMetrics {
                short_term_sentiment: 0.0,
                medium_term_sentiment: 0.0,
                long_term_sentiment: 0.0,
                sentiment_momentum: 0.0,
                sentiment_volatility: 0.0,
                sentiment_trend: 0.0,
            },
            sentiment_weights,
            sentiment_thresholds,
            short_window: 10,
            medium_window: 30,
            long_window: 60,
            max_position_size: 100.0,
            sentiment_decay_factor: 0.95,
        }
    }

    fn calculate_social_sentiment(&self) -> f64 {
        // Simulate social media sentiment based on price and volume patterns
        if self.price_history.len() < 5 || self.volume_history.len() < 5 {
            return 0.0;
        }

        let recent_prices: Vec<f64> = self.price_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let recent_volumes: Vec<f64> = self.volume_history
            .iter()
            .rev()
            .take(5)
            .cloned()
            .collect();

        let price_momentum = (recent_prices.last().unwrap() - recent_prices.first().unwrap()) / recent_prices.first().unwrap();
        let volume_momentum = (recent_volumes.last().unwrap() - recent_volumes.first().unwrap()) / recent_volumes.first().unwrap();

        // Combine price and volume momentum for social sentiment
        let sentiment = (price_momentum * 0.7 + volume_momentum * 0.3).tanh();
        sentiment
    }

    fn calculate_news_sentiment(&self) -> f64 {
        // Simulate news sentiment based on price volatility and trend changes
        if self.price_history.len() < 20 {
            return 0.0;
        }

        let volatility = self.calculate_price_volatility();
        let trend_change = self.calculate_trend_change();

        // Higher volatility and trend changes indicate news impact
        let sentiment = (trend_change * 0.8 + volatility * 0.2).tanh();
        sentiment
    }

    fn calculate_volume_sentiment(&self) -> f64 {
        if self.volume_history.len() < 10 {
            return 0.0;
        }

        let recent_volumes: Vec<f64> = self.volume_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        let avg_volume = recent_volumes.iter().sum::<f64>() / recent_volumes.len() as f64;
        let current_volume = recent_volumes.last().unwrap();

        if avg_volume == 0.0 {
            return 0.0;
        }

        let volume_ratio = current_volume / avg_volume;
        let sentiment = (volume_ratio - 1.0).tanh();
        sentiment
    }

    fn calculate_price_sentiment(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.0;
        }

        let recent_prices: Vec<f64> = self.price_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();

        let current_price = recent_prices.last().unwrap();
        let avg_price = recent_prices.iter().sum::<f64>() / recent_prices.len() as f64;

        let price_deviation = (current_price - avg_price) / avg_price;
        let sentiment = price_deviation.tanh();
        sentiment
    }

    fn calculate_overall_sentiment(&mut self) -> SentimentData {
        let social_sentiment = self.calculate_social_sentiment();
        let news_sentiment = self.calculate_news_sentiment();
        let volume_sentiment = self.calculate_volume_sentiment();
        let price_sentiment = self.calculate_price_sentiment();

        // Weighted combination of all sentiment sources
        let overall_sentiment = 
            social_sentiment * self.sentiment_weights["social"] +
            news_sentiment * self.sentiment_weights["news"] +
            volume_sentiment * self.sentiment_weights["volume"] +
            price_sentiment * self.sentiment_weights["price"];

        let confidence = self.calculate_sentiment_confidence(&social_sentiment, &news_sentiment, &volume_sentiment, &price_sentiment);

        SentimentData {
            social_sentiment,
            news_sentiment,
            volume_sentiment,
            price_sentiment,
            overall_sentiment,
            confidence,
        }
    }

    fn calculate_sentiment_confidence(&self, social: &f64, news: &f64, volume: &f64, price: &f64) -> f64 {
        // Calculate confidence based on agreement between sentiment sources
        let sentiments = vec![*social, *news, *volume, *price];
        let avg_sentiment = sentiments.iter().sum::<f64>() / sentiments.len() as f64;
        
        let variance = sentiments.iter()
            .map(|s| (s - avg_sentiment).powi(2))
            .sum::<f64>() / sentiments.len() as f64;
        
        let std_dev = variance.sqrt();
        let confidence = (1.0 - std_dev).max(0.0).min(1.0);
        confidence
    }

    fn calculate_price_volatility(&self) -> f64 {
        if self.price_history.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = self.price_history
            .iter()
            .zip(self.price_history.iter().skip(1))
            .map(|(prev, curr)| (curr - prev) / prev)
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }

    fn calculate_trend_change(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.0;
        }

        let recent_prices: Vec<f64> = self.price_history
            .iter()
            .rev()
            .take(20)
            .cloned()
            .collect();

        let first_half = recent_prices.iter().take(10).sum::<f64>() / 10.0;
        let second_half = recent_prices.iter().skip(10).sum::<f64>() / 10.0;

        (second_half - first_half) / first_half
    }

    fn update_sentiment_metrics(&mut self) {
        if self.sentiment_history.len() < self.long_window {
            return;
        }

        let short_sentiments: Vec<f64> = self.sentiment_history
            .iter()
            .rev()
            .take(self.short_window)
            .map(|s| s.overall_sentiment)
            .collect();

        let medium_sentiments: Vec<f64> = self.sentiment_history
            .iter()
            .rev()
            .take(self.medium_window)
            .map(|s| s.overall_sentiment)
            .collect();

        let long_sentiments: Vec<f64> = self.sentiment_history
            .iter()
            .rev()
            .take(self.long_window)
            .map(|s| s.overall_sentiment)
            .collect();

        self.sentiment_metrics.short_term_sentiment = short_sentiments.iter().sum::<f64>() / short_sentiments.len() as f64;
        self.sentiment_metrics.medium_term_sentiment = medium_sentiments.iter().sum::<f64>() / medium_sentiments.len() as f64;
        self.sentiment_metrics.long_term_sentiment = long_sentiments.iter().sum::<f64>() / long_sentiments.len() as f64;

        // Calculate sentiment momentum
        if short_sentiments.len() >= 2 {
            self.sentiment_metrics.sentiment_momentum = short_sentiments.last().unwrap() - short_sentiments.first().unwrap();
        }

        // Calculate sentiment volatility
        let mean_sentiment = short_sentiments.iter().sum::<f64>() / short_sentiments.len() as f64;
        let variance = short_sentiments.iter()
            .map(|s| (s - mean_sentiment).powi(2))
            .sum::<f64>() / short_sentiments.len() as f64;
        self.sentiment_metrics.sentiment_volatility = variance.sqrt();

        // Calculate sentiment trend
        self.sentiment_metrics.sentiment_trend = self.sentiment_metrics.short_term_sentiment - self.sentiment_metrics.long_term_sentiment;
    }

    fn generate_sentiment_signal(&self) -> (Signal, f64) {
        let short_sentiment = self.sentiment_metrics.short_term_sentiment;
        let medium_sentiment = self.sentiment_metrics.medium_term_sentiment;
        let long_sentiment = self.sentiment_metrics.long_term_sentiment;
        let sentiment_momentum = self.sentiment_metrics.sentiment_momentum;
        let sentiment_volatility = self.sentiment_metrics.sentiment_volatility;

        // Calculate weighted sentiment score
        let weighted_sentiment = short_sentiment * 0.5 + medium_sentiment * 0.3 + long_sentiment * 0.2;
        
        // Strong buy signal: positive sentiment with momentum
        if weighted_sentiment > 0.2 && sentiment_momentum > 0.05 && sentiment_volatility < 0.8 {
            return (Signal::Buy, 0.7);
        }
        
        // Strong sell signal: negative sentiment with momentum
        if weighted_sentiment < -0.2 && sentiment_momentum < -0.05 && sentiment_volatility < 0.8 {
            return (Signal::Sell, 0.7);
        }
        
        // Moderate buy signal
        if weighted_sentiment > 0.1 && sentiment_momentum > 0.02 {
            return (Signal::Buy, 0.5);
        }
        
        // Moderate sell signal
        if weighted_sentiment < -0.1 && sentiment_momentum < -0.02 {
            return (Signal::Sell, 0.5);
        }
        
        // Weak signals for very strong sentiment
        if weighted_sentiment > 0.3 {
            return (Signal::Buy, 0.4);
        }
        
        if weighted_sentiment < -0.3 {
            return (Signal::Sell, 0.4);
        }
        
        (Signal::Hold, 0.0)
    }
}

#[async_trait]
impl Strategy for SentimentAnalysisStrategy {
    fn get_info(&self) -> String {
        let current_sentiment = if let Some(latest) = self.sentiment_history.back() {
            latest.overall_sentiment
        } else {
            0.0
        };
        format!("Sentiment Analysis Strategy - Current Sentiment: {:.3}", current_sentiment)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.qty);

        if self.price_history.len() > 200 {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > 200 {
            self.volume_history.pop_front();
        }

        // Calculate and store sentiment data
        let sentiment_data = self.calculate_overall_sentiment();
        self.sentiment_history.push_back(sentiment_data);

        if self.sentiment_history.len() > 100 {
            self.sentiment_history.pop_front();
        }

        // Update sentiment metrics
        self.update_sentiment_metrics();
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.sentiment_history.len() < 10 {
            return (Signal::Hold, 0.0);
        }

        self.generate_sentiment_signal()
    }
}
