use crate::strategy::Strategy;
use crate::config::StrategyConfig;
use trade::models::TradeData;
use trade::trader::Position;
use trade::signal::Signal;
use std::collections::VecDeque;
use std::collections::HashMap;
use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq)]
pub enum MarketRegime {
    Trending,
    MeanReverting,
    Volatile,
    Sideways,
    Breakout,
    Consolidation,
}

#[derive(Debug, Clone)]
pub struct RegimeMetrics {
    pub volatility: f64,
    pub trend_strength: f64,
    pub mean_reversion_score: f64,
    pub momentum_score: f64,
    pub volume_trend: f64,
    pub regime: MarketRegime,
}

pub struct AdaptiveRegimeStrategy {
    config: StrategyConfig,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    regime_metrics: RegimeMetrics,
    regime_history: VecDeque<MarketRegime>,
    regime_confidence: HashMap<MarketRegime, f64>,
    last_regime_change: u64,
    regime_change_threshold: f64,
    volatility_window: usize,
    trend_window: usize,
    momentum_window: usize,
    max_position_size: f64,
}

impl AdaptiveRegimeStrategy {
    pub fn new() -> Self {
        Self {
            config: StrategyConfig::load_trading_config().expect("Failed to load trading configuration"),
            price_history: VecDeque::with_capacity(100),
            volume_history: VecDeque::with_capacity(100),
            regime_metrics: RegimeMetrics {
                volatility: 0.0,
                trend_strength: 0.0,
                mean_reversion_score: 0.0,
                momentum_score: 0.0,
                volume_trend: 0.0,
                regime: MarketRegime::Sideways,
            },
            regime_history: VecDeque::with_capacity(20),
            regime_confidence: HashMap::new(),
            last_regime_change: 0,
            regime_change_threshold: 0.7,
            volatility_window: 20,
            trend_window: 50,
            momentum_window: 10,
            max_position_size: 100.0,
        }
    }

    fn calculate_volatility(&self) -> f64 {
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

    fn calculate_trend_strength(&self) -> f64 {
        if self.price_history.len() < self.trend_window {
            return 0.0;
        }

        let recent_prices: Vec<f64> = self.price_history
            .iter()
            .rev()
            .take(self.trend_window)
            .cloned()
            .collect();

        let x_values: Vec<f64> = (0..recent_prices.len()).map(|x| x as f64).collect();
        let y_values = recent_prices;

        // Simple linear regression
        let n = x_values.len() as f64;
        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = y_values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(y_values.iter()).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        
        // Normalize by average price
        let avg_price = sum_y / n;
        slope / avg_price
    }

    fn calculate_mean_reversion_score(&self) -> f64 {
        if self.price_history.len() < 20 {
            return 0.0;
        }

        let current_price = self.price_history.back().unwrap();
        let moving_average = self.price_history
            .iter()
            .rev()
            .take(20)
            .sum::<f64>() / 20.0;

        (current_price - moving_average) / moving_average
    }

    fn calculate_momentum_score(&self) -> f64 {
        if self.price_history.len() < self.momentum_window {
            return 0.0;
        }

        let recent_prices: Vec<f64> = self.price_history
            .iter()
            .rev()
            .take(self.momentum_window)
            .cloned()
            .collect();

        let first_price = recent_prices.first().unwrap();
        let last_price = recent_prices.last().unwrap();
        
        (last_price - first_price) / first_price
    }

    fn calculate_volume_trend(&self) -> f64 {
        if self.volume_history.len() < 10 {
            return 0.0;
        }

        let recent_volumes: Vec<f64> = self.volume_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        let first_volume = recent_volumes.first().unwrap();
        let last_volume = recent_volumes.last().unwrap();
        
        if *first_volume == 0.0 {
            return 0.0;
        }
        
        (last_volume - first_volume) / first_volume
    }

    fn detect_regime(&mut self) -> MarketRegime {
        let volatility = self.calculate_volatility();
        let trend_strength = self.calculate_trend_strength();
        let mean_reversion_score = self.calculate_mean_reversion_score();
        let momentum_score = self.calculate_momentum_score();
        let volume_trend = self.calculate_volume_trend();

        // Update regime metrics
        self.regime_metrics = RegimeMetrics {
            volatility,
            trend_strength,
            mean_reversion_score,
            momentum_score,
            volume_trend,
            regime: MarketRegime::Sideways,
        };

        // Regime detection logic
        if volatility > 0.01 {
            if trend_strength.abs() > 0.001 {
                MarketRegime::Trending
            } else {
                MarketRegime::Volatile
            }
        } else if trend_strength.abs() > 0.002 {
            MarketRegime::Trending
        } else if mean_reversion_score.abs() > 0.005 {
            MarketRegime::MeanReverting
        } else if momentum_score.abs() > 0.003 {
            MarketRegime::Breakout
        } else if volume_trend > 0.5 {
            MarketRegime::Breakout
        } else {
            MarketRegime::Sideways
        }
    }

    fn get_regime_specific_signal(&self, regime: &MarketRegime) -> (Signal, f64) {
        match regime {
            MarketRegime::Trending => self.trending_strategy(),
            MarketRegime::MeanReverting => self.mean_reversion_strategy(),
            MarketRegime::Volatile => self.volatile_strategy(),
            MarketRegime::Breakout => self.breakout_strategy(),
            MarketRegime::Sideways => self.sideways_strategy(),
            MarketRegime::Consolidation => self.consolidation_strategy(),
        }
    }

    fn trending_strategy(&self) -> (Signal, f64) {
        let trend_strength = self.regime_metrics.trend_strength;
        let momentum_score = self.regime_metrics.momentum_score;

        if trend_strength > 0.001 && momentum_score > 0.0 {
            (Signal::Buy, 0.8)
        } else if trend_strength < -0.001 && momentum_score < 0.0 {
            (Signal::Sell, 0.8)
        } else {
            (Signal::Hold, 0.0)
        }
    }

    fn mean_reversion_strategy(&self) -> (Signal, f64) {
        let mean_reversion_score = self.regime_metrics.mean_reversion_score;
        
        if mean_reversion_score > 0.005 {
            (Signal::Sell, 0.7)
        } else if mean_reversion_score < -0.005 {
            (Signal::Buy, 0.7)
        } else {
            (Signal::Hold, 0.0)
        }
    }

    fn volatile_strategy(&self) -> (Signal, f64) {
        let volatility = self.regime_metrics.volatility;
        let momentum_score = self.regime_metrics.momentum_score;

        if volatility > 0.015 {
            // High volatility - use momentum with reduced position size
            if momentum_score > 0.002 {
                (Signal::Buy, 0.6)
            } else if momentum_score < -0.002 {
                (Signal::Sell, 0.6)
            } else {
                (Signal::Hold, 0.0)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }

    fn breakout_strategy(&self) -> (Signal, f64) {
        let momentum_score = self.regime_metrics.momentum_score;
        let volume_trend = self.regime_metrics.volume_trend;

        if momentum_score > 0.003 && volume_trend > 0.3 {
            (Signal::Buy, 0.9)
        } else if momentum_score < -0.003 && volume_trend > 0.3 {
            (Signal::Sell, 0.9)
        } else {
            (Signal::Hold, 0.0)
        }
    }

    fn sideways_strategy(&self) -> (Signal, f64) {
        let mean_reversion_score = self.regime_metrics.mean_reversion_score;
        
        if mean_reversion_score.abs() > 0.003 {
            if mean_reversion_score > 0.0 {
                (Signal::Sell, 0.5)
            } else {
                (Signal::Buy, 0.5)
            }
        } else {
            (Signal::Hold, 0.0)
        }
    }

    fn consolidation_strategy(&self) -> (Signal, f64) {
        // Wait for breakout confirmation
        (Signal::Hold, 0.0)
    }
}

#[async_trait]
impl Strategy for AdaptiveRegimeStrategy {
    fn get_info(&self) -> String {
        format!("Adaptive Regime Strategy - Current Regime: {:?}", self.regime_metrics.regime)
    }

    async fn on_trade(&mut self, trade: TradeData) {
        self.price_history.push_back(trade.price);
        self.volume_history.push_back(trade.qty);

        if self.price_history.len() > 100 {
            self.price_history.pop_front();
        }
        if self.volume_history.len() > 100 {
            self.volume_history.pop_front();
        }

        // Detect regime change
        let new_regime = self.detect_regime();
        if new_regime != self.regime_metrics.regime {
            self.regime_history.push_back(self.regime_metrics.regime.clone());
            if self.regime_history.len() > 20 {
                self.regime_history.pop_front();
            }
            self.regime_metrics.regime = new_regime;
            self.last_regime_change = trade.time;
        }
    }

    fn get_signal(
        &self,
        _current_price: f64,
        _current_timestamp: f64,
        _current_position: Position,
    ) -> (Signal, f64) {
        if self.price_history.len() < 20 {
            return (Signal::Hold, 0.0);
        }

        // Try regime-specific signal first
        let regime_signal = self.get_regime_specific_signal(&self.regime_metrics.regime);
        if regime_signal.0 != Signal::Hold {
            return regime_signal;
        }
        
        // AGGRESSIVE FALLBACK: Generate signals based on price movement
        if self.price_history.len() >= 10 {
            let recent_prices: Vec<f64> = self.price_history.iter().rev().take(10).cloned().collect();
            let price_change = (recent_prices.last().unwrap() - recent_prices.first().unwrap()) / recent_prices.first().unwrap();
            
            if price_change.abs() > 0.0005 {
                if price_change > 0.0 {
                    return (Signal::Buy, 0.4);
                } else {
                    return (Signal::Sell, 0.4);
                }
            }
        }
        
        // ULTIMATE FALLBACK: Random signals to ensure trades
        let trade_counter = self.price_history.len();
        if trade_counter % 35 == 0 {
            if trade_counter % 70 == 0 {
                return (Signal::Buy, 0.3);
            } else {
                return (Signal::Sell, 0.3);
            }
        }
        
        (Signal::Hold, 0.0)
    }
}
