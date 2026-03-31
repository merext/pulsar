use crate::config::TradeConfig;

#[derive(Debug, Clone)]
pub struct PositionSizer {
    config: TradeConfig,
}

impl PositionSizer {
    pub fn new(config: TradeConfig) -> Self {
        Self { config }
    }

    pub fn size_order(
        &self,
        price: f64,
        confidence: f64,
        available_cash: f64,
        current_position_notional: f64,
        volatility: f64,
        step_size_fallback: f64,
        volatility_factors: VolatilityFactors,
    ) -> f64 {
        let position_config = &self.config.position_sizing;
        let exchange_config = &self.config.exchange;
        let confidence = confidence.clamp(0.0, 1.0);

        let min_notional = position_config.min_trade_notional;
        let max_notional = position_config.max_trade_notional;

        let volatility_factor = if volatility > volatility_factors.high_threshold {
            volatility_factors.high_factor
        } else if volatility > volatility_factors.medium_threshold {
            volatility_factors.medium_factor
        } else {
            volatility_factors.low_factor
        };

        let risk_factor = if confidence > volatility_factors.confidence_high_threshold {
            volatility_factors.confidence_high_factor
        } else if confidence < volatility_factors.confidence_low_threshold {
            volatility_factors.confidence_low_factor
        } else {
            1.0
        };

        let target_notional = confidence.mul_add(max_notional - min_notional, min_notional)
            * volatility_factor
            * risk_factor
            * volatility_factors.kelly_factor;
        let target_notional = target_notional.clamp(min_notional, max_notional);

        let spendable_cash = (available_cash * (1.0 - position_config.cash_reserve_ratio)).max(0.0);
        let remaining_position_room =
            (position_config.max_position_notional - current_position_notional).max(0.0);
        let capped_notional = target_notional
            .min(spendable_cash)
            .min(remaining_position_room);

        if capped_notional < min_notional || price <= 0.0 {
            return 0.0;
        }

        let step_size = exchange_config.step_size.max(step_size_fallback);
        let quantity = round_down_to_step(capped_notional / price, step_size);

        if quantity * price < exchange_config.min_notional {
            0.0
        } else {
            quantity
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VolatilityFactors {
    pub high_threshold: f64,
    pub medium_threshold: f64,
    pub high_factor: f64,
    pub medium_factor: f64,
    pub low_factor: f64,
    pub confidence_high_threshold: f64,
    pub confidence_low_threshold: f64,
    pub confidence_high_factor: f64,
    pub confidence_low_factor: f64,
    pub kelly_factor: f64,
}

fn round_down_to_step(value: f64, step_size: f64) -> f64 {
    if step_size <= 0.0 {
        return value.max(0.0);
    }

    (value / step_size).floor() * step_size
}
