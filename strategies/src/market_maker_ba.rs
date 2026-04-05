use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use trade::execution::{DecisionMetric, OrderIntent};
use trade::market::{MarketEvent, MarketState};
use trade::strategy::{
    NoOpStrategyLogger, Strategy, StrategyContext, StrategyDecision, StrategyDiagnostics,
    StrategyLogger,
};

#[derive(Debug, Clone, Deserialize)]
pub struct MarketMakerBaConfig {
    #[serde(default = "default_base_half_spread_bps")]
    pub base_half_spread_bps: f64,
    #[serde(default = "default_min_half_spread_bps")]
    pub min_half_spread_bps: f64,
    #[serde(default = "default_max_half_spread_bps")]
    pub max_half_spread_bps: f64,
    #[serde(default = "default_volatility_reference_bps")]
    pub volatility_reference_bps: f64,
    #[serde(default = "default_volatility_multiplier_min")]
    pub volatility_multiplier_min: f64,
    #[serde(default = "default_volatility_multiplier_max")]
    pub volatility_multiplier_max: f64,
    #[serde(default = "default_cash_fraction")]
    pub cash_fraction: f64,
}

fn default_base_half_spread_bps() -> f64 {
    1.0
}
fn default_min_half_spread_bps() -> f64 {
    0.4
}
fn default_max_half_spread_bps() -> f64 {
    8.0
}
fn default_volatility_reference_bps() -> f64 {
    30.0
}
fn default_volatility_multiplier_min() -> f64 {
    0.8
}
fn default_volatility_multiplier_max() -> f64 {
    3.0
}
fn default_cash_fraction() -> f64 {
    0.04
}

impl Default for MarketMakerBaConfig {
    fn default() -> Self {
        Self {
            base_half_spread_bps: default_base_half_spread_bps(),
            min_half_spread_bps: default_min_half_spread_bps(),
            max_half_spread_bps: default_max_half_spread_bps(),
            volatility_reference_bps: default_volatility_reference_bps(),
            volatility_multiplier_min: default_volatility_multiplier_min(),
            volatility_multiplier_max: default_volatility_multiplier_max(),
            cash_fraction: default_cash_fraction(),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct MarketMakerBaDiagnostics {
    no_quote_ticks: usize,
    last_fair_price: f64,
    last_half_spread_bps: f64,
    last_buy_quantity: f64,
    last_sell_quantity: f64,
}

pub struct MarketMakerBaStrategy {
    config: MarketMakerBaConfig,
    logger: NoOpStrategyLogger,
    diagnostics: MarketMakerBaDiagnostics,
}

impl MarketMakerBaStrategy {
    fn load_config<P: AsRef<Path>>(
        path: P,
    ) -> Result<MarketMakerBaConfig, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: MarketMakerBaConfig = toml::from_str(&content)?;
        Ok(config)
    }

    fn fair_price_and_book(&self, market_state: &MarketState) -> Option<(f64, f64, f64)> {
        if let Some(book) = market_state.top_of_book() {
            let bid = book.bid.price;
            let ask = book.ask.price;
            let fair = (bid + ask) / 2.0;
            if bid > f64::EPSILON && ask > bid && fair > f64::EPSILON {
                return Some((bid, ask, fair));
            }
        }

        let micro = market_state.micro();
        if !micro.mid_initialized() || micro.ema_mid_price <= f64::EPSILON {
            return None;
        }

        let fair = micro.ema_mid_price;
        let spread_bps = micro
            .ema_spread_bps
            .max(self.config.base_half_spread_bps * 2.0);
        let half_spread_frac = spread_bps / 20_000.0;
        let bid = fair * (1.0 - half_spread_frac);
        let ask = fair * (1.0 + half_spread_frac);
        Some((bid, ask, fair))
    }

    fn dynamic_half_spread_bps(&self, market_state: &MarketState) -> f64 {
        let vol_bps = market_state.micro().realized_vol_bps.max(1.0);
        let reference = self.config.volatility_reference_bps.max(1.0);
        let vol_multiplier = (vol_bps / reference).clamp(
            self.config.volatility_multiplier_min,
            self.config.volatility_multiplier_max,
        );
        (self.config.base_half_spread_bps * vol_multiplier).clamp(
            self.config.min_half_spread_bps,
            self.config.max_half_spread_bps,
        )
    }
}

#[async_trait::async_trait]
impl Strategy for MarketMakerBaStrategy {
    fn logger(&self) -> &dyn StrategyLogger {
        &self.logger
    }

    fn from_file<P: AsRef<Path>>(config_path: P) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized,
    {
        let config = Self::load_config(config_path)?;
        Ok(Self {
            config,
            logger: NoOpStrategyLogger,
            diagnostics: MarketMakerBaDiagnostics::default(),
        })
    }

    fn get_info(&self) -> String {
        "MarketMakerBaStrategy - continuous symmetric bid/ask maker".to_string()
    }

    fn diagnostics(&self) -> StrategyDiagnostics {
        let mut counters = BTreeMap::new();
        counters.insert("ba.no_quote_ticks".into(), self.diagnostics.no_quote_ticks);

        let mut gauges = BTreeMap::new();
        gauges.insert(
            "ba.last_fair_price".into(),
            self.diagnostics.last_fair_price,
        );
        gauges.insert(
            "ba.last_half_spread_bps".into(),
            self.diagnostics.last_half_spread_bps,
        );
        gauges.insert(
            "ba.last_buy_quantity".into(),
            self.diagnostics.last_buy_quantity,
        );
        gauges.insert(
            "ba.last_sell_quantity".into(),
            self.diagnostics.last_sell_quantity,
        );

        StrategyDiagnostics { counters, gauges }
    }

    fn market_state_window_millis(&self) -> u64 {
        30_000
    }

    async fn on_event(&mut self, event: &MarketEvent, _market_state: &MarketState) {
        match event {
            MarketEvent::Trade(_) | MarketEvent::BookTicker(_) | MarketEvent::Depth(_) => {}
        }
    }

    fn decide(
        &mut self,
        market_state: &MarketState,
        context: &StrategyContext,
    ) -> StrategyDecision {
        let Some((best_bid, best_ask, fair_price)) = self.fair_price_and_book(market_state) else {
            self.diagnostics.no_quote_ticks += 1;
            return StrategyDecision::no_action();
        };

        let half_spread_bps = self.dynamic_half_spread_bps(market_state);
        let tick_size = context.tick_size.max(f64::EPSILON);

        let mut buy_price = fair_price * (1.0 - half_spread_bps / 10_000.0);
        let mut sell_price = fair_price * (1.0 + half_spread_bps / 10_000.0);

        if buy_price >= best_ask {
            buy_price = best_ask - tick_size;
        }
        if sell_price <= best_bid {
            sell_price = best_bid + tick_size;
        }
        if sell_price <= buy_price {
            sell_price = buy_price + tick_size;
        }

        let quote_quantity = context
            .capped_entry_quantity(buy_price, self.config.cash_fraction, None)
            .unwrap_or(0.0);

        if quote_quantity <= f64::EPSILON {
            return StrategyDecision::no_action();
        }

        self.diagnostics.last_fair_price = fair_price;
        self.diagnostics.last_half_spread_bps = half_spread_bps;
        self.diagnostics.last_buy_quantity = quote_quantity;
        self.diagnostics.last_sell_quantity = quote_quantity;

        let quoted_spread_bps = (sell_price - buy_price) / fair_price * 10_000.0;

        StrategyDecision {
            confidence: 1.0,
            intent: OrderIntent::QuoteBothSides {
                buy_price,
                buy_quantity: quote_quantity,
                sell_price,
                sell_quantity: quote_quantity,
                rationale: "ba_symmetric_quote",
                expected_edge_bps: quoted_spread_bps,
            },
            metrics: vec![
                DecisionMetric {
                    name: "fair_price",
                    value: fair_price,
                },
                DecisionMetric {
                    name: "half_spread_bps",
                    value: half_spread_bps,
                },
                DecisionMetric {
                    name: "buy_quantity",
                    value: quote_quantity,
                },
                DecisionMetric {
                    name: "sell_quantity",
                    value: quote_quantity,
                },
                DecisionMetric {
                    name: "quoted_spread_bps",
                    value: quoted_spread_bps,
                },
            ],
        }
    }
}
