use crate::config::TradeConfig;
use crate::signal::Signal;

#[derive(Debug, Clone, Copy)]
pub struct SimulatedExecution {
    pub execution_price: f64,
    pub executed_quantity: f64,
    pub fee_paid: f64,
    pub notional_value: f64,
    pub latency_seconds: f64,
    pub rejected_reason: Option<&'static str>,
    pub partially_filled: bool,
}

impl SimulatedExecution {
    pub fn total_cost(&self) -> f64 {
        self.notional_value + self.fee_paid
    }

    pub fn net_proceeds(&self) -> f64 {
        self.notional_value - self.fee_paid
    }

    pub fn is_rejected(&self) -> bool {
        self.executed_quantity <= 0.0 || self.rejected_reason.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct BacktestEngine {
    config: TradeConfig,
}

impl BacktestEngine {
    pub fn new(config: TradeConfig) -> Self {
        Self { config }
    }

    pub fn execute(&self, signal: Signal, market_price: f64, quantity: f64) -> SimulatedExecution {
        self.execute_with_constraints(signal, market_price, quantity, f64::INFINITY)
    }

    pub fn execute_with_constraints(
        &self,
        signal: Signal,
        market_price: f64,
        requested_quantity: f64,
        available_cash: f64,
    ) -> SimulatedExecution {
        if matches!(signal, Signal::Hold) {
            return SimulatedExecution {
                execution_price: market_price,
                executed_quantity: 0.0,
                fee_paid: 0.0,
                notional_value: 0.0,
                latency_seconds: 0.0,
                rejected_reason: None,
                partially_filled: false,
            };
        }

        if requested_quantity <= 0.0 {
            return SimulatedExecution {
                execution_price: market_price,
                executed_quantity: 0.0,
                fee_paid: 0.0,
                notional_value: 0.0,
                latency_seconds: 0.0,
                rejected_reason: Some("quantity_below_step"),
                partially_filled: false,
            };
        }

        let latency_seconds = self.estimate_latency_seconds(requested_quantity);
        let spread = self.estimate_spread();
        let slippage = self.estimate_slippage(signal, requested_quantity);
        let latency_impact = self.estimate_latency_impact(latency_seconds);
        let half_spread = spread / 2.0;
        let execution_price = match signal {
            Signal::Buy => market_price * (1.0 + half_spread + slippage + latency_impact),
            Signal::Sell => market_price * (1.0 - half_spread - slippage - latency_impact),
            Signal::Hold => market_price,
        };

        let mut executed_quantity = self.round_down_to_step(requested_quantity);
        let mut partially_filled = false;

        if matches!(signal, Signal::Buy) {
            let max_affordable_quantity = self.round_down_to_step(
                available_cash / (execution_price * (1.0 + self.config.exchange.taker_fee)),
            );
            executed_quantity = executed_quantity.min(max_affordable_quantity);
        }

        if matches!(signal, Signal::Buy) {
            let partial_fill_ratio = self.partial_fill_ratio(requested_quantity, execution_price);
            if partial_fill_ratio < 1.0 {
                let reduced_quantity =
                    self.round_down_to_step(executed_quantity * partial_fill_ratio);
                partially_filled = reduced_quantity > 0.0 && reduced_quantity < executed_quantity;
                executed_quantity = reduced_quantity;
            }
        }

        let notional_value = execution_price * executed_quantity;
        if executed_quantity <= 0.0 {
            return SimulatedExecution {
                execution_price,
                executed_quantity: 0.0,
                fee_paid: 0.0,
                notional_value: 0.0,
                latency_seconds,
                rejected_reason: Some("insufficient_balance"),
                partially_filled: false,
            };
        }

        if notional_value < self.config.exchange.min_notional {
            return SimulatedExecution {
                execution_price,
                executed_quantity: 0.0,
                fee_paid: 0.0,
                notional_value: 0.0,
                latency_seconds,
                rejected_reason: Some("min_notional"),
                partially_filled: false,
            };
        }

        let fee_paid = notional_value * self.config.exchange.taker_fee;

        SimulatedExecution {
            execution_price,
            executed_quantity,
            fee_paid,
            notional_value,
            latency_seconds,
            rejected_reason: None,
            partially_filled,
        }
    }

    fn estimate_slippage(&self, signal: Signal, quantity: f64) -> f64 {
        if matches!(signal, Signal::Hold) {
            return 0.0;
        }

        let min = self.config.slippage.min_slippage;
        let max = self.config.slippage.max_slippage;
        let size_factor = (quantity / self.config.exchange.max_order_size).clamp(0.0, 1.0);
        let volatility_component = self.config.slippage.market_order_volatility;
        let raw = min
            + volatility_component * self.config.slippage.volatility_multiplier
            + size_factor * self.config.slippage.size_multiplier * min;

        raw.clamp(min, max)
    }

    fn estimate_spread(&self) -> f64 {
        self.config
            .backtest_settings
            .as_ref()
            .map_or(0.0, |settings| {
                let volatility_spread =
                    settings.spread_volatility * settings.spread_volatility_factor;
                let widened_spread = if settings.simulate_market_microstructure {
                    settings.base_spread * settings.spread_widening_factor
                } else {
                    settings.base_spread
                };
                (widened_spread + volatility_spread).max(0.0)
            })
    }

    fn estimate_latency_seconds(&self, quantity: f64) -> f64 {
        self.config
            .backtest_settings
            .as_ref()
            .map_or(0.0, |settings| {
                if !settings.simulate_order_delays && !settings.account_for_latency_in_signals {
                    return 0.0;
                }

                let base_latency = settings.propagation_delay
                    + settings.transmission_delay
                    + settings.queuing_delay
                    + settings.processing_delay
                    + settings.exchange_processing_time
                    + settings.order_matching_latency
                    + settings.market_data_latency
                    + settings.signal_generation_time
                    + settings.order_construction_time
                    + settings.risk_check_time;
                let size_factor = (quantity / self.config.exchange.max_order_size).clamp(0.0, 1.0);
                let congestion = 1.0 + size_factor * (settings.congestion_latency_factor - 1.0);
                let spike = if settings.latency_spike_probability > 0.0 && size_factor > 0.8 {
                    settings.latency_spike_multiplier
                } else {
                    1.0
                };

                base_latency * congestion * spike
            })
    }

    fn estimate_latency_impact(&self, latency_seconds: f64) -> f64 {
        self.config
            .backtest_settings
            .as_ref()
            .map_or(0.0, |settings| {
                if latency_seconds <= 0.0 {
                    0.0
                } else {
                    (settings.market_impact_factor + self.config.slippage.market_order_volatility)
                        * latency_seconds
                }
            })
    }

    fn partial_fill_ratio(&self, quantity: f64, execution_price: f64) -> f64 {
        self.config
            .backtest_settings
            .as_ref()
            .map_or(1.0, |settings| {
                if !settings.simulate_partial_fills {
                    return 1.0;
                }

                let notional = quantity * execution_price;
                if notional > self.config.position_sizing.max_trade_notional {
                    settings.partial_fill_ratio.clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
    }

    fn round_down_to_step(&self, quantity: f64) -> f64 {
        let step = self.config.exchange.step_size;
        if step <= 0.0 {
            return quantity.max(0.0);
        }

        (quantity / step).floor() * step
    }
}
