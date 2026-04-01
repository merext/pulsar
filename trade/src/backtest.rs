use crate::config::TradeConfig;
use crate::execution::{ExecutionReport, ExecutionStatus, Side};
use crate::signal::Signal;
use crate::trader::OrderType;

#[derive(Debug, Clone, Copy)]
pub enum MarketPrice {
    Trade(f64),
    Quote { bid: f64, ask: f64 },
}

impl MarketPrice {
    pub fn decision_reference_price(self) -> f64 {
        match self {
            Self::Trade(price) => price,
            Self::Quote { bid, ask } => (bid + ask) / 2.0,
        }
    }

    pub fn execution_reference_price(self, signal: Signal) -> f64 {
        match (self, signal) {
            (Self::Trade(price), _) => price,
            (Self::Quote { ask, .. }, Signal::Buy) => ask,
            (Self::Quote { bid, .. }, Signal::Sell) => bid,
            (Self::Quote { bid, ask }, Signal::Hold) => (bid + ask) / 2.0,
        }
    }

    pub fn has_explicit_quote(self) -> bool {
        matches!(self, Self::Quote { .. })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimulatedExecution {
    pub execution_price: f64,
    pub executed_quantity: f64,
    pub fee_paid: f64,
    pub notional_value: f64,
    pub latency_seconds: f64,
    pub synthetic_half_spread_rate: f64,
    pub slippage_rate: f64,
    pub latency_impact_rate: f64,
    pub market_impact_rate: f64,
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

    pub fn total_price_offset_rate(&self) -> f64 {
        self.synthetic_half_spread_rate
            + self.slippage_rate
            + self.latency_impact_rate
            + self.market_impact_rate
    }
}

#[derive(Debug, Clone)]
pub struct BacktestEngine {
    config: TradeConfig,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueEstimate {
    pub fill_probability: f64,
    pub expected_fill_ratio: f64,
    pub queue_ahead_quantity: f64,
}

impl BacktestEngine {
    pub fn new(config: TradeConfig) -> Self {
        Self { config }
    }

    pub fn execute(&self, signal: Signal, market_price: f64, quantity: f64) -> SimulatedExecution {
        self.execute_with_constraints(signal, market_price, quantity, f64::INFINITY)
    }

    pub fn estimate_passive_fill(
        &self,
        side: Side,
        market_price: MarketPrice,
        requested_quantity: f64,
    ) -> QueueEstimate {
        let quantity = requested_quantity.max(0.0);
        let queue_ahead_quantity = match market_price {
            MarketPrice::Quote { bid, ask } => {
                let reference = ((bid + ask) / 2.0).max(f64::EPSILON);
                quantity * (1.0 + ((ask - bid).abs() / reference * 10_000.0).min(20.0) / 10.0)
            }
            MarketPrice::Trade(_) => quantity * 2.0,
        };

        let settings = self.config.backtest_settings.as_ref();
        let base_fill_rate = settings.map_or(0.2, |cfg| cfg.limit_order_fill_rate.clamp(0.0, 1.0));
        let hft_penalty = settings.map_or(0.0, |cfg| cfg.hft_latency_advantage.max(0.0));
        let size_factor = (quantity / self.config.exchange.max_order_size).clamp(0.0, 1.0);
        let side_bias = match side {
            Side::Buy => 1.0,
            Side::Sell => 0.98,
        };
        let fill_probability =
            (base_fill_rate * side_bias * (1.0 - hft_penalty) * (1.0 - size_factor))
                .clamp(0.0, 1.0);
        let expected_fill_ratio = (fill_probability * (1.0 - size_factor * 0.5)).clamp(0.0, 1.0);

        QueueEstimate {
            fill_probability,
            expected_fill_ratio,
            queue_ahead_quantity,
        }
    }

    pub fn simulate_passive_order(
        &self,
        side: Side,
        market_price: MarketPrice,
        requested_quantity: f64,
        expected_edge_bps: f64,
    ) -> ExecutionReport {
        // Passive (maker) orders rest on the near side of the book:
        // - A passive Buy rests at the bid → fills at bid price
        // - A passive Sell rests at the ask → fills at ask price
        // This is the opposite of taker orders which cross the spread.
        let reference_price = market_price.execution_reference_price(match side {
            Side::Buy => Signal::Sell, // bid price (passive buy rests at bid)
            Side::Sell => Signal::Buy, // ask price (passive sell rests at ask)
        });
        let queue = self.estimate_passive_fill(side, market_price, requested_quantity);
        let executed_quantity =
            self.round_down_to_step(requested_quantity * queue.expected_fill_ratio);

        if executed_quantity <= 0.0 || queue.fill_probability <= 0.0 {
            return ExecutionReport {
                status: ExecutionStatus::Pending,
                symbol: None,
                side: Some(side),
                order_type: Some(OrderType::Maker),
                rationale: None,
                decision_confidence: 0.0,
                decision_metrics: Vec::new(),
                requested_quantity,
                executed_quantity: 0.0,
                execution_price: Some(reference_price),
                fee_paid: 0.0,
                latency_seconds: 0.0,
                synthetic_half_spread_bps: 0.0,
                slippage_bps: 0.0,
                latency_impact_bps: 0.0,
                market_impact_bps: 0.0,
                reason: Some("queue_not_filled"),
                expected_edge_bps,
            };
        }

        let fee_rate = self.config.exchange.maker_fee - self.config.exchange.maker_rebate;
        let fee_paid = reference_price * executed_quantity * fee_rate.max(0.0);
        ExecutionReport {
            status: if executed_quantity < requested_quantity {
                ExecutionStatus::PartiallyFilled
            } else {
                ExecutionStatus::Filled
            },
            symbol: None,
            side: Some(side),
            order_type: Some(OrderType::Maker),
            rationale: None,
            decision_confidence: 0.0,
            decision_metrics: Vec::new(),
            requested_quantity,
            executed_quantity,
            execution_price: Some(reference_price),
            fee_paid,
            latency_seconds: 0.0,
            synthetic_half_spread_bps: 0.0,
            slippage_bps: 0.0,
            latency_impact_bps: 0.0,
            market_impact_bps: 0.0,
            reason: if executed_quantity < requested_quantity {
                Some("partial_queue_fill")
            } else {
                None
            },
            expected_edge_bps,
        }
    }

    pub fn execute_with_constraints(
        &self,
        signal: Signal,
        market_price: f64,
        requested_quantity: f64,
        available_cash: f64,
    ) -> SimulatedExecution {
        self.execute_with_constraints_at(
            signal,
            MarketPrice::Trade(market_price),
            requested_quantity,
            available_cash,
        )
    }

    pub fn execute_with_constraints_at(
        &self,
        signal: Signal,
        market_price: MarketPrice,
        requested_quantity: f64,
        available_cash: f64,
    ) -> SimulatedExecution {
        if matches!(signal, Signal::Hold) {
            let reference_price = market_price.execution_reference_price(signal);
            return SimulatedExecution {
                execution_price: reference_price,
                executed_quantity: 0.0,
                fee_paid: 0.0,
                notional_value: 0.0,
                latency_seconds: 0.0,
                synthetic_half_spread_rate: 0.0,
                slippage_rate: 0.0,
                latency_impact_rate: 0.0,
                market_impact_rate: 0.0,
                rejected_reason: None,
                partially_filled: false,
            };
        }

        if requested_quantity <= 0.0 {
            let reference_price = market_price.execution_reference_price(signal);
            return SimulatedExecution {
                execution_price: reference_price,
                executed_quantity: 0.0,
                fee_paid: 0.0,
                notional_value: 0.0,
                latency_seconds: 0.0,
                synthetic_half_spread_rate: 0.0,
                slippage_rate: 0.0,
                latency_impact_rate: 0.0,
                market_impact_rate: 0.0,
                rejected_reason: Some("quantity_below_step"),
                partially_filled: false,
            };
        }

        let latency_seconds = self.estimate_latency_seconds(requested_quantity);
        let spread = if market_price.has_explicit_quote() {
            0.0
        } else {
            self.estimate_spread()
        };
        let slippage = self.estimate_slippage(signal, requested_quantity);
        let latency_impact = self.estimate_latency_impact(latency_seconds);
        let market_impact = self.estimate_market_impact(requested_quantity);
        let half_spread = spread / 2.0;
        let reference_price = market_price.execution_reference_price(signal);
        let execution_price = match signal {
            Signal::Buy => {
                reference_price * (1.0 + half_spread + slippage + latency_impact + market_impact)
            }
            Signal::Sell => {
                reference_price * (1.0 - half_spread - slippage - latency_impact - market_impact)
            }
            Signal::Hold => reference_price,
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
                synthetic_half_spread_rate: half_spread,
                slippage_rate: slippage,
                latency_impact_rate: latency_impact,
                market_impact_rate: market_impact,
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
                synthetic_half_spread_rate: half_spread,
                slippage_rate: slippage,
                latency_impact_rate: latency_impact,
                market_impact_rate: market_impact,
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
            synthetic_half_spread_rate: half_spread,
            slippage_rate: slippage,
            latency_impact_rate: latency_impact,
            market_impact_rate: market_impact,
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
        let volatility_component = self
            .config
            .slippage
            .market_order_volatility
            .max(self.config.slippage.taker_order_volatility);
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
            .map_or(0.0, |_settings| {
                if latency_seconds <= 0.0 {
                    0.0
                } else {
                    self.config
                        .slippage
                        .market_order_volatility
                        .max(self.config.slippage.taker_order_volatility)
                        * latency_seconds
                }
            })
    }

    fn estimate_market_impact(&self, quantity: f64) -> f64 {
        self.config
            .backtest_settings
            .as_ref()
            .map_or(0.0, |settings| {
                if !settings.simulate_market_impact {
                    return 0.0;
                }

                let size_factor = (quantity / self.config.exchange.max_order_size).clamp(0.0, 1.0);
                if size_factor <= 0.0 {
                    return 0.0;
                }

                let decay_dampener = 1.0 / (1.0 + settings.impact_decay_rate.max(0.0));
                settings.market_impact_factor * size_factor * decay_dampener
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
