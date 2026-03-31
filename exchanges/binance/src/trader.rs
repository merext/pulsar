use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    AccountStatusParams, OrderPlaceParams, OrderPlaceSideEnum, OrderPlaceTypeEnum, WebsocketApi,
};
use futures_util::{Stream, StreamExt};
use rust_decimal::prelude::FromPrimitive;
use rust_decimal::Decimal;
use tracing::info;

use crate::config::BinanceTraderConfig;
use trade::backtest::BacktestEngine;
use trade::config::TradeConfig;
use trade::execution::{ExecutionReport, ExecutionStatus, OrderIntent, Side, TimeInForce};
use trade::logger::{StrategyLoggerAdapter, TradeLogger};
use trade::market::{MarketEvent, MarketState};
use trade::metrics::TradeManager;
use trade::models::Position;
use trade::simulation::{PositionSizer, VolatilityFactors};
use trade::strategy::{Strategy, StrategyContext, StrategyLogger};
use trade::trader::{OrderType, TradeMode, Trader};

pub struct BinanceTrader {
    connection: Option<WebsocketApi>,
    pub trade_manager: TradeManager,
    pub config: TradeConfig,
    pub trader_config: BinanceTraderConfig,
    api_key: Option<String>,
    api_secret: Option<String>,
    logger: TradeLogger,
    exchange_info: trade::trader::ExchangeInfo,
    position_sizer: PositionSizer,
}

impl BinanceTrader {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = std::env::var("BINANCE_API_KEY").ok();
        let api_secret = std::env::var("BINANCE_API_SECRET").ok();

        let trading_config = TradeConfig::load()?;
        let trader_config = BinanceTraderConfig::from_file("config/binance_exchange.toml")?;
        let effective_fee = trading_config.exchange.taker_fee;
        let initial_capital = trading_config
            .backtest_settings
            .as_ref()
            .map_or(0.0, |settings| settings.initial_capital);

        Ok(Self {
            connection: None,
            trade_manager: TradeManager::new(effective_fee, initial_capital),
            config: trading_config.clone(),
            trader_config: trader_config.clone(),
            api_key,
            api_secret,
            logger: TradeLogger::new(trader_config.general.name.clone()),
            exchange_info: trade::trader::ExchangeInfo {
                name: trader_config.general.name.clone(),
                trading_fee: effective_fee,
            },
            position_sizer: PositionSizer::new(trading_config),
        })
    }

    fn credentials(&self) -> Result<(&str, &str), Box<dyn std::error::Error + Send + Sync>> {
        let api_key = self
            .api_key
            .as_deref()
            .ok_or("BINANCE_API_KEY environment variable not set")?;
        let api_secret = self
            .api_secret
            .as_deref()
            .ok_or("BINANCE_API_SECRET environment variable not set")?;

        Ok((api_key, api_secret))
    }

    fn estimate_volatility(&self, _current_price: f64) -> f64 {
        self.trader_config.trading_behavior.default_volatility
    }

    fn current_position(&self, symbol: &str) -> Position {
        self.trade_manager
            .get_position(symbol)
            .cloned()
            .unwrap_or_else(|| Position {
                symbol: symbol.to_string(),
                quantity: 0.0,
                entry_price: 0.0,
                entry_time: 0.0,
            })
    }

    fn validate_entry_trade(&self, symbol: &str, quantity: f64, price: f64) -> bool {
        let notional = quantity * price;
        let max_position = self.config.position_sizing.max_position_notional;
        let current_position = self
            .trade_manager
            .get_position(symbol)
            .map(|p| p.quantity * p.entry_price)
            .unwrap_or(0.0);
        let available_cash = self.trade_manager.available_cash();
        let estimated_fee = notional * self.config.exchange.taker_fee;

        if notional > max_position || (current_position + notional) > max_position {
            self.logger.log_order_error(
                symbol,
                "position_size_limit",
                "exceeded",
                &format!("Trade size {} exceeds max position {}", notional, max_position),
            );
            return false;
        }

        if notional + estimated_fee > available_cash {
            self.logger.log_order_error(
                symbol,
                "insufficient_cash",
                "rejected",
                &format!(
                    "Trade cost {} exceeds available cash {}",
                    notional + estimated_fee,
                    available_cash
                ),
            );
            return false;
        }

        if notional < self.config.exchange.min_notional {
            self.logger.log_order_error(
                symbol,
                "min_notional",
                "below_minimum",
                &format!(
                    "Trade size {} below minimum {}",
                    notional, self.config.exchange.min_notional
                ),
            );
            return false;
        }

        true
    }

    fn validate_exit_trade(&self, symbol: &str, quantity: f64, price: f64) -> bool {
        if quantity <= 0.0 {
            self.logger.log_order_error(
                symbol,
                "exit_validation",
                "missing_position",
                "No open quantity available to close",
            );
            return false;
        }

        let notional = quantity * price;
        if notional < self.config.exchange.min_notional {
            self.logger.log_order_error(
                symbol,
                "min_notional",
                "below_minimum",
                &format!(
                    "Exit size {} below minimum {}",
                    notional, self.config.exchange.min_notional
                ),
            );
            return false;
        }

        true
    }

    fn simulated_report_from_buy(
        &self,
        requested_quantity: f64,
        expected_edge_bps: f64,
        execution: trade::backtest::SimulatedExecution,
    ) -> ExecutionReport {
        ExecutionReport {
            status: if execution.partially_filled {
                ExecutionStatus::PartiallyFilled
            } else {
                ExecutionStatus::Filled
            },
            side: Some(Side::Buy),
            order_type: Some(OrderType::Taker),
            requested_quantity,
            executed_quantity: execution.executed_quantity,
            execution_price: Some(execution.execution_price),
            fee_paid: execution.fee_paid,
            latency_seconds: execution.latency_seconds,
            reason: execution.rejected_reason,
            expected_edge_bps,
        }
    }

    fn simulated_report_from_sell(
        &self,
        requested_quantity: f64,
        expected_edge_bps: f64,
        execution: trade::backtest::SimulatedExecution,
    ) -> ExecutionReport {
        ExecutionReport {
            status: if execution.partially_filled {
                ExecutionStatus::PartiallyFilled
            } else {
                ExecutionStatus::Filled
            },
            side: Some(Side::Sell),
            order_type: Some(OrderType::Taker),
            requested_quantity,
            executed_quantity: execution.executed_quantity,
            execution_price: Some(execution.execution_price),
            fee_paid: execution.fee_paid,
            latency_seconds: execution.latency_seconds,
            reason: execution.rejected_reason,
            expected_edge_bps,
        }
    }
}

#[async_trait]
impl Trader for BinanceTrader {
    fn get_info(&self) -> &trade::trader::ExchangeInfo {
        &self.exchange_info
    }

    fn get_metrics(&self) -> &trade::metrics::PerformanceMetrics {
        self.trade_manager.get_metrics()
    }

    fn get_trade_manager(&self) -> &trade::metrics::TradeManager {
        &self.trade_manager
    }

    async fn account_status(&self) -> Result<(), anyhow::Error> {
        if let Some(connection) = &self.connection {
            let params = AccountStatusParams::builder()
                .omit_zero_balances(true)
                .build()?;
            let status = connection.account_status(params).await?;
            let data = status.data().unwrap();

            if let Some(account_type) = &data.account_type {
                info!(account_type = %account_type, "Account status retrieved");
            }
        }

        Ok(())
    }

    async fn on_order_intent(
        &mut self,
        symbol: &str,
        reference_price: f64,
        intent: OrderIntent,
    ) -> ExecutionReport {
        let Some(connection) = &self.connection else {
            return ExecutionReport {
                status: ExecutionStatus::Rejected,
                side: None,
                order_type: None,
                requested_quantity: 0.0,
                executed_quantity: 0.0,
                execution_price: None,
                fee_paid: 0.0,
                latency_seconds: 0.0,
                reason: Some("connection_missing"),
                expected_edge_bps: 0.0,
            };
        };

        match intent {
            OrderIntent::NoAction => ExecutionReport::ignored(),
            OrderIntent::Cancel { rationale } => ExecutionReport {
                status: ExecutionStatus::Cancelled,
                side: None,
                order_type: None,
                requested_quantity: 0.0,
                executed_quantity: 0.0,
                execution_price: None,
                fee_paid: 0.0,
                latency_seconds: 0.0,
                reason: Some(rationale),
                expected_edge_bps: 0.0,
            },
            OrderIntent::Place {
                side,
                order_type,
                quantity,
                expected_edge_bps,
                ..
            } => {
                let valid = match side {
                    Side::Buy => self.validate_entry_trade(symbol, quantity, reference_price),
                    Side::Sell => self.validate_exit_trade(symbol, quantity, reference_price),
                };

                if !valid {
                    return ExecutionReport {
                        status: ExecutionStatus::Rejected,
                        side: Some(side),
                        order_type: Some(order_type),
                        requested_quantity: quantity,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        reason: Some("risk_validation_failed"),
                        expected_edge_bps,
                    };
                }

                let quantity_decimal = match Decimal::from_f64(quantity) {
                    Some(quantity) => quantity,
                    None => {
                        return ExecutionReport {
                            status: ExecutionStatus::Rejected,
                            side: Some(side),
                            order_type: Some(order_type),
                            requested_quantity: quantity,
                            executed_quantity: 0.0,
                            execution_price: None,
                            fee_paid: 0.0,
                            latency_seconds: 0.0,
                            reason: Some("invalid_quantity"),
                            expected_edge_bps,
                        };
                    }
                };

                let params = OrderPlaceParams::builder(
                    symbol.to_string(),
                    match side {
                        Side::Buy => OrderPlaceSideEnum::Buy,
                        Side::Sell => OrderPlaceSideEnum::Sell,
                    },
                    OrderPlaceTypeEnum::Market,
                )
                .quantity(quantity_decimal)
                .build()
                .expect("Failed to build order parameters");

                let Ok(response) = connection.order_place(params).await else {
                    return ExecutionReport {
                        status: ExecutionStatus::Rejected,
                        side: Some(side),
                        order_type: Some(order_type),
                        requested_quantity: quantity,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        reason: Some("exchange_order_failed"),
                        expected_edge_bps,
                    };
                };

                let Ok(data) = response.data() else {
                    return ExecutionReport {
                        status: ExecutionStatus::Rejected,
                        side: Some(side),
                        order_type: Some(order_type),
                        requested_quantity: quantity,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        reason: Some("missing_order_data"),
                        expected_edge_bps,
                    };
                };

                let Some(fills) = data.fills.as_ref() else {
                    return ExecutionReport {
                        status: ExecutionStatus::Rejected,
                        side: Some(side),
                        order_type: Some(order_type),
                        requested_quantity: quantity,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        reason: Some("missing_fills"),
                        expected_edge_bps,
                    };
                };

                let executed_quantity = data
                    .executed_qty
                    .as_ref()
                    .and_then(|value| value.parse::<f64>().ok())
                    .unwrap_or(0.0);
                let execution_price = fills
                    .first()
                    .and_then(|fill| fill.price.as_ref())
                    .and_then(|value| value.parse::<f64>().ok());
                let fee_paid = execution_price
                    .map(|price| price * executed_quantity * self.config.exchange.taker_fee)
                    .unwrap_or(0.0);

                ExecutionReport {
                    status: ExecutionStatus::Filled,
                    side: Some(side),
                    order_type: Some(order_type),
                    requested_quantity: quantity,
                    executed_quantity,
                    execution_price,
                    fee_paid,
                    latency_seconds: 0.0,
                    reason: None,
                    expected_edge_bps,
                }
            }
        }
    }

    fn calculate_trade_size(
        &self,
        symbol: &str,
        price: f64,
        confidence: f64,
        _trading_size_min: f64,
        _trading_size_max: f64,
        _trading_size_step: f64,
    ) -> f64 {
        let config = &self.trader_config.trading_behavior;
        let volatility = self.estimate_volatility(price);
        let step = self
            .config
            .exchange
            .step_size
            .max(self.trader_config.trading_behavior.step_size_fallback);
        let current_position_notional = self
            .trade_manager
            .get_position(symbol)
            .map_or(0.0, |position| position.quantity * price);

        self.position_sizer.size_order(
            price,
            confidence,
            self.trade_manager.available_cash(),
            current_position_notional,
            volatility,
            step,
            VolatilityFactors {
                high_threshold: config.volatility_high_threshold,
                medium_threshold: config.volatility_medium_threshold,
                high_factor: config.volatility_high_factor,
                medium_factor: config.volatility_medium_factor,
                low_factor: config.volatility_low_factor,
                confidence_high_threshold: config.confidence_high_threshold,
                confidence_low_threshold: config.confidence_low_threshold,
                confidence_high_factor: config.confidence_high_factor,
                confidence_low_factor: config.confidence_low_factor,
                kelly_factor: config.kelly_factor,
            },
        )
    }

    async fn trade(
        &mut self,
        trading_stream: impl Stream<Item = MarketEvent> + Send,
        trading_strategy: &mut dyn Strategy,
        trading_symbol: &str,
        trading_mode: TradeMode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if trading_mode == TradeMode::Real && self.connection.is_none() {
            let (api_key, api_secret) = self.credentials()?;
            let config = ConfigurationWebsocketApi::builder()
                .api_key(api_key)
                .api_secret(api_secret)
                .build()
                .expect("Failed to build Binance API configuration");

            let client = SpotWsApi::production(config);
            self.connection = Some(
                tokio::time::timeout(
                    std::time::Duration::from_secs(self.trader_config.websocket.connection_timeout),
                    client.connect(),
                )
                .await
                .expect("WebSocket connection timeout")
                .expect("Failed to connect to WebSocket API"),
            );
        }

        tokio::pin!(trading_stream);

        let backtest_engine = BacktestEngine::new(self.config.clone());
        let mut market_state = MarketState::new(trading_symbol.to_string(), 1_000);
        let mut last_trade_price = None;
        let mut last_trade_time = None;

        while let Some(event) = trading_stream.next().await {
            self.trade_manager.increment_ticks();
            market_state.apply(&event);
            trading_strategy.on_event(&event, &market_state).await;

            let Some(reference_price) = market_state
                .mid_price()
                .or_else(|| market_state.last_price())
            else {
                continue;
            };

            if let Some(last_trade) = market_state.last_trade() {
                last_trade_price = Some(last_trade.price);
                last_trade_time = Some(last_trade.timestamp_secs());
                let _ = self.trade_manager.mark_to_market(trading_symbol, last_trade.price);
            }

            self.logger.log_market_state_snapshot(
                trading_symbol,
                market_state.last_price(),
                market_state.mid_price(),
                market_state.spread_bps(),
                market_state.trade_flow_imbalance(),
                market_state.order_book_imbalance(),
            );

            let current_position = self.current_position(trading_symbol);
            let decision = trading_strategy.decide(
                &market_state,
                &StrategyContext {
                    symbol: trading_symbol.to_string(),
                    current_position: current_position.clone(),
                    available_cash: self.trade_manager.available_cash(),
                    max_position_notional: self.config.position_sizing.max_position_notional,
                },
            );

            {
                let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                strategy_logger.log_decision(
                    trading_symbol,
                    &decision.intent,
                    decision.confidence,
                    reference_price,
                );
            }

            match trading_mode {
                TradeMode::Real => {
                    let report = self
                        .on_order_intent(trading_symbol, reference_price, decision.intent)
                        .await;

                    let pnl = if report.side == Some(Side::Sell) {
                        report.execution_price.map(|price| {
                            self.trade_manager
                                .close_position(trading_symbol, price, last_trade_time.unwrap_or(0.0))
                        })
                    } else if report.side == Some(Side::Buy)
                        && report.executed_quantity > 0.0
                    {
                        if let Some(price) = report.execution_price {
                            let _ = self.trade_manager.open_position(
                                trading_symbol,
                                price,
                                report.executed_quantity,
                                last_trade_time.unwrap_or(0.0),
                            );
                        }
                        None
                    } else {
                        None
                    };

                    {
                        let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                        strategy_logger.log_execution(
                            trading_symbol,
                            &report,
                            pnl,
                            Some(self.trade_manager.get_metrics().realized_pnl()),
                            Some(self.trade_manager.get_trade_summary()),
                        );
                    }
                }
                TradeMode::Emulated | TradeMode::Backtest => {
                    let report = match decision.intent {
                        OrderIntent::NoAction | OrderIntent::Cancel { .. } => ExecutionReport::ignored(),
                        OrderIntent::Place {
                            side,
                            order_type,
                            quantity,
                            expected_edge_bps,
                            time_in_force,
                            ..
                        } => {
                            let requested_quantity = match side {
                                Side::Buy => quantity,
                                Side::Sell => current_position.quantity.max(quantity),
                            };

                            let signal = match side {
                                Side::Buy => trade::signal::Signal::Buy,
                                Side::Sell => trade::signal::Signal::Sell,
                            };

                            let execution = backtest_engine.execute_with_constraints(
                                signal,
                                reference_price,
                                requested_quantity,
                                self.trade_manager.available_cash(),
                            );

                            if execution.is_rejected() {
                                ExecutionReport {
                                    status: ExecutionStatus::Rejected,
                                    side: Some(side),
                                    order_type: Some(order_type),
                                    requested_quantity,
                                    executed_quantity: 0.0,
                                    execution_price: Some(execution.execution_price),
                                    fee_paid: 0.0,
                                    latency_seconds: execution.latency_seconds,
                                    reason: execution.rejected_reason,
                                    expected_edge_bps,
                                }
                            } else {
                                match (side, time_in_force) {
                                    (Side::Buy, _) => self.simulated_report_from_buy(
                                        requested_quantity,
                                        expected_edge_bps,
                                        execution,
                                    ),
                                    (Side::Sell, TimeInForce::Gtc | TimeInForce::Ioc | TimeInForce::Fok) => {
                                        self.simulated_report_from_sell(
                                            requested_quantity,
                                            expected_edge_bps,
                                            execution,
                                        )
                                    }
                                }
                            }
                        }
                    };

                    let pnl = match report.side {
                        Some(Side::Buy) if report.executed_quantity > 0.0 => {
                            if let Some(price) = report.execution_price {
                                if let Err(reason) = self.trade_manager.open_position(
                                    trading_symbol,
                                    price,
                                    report.executed_quantity,
                                    last_trade_time.unwrap_or(0.0),
                                ) {
                                    self.logger.log_order_error(
                                        trading_symbol,
                                        "simulated_buy",
                                        "failed",
                                        reason,
                                    );
                                }
                            }
                            None
                        }
                        Some(Side::Sell) if report.executed_quantity > 0.0 => report.execution_price.map(|price| {
                            self.trade_manager
                                .close_position(trading_symbol, price, last_trade_time.unwrap_or(0.0))
                        }),
                        _ => None,
                    };

                    {
                        let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                        strategy_logger.log_execution(
                            trading_symbol,
                            &report,
                            pnl,
                            Some(self.trade_manager.get_metrics().realized_pnl()),
                            Some(self.trade_manager.get_trade_summary()),
                        );
                    }
                }
            }
        }

        if matches!(trading_mode, TradeMode::Backtest | TradeMode::Emulated)
            && self.current_position(trading_symbol).quantity > 0.0
            && let (Some(last_price), Some(last_time)) = (last_trade_price, last_trade_time)
        {
            let position = self.current_position(trading_symbol);
            let execution = backtest_engine.execute_with_constraints(
                trade::signal::Signal::Sell,
                last_price,
                position.quantity,
                self.trade_manager.available_cash(),
            );
            if !execution.is_rejected() {
                let report = self.simulated_report_from_sell(position.quantity, 0.0, execution);
                let pnl = self
                    .trade_manager
                    .close_position(trading_symbol, execution.execution_price, last_time);
                let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                strategy_logger.log_execution(
                    trading_symbol,
                    &report,
                    Some(pnl),
                    Some(self.trade_manager.get_metrics().realized_pnl()),
                    Some(self.trade_manager.get_trade_summary()),
                );
            }
        }

        let metrics = self.trade_manager.get_metrics();
        self.logger.log_session_summary(
            trading_symbol,
            self.trade_manager.get_total_ticks(),
            metrics.entry_trades(),
            metrics.closed_trades(),
            metrics.realized_pnl(),
            metrics.fees_paid(),
            metrics.current_cash(),
            metrics.last_equity(),
            metrics.win_rate(),
            metrics.profit_factor(),
            metrics.avg_pnl_per_closed_trade(),
            metrics.max_drawdown(),
        );

        Ok(())
    }
}
