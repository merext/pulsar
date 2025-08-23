use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    AccountStatusParams, OrderPlaceParams, OrderPlaceSideEnum, OrderPlaceTypeEnum, WebsocketApi,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use strategies::strategy::StrategyLogger;
use tracing::info;
use crate::config::BinanceTraderConfig;
use trade::config::TradeConfig;
use trade::logger::{StrategyLoggerAdapter, TradeLogger};
use trade::metrics::TradeManager;
use trade::signal::Signal;
use trade::trader::{TradeMode, Trader};

use futures_util::Stream;
use futures_util::StreamExt;

// Conversion functions to bridge trade and strategies types
fn convert_trade_to_trade_data(trade: &trade::models::Trade) -> strategies::models::TradeData {
    strategies::models::TradeData {
        price: trade.price,
        quantity: trade.quantity,
        timestamp: trade.trade_time as f64,
        symbol: trade.symbol.clone(),
    }
}

fn convert_metrics_position_to_strategies_position(
    position: &trade::metrics::Position,
) -> strategies::models::Position {
    strategies::models::Position {
        symbol: position.symbol.clone(),
        quantity: position.quantity,
        entry_price: position.entry_price,
    }
}

fn convert_strategies_signal_to_trade_signal(
    signal: strategies::models::Signal,
) -> trade::signal::Signal {
    match signal {
        strategies::models::Signal::Buy => trade::signal::Signal::Buy,
        strategies::models::Signal::Sell => trade::signal::Signal::Sell,
        strategies::models::Signal::Hold => trade::signal::Signal::Hold,
    }
}

pub struct BinanceTrader {
    connection: Option<WebsocketApi>,
    pub trade_manager: TradeManager,
    pub config: TradeConfig,
    pub trader_config: BinanceTraderConfig,
    api_key: String,
    api_secret: String,
    logger: TradeLogger,
    exchange_info: trade::trader::ExchangeInfo,
}

impl BinanceTrader {
    /// # Panics
    ///
    /// Panics if the Binance API configuration cannot be built or if the WebSocket API connection fails.
    ///
    /// # Errors
    ///
    /// Will return `Err` if the trading config cannot be loaded.
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = std::env::var("BINANCE_API_KEY")
            .map_err(|_| "BINANCE_API_KEY environment variable not set")?;
        let api_secret = std::env::var("BINANCE_API_SECRET")
            .map_err(|_| "BINANCE_API_SECRET environment variable not set")?;
        
        let trading_config = TradeConfig::load()?;
        let trader_config = BinanceTraderConfig::from_file("config/binance_exchange.toml")?;

        Ok(Self {
            connection: None, // Will be initialized in run_trading_loop if needed
            trade_manager: TradeManager::new(trader_config.general.trading_fee),
            config: trading_config,
            trader_config: trader_config.clone(),
            api_key,
            api_secret,
            logger: TradeLogger::new(trader_config.general.name.clone()),
            exchange_info: trade::trader::ExchangeInfo {
                name: trader_config.general.name.clone(),
                trading_fee: trader_config.general.trading_fee,
            },
        })
    }



    pub fn calculate_trade_size_impl(
        &self,
        _symbol: &str,
        price: f64,
        confidence: f64,
        trading_size_min: f64,
        trading_size_max: f64,
        trading_size_step: f64,
    ) -> f64 {
        // Enhanced dynamic sizing with multiple factors

        // Base size from linear confidence scaling
        let confidence_factor =
            confidence.mul_add(trading_size_max - trading_size_min, trading_size_min);

        // Cache configuration values to avoid repeated struct access
        let config = &self.trader_config.trading_behavior;
        let high_threshold = config.volatility_high_threshold;
        let medium_threshold = config.volatility_medium_threshold;
        let high_factor = config.volatility_high_factor;
        let medium_factor = config.volatility_medium_factor;
        let low_factor = config.volatility_low_factor;
        let confidence_high = config.confidence_high_threshold;
        let confidence_low = config.confidence_low_threshold;
        let confidence_high_factor = config.confidence_high_factor;
        let confidence_low_factor = config.confidence_low_factor;
        let kelly_factor = config.kelly_factor;

        // Volatility adjustment - reduce size in high volatility
        let volatility = self.estimate_volatility(price);
        let volatility_factor = if volatility > high_threshold {
            high_factor
        } else if volatility > medium_threshold {
            medium_factor
        } else {
            low_factor
        };

        // Risk adjustment based on confidence
        let risk_factor = if confidence > confidence_high {
            confidence_high_factor
        } else if confidence < confidence_low {
            confidence_low_factor
        } else {
            1.0
        };

        // Combine all factors
        let dynamic_quantity = confidence_factor * volatility_factor * risk_factor * kelly_factor;

        // Apply step size rounding (round down to nearest step)
        let quantity_to_trade = (dynamic_quantity / trading_size_step).floor() * trading_size_step;

        // Ensure we stay within the min/max bounds after rounding
        quantity_to_trade
            .max(trading_size_min)
            .min(trading_size_max)
    }

    fn estimate_volatility(&self, _current_price: f64) -> f64 {
        // Simple volatility estimation based on recent price movements
        // In a real implementation, this would track price history
        // For now, use the configured default volatility
        self.trader_config.trading_behavior.default_volatility
    }


}

#[async_trait]
impl Trader for BinanceTrader {
    fn get_info(&self) -> &trade::trader::ExchangeInfo {
        &self.exchange_info
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
        // Use TradeConfig for position sizing limits
        let min = self.config.position_sizing.trading_size_min;
        let max = self.config.position_sizing.trading_size_max;
        let step = self.config.exchange.step_size.max(self.trader_config.trading_behavior.step_size_fallback);
        self.calculate_trade_size_impl(symbol, price, confidence, min, max, step)
    }

    #[allow(clippy::too_many_lines)]
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64) {
        let symbol = self.trade_manager.get_current_trade_symbol();
        let quantity = Decimal::from_f64(quantity).expect("Failed to convert quantity to Decimal");

        let Some(connection) = &self.connection else {
            self.logger.log_order_error(
                &symbol,
                "on_signal",
                "connection_missing",
                "WebSocket connection not available"
            );
            return;
        };

        match signal {
            Signal::Buy => {
                if self
                    .trade_manager
                    .get_current_trade()
                    .map_or(0.0, |p| p.quantity)
                    == 0.0
                {
                    let params = OrderPlaceParams::builder(
                        symbol.clone(),
                        OrderPlaceSideEnum::Buy,
                        OrderPlaceTypeEnum::Market,
                    )
                    .quantity(quantity)
                    .build()
                    .expect("Failed to build order parameters");

                    let Ok(response) = connection.order_place(params).await else {
                        self.logger.log_order_error(
                            &symbol,
                            "place_buy_order",
                            "failed",
                            "Failed to place order"
                        );
                        return;
                    };
                    let Ok(data) = response.data() else {
                        self.logger.log_order_error(
                            &symbol,
                            "buy_order_data",
                            "failed",
                            "Failed to get order data"
                        );
                        return;
                    };

                    let Some(fills) = data.fills.as_ref() else {
                        self.logger.log_order_error(
                            &symbol,
                            "buy_order_fills",
                            "missing",
                            "No fills found in order response"
                        );
                        return;
                    };

                    let quantity = data
                        .executed_qty
                        .as_ref()
                        .expect("Executed quantity not found in buy order response")
                        .parse()
                        .expect("Failed to parse executed quantity");
                    let entry_price = fills
                        .first()
                        .expect("No fills found in buy order response")
                        .price
                        .as_ref()
                        .expect("Price not found in fill")
                        .parse()
                        .expect("Failed to parse entry price");

                    self.trade_manager
                        .open_position(&symbol, entry_price, quantity, 0.0);

                    // Buy order executed - position updated
                }
            }
            Signal::Sell => {
                if let Some(current_position) = self.trade_manager.get_current_trade()
                    && current_position.quantity > 0.0
                {
                    let symbol = current_position.symbol.clone();
                    let _pnl = self.trade_manager.close_position(&symbol, price, 0.0);
                    let params = OrderPlaceParams::builder(
                        symbol.clone(),
                        OrderPlaceSideEnum::Sell,
                        OrderPlaceTypeEnum::Market,
                    )
                    .quantity(quantity)
                    .build()
                    .expect("Failed to build order parameters");

                    if let Err(e) = connection.order_place(params).await {
                        self.logger.log_order_error(
                            &symbol,
                            "place_sell_order",
                            "failed",
                            &format!("Failed to place sell order: {}", e)
                        );
                        return;
                    }

                    // Sell order executed - position reset and PnL recorded
                    // Position is already closed by close_position call above
                }
            }
            Signal::Hold => {
                // Do nothing
            }
        }
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

            if let Some(uid) = &data.uid {
                info!(uid = %uid, "Account UID");
            }

            if let Some(can_trade) = data.can_trade {
                info!(can_trade = %can_trade, "Trading permissions");
            }

            if let Some(can_withdraw) = data.can_withdraw {
                info!(can_withdraw = %can_withdraw, "Withdrawal permissions");
            }

            if let Some(can_deposit) = data.can_deposit {
                info!(can_deposit = %can_deposit, "Deposit permissions");
            }

            if let Some(permissions) = &data.permissions {
                info!(permissions = ?permissions, "Account permissions");
            }

            if let Some(rates) = &data.commission_rates {
                info!(
                    maker_rate = %rates.maker.as_deref().unwrap_or("N/A"),
                    taker_rate = %rates.taker.as_deref().unwrap_or("N/A"),
                    buyer_rate = %rates.buyer.as_deref().unwrap_or("N/A"),
                    seller_rate = %rates.seller.as_deref().unwrap_or("N/A"),
                    "Commission rates"
                );
            }

            if let Some(balances) = &data.balances {
                info!(balance_count = balances.len(), "Account balances");
                for b in balances {
                    let asset = b.asset.as_deref().unwrap_or("-");
                    let free = b.free.as_deref().unwrap_or("0");
                    let locked = b.locked.as_deref().unwrap_or("0");
                    info!(
                        asset = %asset,
                        free_balance = %free,
                        locked_balance = %locked,
                        "Balance for asset"
                    );
                }
            }
        }

        Ok(())
    }

    async fn trade(
        &mut self,
        mut trading_stream: impl Stream<Item = trade::models::Trade> + Unpin + Send,
        trading_strategy: &mut dyn strategies::strategy::Strategy,
        trading_symbol: &str,
        trading_mode: TradeMode,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Initialize position symbol
        // Initialize position manager with the trading symbol
        // Position will be created when first trade occurs

        // Initialize connection for real and emulated trading modes
        if (trading_mode == TradeMode::Real || trading_mode == TradeMode::Emulated)
            && self.connection.is_none()
        {
            let config = ConfigurationWebsocketApi::builder()
                .api_key(&self.api_key)
                .api_secret(&self.api_secret)
                .build()
                .expect("Failed to build Binance API configuration");

            let client = SpotWsApi::production(config);
            self.connection = Some(
                tokio::time::timeout(
                    std::time::Duration::from_secs(self.trader_config.websocket.connection_timeout),
                    client.connect()
                )
                .await
                .expect("WebSocket connection timeout")
                .expect("Failed to connect to WebSocket API"),
            );
        }

        let mut current_position = self
            .trade_manager
            .get_current_trade()
            .cloned()
            .unwrap_or_else(|| trade::metrics::Position {
                symbol: trading_symbol.to_string(),
                quantity: 0.0,
                entry_price: 0.0,
                entry_time: 0.0,
            });

        while let Some(trade) = trading_stream.next().await {
            // Increment tick counter for each market data update
            self.trade_manager.increment_ticks();
            
            // Use the instance logger
            let strategy_logger = StrategyLoggerAdapter::new(&self.logger);

            // Update strategy with trade data
            trading_strategy
                .on_trade(convert_trade_to_trade_data(&trade))
                .await;

            let trade_price = trade.price;
            let trade_time = trade.trade_time as f64;

            // Get signal from strategy
            let (final_signal, confidence) = trading_strategy.get_signal(
                convert_metrics_position_to_strategies_position(&current_position),
            );

            // Log signal generated
            strategy_logger.log_signal_generated(trading_symbol, &final_signal, confidence, trade_price);

            // Calculate position size based on confidence and trading config
            let quantity_to_trade = self.calculate_trade_size(
                &current_position.symbol,
                trade_price,
                confidence,
                self.config.position_sizing.trading_size_min,
                self.config.position_sizing.trading_size_max,
                self.trader_config.trading_behavior.step_size_fallback,
            );

            match trading_mode {
                TradeMode::Real => {
                    self.on_signal(
                        convert_strategies_signal_to_trade_signal(final_signal),
                        trade_price,
                        quantity_to_trade,
                    )
                    .await;
                }
                TradeMode::Emulated | TradeMode::Backtest => {
                    // Simulate trade execution for emulated/backtest modes
                    match final_signal {
                        strategies::models::Signal::Buy => {
                            if current_position.quantity == 0.0 {
                                // Update position for emulated trading first
                                self.trade_manager.open_position(
                                    &current_position.symbol,
                                    trade_price,
                                    quantity_to_trade,
                                    trade_time,
                                );

                                // Log buy signal execution after position is updated
                                strategy_logger.log_trade_executed(
                                    trading_symbol,
                                    &final_signal,
                                    trade_price,
                                    quantity_to_trade,
                                    None,
                                    None,
                                    Some(self.trade_manager.get_trade_summary()),
                                );
                            }
                        }
                        strategies::models::Signal::Sell => {
                            if current_position.quantity > 0.0 {
                                let pnl = self.trade_manager.close_position(
                                    &current_position.symbol,
                                    trade_price,
                                    trade_time,
                                );

                                // Reset position after sell
                                self.trade_manager.update_position(
                                    &current_position.symbol,
                                    0.0,
                                    trade_price,
                                    trade_time,
                                );

                                // Log sell signal execution with profit after position is updated
                                let profit = self.trade_manager.get_metrics().realized_pnl();
                                strategy_logger.log_trade_executed(
                                    trading_symbol,
                                    &final_signal,
                                    trade_price,
                                    current_position.quantity,
                                    Some(pnl),
                                    Some(profit),
                                    Some(self.trade_manager.get_trade_summary()),
                                );
                            }
                        }
                        strategies::models::Signal::Hold => {
                            // No action needed for hold signals
                        }
                    }
                }
            }

            // Update current position for display purposes
            current_position = self
                .trade_manager
                .get_current_trade()
                .cloned()
                .unwrap_or_else(|| trade::metrics::Position {
                    symbol: trading_symbol.to_string(),
                    quantity: 0.0,
                    entry_price: 0.0,
                    entry_time: 0.0,
                });
        }

        Ok(())
    }
}
