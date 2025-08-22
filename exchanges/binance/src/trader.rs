use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    AccountStatusParams, OrderPlaceParams, OrderPlaceSideEnum, OrderPlaceTypeEnum, WebsocketApi,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use strategies::strategy::StrategyLogger;
use tracing::{error, info};
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
    api_key: String,
    api_secret: String,
    logger: TradeLogger,
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

        Ok(Self {
            connection: None, // Will be initialized in run_trading_loop if needed
            trade_manager: TradeManager::new(),
            config: trading_config,
            api_key,
            api_secret,
            logger: TradeLogger::new(),
        })
    }

    /// # Panics
    ///
    /// Panics if the Binance API configuration cannot be built or if the WebSocket API connection fails.
    ///
    /// # Errors
    ///
    /// Will return `Err` if the trading config cannot be loaded from the provided path.
    pub async fn new_with_config(
        api_key: &str,
        api_secret: &str,
        config_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let trading_config = TradeConfig::from_file(config_path)?;

        Ok(Self {
            connection: None, // Will be initialized in run_trading_loop if needed
            trade_manager: TradeManager::new(),
            config: trading_config,
            api_key: api_key.to_string(),
            api_secret: api_secret.to_string(),
            logger: TradeLogger::new(),
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

        // Volatility adjustment - reduce size in high volatility
        let volatility = Self::estimate_volatility(price);
        let volatility_factor = if volatility > 0.02 {
            // If volatility > 2%
            0.7 // Reduce size by 30%
        } else if volatility > 0.01 {
            // If volatility > 1%
            0.85 // Reduce size by 15%
        } else {
            1.0 // No adjustment for low volatility
        };

        // Risk adjustment based on confidence
        let risk_factor = if confidence > 0.7 {
            1.2 // Increase size for high confidence
        } else if confidence < 0.3 {
            0.8 // Reduce size for low confidence
        } else {
            1.0
        };

        // Kelly Criterion-inspired sizing (simplified)
        let kelly_factor = Self::estimate_kelly_fraction();

        // Combine all factors
        let dynamic_quantity = confidence_factor * volatility_factor * risk_factor * kelly_factor;

        // Apply step size rounding (round down to nearest step)
        let quantity_to_trade = (dynamic_quantity / trading_size_step).floor() * trading_size_step;

        // Ensure we stay within the min/max bounds after rounding
        quantity_to_trade
            .max(trading_size_min)
            .min(trading_size_max)
    }

    const fn estimate_volatility(_current_price: f64) -> f64 {
        // Simple volatility estimation based on recent price movements
        // In a real implementation, this would track price history
        // For now, use a conservative estimate
        0.015 // 1.5% default volatility
    }

    const fn estimate_kelly_fraction() -> f64 {
        // Kelly Criterion estimation
        // In practice, this would be calculated from historical performance
        // For now, use a conservative multiplier
        0.8 // Reduce position size to 80% of base calculation
    }
}

#[async_trait]
impl Trader for BinanceTrader {
    fn calculate_trade_size(
        &self,
        symbol: &str,
        price: f64,
        confidence: f64,
        _trading_size_min: f64,
        _trading_size_max: f64,
        _trading_size_step: f64,
    ) -> f64 {
        // Read limits from central TradingConfig rather than caller
        let min = self.config.position_sizing.trading_size_min;
        let max = self.config.position_sizing.trading_size_max;
        let step = self.config.exchange.step_size.max(1.0); // default to 1 unit step for spot quantities
        self.calculate_trade_size_impl(symbol, price, confidence, min, max, step)
    }

    #[allow(clippy::too_many_lines)]
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64) {
        let symbol = self.trade_manager.get_current_trade_symbol();
        let quantity = Decimal::from_f64(quantity).expect("Failed to convert quantity to Decimal");

        let Some(connection) = &self.connection else {
            error!(
                exchange = "binance",
                action = "on_signal",
                status = "connection_missing",
                symbol = %symbol
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
                        error!(
                            exchange = "binance",
                            action = "place_buy_order",
                            status = "failed",
                            symbol = %symbol,
                            quantity = %quantity,
                            error = "Failed to place order"
                        );
                        return;
                    };
                    let Ok(data) = response.data() else {
                        error!(
                            exchange = "binance",
                            action = "buy_order_data",
                            status = "failed",
                            symbol = %symbol,
                            error = "Failed to get order data"
                        );
                        return;
                    };

                    let Some(fills) = data.fills.as_ref() else {
                        error!(
                            exchange = "binance",
                            action = "buy_order_fills",
                            status = "missing",
                            symbol = %symbol
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
                        error!(
                            exchange = "binance",
                            action = "place_sell_order",
                            status = "failed",
                            symbol = %symbol,
                            quantity = %quantity,
                            error = %e
                        );
                        std::process::exit(1);
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
                client
                    .connect()
                    .await
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
                1.0, // trading_size_step - use 1.0 for DOGEUSDT
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
                                // Log buy signal execution
                                strategy_logger.log_trade_executed(
                                    trading_symbol,
                                    &final_signal,
                                    trade_price,
                                    quantity_to_trade,
                                    None,
                                    None,
                                );

                                // Update position for emulated trading

                                self.trade_manager.open_position(
                                    &current_position.symbol,
                                    trade_price,
                                    quantity_to_trade,
                                    trade_time,
                                );
                                // Position is now managed by trade_manager

                                // Position change is already captured in buy_executed log
                            }
                        }
                        strategies::models::Signal::Sell => {
                            if current_position.quantity > 0.0 {
                                let pnl = self.trade_manager.close_position(
                                    &current_position.symbol,
                                    trade_price,
                                    trade_time,
                                );

                                // Log sell signal execution with profit
                                let profit = self.trade_manager.get_metrics().realized_pnl();
                                strategy_logger.log_trade_executed(
                                    trading_symbol,
                                    &final_signal,
                                    trade_price,
                                    current_position.quantity,
                                    Some(pnl),
                                    Some(profit),
                                );

                                // Reset position after sell

                                self.trade_manager.update_position(
                                    &current_position.symbol,
                                    0.0,
                                    trade_price,
                                    trade_time,
                                );
                                // Position is now managed by trade_manager

                                // Position change is already captured in buy_executed log
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
