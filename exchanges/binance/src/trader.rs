use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    AccountStatusParams, OpenOrdersCancelAllParams, OrderCancelParams,
    OrderCancelReplaceCancelReplaceModeEnum, OrderCancelReplaceParams,
    OrderCancelReplaceSideEnum,
    OrderCancelReplaceTypeEnum, OrderPlaceParams, OrderPlaceSideEnum,
    OrderPlaceTimeInForceEnum, OrderPlaceTypeEnum, OrderStatusParams, WebsocketApi,
};
use futures_util::{Stream, StreamExt};
use rust_decimal::Decimal;
use std::str::FromStr;
use tracing::{debug, error, info, warn};

use crate::config::BinanceTraderConfig;
use trade::backtest::{BacktestEngine, CumulativePassiveTracker, MarketPrice};
use trade::config::TradeConfig;
use trade::execution::{ExecutionReport, ExecutionStatus, OrderIntent, Side, TimeInForce};
use trade::logger::{StrategyLoggerAdapter, TradeLogger, fmt_price, fmt_usd};
use trade::market::{MarketEvent, MarketState};
use trade::metrics::TradeManager;
use trade::models::Position;
use trade::simulation::{PositionSizer, VolatilityFactors};
use trade::strategy::{Strategy, StrategyContext, StrategyLogger};
use trade::trader::{MarketDataSourceKind, OrderType, TradeMode, Trader};

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
    /// Active limit order tracking for live maker orders.
    /// Contains (order_id, symbol, side, price, quantity) if a limit order is open.
    active_limit_order: Option<ActiveLimitOrder>,
    /// Active bid (buy) limit order for two-sided market-making.
    active_bid_order: Option<ActiveLimitOrder>,
    /// Active ask (sell) limit order for two-sided market-making.
    active_ask_order: Option<ActiveLimitOrder>,
    /// Monotonic counter for generating unique client_order_id per instance.
    order_counter: u64,
    /// Last time we polled order_status (UNIX millis). For throttling.
    last_order_poll_millis: u64,
    /// Last time an exchange rejection occurred (UNIX millis).
    /// Used to apply a cooldown after errors like "insufficient balance"
    /// to prevent spamming Binance with doomed orders on every tick.
    last_rejection_millis: u64,
    /// Count of consecutive exchange rejections (resets on successful placement).
    consecutive_rejections: u32,
    /// Count of consecutive taker order failures (resets on successful taker fill).
    /// After `MAX_TAKER_RETRIES` failures, the virtual position is force-closed
    /// because the on-exchange balance is likely zero / desynchronized.
    consecutive_taker_failures: u32,
    /// Last time a taker order failed (UNIX millis). Used for taker retry cooldown.
    last_taker_failure_millis: u64,
    /// Last time a two-sided order placement failed (UNIX millis).
    /// Prevents spamming Binance with doomed orders every tick.
    last_two_sided_failure_millis: u64,
    /// Count of consecutive two-sided placement failures (resets on success).
    consecutive_two_sided_failures: u32,
    /// Last known market mid-price from on_order_intent reference_price.
    /// Used by rebalance() to determine current market price.
    last_reference_price: f64,
    /// Last known best bid/ask from bookTicker. Used by place_or_replace_side()
    /// as a pre-flight guard to prevent LIMIT_MAKER orders that would cross the spread.
    /// Updated before each on_order_intent call from the current market_state.
    last_best_bid: f64,
    last_best_ask: f64,
    /// Price of the most recent BUY fill. Used as a post-fill guard: after a BUY
    /// fill at price P, the market likely moved down through P, so any new BUY
    /// at >= P will cross the spread. Reset to 0.0 on next fresh bookTicker event.
    last_buy_fill_price: f64,
    /// Price of the most recent SELL fill. Same logic: after SELL fill at P,
    /// any new SELL at <= P will likely cross. Reset to 0.0 on next bookTicker.
    last_sell_fill_price: f64,
    /// Previous two-sided state for change-detection logging.
    /// (bid_active, ask_active, bid_price, bid_qty, ask_price, ask_qty)
    last_two_sided_state: (bool, bool, f64, f64, f64, f64),
    /// Count of consecutive ticks where the strategy returned qty=0 for at least one side.
    /// Triggers automatic rebalance after threshold is exceeded.
    consecutive_zero_qty_ticks: u32,
    /// Millis when the last BID instant fill occurred.
    /// Used to enforce cooldown: no new bid placement for `min_order_rest_millis` after fill.
    last_bid_instant_fill_millis: u64,
    /// Millis when the last ASK instant fill occurred.
    last_ask_instant_fill_millis: u64,
    /// Millis when the last rebalance was executed. Used to enforce minimum interval between rebalances.
    last_rebalance_millis: u64,
    /// Active ladder bid orders for multi-level quoting (QuoteLadder).
    /// Each element is an order at a specific price level.
    active_ladder_bids: Vec<ActiveLimitOrder>,
    /// Active ladder ask orders for multi-level quoting (QuoteLadder).
    active_ladder_asks: Vec<ActiveLimitOrder>,
    /// Previous ladder state for change-detection logging.
    /// (n_bids, n_asks, total_bid_qty, total_ask_qty)
    last_ladder_state: (usize, usize, f64, f64),
}

/// Tracks a pending limit order placed on the exchange.
#[derive(Debug, Clone)]
struct ActiveLimitOrder {
    order_id: i64,
    client_order_id: String,
    symbol: String,
    side: Side,
    price: f64,
    quantity: f64,
    /// UNIX millis when this order was placed — used for min-rest enforcement.
    placed_at_millis: u64,
}

/// Result of `place_or_replace_side()`: the order may be resting on the book,
/// may have filled instantly (LIMIT crossing the spread), or may have failed.
enum PlaceResult {
    /// Order is resting on the book — track it for future cancel-replace/polling.
    Resting(ActiveLimitOrder),
    /// Order filled immediately at placement (LIMIT that crossed the spread).
    /// The caller must register this fill via `open_position`/`close_position`.
    InstantFill {
        avg_price: f64,
        executed_qty: f64,
        fee: f64,
    },
    /// Placement failed (exchange error, insufficient balance, rejected, etc.).
    Failed,
}



impl BinanceTrader {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_with_config_path("config/trading_config.toml").await
    }

    pub async fn new_with_config_path<P: AsRef<std::path::Path>>(
        config_path: P,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = std::env::var("BINANCE_API_KEY").ok();
        let api_secret = std::env::var("BINANCE_API_SECRET").ok();

        let trading_config = TradeConfig::from_file(config_path)?;
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
            active_limit_order: None,
            active_bid_order: None,
            active_ask_order: None,
            order_counter: 0,
            last_order_poll_millis: 0,
            last_rejection_millis: 0,
            consecutive_rejections: 0,
            consecutive_taker_failures: 0,
            last_taker_failure_millis: 0,
            last_two_sided_failure_millis: 0,
            consecutive_two_sided_failures: 0,
            last_reference_price: 0.0,
            last_best_bid: 0.0,
            last_best_ask: 0.0,
            last_buy_fill_price: 0.0,
            last_sell_fill_price: 0.0,
            last_two_sided_state: (false, false, 0.0, 0.0, 0.0, 0.0),
            consecutive_zero_qty_ticks: 0,
            last_bid_instant_fill_millis: 0,
            last_ask_instant_fill_millis: 0,
            last_rebalance_millis: 0,
            active_ladder_bids: Vec::new(),
            active_ladder_asks: Vec::new(),
            last_ladder_state: (0, 0, 0.0, 0.0),
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

    /// Generate a unique client_order_id for this trader instance.
    /// Format: "pulsar_{timestamp_ms}_{counter}" to ensure uniqueness.
    fn next_client_order_id(counter: &mut u64) -> String {
        *counter += 1;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!("pulsar_{}_{}", ts, *counter)
    }

    async fn cancel_two_sided_orders(&mut self) {
        for active_side in [self.active_bid_order.take(), self.active_ask_order.take()]
            .into_iter()
            .flatten()
        {
            let cancel_params = OrderCancelParams::builder(active_side.symbol.clone())
                .orig_client_order_id(active_side.client_order_id.clone())
                .build()
                .expect("Failed to build cancel params");
            let _ = self
                .connection
                .as_ref()
                .unwrap()
                .order_cancel(cancel_params)
                .await;
            info!(
                symbol = %active_side.symbol,
                side = ?active_side.side,
                "Cancelled two-sided order"
            );
        }
        // Also cancel any ladder orders
        self.cancel_ladder_orders().await;
    }

    /// Cancel all active ladder orders on both sides.
    async fn cancel_ladder_orders(&mut self) {
        let bids = std::mem::take(&mut self.active_ladder_bids);
        let asks = std::mem::take(&mut self.active_ladder_asks);
        for order in bids.into_iter().chain(asks.into_iter()) {
            let cancel_params = OrderCancelParams::builder(order.symbol.clone())
                .orig_client_order_id(order.client_order_id.clone())
                .build()
                .expect("Failed to build cancel params");
            let _ = self
                .connection
                .as_ref()
                .unwrap()
                .order_cancel(cancel_params)
                .await;
            debug!(
                symbol = %order.symbol,
                side = ?order.side,
                price = %fmt_price(order.price),
                "Cancelled ladder order"
            );
        }
    }

    /// Poll the exchange for the status of the active limit order.
    /// If the order has been filled asynchronously, record the fill in trade_manager
    /// and clear active_limit_order so the strategy can act on the new position.
    /// Returns true if a fill was detected and processed.
    async fn poll_active_order_status(&mut self, trading_symbol: &str, event_time_secs: f64) -> bool {
        let active = match &self.active_limit_order {
            Some(a) => a.clone(),
            None => return false,
        };
        let Some(connection) = &self.connection else {
            return false;
        };

        // Throttle: poll at most once per 2 seconds to avoid API spam.
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        if now_millis.saturating_sub(self.last_order_poll_millis) < 2000 {
            return false;
        }
        self.last_order_poll_millis = now_millis;

        let params = match OrderStatusParams::builder(active.symbol.clone())
            .orig_client_order_id(active.client_order_id.clone())
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                error!(
                    symbol = %active.symbol,
                    order_id = active.order_id,
                    error = %e,
                    "Failed to build OrderStatusParams"
                );
                return false;
            }
        };

        let response = match connection.order_status(params).await {
            Ok(r) => r,
            Err(e) => {
                // Non-fatal: log and continue, will retry next tick
                info!(
                    symbol = %active.symbol,
                    order_id = active.order_id,
                    error = %e,
                    "order_status query failed, will retry"
                );
                return false;
            }
        };

        let Ok(data) = response.data() else {
            return false;
        };

        let status = data.status.as_deref().unwrap_or("UNKNOWN");
        let executed_qty = data
            .executed_qty
            .as_ref()
            .and_then(|q| q.parse::<f64>().ok())
            .unwrap_or(0.0);
        let cumm_quote_qty = data
            .cummulative_quote_qty
            .as_ref()
            .and_then(|q| q.parse::<f64>().ok())
            .unwrap_or(0.0);

        match status {
            "FILLED" => {
                let avg_price = if executed_qty > 0.0 {
                    cumm_quote_qty / executed_qty
                } else {
                    active.price
                };
                // Use the higher of maker_fee and bnb_discount_fee as conservative
                // estimate. When maker_fee=0 but BNB discount is active, Binance
                // still charges ~7.5 bps in BNB which reduces effective quantity.
                let effective_fee_rate = self.config.exchange.maker_fee
                    .max(self.config.exchange.bnb_discount_fee);
                let fee = executed_qty * avg_price * effective_fee_rate;

                // Async poll has no fills data, so we can't check commission_asset.
                // By default Binance charges fee in the received asset:
                //   Buy → fee in base asset → reduce effective quantity
                //   Sell → fee in quote asset (USDT) → no qty adjustment needed
                let effective_qty = if matches!(active.side, Side::Buy) {
                    let adj = executed_qty * (1.0 - effective_fee_rate);
                    info!(
                        symbol = %active.symbol,
                        order_id = active.order_id,
                        side = ?active.side,
                        price = %fmt_price(avg_price),
                        raw_qty = %fmt_price(executed_qty),
                        estimated_fee_qty = %fmt_price(executed_qty - adj),
                        effective_qty = %fmt_price(adj),
                        fee_usdt = %fmt_price(fee),
                        "Async fill: adjusted qty for estimated base-asset commission"
                    );
                    adj
                } else {
                    info!(
                        symbol = %active.symbol,
                        order_id = active.order_id,
                        side = ?active.side,
                        price = %fmt_price(avg_price),
                        quantity = %fmt_price(executed_qty),
                        fee = %fmt_price(fee),
                        "Async fill detected via order_status poll"
                    );
                    executed_qty
                };

                // Record fill in trade_manager
                match active.side {
                    Side::Buy => {
                        let report = ExecutionReport {
                            status: ExecutionStatus::Filled,
                            symbol: Some(trading_symbol.to_string()),
                            side: Some(Side::Buy),
                            order_type: Some(OrderType::Maker),
                            rationale: Some("passive_buy_at_bid"),
                            decision_confidence: 1.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: active.quantity,
                            executed_quantity: effective_qty,
                            execution_price: Some(avg_price),
                            fee_paid: fee,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: None,
                            expected_edge_bps: 0.0,
                        };
                        self.trade_manager.record_execution_report(&report);
                        let _ = self.trade_manager.open_position(
                            trading_symbol,
                            avg_price,
                            effective_qty,
                            event_time_secs,
                            Some(&report),
                        );
                    }
                    Side::Sell => {
                        let report = ExecutionReport {
                            status: ExecutionStatus::Filled,
                            symbol: Some(trading_symbol.to_string()),
                            side: Some(Side::Sell),
                            order_type: Some(OrderType::Maker),
                            rationale: Some("passive_sell_at_ask"),
                            decision_confidence: 1.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: active.quantity,
                            executed_quantity: executed_qty,
                            execution_price: Some(avg_price),
                            fee_paid: fee,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: None,
                            expected_edge_bps: 0.0,
                        };
                        self.trade_manager.record_execution_report(&report);
                        let _pnl = self.trade_manager.close_position_with_report(
                            trading_symbol,
                            avg_price,
                            event_time_secs,
                            Some(&report),
                        );
                    }
                }

                self.active_limit_order = None;
                true
            }
            "CANCELED" | "EXPIRED" | "REJECTED" | "EXPIRED_IN_MATCH" => {
                info!(
                    symbol = %active.symbol,
                    order_id = active.order_id,
                    status = status,
                    "Active order no longer valid, clearing tracker"
                );
                self.active_limit_order = None;
                false
            }
            "PARTIALLY_FILLED" => {
                // For now, treat partial fill as still resting — will be fully handled later
                info!(
                    symbol = %active.symbol,
                    order_id = active.order_id,
                    executed_qty = %fmt_price(executed_qty),
                    total_qty = %fmt_price(active.quantity),
                    "Partial fill detected, order still resting"
                );
                false
            }
            _ => {
                // "NEW" or unknown — order is still resting, nothing to do
                false
            }
        }
    }

    /// Poll the exchange for status of a specific two-sided order (bid or ask).
    /// If filled, returns (true, avg_price, executed_qty, fee).
    /// If cancelled/expired, returns (false, 0, 0, 0) — meaning "terminal, clear it".
    /// If still resting, returns None — meaning "keep waiting".
    async fn poll_one_sided_order(
        &self,
        active: &ActiveLimitOrder,
    ) -> Option<(bool, f64, f64, f64)> {
        let Some(connection) = &self.connection else {
            return None;
        };

        let params = match OrderStatusParams::builder(active.symbol.clone())
            .orig_client_order_id(active.client_order_id.clone())
            .build()
        {
            Ok(p) => p,
            Err(_) => return None,
        };

        let response = match connection.order_status(params).await {
            Ok(r) => r,
            Err(_) => return None,
        };

        let Ok(data) = response.data() else {
            return None;
        };

        let status = data.status.as_deref().unwrap_or("UNKNOWN");
        let executed_qty = data
            .executed_qty
            .as_ref()
            .and_then(|q| q.parse::<f64>().ok())
            .unwrap_or(0.0);
        let cumm_quote_qty = data
            .cummulative_quote_qty
            .as_ref()
            .and_then(|q| q.parse::<f64>().ok())
            .unwrap_or(0.0);

        match status {
            "FILLED" => {
                let avg_price = if executed_qty > 0.0 {
                    cumm_quote_qty / executed_qty
                } else {
                    active.price
                };
                let effective_fee_rate = self
                    .config
                    .exchange
                    .maker_fee
                    .max(self.config.exchange.bnb_discount_fee);
                let fee = executed_qty * avg_price * effective_fee_rate;
                Some((true, avg_price, executed_qty, fee))
            }
            "CANCELED" | "EXPIRED" | "REJECTED" | "EXPIRED_IN_MATCH" => {
                Some((false, 0.0, 0.0, 0.0))
            }
            _ => None, // still resting
        }
    }

    /// Poll both active two-sided orders for async fills.
    /// Returns true if any fill was detected and processed.
    async fn poll_two_sided_orders(
        &mut self,
        trading_symbol: &str,
        event_time_secs: f64,
    ) -> bool {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        // Throttle: poll at most once per 2 seconds
        if now_millis.saturating_sub(self.last_order_poll_millis) < 2000 {
            return false;
        }
        self.last_order_poll_millis = now_millis;

        let mut any_fill = false;

        // Poll bid order
        if let Some(active_bid) = self.active_bid_order.clone() {
            if let Some((is_filled, avg_price, executed_qty, fee)) =
                self.poll_one_sided_order(&active_bid).await
            {
                if is_filled {
                    let effective_fee_rate = self
                        .config
                        .exchange
                        .maker_fee
                        .max(self.config.exchange.bnb_discount_fee);
                    let effective_qty = executed_qty * (1.0 - effective_fee_rate);
                    info!(
                        symbol = %active_bid.symbol,
                        order_id = active_bid.order_id,
                        side = "BUY",
                        price = %fmt_price(avg_price),
                        effective_qty = %fmt_price(effective_qty),
                        fee = %fmt_price(fee),
                        "Two-sided: async BID fill detected"
                    );
                    let report = ExecutionReport {
                        status: ExecutionStatus::Filled,
                        symbol: Some(trading_symbol.to_string()),
                        side: Some(Side::Buy),
                        order_type: Some(OrderType::Maker),
                        rationale: Some("two_sided_bid_fill"),
                        decision_confidence: 1.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: active_bid.quantity,
                        executed_quantity: effective_qty,
                        execution_price: Some(avg_price),
                        fee_paid: fee,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: None,
                        expected_edge_bps: 0.0,
                    };
                    self.trade_manager.record_execution_report(&report);
                    let _ = self.trade_manager.open_position(
                        trading_symbol,
                        avg_price,
                        effective_qty,
                        event_time_secs,
                        Some(&report),
                    );
                    any_fill = true;
                    self.last_buy_fill_price = avg_price;
                }
                // Clear bid order on any terminal status (filled, cancelled, expired)
                self.active_bid_order = None;
            }
        }

        // Poll ask order
        if let Some(active_ask) = self.active_ask_order.clone() {
            if let Some((is_filled, avg_price, executed_qty, fee)) =
                self.poll_one_sided_order(&active_ask).await
            {
                if is_filled {
                    info!(
                        symbol = %active_ask.symbol,
                        order_id = active_ask.order_id,
                        side = "SELL",
                        price = %fmt_price(avg_price),
                        quantity = %fmt_price(executed_qty),
                        fee = %fmt_price(fee),
                        "Two-sided: async ASK fill detected"
                    );
                    let report = ExecutionReport {
                        status: ExecutionStatus::Filled,
                        symbol: Some(trading_symbol.to_string()),
                        side: Some(Side::Sell),
                        order_type: Some(OrderType::Maker),
                        rationale: Some("two_sided_ask_fill"),
                        decision_confidence: 1.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: active_ask.quantity,
                        executed_quantity: executed_qty,
                        execution_price: Some(avg_price),
                        fee_paid: fee,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: None,
                        expected_edge_bps: 0.0,
                    };
                    self.trade_manager.record_execution_report(&report);
                    // Partial close: reduce position by the sold quantity
                    // instead of removing the entire position.
                    let current_qty = self
                        .trade_manager
                        .get_position(trading_symbol)
                        .map_or(0.0, |p| p.quantity);
                    let remaining = (current_qty - executed_qty).max(0.0);
                    self.trade_manager.update_position(
                        trading_symbol,
                        remaining,
                        avg_price,
                        event_time_secs,
                        Some(fee),
                    );
                    any_fill = true;
                    self.last_sell_fill_price = avg_price;
                }
                // Clear ask order on any terminal status
                self.active_ask_order = None;
            }
        }

        any_fill
    }

    /// Poll all active ladder orders for async fills.
    /// Returns true if any fill was detected and processed.
    async fn poll_ladder_orders(
        &mut self,
        trading_symbol: &str,
        event_time_secs: f64,
    ) -> bool {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        // Throttle: poll at most once per 2 seconds
        // (shares last_order_poll_millis with two-sided polling)
        if now_millis.saturating_sub(self.last_order_poll_millis) < 2000 {
            return false;
        }
        self.last_order_poll_millis = now_millis;

        let mut any_fill = false;

        // Poll bid ladder orders
        let mut bids_to_keep = Vec::with_capacity(self.active_ladder_bids.len());
        let bids = std::mem::take(&mut self.active_ladder_bids);
        for active_bid in bids {
            if let Some((is_filled, avg_price, executed_qty, fee)) =
                self.poll_one_sided_order(&active_bid).await
            {
                if is_filled {
                    let effective_fee_rate = self
                        .config
                        .exchange
                        .maker_fee
                        .max(self.config.exchange.bnb_discount_fee);
                    let effective_qty = executed_qty * (1.0 - effective_fee_rate);
                    info!(
                        symbol = %active_bid.symbol,
                        order_id = active_bid.order_id,
                        side = "BUY",
                        price = %fmt_price(avg_price),
                        level_price = %fmt_price(active_bid.price),
                        effective_qty = %fmt_price(effective_qty),
                        fee = %fmt_price(fee),
                        "Ladder: async BID fill detected"
                    );
                    let report = ExecutionReport {
                        status: ExecutionStatus::Filled,
                        symbol: Some(trading_symbol.to_string()),
                        side: Some(Side::Buy),
                        order_type: Some(OrderType::Maker),
                        rationale: Some("ladder_bid_fill"),
                        decision_confidence: 1.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: active_bid.quantity,
                        executed_quantity: effective_qty,
                        execution_price: Some(avg_price),
                        fee_paid: fee,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: None,
                        expected_edge_bps: 0.0,
                    };
                    self.trade_manager.record_execution_report(&report);
                    let _ = self.trade_manager.open_position(
                        trading_symbol,
                        avg_price,
                        effective_qty,
                        event_time_secs,
                        Some(&report),
                    );
                    any_fill = true;
                    self.last_buy_fill_price = avg_price;
                }
                // Don't keep: terminal status (filled, cancelled, expired)
            } else {
                // Still resting — keep
                bids_to_keep.push(active_bid);
            }
        }
        self.active_ladder_bids = bids_to_keep;

        // Poll ask ladder orders
        let mut asks_to_keep = Vec::with_capacity(self.active_ladder_asks.len());
        let asks = std::mem::take(&mut self.active_ladder_asks);
        for active_ask in asks {
            if let Some((is_filled, avg_price, executed_qty, fee)) =
                self.poll_one_sided_order(&active_ask).await
            {
                if is_filled {
                    info!(
                        symbol = %active_ask.symbol,
                        order_id = active_ask.order_id,
                        side = "SELL",
                        price = %fmt_price(avg_price),
                        level_price = %fmt_price(active_ask.price),
                        quantity = %fmt_price(executed_qty),
                        fee = %fmt_price(fee),
                        "Ladder: async ASK fill detected"
                    );
                    let report = ExecutionReport {
                        status: ExecutionStatus::Filled,
                        symbol: Some(trading_symbol.to_string()),
                        side: Some(Side::Sell),
                        order_type: Some(OrderType::Maker),
                        rationale: Some("ladder_ask_fill"),
                        decision_confidence: 1.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: active_ask.quantity,
                        executed_quantity: executed_qty,
                        execution_price: Some(avg_price),
                        fee_paid: fee,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: None,
                        expected_edge_bps: 0.0,
                    };
                    self.trade_manager.record_execution_report(&report);
                    let current_qty = self
                        .trade_manager
                        .get_position(trading_symbol)
                        .map_or(0.0, |p| p.quantity);
                    let remaining = (current_qty - executed_qty).max(0.0);
                    self.trade_manager.update_position(
                        trading_symbol,
                        remaining,
                        avg_price,
                        event_time_secs,
                        Some(fee),
                    );
                    any_fill = true;
                    self.last_sell_fill_price = avg_price;
                }
                // Don't keep: terminal status
            } else {
                // Still resting — keep
                asks_to_keep.push(active_ask);
            }
        }
        self.active_ladder_asks = asks_to_keep;

        any_fill
    }

    /// Place or replace a single side of a two-sided quote on the exchange.
    /// Returns `PlaceResult::Resting` if the order is live on the book,
    /// `PlaceResult::InstantFill` if the LIMIT order filled immediately (crossed the spread),
    /// or `PlaceResult::Failed` on error.
    async fn place_or_replace_side(
        &mut self,
        symbol: &str,
        side: Side,
        price: f64,
        quantity: f64,
        existing: &Option<ActiveLimitOrder>,
    ) -> PlaceResult {
        if quantity <= f64::EPSILON {
            // Zero quantity — cancel existing order on this side if any
            if let Some(active) = existing {
                let cancel_params = OrderCancelParams::builder(active.symbol.clone())
                    .orig_client_order_id(active.client_order_id.clone())
                    .build()
                    .expect("Failed to build cancel params");
                let _ = self
                    .connection
                    .as_ref()
                    .unwrap()
                    .order_cancel(cancel_params)
                    .await;
            }
            return PlaceResult::Failed;
        }

        // Two-sided cooldown: after a placement failure, wait before retrying.
        // This prevents spamming Binance with doomed orders every tick.
        // NOTE: On tight-spread pairs like DOGEFDUSD (~1 bps spread), LIMIT_MAKER
        // rejections (-2010) are normal and frequent. A 10s cooldown was too
        // aggressive — it caused the bot to go dark on both sides for 10 seconds
        // after a single rejection. Reduced to 1s for faster recovery.
        const TWO_SIDED_COOLDOWN_MS: u64 = 1_000;
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        if self.consecutive_two_sided_failures > 0
            && now_millis.saturating_sub(self.last_two_sided_failure_millis) < TWO_SIDED_COOLDOWN_MS
        {
            // Still in cooldown — keep existing order if any, otherwise skip
            if let Some(active) = existing {
                return PlaceResult::Resting(active.clone());
            }
            return PlaceResult::Failed;
        }

        let tick_size = self.config.exchange.tick_size;
        let step_size = self.config.exchange.step_size;
        let min_notional = self.config.exchange.min_notional;

        // Round price to tick
        let rounded_price = if tick_size > 0.0 {
            let ticks = price / tick_size;
            match side {
                Side::Buy => ticks.floor() * tick_size,
                Side::Sell => ticks.ceil() * tick_size,
            }
        } else {
            price
        };
        // Round quantity to step
        let mut rounded_qty = (quantity / step_size).floor() * step_size;

        // BNB fee adjustment for buys
        if matches!(side, Side::Buy) && self.config.exchange.bnb_discount_fee > 0.0 {
            rounded_qty += step_size;
        }

        // Min notional check
        if rounded_qty * rounded_price < min_notional || rounded_qty < step_size {
            return PlaceResult::Failed;
        }

        // PRE-FLIGHT GUARD: Prevent LIMIT_MAKER orders that would cross the spread.
        // Instead of sending to exchange and getting -2010 rejection, adjust the
        // price to the current best bid/ask. This avoids wasted API calls and
        // keeps the strategy quoting on both sides after fills.
        //
        // A 1-tick safety margin is applied because the bookTicker snapshot used
        // here can be stale by ~80ms (network latency). Without the margin, a
        // BUY at best_ask - 1tick can still get rejected if the ask drops 1 tick
        // before our order reaches the matching engine.
        let rounded_price = if self.last_best_bid > f64::EPSILON && self.last_best_ask > f64::EPSILON {
            match side {
                Side::Buy => {
                    if rounded_price >= self.last_best_ask - tick_size {
                        // BUY within 1 tick of best_ask could cross → clamp to best_bid
                        let clamped = (self.last_best_bid / tick_size).floor() * tick_size;
                        if clamped < tick_size {
                            return PlaceResult::Failed;
                        }
                        clamped
                    } else {
                        rounded_price
                    }
                }
                Side::Sell => {
                    if rounded_price <= self.last_best_bid + tick_size {
                        // SELL within 1 tick of best_bid could cross → clamp to best_ask
                        let clamped = (self.last_best_ask / tick_size).ceil() * tick_size;
                        clamped
                    } else {
                        rounded_price
                    }
                }
            }
        } else {
            rounded_price
        };

        // POST-FILL GUARD: After a fill, the bookTicker may be stale (hasn't
        // updated yet). If we just filled a SELL at price P, the market moved
        // through P upward, so placing a new SELL at <= P will cross. Similarly,
        // after a BUY fill at P, a new BUY at >= P will cross.
        // Skip the order entirely (return Failed) — the next tick with a fresh
        // bookTicker will generate a valid price.
        let rounded_price = match side {
            Side::Buy => {
                if self.last_buy_fill_price > f64::EPSILON
                    && rounded_price >= self.last_buy_fill_price
                {
                    return PlaceResult::Failed;
                }
                rounded_price
            }
            Side::Sell => {
                if self.last_sell_fill_price > f64::EPSILON
                    && rounded_price <= self.last_sell_fill_price
                {
                    return PlaceResult::Failed;
                }
                rounded_price
            }
        };

        // Same-price skip + min rest enforcement
        if let Some(active) = existing {
            let resting_ms = now_millis.saturating_sub(active.placed_at_millis);
            let same_price = (active.price - rounded_price).abs() < tick_size * 0.5;

            // CRITICAL: If price is unchanged, NEVER cancel-replace — even if
            // quantity differs.  On 1-tick spread pairs, queue priority is THE
            // edge.  Ladder-level reshuffling causes qty to oscillate (e.g.
            // 24→12→24→36) at the same price every cycle — wasting API calls
            // and destroying queue position for zero benefit.
            //
            // The marginal qty difference (±12 DOGE ≈ $1) is negligible vs the
            // value of keeping front-of-queue priority.  The qty will be updated
            // naturally the next time the price actually changes.
            if same_price {
                return PlaceResult::Resting(active.clone());
            }
            // Min rest enforcement — don't cancel-replace too rapidly even when
            // the desired price has changed.
            if resting_ms < self.config.order_execution.min_order_rest_millis {
                return PlaceResult::Resting(active.clone());
            }
        }

        let qty_precision = if step_size >= 1.0 { 0 } else { (-step_size.log10()).ceil() as usize };
        let price_precision = (-tick_size.log10()).ceil() as usize;
        let qty_str = format!("{:.prec$}", rounded_qty, prec = qty_precision);
        let price_str = format!("{:.prec$}", rounded_price, prec = price_precision);

        let quantity_decimal = match Decimal::from_str(&qty_str) {
            Ok(q) => q,
            Err(_) => return PlaceResult::Failed,
        };
        let price_decimal = match Decimal::from_str(&price_str) {
            Ok(p) => p,
            Err(_) => return PlaceResult::Failed,
        };

        let client_oid = Self::next_client_order_id(&mut self.order_counter);

        // Use atomic cancel-replace when an existing order needs to be updated.
        // This avoids the race condition where a manual cancel frees the locked
        // balance asynchronously and the subsequent place arrives before the
        // unlock is processed, causing -2010 "insufficient balance".
        if let Some(active) = existing {
            let cr_side = match side {
                Side::Buy => OrderCancelReplaceSideEnum::Buy,
                Side::Sell => OrderCancelReplaceSideEnum::Sell,
            };
            let cr_params = OrderCancelReplaceParams::builder(
                symbol.to_string(),
                OrderCancelReplaceCancelReplaceModeEnum::StopOnFailure,
                cr_side,
                OrderCancelReplaceTypeEnum::LimitMaker,
            )
            .cancel_orig_client_order_id(active.client_order_id.clone())
            .quantity(quantity_decimal)
            .price(price_decimal)
            .new_client_order_id(client_oid.clone())
            .build()
            .expect("Failed to build cancel-replace params");

            let response = match self
                .connection
                .as_ref()
                .unwrap()
                .order_cancel_replace(cr_params)
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    let err_str = format!("{}", e);
                    if err_str.contains("-2011") {
                        // -2011: Unknown order — the old order was already filled or
                        // cancelled. Poll it to check for a fill we must not lose.
                        warn!(
                            symbol = %symbol,
                            side = ?side,
                            old_order_id = active.order_id,
                            "Two-sided: cancel-replace failed — old order already gone, polling for fill"
                        );
                        if let Some((is_filled, avg_price, executed_qty, fee)) =
                            self.poll_one_sided_order(active).await
                        {
                            if is_filled {
                                info!(
                                    symbol = %symbol,
                                    side = ?side,
                                    price = %fmt_price(avg_price),
                                    qty = %fmt_price(executed_qty),
                                    fee = %fmt_price(fee),
                                    "Two-sided: recovered fill from gone order"
                                );
                                return PlaceResult::InstantFill {
                                    avg_price,
                                    executed_qty,
                                    fee,
                                };
                            }
                        }
                    } else {
                        error!(
                            symbol = %symbol,
                            side = ?side,
                            old_price = %fmt_price(active.price),
                            new_price = %price_str,
                            quantity = %qty_str,
                            error = %e,
                            "Two-sided: cancel-replace failed — attempting explicit cancel of old order"
                        );
                        self.consecutive_two_sided_failures += 1;
                        self.last_two_sided_failure_millis = now_millis;

                        // The old order may still be alive on Binance (e.g. -2022).
                        // Try to cancel it explicitly to avoid orphaned orders that
                        // lock balance and cause -2010 on subsequent placements.
                        let cancel_params = OrderCancelParams::builder(symbol.to_string())
                            .orig_client_order_id(active.client_order_id.clone())
                            .build()
                            .expect("Failed to build cancel params");
                        match self
                            .connection
                            .as_ref()
                            .unwrap()
                            .order_cancel(cancel_params)
                            .await
                        {
                            Ok(_) => {
                                info!(
                                    symbol = %symbol,
                                    side = ?side,
                                    old_order_id = active.order_id,
                                    "Two-sided: old order cancelled successfully after cancel-replace failure"
                                );
                            }
                            Err(cancel_err) => {
                                let cancel_err_str = format!("{}", cancel_err);
                                if cancel_err_str.contains("-2011") {
                                    // Old order already gone — check if it was filled
                                    warn!(
                                        symbol = %symbol,
                                        side = ?side,
                                        old_order_id = active.order_id,
                                        "Two-sided: old order already gone during cleanup, polling for fill"
                                    );
                                    if let Some((is_filled, avg_price, executed_qty, fee)) =
                                        self.poll_one_sided_order(active).await
                                    {
                                        if is_filled {
                                            info!(
                                                symbol = %symbol,
                                                side = ?side,
                                                price = %fmt_price(avg_price),
                                                qty = %fmt_price(executed_qty),
                                                fee = %fmt_price(fee),
                                                "Two-sided: recovered fill from old order after cancel-replace failure"
                                            );
                                            return PlaceResult::InstantFill {
                                                avg_price,
                                                executed_qty,
                                                fee,
                                            };
                                        }
                                    }
                                } else {
                                    error!(
                                        symbol = %symbol,
                                        side = ?side,
                                        old_order_id = active.order_id,
                                        error = %cancel_err,
                                        "Two-sided: failed to cancel old order after cancel-replace failure"
                                    );
                                }
                            }
                        }
                    }
                    return PlaceResult::Failed;
                }
            };

            let Ok(data) = response.data() else {
                return PlaceResult::Failed;
            };

            let cancel_result = data.cancel_result.as_deref().unwrap_or("UNKNOWN");
            let new_order_result = data.new_order_result.as_deref().unwrap_or("UNKNOWN");

            if new_order_result != "SUCCESS" {
                warn!(
                    symbol = %symbol,
                    side = ?side,
                    cancel_result = cancel_result,
                    new_order_result = new_order_result,
                    "Two-sided: cancel-replace — new order not placed"
                );
                return PlaceResult::Failed;
            }

            // Reset failure counter on success
            self.consecutive_two_sided_failures = 0;

            // Extract new order details from the response
            let new_resp = match &data.new_order_response {
                Some(r) => r,
                None => return PlaceResult::Failed,
            };
            let order_id = new_resp.order_id.unwrap_or(0);
            let status_str = new_resp.status.as_deref().unwrap_or("UNKNOWN");

            if status_str == "REJECTED" || status_str == "EXPIRED_IN_MATCH" {
                // EXPIRED_IN_MATCH is expected for LIMIT_MAKER cancel-replace:
                // old order was cancelled but new order would cross the spread.
                debug!(
                    symbol = %symbol,
                    side = ?side,
                    order_id = order_id,
                    status = status_str,
                    "Two-sided: cancel-replace — new LIMIT_MAKER rejected (would cross spread)"
                );
                return PlaceResult::Failed;
            }

            // With LIMIT_MAKER, immediate fills should never happen on the new order.
            // Keep this branch as a safety net.
            if status_str == "FILLED" {
                let fill_price = new_resp
                    .price
                    .as_ref()
                    .and_then(|p| p.parse::<f64>().ok())
                    .unwrap_or(rounded_price);
                let fill_qty = new_resp
                    .executed_qty
                    .as_ref()
                    .and_then(|q| q.parse::<f64>().ok())
                    .unwrap_or(rounded_qty);
                // Maker fee is 0% on this pair; actual fee from fills array
                // would be more precise but for 0-fee pairs this is correct.
                let fill_fee = 0.0;
                info!(
                    symbol = %symbol,
                    side = ?side,
                    order_id = order_id,
                    old_price = %fmt_price(active.price),
                    new_price = %price_str,
                    old_qty = %fmt_price(active.quantity),
                    new_qty = %qty_str,
                    status = status_str,
                    "Two-sided: cancel-replace — new order filled immediately"
                );
                return PlaceResult::InstantFill {
                    avg_price: fill_price,
                    executed_qty: fill_qty,
                    fee: fill_fee,
                };
            }

            info!(
                symbol = %symbol,
                side = ?side,
                order_id = order_id,
                old_price = %fmt_price(active.price),
                new_price = %price_str,
                old_qty = %fmt_price(active.quantity),
                new_qty = %qty_str,
                status = status_str,
                "Two-sided: cancel-replace succeeded"
            );

            return PlaceResult::Resting(ActiveLimitOrder {
                order_id,
                client_order_id: client_oid,
                symbol: symbol.to_string(),
                side,
                price: rounded_price,
                quantity: rounded_qty,
                placed_at_millis: now_millis,
            });
        }

        // No existing order — place a fresh LIMIT_MAKER (post-only) order.
        // LIMIT_MAKER is rejected if it would immediately match as taker,
        // guaranteeing we only ever rest on the book as a maker.
        let params = OrderPlaceParams::builder(
            symbol.to_string(),
            match side {
                Side::Buy => OrderPlaceSideEnum::Buy,
                Side::Sell => OrderPlaceSideEnum::Sell,
            },
            OrderPlaceTypeEnum::LimitMaker,
        )
        .quantity(quantity_decimal)
        .price(price_decimal)
        .new_client_order_id(client_oid.clone())
        .build()
        .expect("Failed to build limit order params");

        let response = match self.connection.as_ref().unwrap().order_place(params).await {
            Ok(resp) => resp,
            Err(e) => {
                error!(
                    symbol = %symbol,
                    side = ?side,
                    price = %price_str,
                    quantity = %qty_str,
                    error = %e,
                    "Two-sided: failed to place limit order"
                );
                self.consecutive_two_sided_failures += 1;
                self.last_two_sided_failure_millis = now_millis;
                return PlaceResult::Failed;
            }
        };

        let Ok(data) = response.data() else {
            return PlaceResult::Failed;
        };

        let order_id = data.order_id.unwrap_or(0);
        let status_str = data.status.as_deref().unwrap_or("UNKNOWN");

        if status_str == "REJECTED" || status_str == "EXPIRED_IN_MATCH" {
            // EXPIRED_IN_MATCH is expected for LIMIT_MAKER orders: it means the
            // order would have immediately matched as taker, so the exchange
            // rejected it.  This is correct post-only behaviour — not an error.
            debug!(
                symbol = %symbol,
                side = ?side,
                order_id = order_id,
                price = %price_str,
                status = status_str,
                "Two-sided: LIMIT_MAKER order rejected (would cross spread)"
            );
            return PlaceResult::Failed;
        }

        // Reset failure counter on success
        self.consecutive_two_sided_failures = 0;

        // With LIMIT_MAKER, immediate fills should never happen — the order is
        // rejected instead of crossing.  Keep this branch as a safety net.
        if status_str == "FILLED" {
            let fill_price = data
                .price
                .as_ref()
                .and_then(|p| p.parse::<f64>().ok())
                .unwrap_or(rounded_price);
            let fill_qty = data
                .executed_qty
                .as_ref()
                .and_then(|q| q.parse::<f64>().ok())
                .unwrap_or(rounded_qty);
            info!(
                symbol = %symbol,
                side = ?side,
                order_id = order_id,
                price = %price_str,
                quantity = %qty_str,
                status = status_str,
                "Two-sided: placed limit order — filled immediately"
            );
            return PlaceResult::InstantFill {
                avg_price: fill_price,
                executed_qty: fill_qty,
                fee: 0.0,
            };
        }

        info!(
            symbol = %symbol,
            side = ?side,
            order_id = order_id,
            price = %price_str,
            quantity = %qty_str,
            status = status_str,
            "Two-sided: placed limit order"
        );

        PlaceResult::Resting(ActiveLimitOrder {
            order_id,
            client_order_id: client_oid,
            symbol: symbol.to_string(),
            side,
            price: rounded_price,
            quantity: rounded_qty,
            placed_at_millis: now_millis,
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

    fn validate_exit_trade(&self, symbol: &str, quantity: f64, _price: f64) -> bool {
        if quantity <= 0.0 {
            self.logger.log_order_error(
                symbol,
                "exit_validation",
                "missing_position",
                "No open quantity available to close",
            );
            return false;
        }

        // NOTE: min_notional is intentionally NOT checked for exits.
        // On Binance, you can always sell/close a position regardless of notional.
        // min_notional only applies to opening new positions.
        // Previously this check caused a "dust trap" where partial fills left
        // positions below min_notional that could never be closed, causing the
        // bot to loop forever with rejected sell orders.

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
            symbol: None,
            side: Some(Side::Buy),
            order_type: Some(OrderType::Taker),
            rationale: None,
            decision_confidence: 0.0,
            decision_metrics: Vec::new(),
            requested_quantity,
            executed_quantity: execution.executed_quantity,
            execution_price: Some(execution.execution_price),
            fee_paid: execution.fee_paid,
            latency_seconds: execution.latency_seconds,
            synthetic_half_spread_bps: execution.synthetic_half_spread_rate * 10_000.0,
            slippage_bps: execution.slippage_rate * 10_000.0,
            latency_impact_bps: execution.latency_impact_rate * 10_000.0,
            market_impact_bps: execution.market_impact_rate * 10_000.0,
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
            symbol: None,
            side: Some(Side::Sell),
            order_type: Some(OrderType::Taker),
            rationale: None,
            decision_confidence: 0.0,
            decision_metrics: Vec::new(),
            requested_quantity,
            executed_quantity: execution.executed_quantity,
            execution_price: Some(execution.execution_price),
            fee_paid: execution.fee_paid,
            latency_seconds: execution.latency_seconds,
            synthetic_half_spread_bps: execution.synthetic_half_spread_rate * 10_000.0,
            slippage_bps: execution.slippage_rate * 10_000.0,
            latency_impact_bps: execution.latency_impact_rate * 10_000.0,
            market_impact_bps: execution.market_impact_rate * 10_000.0,
            reason: execution.rejected_reason,
            expected_edge_bps,
        }
    }

    fn market_price_from_state(&self, market_state: &MarketState) -> Option<MarketPrice> {
        market_state
            .top_of_book()
            .map(|book| MarketPrice::Quote {
                bid: book.bid.price,
                ask: book.ask.price,
            })
            .or_else(|| {
                // In trade-only mode, synthesize bid/ask from the last trade price
                // and the EMA spread estimate. This is critical for passive order
                // simulation: without explicit bid/ask, a passive sell at "ask"
                // would execute at the same price as a passive buy at "bid",
                // making spread capture impossible.
                let last_price = market_state.last_price()?;
                let micro = market_state.micro();
                if micro.mid_initialized() && micro.ema_spread_bps > 0.0 {
                    let half_spread_frac = micro.ema_spread_bps / 20_000.0;
                    // Use last trade's direction to infer which side of the
                    // spread the price is on:
                    //   - seller-initiated (is_buyer_market_maker=true) → price near bid
                    //   - buyer-initiated  (is_buyer_market_maker=false) → price near ask
                    let (bid, ask) = if let Some(last_trade) = market_state.last_trade() {
                        if last_trade.is_buyer_market_maker {
                            // Price is at bid; ask is one spread away
                            (last_price, last_price * (1.0 + micro.ema_spread_bps / 10_000.0))
                        } else {
                            // Price is at ask; bid is one spread below
                            (last_price * (1.0 - micro.ema_spread_bps / 10_000.0), last_price)
                        }
                    } else {
                        // No direction info — assume mid
                        (last_price * (1.0 - half_spread_frac), last_price * (1.0 + half_spread_frac))
                    };
                    Some(MarketPrice::Quote { bid, ask })
                } else {
                    Some(MarketPrice::Trade(last_price))
                }
            })
    }

    fn execution_reference_price_for_intent(
        &self,
        market_price: MarketPrice,
        intent: &OrderIntent,
    ) -> f64 {
        match intent {
            OrderIntent::Place { side, .. } => match side {
                Side::Buy => market_price.execution_reference_price(trade::signal::Signal::Buy),
                Side::Sell => market_price.execution_reference_price(trade::signal::Signal::Sell),
            },
            OrderIntent::NoAction | OrderIntent::Cancel { .. } | OrderIntent::QuoteBothSides { .. } | OrderIntent::QuoteLadder { .. } => market_price.decision_reference_price(),
        }
    }

    /// Connect to Binance WebSocket API (for rebalance and other operations).
    pub async fn connect(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.connection.is_some() {
            return Ok(());
        }
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
        Ok(())
    }

    /// Returns (quote_free, base_asset_free) — e.g. (98.6, 0.036) for BNBUSDT.
    /// Quote asset is determined from the config's `exchange.quote_asset` field.
    pub async fn account_balances(
        &self,
        base_asset: &str,
    ) -> Result<(f64, f64), anyhow::Error> {
        let Some(connection) = &self.connection else {
            return Err(anyhow::anyhow!("no connection"));
        };

        let params = AccountStatusParams::builder()
            .omit_zero_balances(true)
            .build()?;
        let status = connection.account_status(params).await?;
        let data = status.data().unwrap();

        if let Some(account_type) = &data.account_type {
            info!(account_type = %account_type, "Account status retrieved");
        }

        let quote_asset = &self.config.exchange.quote_asset;
        let mut quote_free: f64 = 0.0;
        let mut base_free: f64 = 0.0;

        if let Some(balances) = &data.balances {
            for bal in balances {
                let asset = bal.asset.as_deref().unwrap_or("");
                let free: f64 = bal
                    .free
                    .as_deref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let locked: f64 = bal
                    .locked
                    .as_deref()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);

                if free > 0.0 || locked > 0.0 {
                    info!(
                        asset = asset,
                        free = %fmt_usd(free),
                        locked = %fmt_usd(locked),
                        "Account balance"
                    );
                }

                if asset == quote_asset {
                    quote_free = free;
                }
                if asset == base_asset {
                    base_free = free;
                }
            }
        }

        Ok((quote_free, base_free))
    }

    /// Rebalance: отменяет все ордера, делит баланс 50/50 между base и quote
    /// маркет-ордером, синхронизирует виртуальный стейт с биржей.
    /// Возвращает true если после ребалансировки обе стороны имеют достаточный баланс.
    pub async fn rebalance(&mut self, symbol: &str) -> Result<bool, anyhow::Error> {
        let quote_asset = self.config.exchange.quote_asset.clone();
        let base_asset = symbol
            .strip_suffix(quote_asset.as_str())
            .unwrap_or(symbol)
            .to_string();

        // 1. Отменяем все открытые ордера по символу
        if let Some(ref connection) = self.connection {
            let cancel_params = OpenOrdersCancelAllParams::builder(symbol.to_string()).build()?;
            match connection.open_orders_cancel_all(cancel_params).await {
                Ok(response) => {
                    let cancelled = response
                        .raw
                        .as_array()
                        .map(|arr| arr.len())
                        .unwrap_or(0);
                    if cancelled > 0 {
                        info!(symbol = symbol, cancelled = cancelled, "Rebalance: cancelled open orders");
                    }
                }
                Err(e) => {
                    let err_str = format!("{:?}", e);
                    if !err_str.contains("-2011") {
                        error!(symbol = symbol, error = %e, "Rebalance: failed to cancel orders");
                    }
                }
            }
        }
        self.active_limit_order = None;
        self.active_bid_order = None;
        self.active_ask_order = None;

        // Небольшая пауза чтобы Binance обработал отмены
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        // 2. Получаем текущие балансы
        let (quote_free, base_free) = self.account_balances(&base_asset).await?;

        // Определяем рыночную цену: last_reference_price из bookTicker (самый надёжный),
        // иначе — HTTP REST API фоллбэк через /api/v3/ticker/price
        let market_price = if self.last_reference_price > f64::EPSILON {
            self.last_reference_price
        } else {
            // Fallback: fetch price from Binance REST API
            let url = format!(
                "https://api.binance.com/api/v3/ticker/price?symbol={}",
                symbol
            );
            match reqwest::get(&url).await {
                Ok(resp) => {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        let price_str = json["price"].as_str().unwrap_or("0");
                        let price: f64 = price_str.parse().unwrap_or(0.0);
                        if price > f64::EPSILON {
                            info!(symbol = symbol, price = %fmt_price(price), "Rebalance: fetched price from REST API");
                            self.last_reference_price = price;
                            price
                        } else {
                            return Err(anyhow::anyhow!("Cannot determine market price for rebalance — REST API returned zero"));
                        }
                    } else {
                        return Err(anyhow::anyhow!("Cannot determine market price for rebalance — REST API response parse error"));
                    }
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Cannot determine market price for rebalance — REST API error: {}", e));
                }
            }
        };

        if market_price <= f64::EPSILON {
            return Err(anyhow::anyhow!("Market price is zero, cannot rebalance"));
        }

        let base_value = base_free * market_price;
        let total_value = quote_free + base_value;
        let min_notional = self.config.exchange.min_notional;

        if total_value <= min_notional * 2.0 {
            return Err(anyhow::anyhow!(
                "Total value {:.4} too small to rebalance (min {:.4})",
                total_value,
                min_notional * 2.0
            ));
        }

        // 3. Считаем дельту до 50/50
        let target_side_value = total_value / 2.0;
        let value_delta = target_side_value - base_value;

        info!(
            symbol = symbol,
            quote_free = %fmt_usd(quote_free),
            base_free = %fmt_price(base_free),
            base_value = %fmt_usd(base_value),
            total_value = %fmt_usd(total_value),
            target_per_side = %fmt_usd(target_side_value),
            delta = %fmt_usd(value_delta),
            market_price = %fmt_price(market_price),
            "Rebalance: current state"
        );

        // Если дельта меньше min_notional — уже сбалансировано
        if value_delta.abs() < min_notional {
            info!(symbol = symbol, "Rebalance: already balanced, skipping");
            // Синхронизируем виртуальный стейт даже если не торгуем
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            self.trade_manager
                .sync_live_balances(symbol, quote_free, base_free, market_price, timestamp);
            return Ok(true);
        }

        // 4. Размещаем маркет-ордер для ребалансировки
        let side = if value_delta > 0.0 { Side::Buy } else { Side::Sell };
        let desired_qty = value_delta.abs() / market_price;
        let step_size = self.config.exchange.step_size.max(f64::EPSILON);
        let rounded_qty = (desired_qty / step_size).floor() * step_size;

        if rounded_qty < step_size {
            return Ok(false);
        }

        let qty_precision = if step_size > 0.0 && step_size < 1.0 {
            (-step_size.log10()).ceil() as usize
        } else {
            0
        };
        let qty_str = format!("{:.prec$}", rounded_qty, prec = qty_precision);
        let quantity_decimal = Decimal::from_str(&qty_str)?;

        info!(
            symbol = symbol,
            side = ?side,
            quantity = %qty_str,
            market_price = %fmt_price(market_price),
            "Rebalance: placing market order"
        );

        let params = OrderPlaceParams::builder(
            symbol.to_string(),
            match side {
                Side::Buy => OrderPlaceSideEnum::Buy,
                Side::Sell => OrderPlaceSideEnum::Sell,
            },
            OrderPlaceTypeEnum::Market,
        )
        .quantity(quantity_decimal)
        .build()?;

        let response = self
            .connection
            .as_ref()
            .unwrap()
            .order_place(params)
            .await
            .map_err(|e| anyhow::anyhow!("Rebalance market order failed: {}", e))?;
        let data = response
            .data()
            .map_err(|e| anyhow::anyhow!("Rebalance response error: {}", e))?;

        let executed_qty: f64 = data
            .executed_qty
            .as_ref()
            .and_then(|q| q.parse().ok())
            .unwrap_or(0.0);
        let status = data.status.as_deref().unwrap_or("UNKNOWN");
        info!(
            symbol = symbol,
            side = ?side,
            executed_qty = %fmt_price(executed_qty),
            status = status,
            "Rebalance: market order complete"
        );

        // 5. Пауза + синхронизация балансов
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let (quote_after, base_after) = self.account_balances(&base_asset).await?;
        let sync_price = if market_price > f64::EPSILON {
            market_price
        } else {
            quote_after / base_after.max(f64::EPSILON)
        };
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.trade_manager
            .sync_live_balances(symbol, quote_after, base_after, sync_price, timestamp);

        info!(
            symbol = symbol,
            quote = %fmt_usd(quote_after),
            base = %fmt_price(base_after),
            "Rebalance: balances synced"
        );

        let base_notional = base_after * sync_price;
        Ok(quote_after >= min_notional && base_notional >= min_notional)
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

    async fn account_status(&self) -> Result<Option<f64>, anyhow::Error> {
        let (usdt_free, _) = self.account_balances("").await?;
        Ok(Some(usdt_free))
    }

    async fn on_order_intent(
        &mut self,
        symbol: &str,
        reference_price: f64,
        intent: OrderIntent,
    ) -> ExecutionReport {
        if self.connection.is_none() {
            return ExecutionReport {
                status: ExecutionStatus::Rejected,
                symbol: None,
                side: None,
                order_type: None,
                rationale: None,
                decision_confidence: 0.0,
                decision_metrics: Vec::new(),
                requested_quantity: 0.0,
                executed_quantity: 0.0,
                execution_price: None,
                fee_paid: 0.0,
                latency_seconds: 0.0,
                synthetic_half_spread_bps: 0.0,
                slippage_bps: 0.0,
                latency_impact_bps: 0.0,
                market_impact_bps: 0.0,
                reason: Some("connection_missing"),
                expected_edge_bps: 0.0,
            };
        }

        // Запоминаем последнюю рыночную цену для rebalance
        if reference_price > f64::EPSILON {
            self.last_reference_price = reference_price;
        }

        match intent {
            OrderIntent::NoAction => ExecutionReport::ignored(),
            OrderIntent::Cancel { rationale } => {
                self.cancel_two_sided_orders().await;
                ExecutionReport {
                    status: ExecutionStatus::Cancelled,
                    symbol: Some(symbol.to_string()),
                    side: None,
                    order_type: None,
                    rationale: Some(rationale),
                    decision_confidence: 0.0,
                    decision_metrics: Vec::new(),
                    requested_quantity: 0.0,
                    executed_quantity: 0.0,
                    execution_price: None,
                    fee_paid: 0.0,
                    latency_seconds: 0.0,
                    synthetic_half_spread_bps: 0.0,
                    slippage_bps: 0.0,
                    latency_impact_bps: 0.0,
                    market_impact_bps: 0.0,
                    reason: Some(rationale),
                    expected_edge_bps: 0.0,
                }
            }
            OrderIntent::Place {
                side,
                order_type,
                price,
                quantity,
                time_in_force,
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
                        symbol: Some(symbol.to_string()),
                        side: Some(side),
                        order_type: Some(order_type),
                        rationale: None,
                        decision_confidence: 0.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: quantity,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: Some("risk_validation_failed"),
                        expected_edge_bps,
                    };
                }

                // Round quantity to step_size
                let step_size = self.config.exchange.step_size;
                let mut rounded_qty = if step_size > 0.0 {
                    (quantity / step_size).floor() * step_size
                } else {
                    quantity
                };

                // When BNB fee discount is active, buy fills lose a tiny fraction of
                // base qty to commission. Adding 1 step_size to buy qty ensures
                // sell qty (after fee deduction + floor rounding) = original target.
                // Without this, e.g. buy 0.010 BNB → effective 0.0099925 → sell floor = 0.009
                // → 10% dust loss per round-trip.
                if matches!(side, Side::Buy)
                    && self.config.exchange.bnb_discount_fee > 0.0
                    && step_size > 0.0
                {
                    rounded_qty += step_size;
                }
                let qty_precision = if step_size > 0.0 && step_size < 1.0 {
                    (-step_size.log10()).ceil() as usize
                } else {
                    0
                };

                if matches!(order_type, OrderType::Maker) {
                    // === REJECTION COOLDOWN + REBALANCE ===
                    // After an exchange rejection (e.g. insufficient balance):
                    // 1. First rejection: trigger rebalance (cancel orders, 50/50 split, sync balances)
                    // 2. If rebalance recovers enough balance: reset and retry immediately
                    // 3. If not: enter 30s cooldown to avoid spamming Binance
                    if self.consecutive_rejections > 0 {
                        let now_millis = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;

                        // On first rejection, attempt rebalance before entering cooldown
                        if self.consecutive_rejections == 1 {
                            info!(
                                symbol = %symbol,
                                "Insufficient balance — triggering rebalance"
                            );
                            let recovered = self.rebalance(symbol).await.unwrap_or(false);
                            if recovered {
                                info!(
                                    symbol = %symbol,
                                    available_cash = format!("{:.4}", self.trade_manager.available_cash()),
                                    "Rebalance succeeded — resuming trading"
                                );
                                self.consecutive_rejections = 0;
                                // Fall through to place order normally
                            } else {
                                info!(
                                    symbol = %symbol,
                                    available_cash = format!("{:.4}", self.trade_manager.available_cash()),
                                    "Rebalance: insufficient funds — entering 30s cooldown"
                                );
                                self.last_rejection_millis = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64;
                                self.consecutive_rejections = 2;
                                return ExecutionReport {
                                    status: ExecutionStatus::Ignored,
                                    symbol: Some(symbol.to_string()),
                                    side: Some(side),
                                    order_type: Some(order_type),
                                    rationale: None,
                                    decision_confidence: 0.0,
                                    decision_metrics: Vec::new(),
                                    requested_quantity: rounded_qty,
                                    executed_quantity: 0.0,
                                    execution_price: None,
                                    fee_paid: 0.0,
                                    latency_seconds: 0.0,
                                    synthetic_half_spread_bps: 0.0,
                                    slippage_bps: 0.0,
                                    latency_impact_bps: 0.0,
                                    market_impact_bps: 0.0,
                                    reason: Some("rebalance_insufficient"),
                                    expected_edge_bps,
                                };
                            }
                        } else {
                            // Subsequent rejections: enforce cooldown
                            let cooldown_ms: u64 = 30_000; // 30 seconds
                            if now_millis.saturating_sub(self.last_rejection_millis) < cooldown_ms {
                                debug!(
                                    symbol = %symbol,
                                    consecutive_rejections = self.consecutive_rejections,
                                    cooldown_remaining_ms = cooldown_ms.saturating_sub(now_millis.saturating_sub(self.last_rejection_millis)),
                                    "Skipping order — rejection cooldown active"
                                );
                                return ExecutionReport {
                                    status: ExecutionStatus::Ignored,
                                    symbol: Some(symbol.to_string()),
                                    side: Some(side),
                                    order_type: Some(order_type),
                                    rationale: None,
                                    decision_confidence: 0.0,
                                    decision_metrics: Vec::new(),
                                    requested_quantity: rounded_qty,
                                    executed_quantity: 0.0,
                                    execution_price: None,
                                    fee_paid: 0.0,
                                    latency_seconds: 0.0,
                                    synthetic_half_spread_bps: 0.0,
                                    slippage_bps: 0.0,
                                    latency_impact_bps: 0.0,
                                    market_impact_bps: 0.0,
                                    reason: Some("rejection_cooldown"),
                                    expected_edge_bps,
                                };
                            }
                            // Cooldown expired — reset and retry
                            self.consecutive_rejections = 0;
                        }
                    }

                    // === LIVE LIMIT ORDER PLACEMENT ===
                    // Place a LIMIT order at the specified price (or reference_price if None).
                    let limit_price = price.unwrap_or(reference_price);

                    // Round price to tick_size (Binance precision rules)
                    let tick_size = self.config.exchange.tick_size;

                    let rounded_price = if tick_size > 0.0 {
                        let ticks = limit_price / tick_size;
                        match side {
                            Side::Buy => ticks.floor() * tick_size,
                            Side::Sell => ticks.ceil() * tick_size,
                        }
                    } else {
                        limit_price
                    };

                    let price_precision = if tick_size > 0.0 && tick_size < 1.0 {
                        (-tick_size.log10()).ceil() as usize
                    } else {
                        0
                    };

                    let qty_str = format!("{:.prec$}", rounded_qty, prec = qty_precision);
                    let price_str = format!("{:.prec$}", rounded_price, prec = price_precision);

                    let quantity_decimal = match Decimal::from_str(&qty_str) {
                        Ok(q) => q,
                        Err(_) => {
                            return ExecutionReport {
                                status: ExecutionStatus::Rejected,
                                symbol: Some(symbol.to_string()),
                                side: Some(side),
                                order_type: Some(order_type),
                                rationale: None,
                                decision_confidence: 0.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: rounded_qty,
                                executed_quantity: 0.0,
                                execution_price: None,
                                fee_paid: 0.0,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: Some("invalid_quantity"),
                                expected_edge_bps,
                            };
                        }
                    };

                    let price_decimal = match Decimal::from_str(&price_str) {
                        Ok(p) => p,
                        Err(_) => {
                            return ExecutionReport {
                                status: ExecutionStatus::Rejected,
                                symbol: Some(symbol.to_string()),
                                side: Some(side),
                                order_type: Some(order_type),
                                rationale: None,
                                decision_confidence: 0.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: rounded_qty,
                                executed_quantity: 0.0,
                                execution_price: None,
                                fee_paid: 0.0,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: Some("invalid_price"),
                                expected_edge_bps,
                            };
                        }
                    };

                    // Skip cancel+replace if we already have a resting order at the same price and side,
                    // UNLESS the order has been resting too long (staleness timeout).
                    // This avoids API spam and the race condition where cancel frees USDT
                    // slower than new order tries to lock it → "insufficient balance".
                    if let Some(ref active) = self.active_limit_order {
                        // CRITICAL: If same price AND same side, NEVER cancel-replace.
                        // Queue priority is THE edge — destroying it by needless
                        // cancel-replace causes adverse selection.
                        if active.side == side
                            && (active.price - rounded_price).abs() < tick_size * 0.5
                        {
                            return ExecutionReport {
                                status: ExecutionStatus::Pending,
                                symbol: Some(symbol.to_string()),
                                side: Some(side),
                                order_type: Some(order_type),
                                rationale: None,
                                decision_confidence: 0.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: rounded_qty,
                                executed_quantity: 0.0,
                                execution_price: Some(active.price),
                                fee_paid: 0.0,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: Some("limit_order_resting"),
                                expected_edge_bps,
                            };
                        }
                    }

                    // Enforce minimum rest time for maker orders.
                    // If the resting order hasn't been open long enough, skip cancel-replace.
                    // This lets passive fills accumulate instead of churning.
                    // Taker exits (stop_loss, panic_vol, max_hold) are NOT affected
                    // because they go through the Taker branch, not Maker.
                    if let Some(ref active) = self.active_limit_order {
                        let now_millis = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                        let resting_ms = now_millis.saturating_sub(active.placed_at_millis);
                        if resting_ms < self.config.order_execution.min_order_rest_millis {
                            debug!(
                                symbol = %symbol,
                                resting_ms = resting_ms,
                                min_rest_ms = self.config.order_execution.min_order_rest_millis,
                                "Skipping cancel-replace: order has not rested long enough"
                            );
                            return ExecutionReport {
                                status: ExecutionStatus::Pending,
                                symbol: Some(symbol.to_string()),
                                side: Some(side),
                                order_type: Some(order_type),
                                rationale: None,
                                decision_confidence: 0.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: rounded_qty,
                                executed_quantity: 0.0,
                                execution_price: Some(active.price),
                                fee_paid: 0.0,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: Some("limit_order_resting"),
                                expected_edge_bps,
                            };
                        }
                    }

                    // Cancel any existing active limit order before placing new one
                    if let Some(ref active) = self.active_limit_order {
                        let cancel_params = OrderCancelParams::builder(active.symbol.clone())
                            .orig_client_order_id(active.client_order_id.clone())
                            .build()
                            .expect("Failed to build cancel params");
                        // Best-effort cancel — ignore errors (order may already be filled)
                        let _ = self.connection.as_ref().unwrap().order_cancel(cancel_params).await;
                        info!(
                            symbol = %active.symbol,
                            order_id = active.order_id,
                            side = ?active.side,
                            price = %fmt_price(active.price),
                            quantity = %fmt_price(active.quantity),
                            "Cancelled previous limit order before placing new one"
                        );
                    }
                    self.active_limit_order = None;

                    let tif = match time_in_force {
                        TimeInForce::Gtc => OrderPlaceTimeInForceEnum::Gtc,
                        TimeInForce::Ioc => OrderPlaceTimeInForceEnum::Ioc,
                        TimeInForce::Fok => OrderPlaceTimeInForceEnum::Fok,
                    };

                    let client_oid = Self::next_client_order_id(&mut self.order_counter);
                    let params = OrderPlaceParams::builder(
                        symbol.to_string(),
                        match side {
                            Side::Buy => OrderPlaceSideEnum::Buy,
                            Side::Sell => OrderPlaceSideEnum::Sell,
                        },
                        OrderPlaceTypeEnum::Limit,
                    )
                    .quantity(quantity_decimal)
                    .price(price_decimal)
                    .time_in_force(tif)
                    .new_client_order_id(client_oid.clone())
                    .build()
                    .expect("Failed to build limit order parameters");

                    let response = match self.connection.as_ref().unwrap().order_place(params).await {
                        Ok(resp) => {
                            // Successful placement — reset rejection state
                            self.consecutive_rejections = 0;
                            resp
                        }
                        Err(e) => {
                            // Track rejection for cooldown
                            self.consecutive_rejections += 1;
                            self.last_rejection_millis = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis() as u64;
                            error!(
                                symbol = %symbol,
                                side = ?side,
                                quantity = %fmt_price(rounded_qty),
                                price = %fmt_price(rounded_price),
                                consecutive_rejections = self.consecutive_rejections,
                                error = %e,
                                "Limit order placement failed — cooldown 30s activated"
                            );
                            return ExecutionReport {
                                status: ExecutionStatus::Rejected,
                                symbol: Some(symbol.to_string()),
                                side: Some(side),
                                order_type: Some(order_type),
                                rationale: None,
                                decision_confidence: 0.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: rounded_qty,
                                executed_quantity: 0.0,
                                execution_price: None,
                                fee_paid: 0.0,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: Some("exchange_limit_order_failed"),
                                expected_edge_bps,
                            };
                        }
                    };

                    let Ok(data) = response.data() else {
                        return ExecutionReport {
                            status: ExecutionStatus::Rejected,
                            symbol: Some(symbol.to_string()),
                            side: Some(side),
                            order_type: Some(order_type),
                            rationale: None,
                            decision_confidence: 0.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: rounded_qty,
                            executed_quantity: 0.0,
                            execution_price: None,
                            fee_paid: 0.0,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: Some("missing_limit_order_data"),
                            expected_edge_bps,
                        };
                    };

                    // Extract order_id for tracking
                    let order_id = data.order_id.unwrap_or(0);

                    // Check if immediately filled
                    let executed_qty = data
                        .executed_qty
                        .as_ref()
                        .and_then(|q| q.parse::<f64>().ok())
                        .unwrap_or(0.0);

                    let status_str = data.status.as_deref().unwrap_or("");

                    if status_str == "FILLED" || (executed_qty > 0.0 && executed_qty >= quantity * 0.99) {
                        // Fully filled immediately (crossed the spread or liquidity was there)
                        let avg_price = data
                            .price
                            .as_ref()
                            .and_then(|p| p.parse::<f64>().ok())
                            .unwrap_or(limit_price);

                        // Calculate fee and detect commission-in-base-asset
                        let quote_asset = &self.config.exchange.quote_asset;
                        let base_asset = symbol.strip_suffix(quote_asset.as_str()).unwrap_or(symbol);
                        let (fee_paid, base_commission) = if let Some(fills) = data.fills.as_ref() {
                            let mut total_fee = 0.0_f64;
                            let mut total_base_commission = 0.0_f64;
                            for f in fills {
                                let comm = f.commission.as_ref()
                                    .and_then(|c| c.parse::<f64>().ok())
                                    .unwrap_or(0.0);
                                let comm_asset = f.commission_asset.as_deref().unwrap_or("");
                                if comm_asset == base_asset {
                                    // Commission charged in base asset — convert to USDT for fee tracking
                                    total_fee += comm * avg_price;
                                    total_base_commission += comm;
                                } else {
                                    // Commission in USDT or other quote asset
                                    total_fee += comm;
                                }
                            }
                            (total_fee, total_base_commission)
                        } else {
                            (executed_qty * avg_price * self.config.exchange.maker_fee, 0.0)
                        };

                        // When Binance charges fee in base asset on Buy, the actual received
                        // quantity is less than executed_qty. Subtract the commission so the
                        // virtual position matches the real balance.
                        let effective_qty = if matches!(side, Side::Buy) && base_commission > 0.0 {
                            let adj = executed_qty - base_commission;
                            info!(
                                symbol = %symbol,
                                raw_qty = %fmt_price(executed_qty),
                                base_commission = %fmt_price(base_commission),
                                effective_qty = %fmt_price(adj),
                                "Adjusted executed_qty for base-asset commission (immediate limit fill)"
                            );
                            adj
                        } else {
                            executed_qty
                        };

                        self.active_limit_order = None;

                        info!(
                            symbol = %symbol,
                            side = ?side,
                            price = %fmt_price(avg_price),
                            quantity = %fmt_price(effective_qty),
                            order_id = order_id,
                            "Limit order filled immediately"
                        );

                        return ExecutionReport {
                            status: ExecutionStatus::Filled,
                            symbol: Some(symbol.to_string()),
                            side: Some(side),
                            order_type: Some(order_type),
                            rationale: None,
                            decision_confidence: 0.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: rounded_qty,
                            executed_quantity: effective_qty,
                            execution_price: Some(avg_price),
                            fee_paid,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: None,
                            expected_edge_bps,
                        };
                    }

                    // Order is resting on the book — track it
                    self.active_limit_order = Some(ActiveLimitOrder {
                        order_id,
                        client_order_id: client_oid.clone(),
                        symbol: symbol.to_string(),
                        side,
                        price: rounded_price,
                        quantity: rounded_qty,
                        placed_at_millis: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64,
                    });

                    info!(
                        symbol = %symbol,
                        side = ?side,
                        price = %fmt_price(rounded_price),
                        quantity = %fmt_price(rounded_qty),
                        order_id = order_id,
                        client_order_id = %client_oid,
                        status = %status_str,
                        "Limit order placed and resting"
                    );

                    return ExecutionReport {
                        status: ExecutionStatus::Pending,
                        symbol: Some(symbol.to_string()),
                        side: Some(side),
                        order_type: Some(order_type),
                        rationale: None,
                        decision_confidence: 0.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: rounded_qty,
                        executed_quantity: 0.0,
                        execution_price: Some(rounded_price),
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: Some("limit_order_resting"),
                        expected_edge_bps,
                    };
                }

                // === TAKER (MARKET ORDER) ===

                // --- Taker retry guard ---
                // After MAX_TAKER_RETRIES consecutive failures, the virtual position
                // is force-closed because the on-exchange balance is likely zero
                // (e.g. BNB fee discount ate into base qty, or cleanup sold inventory).
                const MAX_TAKER_RETRIES: u32 = 3;
                const TAKER_RETRY_COOLDOWN_MS: u64 = 5_000; // 5 seconds between retries

                if self.consecutive_taker_failures >= MAX_TAKER_RETRIES {
                    error!(
                        symbol = %symbol,
                        side = ?side,
                        failures = self.consecutive_taker_failures,
                        "Taker order failed {} times — force-closing virtual position",
                        MAX_TAKER_RETRIES,
                    );
                    self.consecutive_taker_failures = 0;
                    self.last_taker_failure_millis = 0;
                    // Return a synthetic "filled" report at zero qty so the main loop
                    // does NOT try to close_position_with_report (executed_quantity==0).
                    // The position will be force-closed by the caller via the
                    // "force_close_position" reason.
                    return ExecutionReport {
                        status: ExecutionStatus::Rejected,
                        symbol: Some(symbol.to_string()),
                        side: Some(side),
                        order_type: Some(order_type),
                        rationale: None,
                        decision_confidence: 0.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: rounded_qty,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
                        reason: Some("force_close_position"),
                        expected_edge_bps,
                    };
                }

                if self.consecutive_taker_failures > 0 {
                    let now_millis = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    let elapsed = now_millis.saturating_sub(self.last_taker_failure_millis);
                    if elapsed < TAKER_RETRY_COOLDOWN_MS {
                        debug!(
                            symbol = %symbol,
                            failures = self.consecutive_taker_failures,
                            cooldown_remaining_ms = TAKER_RETRY_COOLDOWN_MS - elapsed,
                            "Taker retry cooldown active — skipping"
                        );
                        return ExecutionReport {
                            status: ExecutionStatus::Ignored,
                            symbol: Some(symbol.to_string()),
                            side: Some(side),
                            order_type: Some(order_type),
                            rationale: None,
                            decision_confidence: 0.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: rounded_qty,
                            executed_quantity: 0.0,
                            execution_price: None,
                            fee_paid: 0.0,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: Some("taker_retry_cooldown"),
                            expected_edge_bps,
                        };
                    }
                }

                // Cancel any resting limit order before placing market order.
                // Without this, the resting order locks the base asset balance
                // and the market order fails with "insufficient balance".
                if let Some(ref active) = self.active_limit_order {
                    let cancel_params = OrderCancelParams::builder(active.symbol.clone())
                        .orig_client_order_id(active.client_order_id.clone())
                        .build()
                        .expect("Failed to build cancel params");
                    // Best-effort cancel — ignore errors (order may already be filled)
                    let _ = self
                        .connection
                        .as_ref()
                        .unwrap()
                        .order_cancel(cancel_params)
                        .await;
                    info!(
                        symbol = %active.symbol,
                        order_id = active.order_id,
                        side = ?active.side,
                        price = %fmt_price(active.price),
                        quantity = %fmt_price(active.quantity),
                        "Cancelled resting limit order before taker exit"
                    );
                }
                self.active_limit_order = None;
                // Also cancel any two-sided orders before taker exit
                self.cancel_two_sided_orders().await;

                let qty_str = format!("{:.prec$}", rounded_qty, prec = qty_precision);

                let quantity_decimal = match Decimal::from_str(&qty_str) {
                    Ok(quantity) => quantity,
                    Err(_) => {
                        return ExecutionReport {
                            status: ExecutionStatus::Rejected,
                            symbol: Some(symbol.to_string()),
                            side: Some(side),
                            order_type: Some(order_type),
                            rationale: None,
                            decision_confidence: 0.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: rounded_qty,
                            executed_quantity: 0.0,
                            execution_price: None,
                            fee_paid: 0.0,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
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

                let response = match self.connection.as_ref().unwrap().order_place(params).await {
                    Ok(resp) => resp,
                    Err(e) => {
                        self.consecutive_taker_failures += 1;
                        self.last_taker_failure_millis = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                        error!(
                            symbol = %symbol,
                            side = ?side,
                            quantity = %fmt_price(rounded_qty),
                            error = %e,
                            attempt = self.consecutive_taker_failures,
                            max_retries = MAX_TAKER_RETRIES,
                            "Taker (market) order placement failed"
                        );
                        return ExecutionReport {
                            status: ExecutionStatus::Rejected,
                            symbol: Some(symbol.to_string()),
                            side: Some(side),
                            order_type: Some(order_type),
                            rationale: None,
                            decision_confidence: 0.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: rounded_qty,
                            executed_quantity: 0.0,
                            execution_price: None,
                            fee_paid: 0.0,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: Some("exchange_order_failed"),
                            expected_edge_bps,
                        };
                    }
                };

                let data = match response.data() {
                    Ok(d) => d,
                    Err(e) => {
                        error!(
                            symbol = %symbol,
                            side = ?side,
                            error = %e,
                            "Taker order response missing data"
                        );
                        return ExecutionReport {
                            status: ExecutionStatus::Rejected,
                            symbol: Some(symbol.to_string()),
                            side: Some(side),
                            order_type: Some(order_type),
                            rationale: None,
                            decision_confidence: 0.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: rounded_qty,
                            executed_quantity: 0.0,
                            execution_price: None,
                            fee_paid: 0.0,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: Some("missing_order_data"),
                            expected_edge_bps,
                        };
                    }
                };

                let Some(fills) = data.fills.as_ref() else {
                    return ExecutionReport {
                        status: ExecutionStatus::Rejected,
                        symbol: Some(symbol.to_string()),
                        side: Some(side),
                        order_type: Some(order_type),
                        rationale: None,
                        decision_confidence: 0.0,
                        decision_metrics: Vec::new(),
                        requested_quantity: rounded_qty,
                        executed_quantity: 0.0,
                        execution_price: None,
                        fee_paid: 0.0,
                        latency_seconds: 0.0,
                        synthetic_half_spread_bps: 0.0,
                        slippage_bps: 0.0,
                        latency_impact_bps: 0.0,
                        market_impact_bps: 0.0,
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

                // Calculate fee and detect commission-in-base-asset
                let quote_asset = &self.config.exchange.quote_asset;
                let base_asset = symbol.strip_suffix(quote_asset.as_str()).unwrap_or(symbol);
                let (fee_paid, base_commission) = {
                    let mut total_fee = 0.0_f64;
                    let mut total_base_commission = 0.0_f64;
                    let fallback_price = execution_price.unwrap_or(0.0);
                    for f in fills {
                        let comm = f.commission.as_ref()
                            .and_then(|c| c.parse::<f64>().ok())
                            .unwrap_or(0.0);
                        let comm_asset = f.commission_asset.as_deref().unwrap_or("");
                        if comm_asset == base_asset {
                            total_fee += comm * fallback_price;
                            total_base_commission += comm;
                        } else {
                            total_fee += comm;
                        }
                    }
                    // Fallback if no fills had commission info
                    if total_fee == 0.0 {
                        total_fee = execution_price
                            .map(|p| p * executed_quantity * self.config.exchange.taker_fee)
                            .unwrap_or(0.0);
                    }
                    (total_fee, total_base_commission)
                };

                // When Binance charges fee in base asset on Buy, reduce effective qty
                let effective_qty = if matches!(side, Side::Buy) && base_commission > 0.0 {
                    let adj = executed_quantity - base_commission;
                    info!(
                        symbol = %symbol,
                        raw_qty = %fmt_price(executed_quantity),
                        base_commission = %fmt_price(base_commission),
                        effective_qty = %fmt_price(adj),
                        "Adjusted executed_qty for base-asset commission (taker fill)"
                    );
                    adj
                } else {
                    executed_quantity
                };

                // Taker order succeeded — reset failure counter
                self.consecutive_taker_failures = 0;

                ExecutionReport {
                    status: ExecutionStatus::Filled,
                    symbol: Some(symbol.to_string()),
                    side: Some(side),
                    order_type: Some(order_type),
                    rationale: None,
                    decision_confidence: 0.0,
                    decision_metrics: Vec::new(),
                    requested_quantity: rounded_qty,
                    executed_quantity: effective_qty,
                    execution_price,
                    fee_paid,
                    latency_seconds: 0.0,
                    synthetic_half_spread_bps: 0.0,
                    slippage_bps: 0.0,
                    latency_impact_bps: 0.0,
                    market_impact_bps: 0.0,
                    reason: None,
                    expected_edge_bps,
                }
            }
            // Two-sided market-making: place/replace both bid and ask limit orders.
            OrderIntent::QuoteBothSides {
                buy_price,
                buy_quantity,
                sell_price,
                sell_quantity,
                expected_edge_bps,
                ..
            } => {
                let trading_symbol = symbol.to_string();
                let event_time_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                let now_millis = (event_time_secs * 1000.0) as u64;
                let instant_fill_cooldown_ms = self.config.order_execution.min_order_rest_millis;

                // When BOTH sides are empty (no resting orders), we MUST re-enter
                // the market immediately — bypass instant-fill cooldown.
                // Otherwise the bot goes dark after both sides fill.
                let both_sides_empty = self.active_bid_order.is_none()
                    && self.active_ask_order.is_none();

                // Place or replace bid side (with instant-fill cooldown)
                let bid_in_cooldown = !both_sides_empty
                    && self.last_bid_instant_fill_millis > 0
                    && now_millis.saturating_sub(self.last_bid_instant_fill_millis)
                        < instant_fill_cooldown_ms;
                let existing_bid = self.active_bid_order.take();
                if bid_in_cooldown && existing_bid.is_none() {
                    // Skip bid placement — still in cooldown after instant fill
                    self.active_bid_order = None;
                } else {
                let bid_result = self
                    .place_or_replace_side(
                        symbol,
                        Side::Buy,
                        buy_price,
                        buy_quantity,
                        &existing_bid,
                    )
                    .await;
                match bid_result {
                    PlaceResult::Resting(order) => {
                        self.active_bid_order = Some(order);
                    }
                    PlaceResult::InstantFill { avg_price, executed_qty, fee } => {
                        // BUY filled immediately — register in position tracker
                        let effective_fee_rate = self
                            .config
                            .exchange
                            .maker_fee
                            .max(self.config.exchange.bnb_discount_fee);
                        let effective_qty = executed_qty * (1.0 - effective_fee_rate);
                        info!(
                            symbol = %trading_symbol,
                            side = "BUY",
                            price = %fmt_price(avg_price),
                            effective_qty = %fmt_price(effective_qty),
                            fee = %fmt_price(fee),
                            "Two-sided: immediate BID fill registered"
                        );
                        let report = ExecutionReport {
                            status: ExecutionStatus::Filled,
                            symbol: Some(trading_symbol.clone()),
                            side: Some(Side::Buy),
                            order_type: Some(OrderType::Maker),
                            rationale: Some("two_sided_bid_instant_fill"),
                            decision_confidence: 1.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: buy_quantity,
                            executed_quantity: effective_qty,
                            execution_price: Some(avg_price),
                            fee_paid: fee,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: None,
                            expected_edge_bps: 0.0,
                        };
                        self.trade_manager.record_execution_report(&report);
                        let _ = self.trade_manager.open_position(
                            &trading_symbol,
                            avg_price,
                            effective_qty,
                            event_time_secs,
                            Some(&report),
                        );
                        self.active_bid_order = None;
                        self.last_bid_instant_fill_millis = now_millis;
                        self.last_buy_fill_price = avg_price;
                    }
                    PlaceResult::Failed => {
                        self.active_bid_order = None;
                    }
                }
                } // end bid cooldown else

                // Place or replace ask side (with instant-fill cooldown)
                // Re-check both_sides_empty after bid placement (bid may have
                // been placed successfully, making it no longer empty)
                let both_sides_empty_for_ask = self.active_bid_order.is_none()
                    && self.active_ask_order.is_none();
                let ask_in_cooldown = !both_sides_empty_for_ask
                    && self.last_ask_instant_fill_millis > 0
                    && now_millis.saturating_sub(self.last_ask_instant_fill_millis)
                        < instant_fill_cooldown_ms;
                let existing_ask = self.active_ask_order.take();
                if ask_in_cooldown && existing_ask.is_none() {
                    // Skip ask placement — still in cooldown after instant fill
                    self.active_ask_order = None;
                } else {
                let ask_result = self
                    .place_or_replace_side(
                        symbol,
                        Side::Sell,
                        sell_price,
                        sell_quantity,
                        &existing_ask,
                    )
                    .await;
                match ask_result {
                    PlaceResult::Resting(order) => {
                        self.active_ask_order = Some(order);
                    }
                    PlaceResult::InstantFill { avg_price, executed_qty, fee } => {
                        // SELL filled immediately — close position
                        info!(
                            symbol = %trading_symbol,
                            side = "SELL",
                            price = %fmt_price(avg_price),
                            quantity = %fmt_price(executed_qty),
                            fee = %fmt_price(fee),
                            "Two-sided: immediate ASK fill registered"
                        );
                        let report = ExecutionReport {
                            status: ExecutionStatus::Filled,
                            symbol: Some(trading_symbol.clone()),
                            side: Some(Side::Sell),
                            order_type: Some(OrderType::Maker),
                            rationale: Some("two_sided_ask_instant_fill"),
                            decision_confidence: 1.0,
                            decision_metrics: Vec::new(),
                            requested_quantity: sell_quantity,
                            executed_quantity: executed_qty,
                            execution_price: Some(avg_price),
                            fee_paid: fee,
                            latency_seconds: 0.0,
                            synthetic_half_spread_bps: 0.0,
                            slippage_bps: 0.0,
                            latency_impact_bps: 0.0,
                            market_impact_bps: 0.0,
                            reason: None,
                            expected_edge_bps: 0.0,
                        };
                        self.trade_manager.record_execution_report(&report);
                        // Partial close: reduce position by the sold quantity
                        // instead of removing the entire position.
                        let current_qty = self
                            .trade_manager
                            .get_position(&trading_symbol)
                            .map_or(0.0, |p| p.quantity);
                        let remaining = (current_qty - executed_qty).max(0.0);
                        self.trade_manager.update_position(
                            &trading_symbol,
                            remaining,
                            avg_price,
                            event_time_secs,
                            Some(fee),
                        );
                        self.active_ask_order = None;
                        self.last_ask_instant_fill_millis = now_millis;
                        self.last_sell_fill_price = avg_price;
                    }
                    PlaceResult::Failed => {
                        self.active_ask_order = None;
                    }
                }
                } // end ask cooldown else

                // === TWO-SIDED STATE LOG (change-detection for both branches) ===
                let bid_active = self.active_bid_order.is_some();
                let ask_active = self.active_ask_order.is_some();
                if !bid_active || !ask_active {
                    let new_state = (bid_active, ask_active, 0.0, 0.0, 0.0, 0.0);
                    if new_state != self.last_two_sided_state {
                        self.last_two_sided_state = new_state;
                        warn!(
                            symbol = %trading_symbol,
                            bid_active = bid_active,
                            ask_active = ask_active,
                            buy_qty_requested = %fmt_price(buy_quantity),
                            sell_qty_requested = %fmt_price(sell_quantity),
                            "Two-sided: NOT both sides active after placement"
                        );
                    }
                } else {
                    let bp = self.active_bid_order.as_ref().unwrap().price;
                    let bq = self.active_bid_order.as_ref().unwrap().quantity;
                    let ap = self.active_ask_order.as_ref().unwrap().price;
                    let aq = self.active_ask_order.as_ref().unwrap().quantity;
                    let new_state = (true, true, bp, bq, ap, aq);
                    if new_state != self.last_two_sided_state {
                        self.last_two_sided_state = new_state;
                        info!(
                            symbol = %trading_symbol,
                            bid_price = %fmt_price(bp),
                            bid_qty = %fmt_price(bq),
                            ask_price = %fmt_price(ap),
                            ask_qty = %fmt_price(aq),
                            "Two-sided: both sides active"
                        );
                    }
                }

                // === TWO-SIDED REBALANCE ===
                // Track ticks where the strategy returned qty=0 for BOTH sides
                // (completely unable to quote). After threshold, auto-rebalance.
                // Note: one-sided quoting (e.g. trend filter blocking buys) is
                // intentional and should NOT trigger rebalance.
                let buy_zero = buy_quantity < f64::EPSILON;
                let sell_zero = sell_quantity < f64::EPSILON;
                if buy_zero && sell_zero {
                    self.consecutive_zero_qty_ticks += 1;
                } else {
                    self.consecutive_zero_qty_ticks = 0;
                }

                // Trigger rebalance if:
                // (a) either side FAILED placement (order rejected / insufficient balance)
                //     AND at least 3 consecutive failures (avoids rebalance on transient cancel-replace errors)
                //     AND at least 30s since last rebalance
                // (b) strategy returned qty=0 for one side for 10+ consecutive ticks
                //     (inventory exhausted, A-S scaling drove qty to zero)
                const ZERO_QTY_REBALANCE_THRESHOLD: u32 = 100;
                const MIN_REBALANCE_INTERVAL_MS: u64 = 30_000; // 30 seconds
                const FAILURE_REBALANCE_THRESHOLD: u32 = 3;
                let bid_failed = self.active_bid_order.is_none() && buy_quantity > f64::EPSILON;
                let ask_failed = self.active_ask_order.is_none() && sell_quantity > f64::EPSILON;
                let zero_qty_trigger = self.consecutive_zero_qty_ticks >= ZERO_QTY_REBALANCE_THRESHOLD;

                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let since_last_rebalance = now_ms.saturating_sub(self.last_rebalance_millis);
                let rebalance_cooldown_ok = self.last_rebalance_millis == 0 || since_last_rebalance >= MIN_REBALANCE_INTERVAL_MS;

                let failure_trigger = (bid_failed || ask_failed) && self.consecutive_two_sided_failures >= FAILURE_REBALANCE_THRESHOLD && rebalance_cooldown_ok;
                let should_rebalance = failure_trigger || (zero_qty_trigger && rebalance_cooldown_ok);

                if should_rebalance {
                    info!(
                        symbol = %trading_symbol,
                        bid_failed = bid_failed,
                        ask_failed = ask_failed,
                        consecutive_failures = self.consecutive_two_sided_failures,
                        zero_qty_ticks = self.consecutive_zero_qty_ticks,
                        since_last_rebalance_ms = since_last_rebalance,
                        "Two-sided: triggering rebalance (placement failure or inventory exhausted)"
                    );
                    let recovered = self.rebalance(symbol).await.unwrap_or(false);
                    self.last_rebalance_millis = now_ms;
                    if recovered {
                        info!(
                            symbol = %trading_symbol,
                            available_cash = format!("{:.4}", self.trade_manager.available_cash()),
                            "Two-sided: rebalance succeeded — resuming trading"
                        );
                        self.consecutive_two_sided_failures = 0;
                        self.last_two_sided_failure_millis = 0;
                        self.consecutive_zero_qty_ticks = 0;
                    } else {
                        info!(
                            symbol = %trading_symbol,
                            available_cash = format!("{:.4}", self.trade_manager.available_cash()),
                            "Two-sided: rebalance insufficient — entering cooldown"
                        );
                        self.consecutive_two_sided_failures = 0; // reset so we accumulate fresh
                        self.consecutive_zero_qty_ticks = 0;
                    }
                }

                // Return a summary report
                ExecutionReport {
                    status: ExecutionStatus::Pending,
                    symbol: Some(symbol.to_string()),
                    side: None,
                    order_type: Some(OrderType::Maker),
                    rationale: Some("two_sided_quote"),
                    decision_confidence: 1.0,
                    decision_metrics: Vec::new(),
                    requested_quantity: buy_quantity + sell_quantity,
                    executed_quantity: 0.0,
                    execution_price: None,
                    fee_paid: 0.0,
                    latency_seconds: 0.0,
                    synthetic_half_spread_bps: 0.0,
                    slippage_bps: 0.0,
                    latency_impact_bps: 0.0,
                    market_impact_bps: 0.0,
                    reason: Some("two_sided_orders_placed"),
                    expected_edge_bps,
                }
            }
            OrderIntent::QuoteLadder {
                bids,
                asks,
                expected_edge_bps,
                ..
            } => {
                let trading_symbol = symbol.to_string();
                let event_time_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();

                // ═══════════════════════════════════════════════════════════
                // Multi-level ladder: diff desired levels vs existing orders.
                //
                // Algorithm per side:
                //   1. Index existing orders by price → HashMap<price, order>
                //   2. For each desired (price, qty):
                //      a. If existing order at same price & qty → keep (skip)
                //      b. If existing at same price, different qty → cancel-replace
                //      c. If no existing at this price → place new
                //   3. Cancel any existing orders NOT in the desired set
                //
                // This minimizes exchange API calls and preserves queue priority
                // for unchanged levels.
                // ═══════════════════════════════════════════════════════════

                // --- BID side ---
                let existing_bids = std::mem::take(&mut self.active_ladder_bids);
                let mut new_active_bids: Vec<ActiveLimitOrder> = Vec::with_capacity(bids.len());

                // Index existing bids by rounded price for matching
                let tick = self.config.exchange.tick_size.max(f64::EPSILON);
                let mut existing_bid_map: std::collections::HashMap<i64, ActiveLimitOrder> =
                    std::collections::HashMap::new();
                for order in existing_bids {
                    let price_ticks = (order.price / tick).round() as i64;
                    existing_bid_map.insert(price_ticks, order);
                }

                for &(desired_price, desired_qty) in &bids {
                    if desired_qty < f64::EPSILON || desired_price < f64::EPSILON {
                        continue;
                    }
                    let price_ticks = (desired_price / tick).round() as i64;
                    let existing = existing_bid_map.remove(&price_ticks);

                    let result = self
                        .place_or_replace_side(
                            symbol,
                            Side::Buy,
                            desired_price,
                            desired_qty,
                            &existing,
                        )
                        .await;
                    match result {
                        PlaceResult::Resting(order) => {
                            new_active_bids.push(order);
                        }
                        PlaceResult::InstantFill { avg_price, executed_qty, fee } => {
                            let effective_fee_rate = self
                                .config
                                .exchange
                                .maker_fee
                                .max(self.config.exchange.bnb_discount_fee);
                            let effective_qty = executed_qty * (1.0 - effective_fee_rate);
                            info!(
                                symbol = %trading_symbol,
                                side = "BUY",
                                price = %fmt_price(avg_price),
                                level_price = %fmt_price(desired_price),
                                effective_qty = %fmt_price(effective_qty),
                                fee = %fmt_price(fee),
                                "Ladder: instant BID fill"
                            );
                            let report = ExecutionReport {
                                status: ExecutionStatus::Filled,
                                symbol: Some(trading_symbol.clone()),
                                side: Some(Side::Buy),
                                order_type: Some(OrderType::Maker),
                                rationale: Some("ladder_bid_instant_fill"),
                                decision_confidence: 1.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: desired_qty,
                                executed_quantity: effective_qty,
                                execution_price: Some(avg_price),
                                fee_paid: fee,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: None,
                                expected_edge_bps: 0.0,
                            };
                            self.trade_manager.record_execution_report(&report);
                            let _ = self.trade_manager.open_position(
                                &trading_symbol,
                                avg_price,
                                effective_qty,
                                event_time_secs,
                                Some(&report),
                            );
                            self.last_buy_fill_price = avg_price;
                        }
                        PlaceResult::Failed => {}
                    }
                }

                // Cancel stale bids (existing orders not in desired set)
                for (_price_ticks, stale_order) in existing_bid_map {
                    let cancel_params = OrderCancelParams::builder(stale_order.symbol.clone())
                        .orig_client_order_id(stale_order.client_order_id.clone())
                        .build()
                        .expect("Failed to build cancel params");
                    let _ = self
                        .connection
                        .as_ref()
                        .unwrap()
                        .order_cancel(cancel_params)
                        .await;
                    debug!(
                        symbol = %stale_order.symbol,
                        price = %fmt_price(stale_order.price),
                        side = "BUY",
                        "Ladder: cancelled stale bid level"
                    );
                }

                self.active_ladder_bids = new_active_bids;

                // --- ASK side ---
                let existing_asks = std::mem::take(&mut self.active_ladder_asks);
                let mut new_active_asks: Vec<ActiveLimitOrder> = Vec::with_capacity(asks.len());

                let mut existing_ask_map: std::collections::HashMap<i64, ActiveLimitOrder> =
                    std::collections::HashMap::new();
                for order in existing_asks {
                    let price_ticks = (order.price / tick).round() as i64;
                    existing_ask_map.insert(price_ticks, order);
                }

                for &(desired_price, desired_qty) in &asks {
                    if desired_qty < f64::EPSILON || desired_price < f64::EPSILON {
                        continue;
                    }
                    let price_ticks = (desired_price / tick).round() as i64;
                    let existing = existing_ask_map.remove(&price_ticks);

                    let result = self
                        .place_or_replace_side(
                            symbol,
                            Side::Sell,
                            desired_price,
                            desired_qty,
                            &existing,
                        )
                        .await;
                    match result {
                        PlaceResult::Resting(order) => {
                            new_active_asks.push(order);
                        }
                        PlaceResult::InstantFill { avg_price, executed_qty, fee } => {
                            info!(
                                symbol = %trading_symbol,
                                side = "SELL",
                                price = %fmt_price(avg_price),
                                level_price = %fmt_price(desired_price),
                                quantity = %fmt_price(executed_qty),
                                fee = %fmt_price(fee),
                                "Ladder: instant ASK fill"
                            );
                            let report = ExecutionReport {
                                status: ExecutionStatus::Filled,
                                symbol: Some(trading_symbol.clone()),
                                side: Some(Side::Sell),
                                order_type: Some(OrderType::Maker),
                                rationale: Some("ladder_ask_instant_fill"),
                                decision_confidence: 1.0,
                                decision_metrics: Vec::new(),
                                requested_quantity: desired_qty,
                                executed_quantity: executed_qty,
                                execution_price: Some(avg_price),
                                fee_paid: fee,
                                latency_seconds: 0.0,
                                synthetic_half_spread_bps: 0.0,
                                slippage_bps: 0.0,
                                latency_impact_bps: 0.0,
                                market_impact_bps: 0.0,
                                reason: None,
                                expected_edge_bps: 0.0,
                            };
                            self.trade_manager.record_execution_report(&report);
                            let current_qty = self
                                .trade_manager
                                .get_position(&trading_symbol)
                                .map_or(0.0, |p| p.quantity);
                            let remaining = (current_qty - executed_qty).max(0.0);
                            self.trade_manager.update_position(
                                &trading_symbol,
                                remaining,
                                avg_price,
                                event_time_secs,
                                Some(fee),
                            );
                            self.last_sell_fill_price = avg_price;
                        }
                        PlaceResult::Failed => {}
                    }
                }

                // Cancel stale asks
                for (_price_ticks, stale_order) in existing_ask_map {
                    let cancel_params = OrderCancelParams::builder(stale_order.symbol.clone())
                        .orig_client_order_id(stale_order.client_order_id.clone())
                        .build()
                        .expect("Failed to build cancel params");
                    let _ = self
                        .connection
                        .as_ref()
                        .unwrap()
                        .order_cancel(cancel_params)
                        .await;
                    debug!(
                        symbol = %stale_order.symbol,
                        price = %fmt_price(stale_order.price),
                        side = "SELL",
                        "Ladder: cancelled stale ask level"
                    );
                }

                self.active_ladder_asks = new_active_asks;

                // === LADDER STATE LOG (change-detection) ===
                let n_bids = self.active_ladder_bids.len();
                let n_asks = self.active_ladder_asks.len();
                let total_bid_qty: f64 = self.active_ladder_bids.iter().map(|o| o.quantity).sum();
                let total_ask_qty: f64 = self.active_ladder_asks.iter().map(|o| o.quantity).sum();
                let new_state = (n_bids, n_asks, total_bid_qty, total_ask_qty);
                if new_state != self.last_ladder_state {
                    self.last_ladder_state = new_state;
                    info!(
                        symbol = %trading_symbol,
                        bid_levels = n_bids,
                        ask_levels = n_asks,
                        total_bid_qty = %fmt_price(total_bid_qty),
                        total_ask_qty = %fmt_price(total_ask_qty),
                        "Ladder: levels updated"
                    );
                }

                // === LADDER REBALANCE TRIGGER ===
                let all_bids_zero = bids.iter().all(|(_, q)| *q < f64::EPSILON);
                let all_asks_zero = asks.iter().all(|(_, q)| *q < f64::EPSILON);
                if all_bids_zero && all_asks_zero {
                    self.consecutive_zero_qty_ticks += 1;
                } else {
                    self.consecutive_zero_qty_ticks = 0;
                }

                const ZERO_QTY_REBALANCE_THRESHOLD: u32 = 100;
                const MIN_REBALANCE_INTERVAL_MS: u64 = 30_000;
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                let since_last_rebalance = now_ms.saturating_sub(self.last_rebalance_millis);
                let rebalance_cooldown_ok = self.last_rebalance_millis == 0 || since_last_rebalance >= MIN_REBALANCE_INTERVAL_MS;

                if self.consecutive_zero_qty_ticks >= ZERO_QTY_REBALANCE_THRESHOLD && rebalance_cooldown_ok {
                    info!(
                        symbol = %trading_symbol,
                        zero_qty_ticks = self.consecutive_zero_qty_ticks,
                        "Ladder: triggering rebalance (both sides empty)"
                    );
                    let recovered = self.rebalance(symbol).await.unwrap_or(false);
                    self.last_rebalance_millis = now_ms;
                    if recovered {
                        info!(
                            symbol = %trading_symbol,
                            available_cash = format!("{:.4}", self.trade_manager.available_cash()),
                            "Ladder: rebalance succeeded"
                        );
                        self.consecutive_zero_qty_ticks = 0;
                    } else {
                        info!(
                            symbol = %trading_symbol,
                            "Ladder: rebalance insufficient"
                        );
                        self.consecutive_zero_qty_ticks = 0;
                    }
                }

                ExecutionReport {
                    status: ExecutionStatus::Pending,
                    symbol: Some(symbol.to_string()),
                    side: None,
                    order_type: Some(OrderType::Maker),
                    rationale: Some("ladder_quote"),
                    decision_confidence: 1.0,
                    decision_metrics: Vec::new(),
                    requested_quantity: bids.iter().map(|(_, q)| q).sum::<f64>() + asks.iter().map(|(_, q)| q).sum::<f64>(),
                    executed_quantity: 0.0,
                    execution_price: None,
                    fee_paid: 0.0,
                    latency_seconds: 0.0,
                    synthetic_half_spread_bps: 0.0,
                    slippage_bps: 0.0,
                    latency_impact_bps: 0.0,
                    market_impact_bps: 0.0,
                    reason: Some("ladder_orders_placed"),
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
        market_data_source: MarketDataSourceKind,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let strategy_info = trading_strategy.get_info();
        let preserve_startup_inventory = strategy_info.contains("continuous bid/ask");

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

        // ── Cancel all open orders on startup to clear stale orders from previous sessions ──
        if trading_mode == TradeMode::Real {
            if let Some(ref connection) = self.connection {
                info!(
                    symbol = trading_symbol,
                    "Cancelling all open orders on startup"
                );
                let cancel_params =
                    OpenOrdersCancelAllParams::builder(trading_symbol.to_string()).build()?;
                match connection.open_orders_cancel_all(cancel_params).await {
                    Ok(response) => {
                        let cancelled_count = response
                            .raw
                            .as_array()
                            .map(|arr| arr.len())
                            .unwrap_or(0);
                        info!(
                            symbol = trading_symbol,
                            cancelled_count = cancelled_count,
                            "Open orders cancelled on startup"
                        );
                    }
                    Err(e) => {
                        // Error code -2011 means "no open orders" — not a real error
                        let err_str = format!("{:?}", e);
                        if err_str.contains("-2011") {
                            info!(
                                symbol = trading_symbol,
                                "No open orders to cancel on startup"
                            );
                        } else {
                            error!(
                                symbol = trading_symbol,
                                error = %e,
                                "Failed to cancel open orders on startup"
                            );
                        }
                    }
                }
                // Small delay to let Binance unlock balances after cancellation
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }

        // ── Live startup: sync real balances into virtual state ──
        if trading_mode == TradeMode::Real {
            let quote_asset = self.config.exchange.quote_asset.clone();
            let base_asset = trading_symbol.strip_suffix(quote_asset.as_str()).unwrap_or(trading_symbol);

            match self.account_balances(base_asset).await {
                Ok((quote_free_before, base_free_before)) => {
                    // Do NOT compute startup reference price from balance ratio
                    // (quote_free / base_free) — this gives a meaningless number
                    // (e.g. 66 FDUSD / 79 DOGE = 0.835 when real price ≈ 0.092).
                    // Use 0.0 so sync_live_balances records balances without creating
                    // a phantom position at a bogus entry price.
                    // last_reference_price will be set correctly on first bookTicker.
                    let startup_reference_price = 0.0;
                    let event_time_secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    self.trade_manager.sync_live_balances(
                        trading_symbol,
                        quote_free_before,
                        base_free_before,
                        startup_reference_price,
                        event_time_secs,
                    );

                    info!(
                        symbol = trading_symbol,
                        quote_asset = %quote_asset,
                        base_asset = base_asset,
                        quote_free = %fmt_usd(quote_free_before),
                        base_free = %fmt_price(base_free_before),
                        reference_price = %fmt_price(startup_reference_price),
                        preserve_inventory = preserve_startup_inventory,
                        "Startup balance baseline synced"
                    );
                }
                Err(e) => {
                    error!(
                        symbol = trading_symbol,
                        error = %e,
                        "Balance check failed on startup — continuing anyway"
                    );
                }
            }
        }

        self.logger
            .log_market_data_source(
                trading_symbol,
                market_data_source.source(),
                market_data_source.status(),
            );

        tokio::pin!(trading_stream);

        let backtest_engine = BacktestEngine::new(self.config.clone());
        let mut market_state =
            MarketState::new(trading_symbol.to_string(), trading_strategy.market_state_window_millis());
        let mut last_trade_price = None;
        // Cumulative passive tracker: counts consecutive ticks where strategy
        // wants a Maker order on the same side. Used to boost fill probability
        // via cumulative formula: 1 - (1-p)^N.
        let mut cumulative_tracker = CumulativePassiveTracker::new();

        while let Some(event) = trading_stream.next().await {
            self.trade_manager.increment_ticks();
            market_state.apply(&event);
            trading_strategy.on_event(&event, &market_state).await;

            let Some(market_price) = self.market_price_from_state(&market_state) else {
                continue;
            };
            let decision_reference_price = market_price.decision_reference_price();

            // On first valid market price, fix up any startup position
            // that was synced with entry_price=0 (because we didn't have
            // a market price yet during sync_live_balances).
            if decision_reference_price > f64::EPSILON {
                if let Some(pos) = self.trade_manager.get_position(trading_symbol) {
                    if pos.entry_price <= f64::EPSILON {
                        info!(
                            symbol = trading_symbol,
                            new_entry_price = %fmt_price(decision_reference_price),
                            quantity = %fmt_price(pos.quantity),
                            "Startup position: fixing entry_price from first bookTicker"
                        );
                        self.trade_manager.fix_entry_price(
                            trading_symbol,
                            decision_reference_price,
                        );
                    }
                }
            }

            if let Some(last_trade) = market_state.last_trade() {
                last_trade_price = Some(last_trade.price);
                let _ = self.trade_manager.mark_to_market(trading_symbol, last_trade.price);
            }

            if let Some(mark_price) = market_state.last_price() {
                let current_drawdown = self.trade_manager.current_drawdown(trading_symbol, mark_price);
                let drawdown_disabled = self
                    .config
                    .backtest_settings
                    .as_ref()
                    .map(|s| s.disable_drawdown_limit)
                    .unwrap_or(false);

                let max_allowed_drawdown = if drawdown_disabled {
                    None // No limit — drawdown guardrail fully disabled
                } else {
                    Some(
                        self.config
                            .backtest_settings
                            .as_ref()
                            .and_then(|s| s.max_drawdown_override)
                            .unwrap_or(self.config.risk_management.max_drawdown),
                    )
                };

                if let Some(max_dd) = max_allowed_drawdown {
                    if current_drawdown >= max_dd {
                    self.logger.log_warning(
                        trading_symbol,
                        "drawdown_guardrail",
                        "Current drawdown exceeded configured limit; suppressing new entries",
                    );

                    if self.current_position(trading_symbol).quantity > 0.0 {
                        // Reset cumulative tracker on forced drawdown exit
                        cumulative_tracker.reset();
                        let position_quantity = self.current_position(trading_symbol).quantity;
                        let execution = backtest_engine.execute_with_constraints_at(
                            trade::signal::Signal::Sell,
                            market_price,
                            position_quantity,
                            self.trade_manager.available_cash(),
                        );
                        if !execution.is_rejected() {
                            let report = self.simulated_report_from_sell(position_quantity, 0.0, execution);
                            self.trade_manager.record_execution_report(&report);
                            let pnl = self.trade_manager.close_position_with_report(
                                trading_symbol,
                                execution.execution_price,
                                market_state.last_event_time_secs().unwrap_or(0.0),
                                Some(&report),
                            );
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

                    continue;
                    }
                }
            }

            self.logger.log_market_state_snapshot(
                trading_symbol,
                market_state.last_price(),
                market_state.mid_price(),
                market_state.spread_bps(),
                market_state.trade_flow_imbalance(),
                market_state.order_book_imbalance(),
            );

            // Poll exchange for async fills on active limit orders (live mode only).
            // This detects fills that happened between ticks and updates trade_manager
            // so the strategy sees the correct position state.
            if matches!(trading_mode, TradeMode::Real) && self.active_limit_order.is_some() {
                let event_time = market_state.last_event_time_secs().unwrap_or(0.0);
                self.poll_active_order_status(trading_symbol, event_time).await;
            }
            // Poll two-sided orders (bid + ask) for async fills
            if matches!(trading_mode, TradeMode::Real)
                && (self.active_bid_order.is_some() || self.active_ask_order.is_some())
            {
                let event_time = market_state.last_event_time_secs().unwrap_or(0.0);
                self.poll_two_sided_orders(trading_symbol, event_time).await;
            }
            // Poll ladder orders for async fills
            if matches!(trading_mode, TradeMode::Real)
                && (!self.active_ladder_bids.is_empty() || !self.active_ladder_asks.is_empty())
            {
                let event_time = market_state.last_event_time_secs().unwrap_or(0.0);
                self.poll_ladder_orders(trading_symbol, event_time).await;
            }

            let current_position = self.current_position(trading_symbol);
            // Inventory guardrail: skip strategy entirely if position exceeds max.
            // BUT: two-sided strategies manage their own inventory — they set
            // buy_quantity=0 when full and still need sell-side to unwind.
            // So we only `continue` for single-sided (Place) strategies.
            // For two-sided, we let decide() run and trust the strategy's skew logic.
            let inventory_exceeded = current_position.quantity > 0.0
                && current_position.quantity * decision_reference_price
                    > self.config.position_sizing.max_position_notional;
            let decision = trading_strategy.decide(
                &market_state,
                &StrategyContext {
                    symbol: trading_symbol.to_string(),
                    current_position: current_position.clone(),
                    available_cash: self.trade_manager.available_cash(),
                    max_position_notional: self.config.position_sizing.max_position_notional,
                    initial_capital: self.trade_manager.initial_capital(),
                    tick_size: self.config.exchange.tick_size,
                    step_size: Some(self.config.exchange.step_size),
                    min_notional: Some(self.config.exchange.min_notional),
                },
            );

            // Post-decide inventory guardrail for single-sided strategies:
            // Skip new BUY entries (Place { side: Buy }) when inventory exceeded.
            // Two-sided QuoteBothSides already handles this via buy_quantity=0.
            if inventory_exceeded {
                match &decision.intent {
                    OrderIntent::Place { side: Side::Buy, .. } => {
                        self.logger.log_warning(
                            trading_symbol,
                            "inventory_guardrail",
                            "Current position notional exceeds configured max; suppressing new entries",
                        );
                        continue;
                    }
                    // Allow sells, cancels, no_action, and QuoteBothSides through
                    _ => {}
                }
            }

            {
                let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                strategy_logger.log_decision(
                    trading_symbol,
                    &decision.intent,
                    decision.confidence,
                    decision_reference_price,
                );
            }

            let execution_reference_price =
                self.execution_reference_price_for_intent(market_price, &decision.intent);

            // Update last known BBO for pre-flight guard in place_or_replace_side.
            // Also clear post-fill guard prices: if BBO changed since the fill,
            // the market has stabilized and we can quote normally again.
            if let Some(book) = market_state.top_of_book() {
                let new_bid = book.bid.price;
                let new_ask = book.ask.price;
                if (new_bid - self.last_best_bid).abs() > f64::EPSILON
                    || (new_ask - self.last_best_ask).abs() > f64::EPSILON
                {
                    // BBO has changed since our last snapshot — safe to clear fill guards
                    self.last_buy_fill_price = 0.0;
                    self.last_sell_fill_price = 0.0;
                }
                self.last_best_bid = new_bid;
                self.last_best_ask = new_ask;
            }

            match trading_mode {
                TradeMode::Real => {
                    let intent = decision.intent.clone();
                    let mut report = self
                        .on_order_intent(
                            trading_symbol,
                            execution_reference_price,
                            intent,
                        )
                        .await;
                    report.rationale = match &decision.intent {
                        OrderIntent::Place { rationale, .. } | OrderIntent::Cancel { rationale } | OrderIntent::QuoteBothSides { rationale, .. } | OrderIntent::QuoteLadder { rationale, .. } => Some(*rationale),
                        OrderIntent::NoAction => None,
                    };
                    report.decision_confidence = decision.confidence;
                    report.decision_metrics = decision.metrics.clone();
                    report.symbol = Some(trading_symbol.to_string());

                    let pnl = if report.reason == Some("force_close_position") {
                        // Taker sell failed MAX_TAKER_RETRIES times — force-close virtual
                        // position at entry price (zero PnL). The real asset is missing.
                        let pos = self.current_position(trading_symbol);
                        if pos.quantity > 0.0 {
                            error!(
                                symbol = %trading_symbol,
                                position_qty = %fmt_price(pos.quantity),
                                entry_price = %fmt_price(pos.entry_price),
                                "Force-closing virtual position — asset missing on exchange"
                            );
                            Some(self.trade_manager.close_position_with_report(
                                trading_symbol,
                                pos.entry_price, // close at entry → zero gross PnL
                                market_state.last_event_time_secs().unwrap_or(0.0),
                                Some(&report),
                            ))
                        } else {
                            None
                        }
                    } else if report.side == Some(Side::Sell)
                        && report.executed_quantity > 0.0
                    {
                        report.execution_price.map(|price| {
                            self.trade_manager.close_position_with_report(
                                trading_symbol,
                                price,
                                market_state.last_event_time_secs().unwrap_or(0.0),
                                Some(&report),
                            )
                        })
                    } else if report.side == Some(Side::Buy)
                        && report.executed_quantity > 0.0
                    {
                        if let Some(price) = report.execution_price {
                            let _ = self.trade_manager.open_position(
                                trading_symbol,
                                price,
                                report.executed_quantity,
                                market_state.last_event_time_secs().unwrap_or(0.0),
                                Some(&report),
                            );
                        }
                        None
                    } else {
                        None
                    };

                    {
                        self.trade_manager.record_execution_report(&report);
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
                    let mut report = match decision.intent {
                        OrderIntent::NoAction => {
                            cumulative_tracker.reset();
                            ExecutionReport::ignored()
                        }
                        OrderIntent::Cancel { .. } => {
                            cumulative_tracker.reset();
                            ExecutionReport::ignored()
                        }
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

                            if matches!(order_type, OrderType::Maker) {
                                let n = cumulative_tracker.record_maker_intent(side);
                                backtest_engine.simulate_passive_order_cumulative(
                                    side,
                                    market_price,
                                    requested_quantity,
                                    expected_edge_bps,
                                    n,
                                )
                            } else {
                                // Taker order: reset cumulative tracker
                                cumulative_tracker.reset();
                                let execution = backtest_engine.execute_with_constraints_at(
                                    signal,
                                    market_price,
                                    requested_quantity,
                                    self.trade_manager.available_cash(),
                                );

                                if execution.is_rejected() {
                                    ExecutionReport {
                                        status: ExecutionStatus::Rejected,
                                        symbol: Some(trading_symbol.to_string()),
                                        side: Some(side),
                                        order_type: Some(order_type),
                                        rationale: None,
                                        decision_confidence: 0.0,
                                        decision_metrics: Vec::new(),
                                        requested_quantity,
                                        executed_quantity: 0.0,
                                        execution_price: Some(execution.execution_price),
                                        fee_paid: 0.0,
                                        latency_seconds: execution.latency_seconds,
                                        synthetic_half_spread_bps: execution.synthetic_half_spread_rate * 10_000.0,
                                        slippage_bps: execution.slippage_rate * 10_000.0,
                                        latency_impact_bps: execution.latency_impact_rate * 10_000.0,
                                        market_impact_bps: execution.market_impact_rate * 10_000.0,
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
                        }
                        OrderIntent::QuoteBothSides {
                            buy_price,
                            buy_quantity,
                            sell_price,
                            sell_quantity,
                            expected_edge_bps,
                            ..
                        } => {
                            // Reset single-sided tracker (not used in two-sided mode)
                            cumulative_tracker.reset();

                            // ═══════════════════════════════════════════════════════
                            // Trade-through fill model for two-sided market-making.
                            //
                            // Instead of probabilistic cumulative fill, we check if
                            // the latest trade crosses our quote prices:
                            //   - Buy fills when seller-initiated trade at or below buy_price
                            //   - Sell fills when buyer-initiated trade at or above sell_price
                            //
                            // This is far more realistic than the cumulative model
                            // because it respects price levels and trade direction.
                            // ═══════════════════════════════════════════════════════

                            if let Some(last_trade) = market_state.last_trade() {
                                let trade_price = last_trade.price;
                                let trade_qty = last_trade.quantity;
                                let step = self.config.exchange.step_size;
                                let tick_size = self.config.exchange.tick_size;
                                let min_notional = self.config.exchange.min_notional;
                                let maker_fee = self.config.exchange.maker_fee;

                                let rounded_buy_price = if tick_size > 0.0 {
                                    let ticks = buy_price / tick_size;
                                    ticks.floor() * tick_size
                                } else {
                                    buy_price
                                };
                                let rounded_sell_price = if tick_size > 0.0 {
                                    let ticks = sell_price / tick_size;
                                    ticks.ceil() * tick_size
                                } else {
                                    sell_price
                                };
                                let rounded_buy_qty = if step > 0.0 {
                                    (buy_quantity / step).floor() * step
                                } else {
                                    buy_quantity
                                };
                                let rounded_sell_qty = if sell_quantity <= step && sell_quantity > 0.0 {
                                    sell_quantity
                                } else {
                                    backtest_engine.round_down_to_step(sell_quantity)
                                };
                                let buy_order_is_live = rounded_buy_qty >= step
                                    && rounded_buy_price > 0.0
                                    && rounded_buy_qty * rounded_buy_price >= min_notional;

                                // --- BUY side: trade-through check ---
                                // Passive buy rests at buy_price. Fills when a seller-
                                // initiated trade occurs at or below our price.
                                // is_buyer_market_maker == true means seller crossed the
                                // spread (seller-initiated → price is at bid side).
                                if buy_order_is_live
                                    && last_trade.is_buyer_market_maker
                                    && trade_price <= rounded_buy_price
                                {
                                    // Fill quantity: min of our order, trade size, and
                                    // a fraction to model queue position (we're not first).
                                    // Use 30% of trade volume as fill — we share the queue
                                    // with other market makers.
                                    let queue_fill_fraction = 0.3;
                                    let available_fill = (trade_qty * queue_fill_fraction).max(step);
                                    let fill_qty = backtest_engine.round_down_to_step(
                                        rounded_buy_qty.min(available_fill),
                                    );

                                    if fill_qty > 0.0 {
                                        let fill_price = rounded_buy_price; // maker fills at own price
                                        let fee_paid = fill_price * fill_qty * maker_fee;
                                        let buy_report = ExecutionReport {
                                            status: if fill_qty < rounded_buy_qty {
                                                ExecutionStatus::PartiallyFilled
                                            } else {
                                                ExecutionStatus::Filled
                                            },
                                            symbol: Some(trading_symbol.to_string()),
                                            side: Some(Side::Buy),
                                            order_type: Some(OrderType::Maker),
                                            rationale: Some("two_sided_buy_fill"),
                                            decision_confidence: 1.0,
                                            decision_metrics: Vec::new(),
                                            requested_quantity: rounded_buy_qty,
                                            executed_quantity: fill_qty,
                                            execution_price: Some(fill_price),
                                            fee_paid,
                                            latency_seconds: 0.0,
                                            synthetic_half_spread_bps: 0.0,
                                            slippage_bps: 0.0,
                                            latency_impact_bps: 0.0,
                                            market_impact_bps: 0.0,
                                            reason: if fill_qty < rounded_buy_qty {
                                                Some("partial_queue_fill")
                                            } else {
                                                None
                                            },
                                            expected_edge_bps,
                                        };

                                        let _ = self.trade_manager.open_position(
                                            trading_symbol,
                                            fill_price,
                                            fill_qty,
                                            market_state.last_event_time_secs().unwrap_or(0.0),
                                            Some(&buy_report),
                                        );
                                        self.trade_manager.record_execution_report(&buy_report);
                                        let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                                        strategy_logger.log_execution(
                                            trading_symbol,
                                            &buy_report,
                                            None,
                                            Some(self.trade_manager.get_metrics().realized_pnl()),
                                            Some(self.trade_manager.get_trade_summary()),
                                        );
                                    }
                                }

                                // --- SELL side: trade-through check ---
                                // Passive sell rests at sell_price. Fills when a buyer-
                                // initiated trade occurs at or above our price.
                                // is_buyer_market_maker == false means buyer crossed the
                                // spread (buyer-initiated → price is at ask side).
                                let pos = self.current_position(trading_symbol);
                                if rounded_sell_qty > 0.0
                                    && pos.quantity > 0.0
                                    && !last_trade.is_buyer_market_maker
                                    && trade_price >= rounded_sell_price
                                {
                                    let sell_qty = rounded_sell_qty.min(pos.quantity);
                                    let queue_fill_fraction = 0.3;
                                    let available_fill = (trade_qty * queue_fill_fraction).max(step);
                                    let fill_qty = if sell_qty <= step && available_fill >= step {
                                        // All-or-nothing for min lot
                                        sell_qty
                                    } else {
                                        backtest_engine.round_down_to_step(
                                            sell_qty.min(available_fill),
                                        )
                                    };

                                    if fill_qty > 0.0 {
                                        let fill_price = rounded_sell_price; // maker fills at own price
                                        let fee_paid = fill_price * fill_qty * maker_fee;
                                        let sell_report = ExecutionReport {
                                            status: if fill_qty < sell_qty {
                                                ExecutionStatus::PartiallyFilled
                                            } else {
                                                ExecutionStatus::Filled
                                            },
                                            symbol: Some(trading_symbol.to_string()),
                                            side: Some(Side::Sell),
                                            order_type: Some(OrderType::Maker),
                                            rationale: Some("two_sided_sell_fill"),
                                            decision_confidence: 1.0,
                                            decision_metrics: Vec::new(),
                                            requested_quantity: sell_qty,
                                            executed_quantity: fill_qty,
                                            execution_price: Some(fill_price),
                                            fee_paid,
                                            latency_seconds: 0.0,
                                            synthetic_half_spread_bps: 0.0,
                                            slippage_bps: 0.0,
                                            latency_impact_bps: 0.0,
                                            market_impact_bps: 0.0,
                                            reason: if fill_qty < sell_qty {
                                                Some("partial_queue_fill")
                                            } else {
                                                None
                                            },
                                            expected_edge_bps,
                                        };

                                        let remaining = pos.quantity - fill_qty;
                                        let pnl = if remaining > step * 0.5 {
                                            let actual_fee = fill_price * fill_qty * maker_fee;
                                            self.trade_manager.update_position(
                                                trading_symbol,
                                                remaining,
                                                fill_price,
                                                market_state.last_event_time_secs().unwrap_or(0.0),
                                                Some(actual_fee),
                                            );
                                            let gross = (fill_price - pos.entry_price) * fill_qty;
                                            gross - actual_fee
                                        } else {
                                            self.trade_manager.close_position_with_report(
                                                trading_symbol,
                                                fill_price,
                                                market_state.last_event_time_secs().unwrap_or(0.0),
                                                Some(&sell_report),
                                            )
                                        };

                                        self.trade_manager.record_execution_report(&sell_report);
                                        let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                                        strategy_logger.log_execution(
                                            trading_symbol,
                                            &sell_report,
                                            Some(pnl),
                                            Some(self.trade_manager.get_metrics().realized_pnl()),
                                            Some(self.trade_manager.get_trade_summary()),
                                        );
                                    }
                                }
                            }

                            // Return ignored for the main report pipeline — we handled
                            // both sides above with their own logging.
                            ExecutionReport::ignored()
                        }
                        OrderIntent::QuoteLadder {
                            ref bids,
                            ref asks,
                            expected_edge_bps,
                            ..
                        } => {
                            // Ladder backtest: trade-through fill model for each level.
                            // Same logic as QuoteBothSides but applied to each level.
                            cumulative_tracker.reset();

                            if let Some(last_trade) = market_state.last_trade() {
                                let trade_price = last_trade.price;
                                let trade_qty = last_trade.quantity;
                                let step = self.config.exchange.step_size;
                                let tick_size = self.config.exchange.tick_size;
                                let min_notional = self.config.exchange.min_notional;
                                let maker_fee = self.config.exchange.maker_fee;

                                // --- BID levels: trade-through check ---
                                for &(bid_price, bid_qty) in bids {
                                    let rounded_buy_price = if tick_size > 0.0 {
                                        (bid_price / tick_size).floor() * tick_size
                                    } else {
                                        bid_price
                                    };
                                    let rounded_buy_qty = if step > 0.0 {
                                        (bid_qty / step).floor() * step
                                    } else {
                                        bid_qty
                                    };
                                    let buy_order_is_live = rounded_buy_qty >= step
                                        && rounded_buy_price > 0.0
                                        && rounded_buy_qty * rounded_buy_price >= min_notional;

                                    if buy_order_is_live
                                        && last_trade.is_buyer_market_maker
                                        && trade_price <= rounded_buy_price
                                    {
                                        let queue_fill_fraction = 0.3;
                                        let available_fill = (trade_qty * queue_fill_fraction).max(step);
                                        let fill_qty = backtest_engine.round_down_to_step(
                                            rounded_buy_qty.min(available_fill),
                                        );

                                        if fill_qty > 0.0 {
                                            let fill_price = rounded_buy_price;
                                            let fee_paid = fill_price * fill_qty * maker_fee;
                                            let buy_report = ExecutionReport {
                                                status: if fill_qty < rounded_buy_qty {
                                                    ExecutionStatus::PartiallyFilled
                                                } else {
                                                    ExecutionStatus::Filled
                                                },
                                                symbol: Some(trading_symbol.to_string()),
                                                side: Some(Side::Buy),
                                                order_type: Some(OrderType::Maker),
                                                rationale: Some("ladder_buy_fill"),
                                                decision_confidence: 1.0,
                                                decision_metrics: Vec::new(),
                                                requested_quantity: rounded_buy_qty,
                                                executed_quantity: fill_qty,
                                                execution_price: Some(fill_price),
                                                fee_paid,
                                                latency_seconds: 0.0,
                                                synthetic_half_spread_bps: 0.0,
                                                slippage_bps: 0.0,
                                                latency_impact_bps: 0.0,
                                                market_impact_bps: 0.0,
                                                reason: if fill_qty < rounded_buy_qty {
                                                    Some("partial_queue_fill")
                                                } else {
                                                    None
                                                },
                                                expected_edge_bps,
                                            };

                                            let _ = self.trade_manager.open_position(
                                                trading_symbol,
                                                fill_price,
                                                fill_qty,
                                                market_state.last_event_time_secs().unwrap_or(0.0),
                                                Some(&buy_report),
                                            );
                                            self.trade_manager.record_execution_report(&buy_report);
                                            let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                                            strategy_logger.log_execution(
                                                trading_symbol,
                                                &buy_report,
                                                None,
                                                Some(self.trade_manager.get_metrics().realized_pnl()),
                                                Some(self.trade_manager.get_trade_summary()),
                                            );
                                        }
                                        // Only one bid level fills per trade event
                                        break;
                                    }
                                }

                                // --- ASK levels: trade-through check ---
                                let pos = self.current_position(trading_symbol);
                                if pos.quantity > 0.0 {
                                    for &(ask_price, ask_qty) in asks {
                                        let rounded_sell_price = if tick_size > 0.0 {
                                            (ask_price / tick_size).ceil() * tick_size
                                        } else {
                                            ask_price
                                        };
                                        let rounded_sell_qty = if ask_qty <= step && ask_qty > 0.0 {
                                            ask_qty
                                        } else {
                                            backtest_engine.round_down_to_step(ask_qty)
                                        };

                                        if rounded_sell_qty > 0.0
                                            && !last_trade.is_buyer_market_maker
                                            && trade_price >= rounded_sell_price
                                        {
                                            let sell_qty = rounded_sell_qty.min(pos.quantity);
                                            let queue_fill_fraction = 0.3;
                                            let available_fill = (trade_qty * queue_fill_fraction).max(step);
                                            let fill_qty = if sell_qty <= step && available_fill >= step {
                                                sell_qty
                                            } else {
                                                backtest_engine.round_down_to_step(
                                                    sell_qty.min(available_fill),
                                                )
                                            };

                                            if fill_qty > 0.0 {
                                                let fill_price = rounded_sell_price;
                                                let fee_paid = fill_price * fill_qty * maker_fee;
                                                let sell_report = ExecutionReport {
                                                    status: if fill_qty < sell_qty {
                                                        ExecutionStatus::PartiallyFilled
                                                    } else {
                                                        ExecutionStatus::Filled
                                                    },
                                                    symbol: Some(trading_symbol.to_string()),
                                                    side: Some(Side::Sell),
                                                    order_type: Some(OrderType::Maker),
                                                    rationale: Some("ladder_sell_fill"),
                                                    decision_confidence: 1.0,
                                                    decision_metrics: Vec::new(),
                                                    requested_quantity: sell_qty,
                                                    executed_quantity: fill_qty,
                                                    execution_price: Some(fill_price),
                                                    fee_paid,
                                                    latency_seconds: 0.0,
                                                    synthetic_half_spread_bps: 0.0,
                                                    slippage_bps: 0.0,
                                                    latency_impact_bps: 0.0,
                                                    market_impact_bps: 0.0,
                                                    reason: if fill_qty < sell_qty {
                                                        Some("partial_queue_fill")
                                                    } else {
                                                        None
                                                    },
                                                    expected_edge_bps,
                                                };

                                                let remaining = pos.quantity - fill_qty;
                                                let pnl = if remaining > step * 0.5 {
                                                    let actual_fee = fill_price * fill_qty * maker_fee;
                                                    self.trade_manager.update_position(
                                                        trading_symbol,
                                                        remaining,
                                                        fill_price,
                                                        market_state.last_event_time_secs().unwrap_or(0.0),
                                                        Some(actual_fee),
                                                    );
                                                    let gross = (fill_price - pos.entry_price) * fill_qty;
                                                    gross - actual_fee
                                                } else {
                                                    self.trade_manager.close_position_with_report(
                                                        trading_symbol,
                                                        fill_price,
                                                        market_state.last_event_time_secs().unwrap_or(0.0),
                                                        Some(&sell_report),
                                                    )
                                                };

                                                self.trade_manager.record_execution_report(&sell_report);
                                                let strategy_logger = StrategyLoggerAdapter::new(&self.logger);
                                                strategy_logger.log_execution(
                                                    trading_symbol,
                                                    &sell_report,
                                                    Some(pnl),
                                                    Some(self.trade_manager.get_metrics().realized_pnl()),
                                                    Some(self.trade_manager.get_trade_summary()),
                                                );
                                            }
                                            // Only one ask level fills per trade event
                                            break;
                                        }
                                    }
                                }
                            }

                            ExecutionReport::ignored()
                        }
                    };
                    report.rationale = match &decision.intent {
                        OrderIntent::Place { rationale, .. } | OrderIntent::Cancel { rationale } | OrderIntent::QuoteBothSides { rationale, .. } | OrderIntent::QuoteLadder { rationale, .. } => Some(*rationale),
                        OrderIntent::NoAction => None,
                    };
                    report.decision_confidence = decision.confidence;
                    report.decision_metrics = decision.metrics.clone();
                    report.symbol = Some(trading_symbol.to_string());

                    let pnl = match report.side {
                        Some(Side::Buy) if report.executed_quantity > 0.0 => {
                            // Reset cumulative tracker after successful buy fill
                            cumulative_tracker.reset();
                            if let Some(price) = report.execution_price {
                                if let Err(reason) = self.trade_manager.open_position(
                                    trading_symbol,
                                    price,
                                    report.executed_quantity,
                                    market_state.last_event_time_secs().unwrap_or(0.0),
                                    Some(&report),
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
                        Some(Side::Sell) if report.executed_quantity > 0.0 => {
                            // Reset cumulative tracker after sell fill
                            cumulative_tracker.reset();
                            // Check if this is a partial fill — only reduce position
                            // instead of closing entirely
                            let is_partial = matches!(report.status, ExecutionStatus::PartiallyFilled);
                            if is_partial {
                                report.execution_price.map(|price| {
                                    let position = self.current_position(trading_symbol);
                                    let remaining = position.quantity - report.executed_quantity;
                                    if remaining > self.config.exchange.step_size * 0.5 {
                                        // Partial close: reduce position by executed amount
                                        let exit_fee = price * report.executed_quantity * self.config.exchange.maker_fee;
                                        self.trade_manager.update_position(
                                            trading_symbol,
                                            remaining,
                                            price,
                                            market_state.last_event_time_secs().unwrap_or(0.0),
                                            Some(exit_fee),
                                        );
                                        // Return approximate PnL for the partial
                                        let entry_price = position.entry_price;
                                        let gross_pnl = (price - entry_price) * report.executed_quantity;
                                        let exit_fee = price * report.executed_quantity * self.config.exchange.maker_fee;
                                        gross_pnl - exit_fee
                                    } else {
                                        // Remaining is dust — close entire position
                                        self.trade_manager.close_position_with_report(
                                            trading_symbol,
                                            price,
                                            market_state.last_event_time_secs().unwrap_or(0.0),
                                            Some(&report),
                                        )
                                    }
                                })
                            } else {
                                report.execution_price.map(|price| {
                                    self.trade_manager.close_position_with_report(
                                        trading_symbol,
                                        price,
                                        market_state.last_event_time_secs().unwrap_or(0.0),
                                        Some(&report),
                                    )
                                })
                            }
                        }
                        _ => None,
                    };

                    {
                        self.trade_manager.record_execution_report(&report);
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

        // ── Cancel all open orders on shutdown (live mode) ──
        if trading_mode == TradeMode::Real {
            if let Some(ref connection) = self.connection {
                info!(
                    symbol = trading_symbol,
                    "Cancelling all open orders on shutdown"
                );
                let cancel_params =
                    OpenOrdersCancelAllParams::builder(trading_symbol.to_string()).build()?;
                match connection.open_orders_cancel_all(cancel_params).await {
                    Ok(response) => {
                        let cancelled_count = response
                            .raw
                            .as_array()
                            .map(|arr| arr.len())
                            .unwrap_or(0);
                        info!(
                            symbol = trading_symbol,
                            cancelled_count = cancelled_count,
                            "Open orders cancelled on shutdown"
                        );
                    }
                    Err(e) => {
                        let err_str = format!("{:?}", e);
                        if err_str.contains("-2011") {
                            info!(
                                symbol = trading_symbol,
                                "No open orders to cancel on shutdown"
                            );
                        } else {
                            error!(
                                symbol = trading_symbol,
                                error = %e,
                                "Failed to cancel open orders on shutdown"
                            );
                        }
                    }
                }
            }
        }

        if matches!(trading_mode, TradeMode::Backtest | TradeMode::Emulated)
            && self.current_position(trading_symbol).quantity > 0.0
        {
            let position = self.current_position(trading_symbol);
            let liquidation_market_price = self
                .market_price_from_state(&market_state)
                .or_else(|| last_trade_price.map(MarketPrice::Trade));

            if let Some(liquidation_market_price) = liquidation_market_price {
                let execution = backtest_engine.execute_with_constraints_at(
                    trade::signal::Signal::Sell,
                    liquidation_market_price,
                    position.quantity,
                    self.trade_manager.available_cash(),
                );
                if !execution.is_rejected() {
                    let report = self.simulated_report_from_sell(position.quantity, 0.0, execution);
                    self.trade_manager.record_execution_report(&report);
                    let pnl = self.trade_manager.close_position_with_report(
                        trading_symbol,
                        execution.execution_price,
                        market_state.last_event_time_secs().unwrap_or(0.0),
                        Some(&report),
                    );
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
        }

        let metrics = self.trade_manager.get_metrics();
        let event_mix = market_state.event_mix_diagnostics();
        self.logger.log_replay_event_mix(
            trading_symbol,
            event_mix.trade_events,
            event_mix.book_ticker_events,
            event_mix.depth_events,
            event_mix.trade_without_quote_events,
            event_mix.stale_quote_events,
            event_mix.stale_depth_events,
        );
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
            metrics.fill_ratio(),
            metrics.rejection_rate(),
            metrics.avg_latency_seconds(),
            metrics.avg_synthetic_half_spread_bps(),
            metrics.avg_slippage_bps(),
            metrics.avg_latency_impact_bps(),
            metrics.avg_market_impact_bps(),
            metrics.avg_expected_edge_bps(),
        );

        Ok(())
    }
}
