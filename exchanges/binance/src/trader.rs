use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    AccountStatusParams, OrderPlaceParams, OrderPlaceSideEnum, OrderPlaceTypeEnum, WebsocketApi,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal_macros::dec;
use tracing::{error, info};
use trade::signal::Signal;
use trade::trader::{Position, TradeMode, Trader};

pub struct BinanceTrader {
    connection: Option<WebsocketApi>,
    pub position: Position,
    pub realized_pnl: f64,
}

impl BinanceTrader {
    pub async fn new(symbol: &str, api_key: &str, api_secret: &str, mode: TradeMode) -> Self {
        let connection = if mode == TradeMode::Real {
            let config = ConfigurationWebsocketApi::builder()
                .api_key(api_key)
                .api_secret(api_secret)
                .build()
                .expect("Failed to build Binance API configuration");

            let client = SpotWsApi::production(config);

            Some(
                client
                    .connect()
                    .await
                    .expect("Failed to connect to WebSocket API"),
            )
        } else {
            None
        };

        BinanceTrader {
            connection,
            position: Position {
                symbol: symbol.to_string(),
                quantity: 0.0,
                entry_price: 0.0,
            },
            realized_pnl: 0.0,
        }
    }
}

#[async_trait]
impl Trader for BinanceTrader {
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64) {
        let symbol = self.position.symbol.clone();
        let quantity = Decimal::from_f64(quantity).expect("Failed to convert quantity to Decimal");

        let connection = match &self.connection {
            Some(c) => c,
            None => {
                error!(
                    exchange = "binance",
                    action = "on_signal",
                    status = "connection_missing",
                    symbol = %symbol
                );
                return;
            }
        };

        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 {
                    let params = OrderPlaceParams::builder(
                        symbol.clone(),
                        OrderPlaceSideEnum::Buy,
                        OrderPlaceTypeEnum::Market,
                    )
                    .quantity(quantity)
                    .build()
                    .expect("Failed to build order parameters");

                    let response = match connection.order_place(params).await {
                        Ok(r) => r,
                        Err(e) => {
                            error!(
                                exchange = "binance",
                                action = "place_buy_order",
                                status = "failed",
                                symbol = %symbol,
                                quantity = %quantity,
                                error = %e
                            );
                            return;
                        }
                    };
                    let data = match response.data() {
                        Ok(d) => d,
                        Err(e) => {
                            error!(
                                exchange = "binance",
                                action = "buy_order_data",
                                status = "failed",
                                symbol = %symbol,
                                error = %e
                            );
                            return;
                        }
                    };

                    let fills = match data.fills.as_ref() {
                        Some(f) => f,
                        None => {
                            error!(
                                exchange = "binance",
                                action = "buy_order_fills",
                                status = "missing",
                                symbol = %symbol
                            );
                            return;
                        }
                    };

                    self.position.quantity = data
                        .executed_qty
                        .as_ref()
                        .expect("Executed quantity not found in buy order response")
                        .parse()
                        .expect("Failed to parse executed quantity");
                    self.position.entry_price = fills
                        .first()
                        .expect("No fills found in buy order response")
                        .price
                        .as_ref()
                        .expect("Price not found in fill")
                        .parse()
                        .expect("Failed to parse entry price");
                }
            }
            Signal::Sell => {
                if self.position.quantity > 0.0 {
                    let pnl = (price - self.position.entry_price) * self.position.quantity;
                    self.realized_pnl += pnl;
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
                        return;
                    }

                    self.position.quantity = 0.0; // Assuming full sell
                    self.position.entry_price = 0.0;
                }
            }
            Signal::Hold => {
                // Do nothing
            }
        }
    }

    async fn on_emulate(&mut self, signal: Signal, price: f64, quantity: f64) {
        let symbol = self.position.symbol.clone();
        let quantity = Decimal::from_f64(quantity).unwrap_or(dec!(0.0)); // Updated quantity conversion

        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 {
                    info!(
                        exchange = "binance",
                        mode = "emulated",
                        action = "place_buy_order",
                        symbol = %symbol,
                        price = %format!("{:.6}", price),
                        quantity = %format!("{:.6}", quantity.to_f64().unwrap_or_default()),
                        profit = %format!("{:.6}", self.realized_pnl)
                    );
                    // Simulate a buy in emulated mode
                    self.position.quantity = quantity.to_f64().unwrap_or(0.0);
                    self.position.entry_price = price;
                }
            }
            Signal::Sell => {
                if self.position.quantity > 0.0 {
                    let pnl = (price - self.position.entry_price) * self.position.quantity;
                    self.realized_pnl += pnl;
                    info!(
                        exchange = "binance",
                        mode = "emulated",
                        action = "place_sell_order",
                        symbol = %symbol,
                        price = %format!("{:.6}", price),
                        quantity = %format!("{:.6}", self.position.quantity),
                        pnl = %format!("{:.6}", pnl),
                        profit = %format!("{:.6}", self.realized_pnl)
                    );
                    // Simulate a sell in emulated mode
                    self.position.quantity = 0.0;
                    self.position.entry_price = 0.0;
                    self.realized_pnl += pnl;
                }
            }
            Signal::Hold => {
                // Do nothing
            }
        }
    }

    fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.position.quantity > 0.0 {
            (current_price - self.position.entry_price) * self.position.quantity
        } else {
            0.0
        }
    }

    fn realized_pnl(&self) -> f64 {
        self.realized_pnl
    }

    fn position(&self) -> Position {
        self.position.clone()
    }

    async fn account_status(&self) -> Result<(), anyhow::Error> {
        if let Some(connection) = &self.connection {
            let params = AccountStatusParams::builder()
                .omit_zero_balances(true)
                .build()?;
            let status = connection.account_status(params).await?;
            let data = status.data().unwrap();

            if let Some(account_type) = &data.account_type {
                println!("Account Type:\t{}", account_type);
            }

            if let Some(uid) = &data.uid {
                println!("UID:\t{}", uid);
            }

            if let Some(can_trade) = data.can_trade {
                println!("Can Trade:\t{}", can_trade);
            }

            if let Some(can_withdraw) = data.can_withdraw {
                println!("Can Withdraw:\t{}", can_withdraw);
            }

            if let Some(can_deposit) = data.can_deposit {
                println!("Can Deposit:\t{}", can_deposit);
            }

            if let Some(permissions) = &data.permissions {
                println!("Permissions:\t{:?}", permissions);
            }

            if let Some(rates) = &data.commission_rates {
                println!("\nCommission Rates:");
                println!("  Maker:\t{}", rates.maker.as_deref().unwrap_or("N/A"));
                println!("  Taker:\t{}", rates.taker.as_deref().unwrap_or("N/A"));
                println!("  Buyer:\t{}", rates.buyer.as_deref().unwrap_or("N/A"));
                println!("  Seller:\t{}", rates.seller.as_deref().unwrap_or("N/A"));
            }

            if let Some(balances) = &data.balances {
                println!("\nAsset\tFree\t\tLocked");
                println!("-----\t--------\t--------");
                for b in balances {
                    let asset = b.asset.as_deref().unwrap_or("-");
                    let free = b.free.as_deref().unwrap_or("0");
                    let locked = b.locked.as_deref().unwrap_or("0");
                    println!("{}\t{}\t{}", asset, free, locked);
                }
            }
        }

        Ok(())
    }
}
