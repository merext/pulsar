use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    OrderPlaceParams, OrderPlaceSideEnum, OrderPlaceTypeEnum, WebsocketApi,
};
use log::{error, info};
use rust_decimal::Decimal;
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal_macros::dec;
use trade::signal::Signal;
use trade::trader::{Position, TradeMode, Trader};

pub struct BinanceTrader {
    connection: WebsocketApi,
    pub position: Position,
    pub realized_pnl: f64,
}

impl BinanceTrader {
    pub async fn new(symbol: &str, api_key: &str, api_secret: &str) -> Self {
        let config = ConfigurationWebsocketApi::builder()
            .api_key(api_key)
            .api_secret(api_secret)
            .build()
            .expect("Failed to build Binance API configuration");

        let client = SpotWsApi::production(config);

        let connection = client
            .connect()
            .await
            .expect("Failed to connect to WebSocket API");

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
    async fn on_signal(&mut self, signal: Signal, price: f64, quantity: f64, mode: TradeMode) {
        let symbol = self.position.symbol.clone();
        let quantity = Decimal::from_f64(quantity).unwrap_or(dec!(0.0)); // Updated quantity conversion

        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 {
                    info!(
                        "BinanceTrader: Placing BUY order for {} at {}",
                        symbol, price
                    );
                    let params = OrderPlaceParams::builder(
                        symbol.clone(),
                        OrderPlaceSideEnum::Buy,
                        OrderPlaceTypeEnum::Market,
                    )
                    .quantity(quantity)
                    .build()
                    .unwrap();

                    if mode == TradeMode::Emulated {
                        info!("BinanceTrader (emulated): Buy order placed successfully");
                        // Simulate a buy in emulated mode
                        self.position.quantity = quantity.to_f64().unwrap_or(0.0);
                        self.position.entry_price = price;
                    } else {
                        match self.connection.order_place(params).await {
                            Ok(response) => {
                                let data = response.data().unwrap();
                                info!("BinanceTrader: Buy order placed successfully: {:?}", data);
                                self.position.quantity = data
                                    .executed_qty
                                    .as_ref()
                                    .and_then(|qty| qty.parse().ok())
                                    .unwrap_or_default();
                                self.position.entry_price = data
                                    .fills
                                    .as_ref()
                                    .and_then(|fills| fills.first())
                                    .and_then(|f| f.price.as_ref().and_then(|p| p.parse().ok()))
                                    .unwrap_or(price);
                            }
                            Err(e) => {
                                error!("BinanceTrader: Failed to place buy order: {:?}", e);
                            }
                        }
                    }
                }
            }
            Signal::Sell => {
                if self.position.quantity > 0.0 {
                    let pnl = (price - self.position.entry_price) * self.position.quantity;
                    self.realized_pnl += pnl;

                    info!(
                        "BinanceTrader: Placing SELL order for {} at {}",
                        symbol, price
                    );
                    let params = OrderPlaceParams::builder(
                        symbol.clone(),
                        OrderPlaceSideEnum::Sell,
                        OrderPlaceTypeEnum::Market,
                    )
                    .quantity(quantity)
                    .build()
                    .unwrap();

                    if mode == TradeMode::Emulated {
                        info!(
                            "BinanceTrader (emulated): Placing SELL order for {} at {}",
                            symbol, price
                        );
                        // Simulate a sell in emulated mode
                        self.position.quantity = 0.0;
                        self.position.entry_price = 0.0;
                        self.realized_pnl += pnl;
                    } else {
                        match self.connection.order_place(params).await {
                            Ok(response) => {
                                let data = response.data().unwrap();
                                info!("BinanceTrader: Sell order placed successfully: {:?}", data);
                                self.position.quantity = 0.0; // Assuming full sell
                                self.position.entry_price = 0.0;
                            }
                            Err(e) => {
                                error!("BinanceTrader: Failed to sell order: {:?}", e);
                            }
                        }
                    }
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
}
