use async_trait::async_trait;
use binance_sdk::config::ConfigurationWebsocketApi;
use binance_sdk::spot::SpotWsApi;
use binance_sdk::spot::websocket_api::{
    OrderPlaceParams, OrderPlaceSideEnum, OrderPlaceTypeEnum, WebsocketApi,
};
use log::error;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use rust_decimal_macros::dec;
use trade::signal::Signal;
use trade::trader::{Position, Trader};

pub struct BinanceTrader {
    connection: WebsocketApi,
    pub position: Position,
    pub realized_pnl: f64,
}

impl BinanceTrader {
    pub async fn new(symbol: String, api_key: &str, api_secret: &str) -> Self {
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
                symbol,
                quantity: 0.0,
                entry_price: 0.0,
            },
            realized_pnl: 0.0,
        }
    }
}

#[async_trait]
impl Trader for BinanceTrader {
    async fn on_signal(&mut self, signal: Signal, price: f64) {
        let symbol = self.position.symbol.clone();
        let quantity = dec!(10.0); // Example fixed quantity, converted to Decimal

        match signal {
            Signal::Buy => {
                if self.position.quantity == 0.0 {
                    println!(
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

                    match self.connection.order_place(params).await {
                        Ok(response) => {
                            let data = response.data().unwrap();
                            println!("BinanceTrader: Buy order placed successfully: {:?}", data);
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
            Signal::Sell => {
                if self.position.quantity > 0.0 {
                    let pnl = (price - self.position.entry_price) * self.position.quantity;
                    self.realized_pnl += pnl;

                    println!(
                        "BinanceTrader: Placing SELL order for {} at {}",
                        symbol, price
                    );
                    let params = OrderPlaceParams::builder(
                        symbol.clone(),
                        OrderPlaceSideEnum::Sell,
                        OrderPlaceTypeEnum::Market,
                    )
                    .quantity(Decimal::from_f64(self.position.quantity).unwrap_or_default())
                    .build()
                    .unwrap();

                    match self.connection.order_place(params).await {
                        Ok(response) => {
                            let data = response.data().unwrap();
                            println!("BinanceTrader: Sell order placed successfully: {:?}", data);
                            self.position.quantity = 0.0; // Assuming full sell
                            self.position.entry_price = 0.0;
                        }
                        Err(e) => {
                            error!("BinanceTrader: Failed to sell order: {:?}", e);
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
