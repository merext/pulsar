use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use std::env;
use std::time::Duration;
use strategies::strategy::Strategy;
use strategies::zscore_strategy::ZScoreStrategy;
use tokio_stream::StreamExt;
use tracing::{debug, error, warn};
use trade::trader::{TradeMode, Trader};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    let timeout = 30;
    tracing_subscriber::fmt::init();

    // Instantiate the strategy with required state
    // let period = 14;
    // let overbought = 70.0;
    // let oversold = 30.0;
    // let mut strategy = RsiStrategy::new(period, overbought, oversold);

    // Kalman Filter Strategy
    // let signal_threshold = 0.0001; // Example threshold, needs tuning
    // let mut strategy = KalmanFilterStrategy::new(signal_threshold);

    // Mean Reversion Strategy
    // let window_size = 20; // Example window size
    // let max_trade_window = 10; // Example max trade window
    // let mut strategy = MeanReversionStrategy::new(window_size, max_trade_window);

    // Momentum Scalping Strategy
    // let trade_window_size = 5; // Example window size
    // let price_change_threshold = 0.00001; // Example threshold
    // let mut strategy = MomentumScalping::new(trade_window_size, price_change_threshold);

    // Order Book Imbalance Strategy
    // let period = 100; // Example period (number of trades)
    // let buy_threshold = 0.1; // Example buy threshold
    // let sell_threshold = -0.1; // Example sell threshold
    // let mut strategy = OrderBookImbalance::new(period, buy_threshold, sell_threshold);

    // Spline Strategy
    // let window_size = 10; // Example window size
    // let interpolation = Interpolation::Linear; // Example interpolation type
    // let mut strategy = SplineStrategy::new(window_size, interpolation);

    // VWAP Deviation Strategy (Placeholder - requires re-implementation)
    // let period = 100; // Example period
    // let deviation_threshold = 0.001; // Example threshold
    // let mut strategy = VwapDeviationStrategy::new(period, deviation_threshold);

    // Z-Score Strategy
    let period = 50; // Example period
    let buy_threshold = -1.5; // Example buy threshold
    let sell_threshold = 1.5; // Example sell threshold
    let mut strategy = ZScoreStrategy::new(period, buy_threshold, sell_threshold);

    // Initialize trader with API credentials
    let trading_symbol = "DOGEUSDT";
    // IMPORTANT: Replace with your actual Binance API Key and Secret
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set in the environment");
    let api_secret =
        env::var("BINANCE_API_SECRET").expect("API_SECRET must be set in the environment");

    let mut binance_trader = BinanceTrader::new(&trading_symbol, &api_key, &api_secret).await;

    binance_trader.account_status().await?;

    loop {
        let mut trade_stream = loop {
            match BinanceClient::new().await {
                Ok(binance_client) => match binance_client.trade_stream(&trading_symbol).await {
                    Ok(stream) => break stream,
                    Err(e) => {
                        error!(
                            "action=create_trade_stream status=failed error={:?} retry_in_seconds={}",
                            e, timeout
                        );
                        tokio::time::sleep(tokio::time::Duration::from_secs(10)).await;
                    }
                },
                Err(e) => {
                    error!(
                        "action=connect_to_binance status=failed error={:?} retry_in_seconds={}",
                        e, timeout
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(timeout)).await;
                }
            }
        };

        // Process trade stream
        #[allow(unreachable_code)] // This loop is intended to run indefinitely for a live bot
        loop {
            tokio::select! {
                trade_result = tokio::time::timeout(Duration::from_secs(timeout), trade_stream.next()) => {
                    match trade_result {
                        Ok(Some(trade)) => {
                            // log::info!("Received trade: {:?}", trade);
                            strategy.on_trade(trade.clone().into()).await;

                            let trade_price = trade.price;
                            let trade_time = trade.trade_time as f64;

                            let (signal, confidence) = strategy.get_signal(trade_price, trade_time, binance_trader.position());


                            let min_notional = 1.0 + 4.0 * confidence;
                            let raw_quantity = min_notional / trade_price;
                            let quantity_step = 1.0; // get this from exchangeInfo or hardcode per symbol
                            let quantity_to_trade = (raw_quantity / quantity_step).ceil() * quantity_step;
                            binance_trader.on_signal(signal, trade_price, quantity_to_trade, TradeMode::Emulated).await;

                            debug!(
                                signal = %signal,
                                confidence = %confidence,
                                position = ?binance_trader.position(),
                                unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
                                realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
                            );
                        }
                        Ok(None) => {
                            // Stream ended, break the loop to reconnect
                            break;
                        }
                        Err(_) => {
                            warn!("action=trade_stream_timeout status=reconnecting");
                            break;
                        }
                    }
                },
            }
        }
    }
}
