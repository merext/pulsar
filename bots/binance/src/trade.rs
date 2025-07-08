use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use trade::trader::{TradeMode, Trader};

pub async fn run_trading(
    trading_symbol: &str,
    _api_key: &str,
    _api_secret: &str,
    mut strategy: impl Strategy + Send,
    binance_trader: &mut BinanceTrader,
    trade_mode: TradeMode,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new().await.expect("Failed to create BinanceClient");
    let mut trade_stream = binance_client.trade_stream(trading_symbol).await.expect("Failed to create trade stream");

    // Process trade stream
    #[allow(unreachable_code)] // This loop is intended to run indefinitely for a live bot
    loop {
        let trade = trade_stream.next().await.expect("Trade stream ended unexpectedly");
        strategy.on_trade(trade.clone().into()).await;

        let trade_price = trade.price;
        let trade_time = trade.trade_time as f64;

        let (signal, confidence) = strategy.get_signal(trade_price, trade_time, binance_trader.position());

        let min_notional = 1.0 + 4.0 * confidence;
        let raw_quantity = min_notional / trade_price;
        let quantity_step = 1.0; // get this from exchangeInfo or hardcode per symbol
        let quantity_to_trade = (raw_quantity / quantity_step).ceil() * quantity_step;

        match trade_mode {
            TradeMode::Real => {
                binance_trader.on_signal(signal, trade_price, quantity_to_trade).await;
            }
            TradeMode::Emulated => {
                binance_trader.on_emulate(signal, trade_price, quantity_to_trade).await;
            }
        }
    }
}
