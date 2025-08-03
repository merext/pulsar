// RUST_LOG=info,binance_sdk::common::websocket=error cargo run backtest --url https://data.binance.vision/data/spot/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-05-30.zip
use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::debug;
use trade::models::Trade;
use trade::trader::Trader;

pub async fn run_backtest(
    source: &str,
    mut strategy: impl Strategy + Send,
    binance_trader: &mut BinanceTrader,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new().await?;
    let mut trade_stream: Box<dyn futures_util::Stream<Item = Trade> + Unpin> =
        if source.starts_with("http") {
            Box::new(binance_client.trade_data(source).await?)
        } else {
            Box::new(binance_client.trade_data_from_path(source).await?)
        };

    while let Some(trade) = trade_stream.next().await {
        strategy.on_trade(trade.clone().into()).await;

        let trade_price = trade.price;
        let trade_time = trade.trade_time as f64;

        let (signal, confidence) =
            strategy.get_signal(trade_price, trade_time, binance_trader.position());

        let min_notional = 1.0 + 3.0 * confidence;
        let raw_quantity = min_notional / trade_price;
        let quantity_step = 1.0; // get this from exchangeInfo or hardcode per symbol
        let quantity_to_trade = (raw_quantity / quantity_step).ceil() * quantity_step;
        binance_trader
            .on_emulate(signal, trade_price, quantity_to_trade)
            .await;

        debug!(
            signal = %signal,
            confidence = %confidence,
            position = ?binance_trader.position(),
            unrealized_pnl = format!("{:.6}", binance_trader.unrealized_pnl(trade_price)),
            realized_pnl = format!("{:.6}", binance_trader.realized_pnl())
        );
    }

    Ok(())
}
