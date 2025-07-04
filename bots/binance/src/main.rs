use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use env_logger::Builder;
use log::LevelFilter;
use std::env;
use std::io::Write;
use strategies::rsi_strategy::RsiStrategy;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use trade::trader::{TradeMode, Trader}; // For using .next() on streams

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    Builder::new()
        .filter_level(LevelFilter::Debug) // Set fixed log level
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {}:{}] {}",
                record.level(),
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .init();

    // Instantiate the strategy with required state
    let period = 14;
    let overbought = 70.0;
    let oversold = 30.0;
    let mut strategy = RsiStrategy::new(period, overbought, oversold);

    // Initialize trader with API credentials
    let trading_symbol = "DOGEUSDT";
    // IMPORTANT: Replace with your actual Binance API Key and Secret
    let api_key = env::var("BINANCE_API_KEY").expect("API_KEY must be set in the environment");
    let api_secret =
        env::var("BINANCE_API_SECRET").expect("API_SECRET must be set in the environment");

    let mut binance_trader = BinanceTrader::new(&trading_symbol, &api_key, &api_secret).await;
    let binance_client = BinanceClient::new().await;

    let mut kline_stream = binance_client.kline_stream(&trading_symbol, "1m").await?;

    // Process remaining klines
    #[allow(unreachable_code)] // This loop is intended to run indefinitely for a live bot
    loop {
        tokio::select! {
            kline_result = kline_stream.next() => {
                if let Some(kline) = kline_result {
                    let close_price = kline.close;
                    let close_time = kline.close_time as f64;

                    let signal = strategy.get_signal(close_price, close_time, binance_trader.position());

                    binance_trader.on_signal(signal, close_price, 0.01, TradeMode::Emulated).await;

                    log::info!(
                        "Symbol: {}, Signal: {}, Position: {}, Unrealized PnL: {:.5}, Realized PnL: {:.5}",
                        binance_trader.position().symbol,
                        signal,
                        binance_trader.position(),
                        binance_trader.unrealized_pnl(close_price),
                        binance_trader.realized_pnl()
                    );

                    strategy.on_kline(kline).await;
                } else {
                    // In live, kline stream might just be slow, keep waiting for trades
                    {}
                }
            },
        }
    }
}
