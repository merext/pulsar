use binance_exchange::client::BinanceClient;
use env_logger::Builder;
use log::LevelFilter;
use strategies::mean_reversion::MeanReversionStrategy;
use strategies::position::Position;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt; // For using .next() on streams

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .init();

    // Instantiate the strategy with required state
    let mut strategy = MeanReversionStrategy {
        window_size: 20,
        prices: Vec::new(),
        last_sma: None,
        recent_trades: Vec::new(),
        max_trade_window: 40,
        position: Position::Flat,
    };

    // Subscribe to Kline and Trade streams
    // let mut trade_stream = BinanceClient::subscribe_trades("DOGEUSDT").await?;
    // let mut kline_stream = BinanceClient::subscribe_klines("DOGEUSDT").await?;
    let mut kline_stream = BinanceClient::backtest_klines(
        "https://data.binance.vision/data/spot/daily/klines/DOGEUSDT/1m/DOGEUSDT-1m-2025-03-04.zip",
        "DOGEUSDT",
        "1m",
    )
    .await?;
    while let Some(kline) = kline_stream.next().await {
        strategy.on_kline(kline).await;
    }

    // Concurrently process both streams
    loop {
        tokio::select! {
            // Some(trade) = trade_stream.next() => {
            //     strategy.on_trade(trade).await;
            // }

            Some(kline) = kline_stream.next() => {
                strategy.on_kline(kline).await;
            }

            else => break,
        }
    }

    Ok(())
}
