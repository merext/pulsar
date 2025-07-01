use binance_exchange::client::BinanceClient;
use env_logger::Builder;
use log::LevelFilter;
use std::env;
use std::io::Write;
use strategies::mean_reversion::MeanReversionStrategy;
use strategies::position::Position;
use strategies::strategy::Strategy;
use strategies::trader::{Signal, VirtualTrader};
use tokio_stream::StreamExt; // For using .next() on streams

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

    // Instantiate the strategy with required state (position removed)
    let mut strategy = MeanReversionStrategy {
        window_size: 20,
        prices: Vec::new(),
        last_sma: None,
        recent_trades: Vec::new(),
        max_trade_window: 40,
    };

    // Initialize virtual trader to track position and PnL
    let mut trader = VirtualTrader::new();

    let mut kline_stream = BinanceClient::backtest_klines(
        "https://data.binance.vision/data/spot/daily/klines/DOGEUSDT/1m/DOGEUSDT-1m-2025-03-14.zip",
        "DOGEUSDT",
        "1m",
    )
    .await?;

    // let mut kline_stream = BinanceClient::subscribe_klines("DOGEUSDT").await?;

    while let Some(kline) = kline_stream.next().await {
        strategy.on_kline(kline.clone()).await;

        let close_price = kline.close_price.parse().unwrap_or(0.0);
        let sma = strategy.last_sma.unwrap_or(close_price);

        let signal = strategy.get_signal(close_price, sma, trader.position);

        trader.on_signal(signal, close_price);

        log::debug!(
            "Signal: {}, Position: {:?}, Realized PnL: {:.5}, Unrealized PnL: {:.5}",
            signal,
            trader.position,
            trader.realized_pnl,
            trader.unrealized_pnl(close_price)
        );
    }

    Ok(())
}
