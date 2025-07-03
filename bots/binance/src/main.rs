use futures_util::future::FutureExt; // For .now_or_never()
use binance_exchange::client::BinanceClient;
use env_logger::Builder;
use log::LevelFilter;
use std::io::Write;
use strategies::rsi_strategy::RsiStrategy;
use strategies::strategy::Strategy;
use trade::trader::{VirtualTrader, Trader};
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

    // Instantiate the strategy with required state
    let period = 14;
    let overbought = 70.0;
    let oversold = 30.0;
    let mut strategy = RsiStrategy::new(period, overbought, oversold);

    // Initialize virtual trader to track position and PnL
    let mut trader = VirtualTrader::new();

    #[cfg(feature = "backtest")]
    let mut kline_stream = BinanceClient::backtest_klines(
        "https://data.binance.vision/data/spot/daily/klines/DOGEUSDT/1m/DOGEUSDT-1m-2025-02-21.zip",
        "DOGEUSDT",
        "1m",
    )
    .await?;

    #[cfg(not(feature = "backtest"))]
    let mut kline_stream = BinanceClient::kline_stream("dogeusdt", "1m").await?;

    #[cfg(not(feature = "backtest"))]
    let mut trade_stream = BinanceClient::subscribe_trades("dogeusdt").await?;

    // Process remaining klines
    loop {
        tokio::select! {
            kline_result = kline_stream.next() => {
                if let Some(kline) = kline_result {
                    let close_price = kline.close;
                    let close_time = kline.close_time as f64;

                    let signal = strategy.get_signal(close_price, close_time, trader.position());

                    trader.on_signal(signal, close_price).await;

                    log::info!(
                        "Signal: {:?}, Position: {:?}, Unrealized PnL: {:.5}",
                        signal,
                        trader.position(),
                        trader.unrealized_pnl(close_price)
                    );

                    strategy.on_kline(kline).await;
                } else {
                    // Kline stream exhausted
                    #[cfg(feature = "backtest")]
                    break; // In backtest, if kline stream is done, we are done
                    #[cfg(not(feature = "backtest"))]
                    // In live, kline stream might just be slow, keep waiting for trades
                    {}
                }
            },
            trade_result = async { 
                #[cfg(not(feature = "backtest"))] {
                    trade_stream.next().await
                }
                #[cfg(feature = "backtest")] {
                    futures::future::pending().await
                }
            } => {
                #[cfg(not(feature = "backtest"))]
                if let Some(trade) = trade_result {
                    strategy.on_trade(trade).await;
                } else {
                    // Trade stream exhausted, if kline stream is also exhausted, break
                    if kline_stream.next().now_or_never().is_none() {
                        break;
                    }
                }
            },
        }
    }

    Ok(())
}
