use binance_exchange::client::BinanceClient;
use binance_exchange::trader::BinanceTrader;
use chrono::Utc;
use env_logger::Builder;
use log::LevelFilter;
use std::io::Write;
use strategies::rsi_strategy::RsiStrategy;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use trade::trader::Trader;
// For using .next() on streams

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
    let trading_symbol = "DOGEUSDT".to_string();
    let mut trader = BinanceTrader::new(trading_symbol.clone());

    let mut kline_stream = BinanceClient::kline_stream(&trading_symbol, "1m").await?;

    let mut trade_stream = BinanceClient::subscribe_trades(&trading_symbol).await?;

    // Process remaining klines
    #[allow(unreachable_code)] // This loop is intended to run indefinitely for a live bot
    loop {
        tokio::select! {
            kline_result = kline_stream.next() => {
                if let Some(kline) = kline_result {
                    let close_price = kline.close;
                    let close_time = kline.close_time as f64;

                    let signal = strategy.get_signal(close_price, close_time, trader.position());

                    trader.on_signal(signal, close_price).await;

                    let position = trader.position();

                    log::info!(
                        "{} | {} | Position: {}, Unrealized PnL: {:.5}, Realized PnL: {:.5}",
                        Utc::now().timestamp(), // Unix timestamp (in seconds)
                        signal,
                        position.to_string(),
                        trader.unrealized_pnl(close_price),
                        trader.realized_pnl()
                    );

                    strategy.on_kline(kline).await;
                } else {
                    // In live, kline stream might just be slow, keep waiting for trades
                    {}
                }
            },
            trade_result = trade_stream.next() => {
                if let Some(trade) = trade_result {
                    strategy.on_trade(trade).await;
                }
            },
        }
    }
}
