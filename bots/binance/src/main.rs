use binance_exchange::client::BinanceClient;
use env_logger::Builder;
use log::LevelFilter;

#[tokio::main]
async fn main() {
    Builder::from_default_env()
        .filter_level(LevelFilter::Info)
        .init();

    if let Err(e) = BinanceClient::subscribe_klines("BTCUSDT").await {
        eprintln!("WebSocket subscription failed: {}", e);
    }
}