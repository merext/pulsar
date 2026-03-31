use binance_exchange::BinanceClient;
use futures_util::StreamExt;
use trade::market::MarketEvent;

#[tokio::test]
async fn parses_simple_csv_fixture() {
    let trades: Vec<_> = BinanceClient::trade_data_from_uri(
        "../../data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip",
    )
        .await
        .expect("fixture loads")
        .collect::<Vec<_>>()
        .await;

    assert!(!trades.is_empty());
    assert_eq!(trades[0].price, 0.16105);
    assert_eq!(trades[0].quantity, 62.0);
}

#[tokio::test]
async fn historical_trade_stream_maps_to_market_events() {
    let events: Vec<_> = BinanceClient::trade_data_from_uri(
        "../../data/binance/daily/trades/DOGEUSDT/DOGEUSDT-trades-2025-06-28.zip",
    )
    .await
    .expect("fixture loads")
    .map(MarketEvent::Trade)
    .take(2)
    .collect::<Vec<_>>()
    .await;

    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], MarketEvent::Trade(_)));
}
