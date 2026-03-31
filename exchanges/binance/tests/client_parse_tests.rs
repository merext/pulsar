use binance_exchange::BinanceClient;
use futures_util::StreamExt;
use trade::market::MarketEvent;

#[test]
fn live_quote_and_depth_messages_require_exchange_event_time() {
    let book_payload = r#"{"stream":"dogeusdt@bookTicker","data":{"u":24126432035,"b":"0.09117","B":"508947.0","a":"0.09118","A":"26021.0"}}"#;
    let depth_payload = r#"{"stream":"dogeusdt@depth5@100ms","data":{"lastUpdateId":24126432029,"bids":[["0.09117","510725.0"]],"asks":[["0.09118","26021.0"]]}}"#;

    assert!(BinanceClient::parse_market_event_message_for_test(book_payload).is_none());
    assert!(BinanceClient::parse_market_event_message_for_test(depth_payload).is_none());
}

#[tokio::test]
async fn market_event_stream_is_constructible() {
    let client = BinanceClient::new().await.expect("client initializes");
    let _stream = client
        .market_event_stream("DOGEUSDT")
        .await
        .expect("market event stream builds");
}

#[tokio::test]
async fn market_event_stream_with_depth_is_constructible() {
    let client = BinanceClient::new().await.expect("client initializes");
    let _stream = client
        .market_event_stream_with_depth("DOGEUSDT", 5)
        .await
        .expect("market event stream with depth builds");
}

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

#[tokio::test]
async fn parses_captured_jsonl_fixture_into_market_events() {
    let events: Vec<_> = BinanceClient::market_event_data_from_uri("/tmp/pulsar_capture.jsonl")
        .await
        .expect("captured fixture loads")
        .take(4)
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 4);
    assert!(matches!(events[0], MarketEvent::Depth(_)));
    assert!(matches!(events[1], MarketEvent::BookTicker(_)));
    assert!(matches!(events[2], MarketEvent::Depth(_)));
    assert!(matches!(events[3], MarketEvent::Trade(_)));
}

#[tokio::test]
async fn summarizes_captured_jsonl_fixture() {
    let parsed = BinanceClient::load_captured_market_event_data_from_uri("/tmp/pulsar_capture.jsonl")
        .await
        .expect("captured fixture loads");

    assert_eq!(parsed.summary.parsed_events, parsed.events.len());
    assert_eq!(parsed.summary.trade_events, 3);
    assert_eq!(parsed.summary.book_ticker_events, 18);
    assert_eq!(parsed.summary.depth_events, 11);
    assert_eq!(parsed.summary.parse_errors, 0);
    assert_eq!(parsed.summary.event_time_regressions, 2);
    assert_eq!(parsed.summary.symbols, vec!["DOGEUSDT"]);
}
