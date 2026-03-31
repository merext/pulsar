use trade::market::{BookLevel, BookTicker, MarketEvent, MarketState};
use trade::Trade;

#[test]
fn market_state_computes_quote_metrics() {
    let mut state = MarketState::new("DOGEUSDT", 1_000);
    state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 200.0,
        },
        ask: BookLevel {
            price: 0.1002,
            quantity: 100.0,
        },
        event_time: 1,
    }));

    let mid = state.mid_price().expect("mid price available");
    let microprice = state.microprice().expect("microprice available");
    let imbalance = state.order_book_imbalance().expect("imbalance available");

    assert!((mid - 0.1001).abs() < 1e-12);
    assert!(microprice > mid);
    assert!(imbalance > 0.0);
}

#[test]
fn market_state_tracks_trade_flow_imbalance() {
    let mut state = MarketState::new("DOGEUSDT", 1_000);

    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1000,
        quantity: 100.0,
        trade_time: 1_000,
        is_buyer_market_maker: false,
        ..Default::default()
    }));
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1001,
        quantity: 50.0,
        trade_time: 1_200,
        is_buyer_market_maker: true,
        ..Default::default()
    }));

    let stats = state.trade_window_stats();
    let imbalance = state.trade_flow_imbalance();

    assert_eq!(stats.trade_count, 2);
    assert_eq!(stats.volume, 150.0);
    assert!(imbalance > 0.0);
}

#[test]
fn market_state_trims_old_trades() {
    let mut state = MarketState::new("DOGEUSDT", 500);

    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1000,
        quantity: 100.0,
        trade_time: 1_000,
        ..Default::default()
    }));
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1001,
        quantity: 100.0,
        trade_time: 1_700,
        ..Default::default()
    }));

    let stats = state.trade_window_stats();
    assert_eq!(stats.trade_count, 1);
    assert_eq!(stats.last_price, 0.1001);
}
