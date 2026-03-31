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

#[test]
fn market_state_tracks_latest_event_time_across_event_types() {
    let mut state = MarketState::new("DOGEUSDT", 1_000);

    state.apply(&MarketEvent::Trade(Trade {
        trade_time: 1_000,
        ..Default::default()
    }));
    state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 100.0,
        },
        ask: BookLevel {
            price: 0.1001,
            quantity: 100.0,
        },
        event_time: 1_500,
    }));

    assert_eq!(state.last_event_time_millis(), Some(1_500));
    assert_eq!(state.last_event_time_secs(), Some(1.5));
}

#[test]
fn market_state_computes_recent_trade_stats_and_vwap() {
    let mut state = MarketState::new("DOGEUSDT", 1_000);

    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1000,
        quantity: 10.0,
        trade_time: 1_000,
        is_buyer_market_maker: true,
        ..Default::default()
    }));
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1002,
        quantity: 20.0,
        trade_time: 1_100,
        is_buyer_market_maker: false,
        ..Default::default()
    }));
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1004,
        quantity: 30.0,
        trade_time: 1_200,
        is_buyer_market_maker: false,
        ..Default::default()
    }));

    let recent_stats = state.recent_trade_window_stats(2);
    assert_eq!(recent_stats.trade_count, 2);
    assert_eq!(recent_stats.last_price, 0.1004);
    assert!(state.recent_trade_flow_imbalance(2) > 0.0);

    let recent_vwap = state
        .recent_trade_window_vwap(2)
        .expect("recent vwap available");
    assert!(recent_vwap > 0.1002);
    assert!(recent_vwap < 0.1004);
}

#[test]
fn market_state_tracks_event_mix_diagnostics() {
    let mut state = MarketState::new("DOGEUSDT", 1_000);

    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1000,
        quantity: 1.0,
        trade_time: 1_000,
        ..Default::default()
    }));
    state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.0999,
            quantity: 10.0,
        },
        ask: BookLevel {
            price: 0.1001,
            quantity: 12.0,
        },
        event_time: 1_100,
    }));
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1002,
        quantity: 2.0,
        trade_time: 1_050,
        ..Default::default()
    }));

    let diagnostics = state.event_mix_diagnostics();
    assert_eq!(diagnostics.trade_events, 2);
    assert_eq!(diagnostics.book_ticker_events, 1);
    assert_eq!(diagnostics.trade_without_quote_events, 1);
    assert_eq!(diagnostics.stale_quote_events, 1);
}
