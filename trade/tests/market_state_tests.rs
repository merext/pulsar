use trade::market::{BookLevel, BookTicker, DepthLevel, DepthSnapshot, MarketEvent, MarketState};
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

#[test]
fn microstructure_state_updates_ema_mid_and_spread() {
    let mut state = MarketState::new("DOGEUSDT", 5_000);

    // First BookTicker seeds the EMA
    state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 100.0,
        },
        ask: BookLevel {
            price: 0.1002,
            quantity: 100.0,
        },
        event_time: 1_000,
    }));

    let micro = state.micro();
    assert!(
        (micro.ema_mid_price - 0.1001).abs() < 1e-10,
        "first mid seeds EMA"
    );
    let expected_spread_bps = (0.1002 - 0.1000) / 0.1001 * 10_000.0;
    assert!(
        (micro.ema_spread_bps - expected_spread_bps).abs() < 0.1,
        "first spread seeds EMA"
    );

    // Second BookTicker with wider spread updates via EMA
    state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 100.0,
        },
        ask: BookLevel {
            price: 0.1006,
            quantity: 100.0,
        },
        event_time: 1_100,
    }));

    let micro = state.micro();
    // EMA mid should be between old and new
    let new_mid = 0.1003;
    assert!(
        micro.ema_mid_price > 0.1001 && micro.ema_mid_price < new_mid,
        "EMA mid should lag toward new mid: {}",
        micro.ema_mid_price
    );
    // Spread EMA should have increased
    assert!(
        micro.ema_spread_bps > expected_spread_bps,
        "spread EMA should increase: {}",
        micro.ema_spread_bps
    );
}

#[test]
fn microstructure_state_computes_realized_volatility() {
    let mut state = MarketState::new("DOGEUSDT", 5_000);

    // Need at least two trades for volatility
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1000,
        quantity: 10.0,
        trade_time: 1_000,
        ..Default::default()
    }));

    let micro = state.micro();
    assert_eq!(micro.realized_vol_bps, 0.0, "vol is zero after first trade");

    // Second trade creates a return
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1010,
        quantity: 10.0,
        trade_time: 1_100,
        ..Default::default()
    }));

    let micro = state.micro();
    assert!(
        micro.realized_vol_bps > 0.0,
        "vol should be positive after two trades: {}",
        micro.realized_vol_bps
    );

    // Trade rate should also be populated
    assert!(
        micro.trade_rate_per_second > 0.0,
        "trade rate should be positive: {}",
        micro.trade_rate_per_second
    );
    // 100ms apart = 10 trades/second
    assert!(
        (micro.trade_rate_per_second - 10.0).abs() < 1.0,
        "trade rate ~10/s: {}",
        micro.trade_rate_per_second
    );
}

#[test]
fn microstructure_state_computes_depth_imbalance() {
    let mut state = MarketState::new("DOGEUSDT", 5_000);

    // Bid-heavy depth
    state.apply(&MarketEvent::Depth(DepthSnapshot {
        bids: vec![
            DepthLevel {
                price: 0.1000,
                quantity: 500.0,
            },
            DepthLevel {
                price: 0.0999,
                quantity: 300.0,
            },
        ],
        asks: vec![
            DepthLevel {
                price: 0.1001,
                quantity: 100.0,
            },
            DepthLevel {
                price: 0.1002,
                quantity: 50.0,
            },
        ],
        event_time: 1_000,
    }));

    let micro = state.micro();
    assert!(
        micro.depth_imbalance > 0.0,
        "should be bid-heavy: {}",
        micro.depth_imbalance
    );

    // Ask-heavy depth
    state.apply(&MarketEvent::Depth(DepthSnapshot {
        bids: vec![DepthLevel {
            price: 0.1000,
            quantity: 50.0,
        }],
        asks: vec![
            DepthLevel {
                price: 0.1001,
                quantity: 500.0,
            },
            DepthLevel {
                price: 0.1002,
                quantity: 300.0,
            },
        ],
        event_time: 1_100,
    }));

    let micro = state.micro();
    assert!(
        micro.depth_imbalance < 0.0,
        "should be ask-heavy: {}",
        micro.depth_imbalance
    );
}

#[test]
fn microstructure_state_trade_rate_handles_batched_trades() {
    let mut state = MarketState::new("DOGEUSDT", 5_000);

    // Same-timestamp trades (batched)
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1000,
        quantity: 10.0,
        trade_time: 1_000,
        ..Default::default()
    }));
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1001,
        quantity: 10.0,
        trade_time: 1_000,
        ..Default::default()
    }));

    let micro = state.micro();
    // Same timestamp trades should not update rate (dt_millis == 0)
    assert_eq!(
        micro.trade_rate_per_second, 0.0,
        "same-timestamp trades should not produce rate: {}",
        micro.trade_rate_per_second
    );

    // Now a trade 100ms later
    state.apply(&MarketEvent::Trade(Trade {
        price: 0.1002,
        quantity: 10.0,
        trade_time: 1_100,
        ..Default::default()
    }));

    let micro = state.micro();
    assert!(
        micro.trade_rate_per_second > 0.0,
        "should have rate after time gap"
    );
}
