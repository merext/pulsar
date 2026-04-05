use std::fs;
use strategies::SpreadRegimeCaptureStrategy;
use trade::market::{BookLevel, BookTicker, DepthLevel, DepthSnapshot, MarketEvent, MarketState};
use trade::strategy::{Strategy, StrategyContext};
use trade::trader::OrderType;
use trade::{OrderIntent, Position, Side, Trade};

fn flat_context() -> StrategyContext {
    StrategyContext {
        symbol: "DOGEUSDT".to_string(),
        current_position: Position {
            symbol: "DOGEUSDT".to_string(),
            quantity: 0.0,
            entry_price: 0.0,
            entry_time: 0.0,
        },
        available_cash: 100.0,
        max_position_notional: 35.0,
        initial_capital: 100.0,
        tick_size: 0.01,
        step_size: None,
        min_notional: None,
    }
}

fn long_context(quantity: f64, entry_price: f64, entry_time: f64) -> StrategyContext {
    StrategyContext {
        symbol: "DOGEUSDT".to_string(),
        current_position: Position {
            symbol: "DOGEUSDT".to_string(),
            quantity,
            entry_price,
            entry_time,
        },
        available_cash: 100.0,
        max_position_notional: 35.0,
        initial_capital: 100.0,
        tick_size: 0.01,
        step_size: None,
        min_notional: None,
    }
}

/// Helper: build a strategy with relaxed thresholds for testing.
fn test_config_content() -> String {
    r#"
trade_window_millis = 5000
fair_value_w_microprice = 0.5
fair_value_w_ema_mid = 0.3
fair_value_w_vwap = 0.2
base_threshold_bps = 2.0
vol_scale = 0.0
spread_scale = 0.0
mean_reversion_factor = 1.0
half_round_trip_cost_bps = 0.0
min_edge_after_cost_bps = 0.0
min_vol_bps = 0.0
max_vol_bps = 10000.0
max_entry_spread_bps = 100.0
max_adverse_depth = 1.0
min_trades_in_window = 3
min_flow_imbalance = 0.0
stop_loss_bps = 15.0
take_profit_bps = 12.0
max_hold_millis = 5000
panic_vol_bps = 50000.0
exit_dislocation_reversal_bps = 8.0
entry_cooldown_millis = 0
assumed_round_trip_taker_cost_bps = 0.0
min_expected_edge_after_cost_bps = 0.0
"#
    .to_string()
}

/// Seed market state with a BookTicker and several trades at a stable price.
async fn seed_market(
    strategy: &mut SpreadRegimeCaptureStrategy,
    market_state: &mut MarketState,
    base_price: f64,
) {
    // Book ticker establishes mid price and spread
    let book = MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: base_price - 0.00005,
            quantity: 500.0,
        },
        ask: BookLevel {
            price: base_price + 0.00005,
            quantity: 500.0,
        },
        event_time: 1_000,
    });
    strategy.on_event(&book, market_state).await;
    market_state.apply(&book);

    // Several trades at base price to establish VWAP and initialize vol
    for i in 0..6 {
        let trade = MarketEvent::Trade(Trade {
            price: base_price + (i as f64 * 0.00001 - 0.000025),
            quantity: 100.0,
            trade_time: 1_100 + i as u64 * 100,
            is_buyer_market_maker: i % 2 == 0,
            ..Default::default()
        });
        strategy.on_event(&trade, market_state).await;
        market_state.apply(&trade);
    }
}

#[tokio::test]
async fn enters_long_on_price_dislocation_below_fair_value() {
    let config_path = std::env::temp_dir().join("src_test_enter_long.toml");
    fs::write(&config_path, test_config_content()).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Now a large dislocation downward: price drops well below fair value
    let dislocation_trade = MarketEvent::Trade(Trade {
        price: base_price * 0.997, // ~30 bps below
        quantity: 200.0,
        trade_time: 2_000,
        is_buyer_market_maker: true,
        ..Default::default()
    });
    strategy.on_event(&dislocation_trade, &market_state).await;
    market_state.apply(&dislocation_trade);

    let decision = strategy.decide(&market_state, &flat_context());
    match &decision.intent {
        OrderIntent::Place {
            side,
            order_type,
            rationale,
            expected_edge_bps,
            ..
        } => {
            assert_eq!(
                *side,
                Side::Buy,
                "should enter long on downward dislocation"
            );
            assert_eq!(
                *order_type,
                OrderType::Maker,
                "entry should use maker order"
            );
            assert_eq!(*rationale, "spread_regime_capture_entry");
            assert!(*expected_edge_bps > 0.0, "edge should be positive");
        }
        _ => panic!("expected buy intent, got {:?}", decision.intent),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn no_entry_on_upward_dislocation_long_only() {
    let config_path = std::env::temp_dir().join("src_test_no_short.toml");
    fs::write(&config_path, test_config_content()).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Large dislocation upward: price jumps above fair value
    // In the old strategy this would trigger a short entry.
    // Now it should produce no action (long-only).
    let dislocation_trade = MarketEvent::Trade(Trade {
        price: base_price * 1.003, // ~30 bps above
        quantity: 200.0,
        trade_time: 2_000,
        is_buyer_market_maker: false,
        ..Default::default()
    });
    strategy.on_event(&dislocation_trade, &market_state).await;
    market_state.apply(&dislocation_trade);

    let decision = strategy.decide(&market_state, &flat_context());
    assert!(
        matches!(decision.intent, OrderIntent::NoAction),
        "should NOT enter on upward dislocation (long-only strategy)"
    );

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn no_entry_when_dislocation_is_small() {
    let config_path = std::env::temp_dir().join("src_test_no_entry.toml");
    fs::write(&config_path, test_config_content()).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Small dislocation — below threshold
    let small_trade = MarketEvent::Trade(Trade {
        price: base_price * 0.99999, // ~0.1 bps, well below 2 bps threshold
        quantity: 100.0,
        trade_time: 2_000,
        is_buyer_market_maker: false,
        ..Default::default()
    });
    strategy.on_event(&small_trade, &market_state).await;
    market_state.apply(&small_trade);

    let decision = strategy.decide(&market_state, &flat_context());
    assert!(
        matches!(decision.intent, OrderIntent::NoAction),
        "should not enter on small dislocation"
    );

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_on_stop_loss_via_taker() {
    let config_path = std::env::temp_dir().join("src_test_stop_loss.toml");
    fs::write(&config_path, test_config_content()).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    // Seed then simulate an entry
    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Simulate that we entered long
    strategy.entry_side = Some(Side::Buy);
    strategy.last_entry_time_millis = Some(1_500);

    // Now price drops below stop loss (15 bps)
    let book = MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: base_price * (1.0 - 0.0020), // ~20 bps below entry
            quantity: 500.0,
        },
        ask: BookLevel {
            price: base_price * (1.0 - 0.0018),
            quantity: 500.0,
        },
        event_time: 3_000,
    });
    strategy.on_event(&book, &market_state).await;
    market_state.apply(&book);

    let ctx = long_context(300.0, base_price, 1.5);
    let decision = strategy.decide(&market_state, &ctx);

    match &decision.intent {
        OrderIntent::Place {
            side,
            order_type,
            rationale,
            ..
        } => {
            assert_eq!(*side, Side::Sell, "should exit long position");
            assert_eq!(
                *order_type,
                OrderType::Taker,
                "stop loss should use taker (emergency)"
            );
            assert_eq!(*rationale, "stop_loss");
        }
        _ => panic!("expected stop loss exit, got {:?}", decision.intent),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_on_max_hold_time_via_taker() {
    let config_path = std::env::temp_dir().join("src_test_max_hold.toml");
    fs::write(&config_path, test_config_content()).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Simulate entry at time 1.5s
    strategy.entry_side = Some(Side::Buy);
    strategy.last_entry_time_millis = Some(1_500);

    // Trade at time 7s (5.5s later > 5s max_hold)
    let late_trade = MarketEvent::Trade(Trade {
        price: base_price,
        quantity: 100.0,
        trade_time: 7_000,
        ..Default::default()
    });
    strategy.on_event(&late_trade, &market_state).await;
    market_state.apply(&late_trade);

    // Also need a book ticker to provide bid price for exit
    let book = MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: base_price,
            quantity: 500.0,
        },
        ask: BookLevel {
            price: base_price + 0.00005,
            quantity: 500.0,
        },
        event_time: 7_100,
    });
    strategy.on_event(&book, &market_state).await;
    market_state.apply(&book);

    let ctx = long_context(300.0, base_price, 1.5); // entry_time = 1.5s
    let decision = strategy.decide(&market_state, &ctx);

    match &decision.intent {
        OrderIntent::Place {
            side,
            order_type,
            rationale,
            ..
        } => {
            assert_eq!(*side, Side::Sell);
            assert_eq!(
                *order_type,
                OrderType::Taker,
                "max_hold exit should use taker"
            );
            assert_eq!(*rationale, "max_hold_time");
        }
        _ => panic!("expected max_hold exit, got {:?}", decision.intent),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_take_profit_via_maker() {
    let config_path = std::env::temp_dir().join("src_test_take_profit.toml");
    fs::write(&config_path, test_config_content()).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Simulate that we entered long at base_price
    strategy.entry_side = Some(Side::Buy);
    strategy.last_entry_time_millis = Some(1_500);

    // Price rises above take_profit_bps (12 bps)
    let book = MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: base_price * 1.0015, // ~15 bps above entry
            quantity: 500.0,
        },
        ask: BookLevel {
            price: base_price * 1.0017,
            quantity: 500.0,
        },
        event_time: 3_000,
    });
    strategy.on_event(&book, &market_state).await;
    market_state.apply(&book);

    let ctx = long_context(300.0, base_price, 1.5);
    let decision = strategy.decide(&market_state, &ctx);

    match &decision.intent {
        OrderIntent::Place {
            side,
            order_type,
            rationale,
            ..
        } => {
            assert_eq!(*side, Side::Sell, "should exit long position");
            assert_eq!(
                *order_type,
                OrderType::Maker,
                "take profit should use maker order"
            );
            assert_eq!(*rationale, "take_profit");
        }
        _ => panic!("expected take_profit exit, got {:?}", decision.intent),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exposes_diagnostics_after_blocked_entry() {
    let mut strategy = SpreadRegimeCaptureStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    // Only 2 trades — below min_trades_in_window (5 by default)
    let book = MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 100.0,
        },
        ask: BookLevel {
            price: 0.1001,
            quantity: 100.0,
        },
        event_time: 1_000,
    });
    strategy.on_event(&book, &market_state).await;
    market_state.apply(&book);

    for i in 0..2 {
        let trade = MarketEvent::Trade(Trade {
            price: 0.1000,
            quantity: 10.0,
            trade_time: 1_100 + i * 100,
            ..Default::default()
        });
        strategy.on_event(&trade, &market_state).await;
        market_state.apply(&trade);
    }

    let _ = strategy.decide(&market_state, &flat_context());
    let diagnostics = strategy.diagnostics();
    assert_eq!(
        diagnostics.counters.get("src.blocked_min_trades"),
        Some(&1),
        "should block on min_trades"
    );
}

#[tokio::test]
async fn adverse_depth_blocks_entry() {
    let config_path = std::env::temp_dir().join("src_test_depth_block.toml");
    // Strict adverse depth filter (max_adverse_depth = 0.1)
    let mut config = test_config_content();
    config = config.replace("max_adverse_depth = 1.0", "max_adverse_depth = 0.1");
    fs::write(&config_path, &config).expect("write config");

    let mut strategy =
        SpreadRegimeCaptureStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let base_price = 0.1000;
    seed_market(&mut strategy, &mut market_state, base_price).await;

    // Heavy ask-side depth (adverse for a buy entry)
    let depth = MarketEvent::Depth(DepthSnapshot {
        bids: vec![DepthLevel {
            price: 0.0999,
            quantity: 10.0,
        }],
        asks: vec![
            DepthLevel {
                price: 0.1001,
                quantity: 10000.0,
            },
            DepthLevel {
                price: 0.1002,
                quantity: 5000.0,
            },
        ],
        event_time: 1_800,
    });
    strategy.on_event(&depth, &market_state).await;
    market_state.apply(&depth);

    // Dislocation downward -> would try to buy, but depth is strongly adverse
    let trade = MarketEvent::Trade(Trade {
        price: base_price * 0.997,
        quantity: 200.0,
        trade_time: 2_000,
        is_buyer_market_maker: true,
        ..Default::default()
    });
    strategy.on_event(&trade, &market_state).await;
    market_state.apply(&trade);

    let decision = strategy.decide(&market_state, &flat_context());
    assert!(
        matches!(decision.intent, OrderIntent::NoAction),
        "should block on adverse depth"
    );

    let diagnostics = strategy.diagnostics();
    assert!(
        diagnostics
            .counters
            .get("src.blocked_adverse_depth")
            .is_some_and(|v| *v > 0),
        "should count adverse depth blocks"
    );

    let _ = fs::remove_file(config_path);
}
