use std::fs;
use strategies::LiquiditySweepReversalStrategy;
use trade::market::{BookLevel, BookTicker, MarketEvent, MarketState};
use trade::strategy::{Strategy, StrategyContext};
use trade::{OrderIntent, Position, Trade};

fn context_with_position(quantity: f64, entry_price: f64, entry_time: f64) -> StrategyContext {
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

#[tokio::test]
async fn enters_after_local_sweep_and_reclaim() {
    let config_path = std::env::temp_dir().join("liquidity_sweep_reversal_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 2000\nmin_trades_in_window = 18\nmin_sweep_drop_bps = 10.0\nmin_buyer_reclaim_imbalance = -0.05\nmin_recent_buyer_imbalance = -0.05\nmin_reclaim_bps = 2.0\nmax_reclaim_bps = 20.0\nmax_reclaim_above_vwap_bps = 50.0\nmax_spread_bps = 14.0\nmin_large_trade_ratio = 0.10\nmin_order_book_imbalance = -1.0\nstop_loss_bps = 16.0\ntake_profit_bps = 20.0\nhold_time_millis = 5000\nentry_cooldown_millis = 3000\nassumed_round_trip_taker_cost_bps = 0.0\nmin_expected_edge_after_cost_bps = 0.0\n",
    )
    .expect("write temp config");

    let mut strategy =
        LiquiditySweepReversalStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let prices = [
        0.1012, 0.1010, 0.1008, 0.1006, 0.1002, 0.0998, 0.0994, 0.0990, 0.0988, 0.0989, 0.0991,
        0.0993, 0.0995, 0.0997, 0.0999, 0.1000, 0.1001, 0.1002,
    ];

    for (index, price) in prices.into_iter().enumerate() {
        let event = MarketEvent::Trade(Trade {
            price,
            quantity: if index < 3 {
                6000.0
            } else if index < 9 {
                2200.0
            } else {
                3200.0
            },
            trade_time: 100 + index as u64 * 90,
            is_buyer_market_maker: index < 9,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.0989,
            quantity: 900.0,
        },
        ask: BookLevel {
            price: 0.09895,
            quantity: 1100.0,
        },
        event_time: 2_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    match decision.intent {
        OrderIntent::Place {
            side, rationale, ..
        } => {
            assert_eq!(side, trade::Side::Buy);
            assert_eq!(rationale, "liquidity_sweep_reversal_entry");
        }
        _ => panic!("expected buy intent"),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_when_reversal_fails() {
    let mut strategy =
        LiquiditySweepReversalStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.0992,
            quantity: 1000.0,
        },
        ask: BookLevel {
            price: 0.0993,
            quantity: 1000.0,
        },
        event_time: 6_000,
    }));

    for index in 0..20 {
        let event = MarketEvent::Trade(Trade {
            price: 0.0995 - (index as f64 * 0.00001),
            quantity: 1800.0,
            trade_time: 4_000 + index as u64 * 70,
            is_buyer_market_maker: true,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let decision = strategy.decide(&market_state, &context_with_position(250.0, 0.09925, 3.0));
    match decision.intent {
        OrderIntent::Place {
            side, rationale, ..
        } => {
            assert_eq!(side, trade::Side::Sell);
            assert_eq!(rationale, "reversal_failed");
        }
        _ => panic!("expected sell intent"),
    }
}

#[tokio::test]
async fn rejects_entry_when_recent_flow_or_book_support_is_weak() {
    let mut strategy =
        LiquiditySweepReversalStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let prices = [
        0.1012, 0.1010, 0.1008, 0.1006, 0.1002, 0.0998, 0.0994, 0.0990, 0.0988, 0.0989, 0.0990,
        0.0991, 0.09915, 0.09918, 0.09920, 0.09919, 0.09918, 0.09917,
    ];

    for (index, price) in prices.into_iter().enumerate() {
        let event = MarketEvent::Trade(Trade {
            price,
            quantity: 2200.0,
            trade_time: 100 + index as u64 * 90,
            is_buyer_market_maker: index < 12,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.09915,
            quantity: 900.0,
        },
        ask: BookLevel {
            price: 0.09920,
            quantity: 1600.0,
        },
        event_time: 2_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    assert!(matches!(decision.intent, OrderIntent::NoAction));
}

#[tokio::test]
async fn exposes_sweep_diagnostics_after_blocked_entry() {
    let mut strategy =
        LiquiditySweepReversalStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    for index in 0..5 {
        let event = MarketEvent::Trade(Trade {
            price: 0.1000 - index as f64 * 0.00001,
            quantity: 50.0,
            trade_time: 100 + index as u64 * 100,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let _ = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    let diagnostics = strategy.diagnostics();
    assert_eq!(
        diagnostics.counters.get("sweep.blocked_min_trades"),
        Some(&1)
    );
}

#[tokio::test]
async fn rejects_entry_when_reclaim_edge_does_not_clear_cost_gate() {
    let config_path = std::env::temp_dir().join("liquidity_sweep_reversal_cost_gate_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 2000\nmin_trades_in_window = 18\nmin_sweep_drop_bps = 10.0\nmin_buyer_reclaim_imbalance = -0.05\nmin_recent_buyer_imbalance = -0.05\nmin_reclaim_bps = 2.0\nmax_reclaim_bps = 20.0\nmax_reclaim_above_vwap_bps = 50.0\nmax_spread_bps = 14.0\nmin_large_trade_ratio = 0.10\nmin_order_book_imbalance = -1.0\nstop_loss_bps = 16.0\ntake_profit_bps = 20.0\nhold_time_millis = 5000\nentry_cooldown_millis = 3000\nassumed_round_trip_taker_cost_bps = 22.7\nmin_expected_edge_after_cost_bps = 0.0\n",
    )
    .expect("write temp config");

    let mut strategy =
        LiquiditySweepReversalStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let prices = [
        0.1012, 0.1010, 0.1008, 0.1006, 0.1002, 0.0998, 0.0994, 0.0990, 0.0988, 0.0989, 0.0991,
        0.0993, 0.0995, 0.0997, 0.0999, 0.1000, 0.1001, 0.1002,
    ];

    for (index, price) in prices.into_iter().enumerate() {
        let event = MarketEvent::Trade(Trade {
            price,
            quantity: if index < 3 {
                6000.0
            } else if index < 9 {
                2200.0
            } else {
                3200.0
            },
            trade_time: 100 + index as u64 * 90,
            is_buyer_market_maker: index < 9,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.0989,
            quantity: 900.0,
        },
        ask: BookLevel {
            price: 0.09895,
            quantity: 1100.0,
        },
        event_time: 2_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    assert!(matches!(decision.intent, OrderIntent::NoAction));

    let diagnostics = strategy.diagnostics();
    assert_eq!(
        diagnostics.counters.get("sweep.blocked_cost_gate"),
        Some(&1)
    );
    assert!(
        diagnostics
            .gauges
            .get("sweep.last_edge_after_cost_bps")
            .is_some_and(|value| *value < 0.0)
    );

    let _ = fs::remove_file(config_path);
}
