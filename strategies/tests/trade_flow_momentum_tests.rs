use strategies::TradeFlowMomentumStrategy;
use std::fs;
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
    }
}

#[tokio::test]
async fn enters_on_strong_trade_flow_burst() {
    let config_path = std::env::temp_dir().join("trade_flow_momentum_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 1500\nmin_trades_in_window = 12\nmin_trade_flow_imbalance = 0.18\nmin_recent_trade_flow_imbalance = 0.10\nmin_price_drift_bps = 6.0\nmax_price_drift_bps = 35.0\nmax_drift_above_vwap_bps = 20.0\nmax_spread_bps = 12.0\nmin_burst_per_second = 10.0\nmin_order_book_imbalance = -1.0\nposition_size_confidence_floor = 0.72\nentry_cooldown_millis = 2500\nhold_time_millis = 4000\nexit_on_flow_reversal = -0.05\nstop_loss_bps = 18.0\ntake_profit_bps = 24.0\nassumed_round_trip_taker_cost_bps = 0.0\nmin_expected_edge_after_cost_bps = 0.0\n",
    )
    .expect("write temp config");

    let mut strategy = TradeFlowMomentumStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 1000.0,
        },
        ask: BookLevel {
            price: 0.1001,
            quantity: 1000.0,
        },
        event_time: 1_500,
    }));

    for index in 0..16 {
        let event = MarketEvent::Trade(Trade {
            price: 0.1000 + (index as f64 * 0.00001),
            quantity: 1500.0,
            trade_time: 100 + index as u64 * 70,
            is_buyer_market_maker: false,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    match decision.intent {
        OrderIntent::Place { side, rationale, .. } => {
            assert_eq!(side, trade::Side::Buy);
            assert_eq!(rationale, "trade_flow_momentum_entry");
            assert!(decision.confidence > 0.0);
        }
        _ => panic!("expected buy intent"),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_on_flow_reversal_when_position_open() {
    let mut strategy = TradeFlowMomentumStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1003,
            quantity: 800.0,
        },
        ask: BookLevel {
            price: 0.1004,
            quantity: 1200.0,
        },
        event_time: 5_000,
    }));

    for index in 0..14 {
        let event = MarketEvent::Trade(Trade {
            price: 0.1002 - (index as f64 * 0.00001),
            quantity: 1200.0,
            trade_time: 3_000 + index as u64 * 80,
            is_buyer_market_maker: true,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let decision = strategy.decide(&market_state, &context_with_position(300.0, 0.1001, 2.0));
    match decision.intent {
        OrderIntent::Place { side, rationale, .. } => {
            assert_eq!(side, trade::Side::Sell);
            assert_eq!(rationale, "flow_reversal");
        }
        _ => panic!("expected sell intent"),
    }
}

#[tokio::test]
async fn rejects_entry_when_recent_flow_or_book_support_is_weak() {
    let mut strategy = TradeFlowMomentumStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1002,
            quantity: 800.0,
        },
        ask: BookLevel {
            price: 0.1003,
            quantity: 1400.0,
        },
        event_time: 1_500,
    }));

    let prices = [
        0.1000, 0.1001, 0.1002, 0.1003, 0.1004, 0.1005, 0.1006, 0.1007, 0.1008, 0.1009, 0.1010,
        0.1011,
    ];

    for (index, price) in prices.into_iter().enumerate() {
        let event = MarketEvent::Trade(Trade {
            price,
            quantity: 1400.0,
            trade_time: 100 + index as u64 * 90,
            is_buyer_market_maker: index < 8,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    assert!(matches!(decision.intent, OrderIntent::NoAction));
}

#[tokio::test]
async fn exposes_momentum_diagnostics_after_blocked_entry() {
    let mut strategy = TradeFlowMomentumStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    for index in 0..4 {
        let event = MarketEvent::Trade(Trade {
            price: 0.1000 + index as f64 * 0.00001,
            quantity: 10.0,
            trade_time: 100 + index as u64 * 100,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let _ = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    let diagnostics = strategy.diagnostics();
    assert_eq!(diagnostics.counters.get("momentum.blocked_min_trades"), Some(&1));
}

#[tokio::test]
async fn rejects_entry_when_expected_edge_does_not_clear_cost_gate() {
    let config_path = std::env::temp_dir().join("trade_flow_momentum_cost_gate_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 1500\nmin_trades_in_window = 12\nmin_trade_flow_imbalance = 0.18\nmin_recent_trade_flow_imbalance = 0.10\nmin_price_drift_bps = 6.0\nmax_price_drift_bps = 35.0\nmax_drift_above_vwap_bps = 20.0\nmax_spread_bps = 12.0\nmin_burst_per_second = 10.0\nmin_order_book_imbalance = -1.0\nposition_size_confidence_floor = 0.72\nentry_cooldown_millis = 2500\nhold_time_millis = 4000\nexit_on_flow_reversal = -0.05\nstop_loss_bps = 18.0\ntake_profit_bps = 24.0\nassumed_round_trip_taker_cost_bps = 22.7\nmin_expected_edge_after_cost_bps = 0.0\n",
    )
    .expect("write temp config");

    let mut strategy = TradeFlowMomentumStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 1000.0,
        },
        ask: BookLevel {
            price: 0.1001,
            quantity: 1000.0,
        },
        event_time: 1_500,
    }));

    for index in 0..16 {
        let event = MarketEvent::Trade(Trade {
            price: 0.1000 + (index as f64 * 0.00001),
            quantity: 1500.0,
            trade_time: 100 + index as u64 * 70,
            is_buyer_market_maker: false,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    assert!(matches!(decision.intent, OrderIntent::NoAction));

    let diagnostics = strategy.diagnostics();
    assert_eq!(diagnostics.counters.get("momentum.blocked_cost_gate"), Some(&1));
    assert!(diagnostics.gauges.get("momentum.last_edge_after_cost_bps").is_some_and(|value| *value < 0.0));

    let _ = fs::remove_file(config_path);
}
