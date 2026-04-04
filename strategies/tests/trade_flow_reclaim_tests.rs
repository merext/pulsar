use std::fs;
use strategies::TradeFlowReclaimStrategy;
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
    }
}

#[tokio::test]
async fn enters_on_pullback_reclaim_with_positive_flow() {
    let config_path = std::env::temp_dir().join("trade_flow_reclaim_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 2000\nmin_trades_in_window = 14\nmin_pullback_from_high_bps = 4.0\nmax_pullback_from_high_bps = 60.0\nmin_reclaim_from_low_bps = 2.0\nmin_trade_flow_imbalance = -0.20\nmin_recent_trade_flow_imbalance = -0.20\nmax_spread_bps = 20.0\nstop_loss_bps = 14.0\ntake_profit_bps = 18.0\nhold_time_millis = 4000\nentry_cooldown_millis = 2000\nassumed_round_trip_taker_cost_bps = 0.0\nmin_expected_edge_after_cost_bps = 0.0\n",
    )
    .expect("write temp config");
    let mut strategy = TradeFlowReclaimStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let prices = [
        0.1008, 0.1009, 0.1010, 0.1007, 0.1005, 0.1003, 0.1002, 0.1001, 0.10015, 0.10020, 0.10025,
        0.10030, 0.10035, 0.10040, 0.10045, 0.10050,
    ];

    for (index, price) in prices.into_iter().enumerate() {
        let event = MarketEvent::Trade(Trade {
            price,
            quantity: 1800.0,
            trade_time: 100 + index as u64 * 100,
            is_buyer_market_maker: index < 8,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.10045,
            quantity: 1200.0,
        },
        ask: BookLevel {
            price: 0.10055,
            quantity: 1100.0,
        },
        event_time: 2_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    match decision.intent {
        OrderIntent::Place { side, rationale, .. } => {
            assert_eq!(side, trade::Side::Buy);
            assert_eq!(rationale, "trade_flow_reclaim_entry");
        }
        _ => panic!("expected buy intent"),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_on_flow_reversal() {
    let mut strategy = TradeFlowReclaimStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 800.0,
        },
        ask: BookLevel {
            price: 0.1001,
            quantity: 800.0,
        },
        event_time: 6_000,
    }));

    for index in 0..12 {
        let event = MarketEvent::Trade(Trade {
            price: 0.1000 - (index as f64 * 0.00001),
            quantity: 1500.0,
            trade_time: 4_000 + index as u64 * 90,
            is_buyer_market_maker: true,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    let decision = strategy.decide(&market_state, &context_with_position(200.0, 0.1001, 2.0));
    match decision.intent {
        OrderIntent::Place { side, rationale, .. } => {
            assert_eq!(side, trade::Side::Sell);
            assert_eq!(rationale, "flow_reversal");
        }
        _ => panic!("expected sell intent"),
    }
}

#[tokio::test]
async fn rejects_entry_when_reclaim_edge_does_not_clear_cost_gate() {
    let config_path = std::env::temp_dir().join("trade_flow_reclaim_cost_gate_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 2000\nmin_trades_in_window = 14\nmin_pullback_from_high_bps = 4.0\nmax_pullback_from_high_bps = 60.0\nmin_reclaim_from_low_bps = 2.0\nmin_trade_flow_imbalance = -0.20\nmin_recent_trade_flow_imbalance = -0.20\nmax_spread_bps = 20.0\nstop_loss_bps = 14.0\ntake_profit_bps = 18.0\nhold_time_millis = 4000\nentry_cooldown_millis = 2000\nassumed_round_trip_taker_cost_bps = 80.0\nmin_expected_edge_after_cost_bps = 0.0\n",
    )
    .expect("write temp config");
    let mut strategy = TradeFlowReclaimStrategy::from_file(&config_path).expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    let prices = [
        0.1008, 0.1009, 0.1010, 0.1007, 0.1005, 0.1003, 0.1002, 0.1001, 0.10015, 0.10020, 0.10025,
        0.10030, 0.10035, 0.10040, 0.10045, 0.10050,
    ];

    for (index, price) in prices.into_iter().enumerate() {
        let event = MarketEvent::Trade(Trade {
            price,
            quantity: 1800.0,
            trade_time: 100 + index as u64 * 100,
            is_buyer_market_maker: index < 8,
            ..Default::default()
        });
        strategy.on_event(&event, &market_state).await;
        market_state.apply(&event);
    }

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.10045,
            quantity: 1200.0,
        },
        ask: BookLevel {
            price: 0.10055,
            quantity: 1100.0,
        },
        event_time: 2_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    assert!(matches!(decision.intent, OrderIntent::NoAction));

    let diagnostics = strategy.diagnostics();
    assert_eq!(diagnostics.counters.get("reclaim.blocked_cost_gate"), Some(&1));
    assert!(diagnostics.gauges.get("reclaim.last_edge_after_cost_bps").is_some_and(|value| *value < 0.0));

    let _ = fs::remove_file(config_path);
}
