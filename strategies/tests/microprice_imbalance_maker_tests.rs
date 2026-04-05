use strategies::MicropriceImbalanceMakerStrategy;
use trade::market::{BookLevel, BookTicker, MarketEvent, MarketState};
use trade::strategy::{Strategy, StrategyContext};
use trade::{OrderIntent, Position};

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
async fn enters_on_positive_microprice_edge() {
    let mut strategy =
        MicropriceImbalanceMakerStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 5000.0,
        },
        ask: BookLevel {
            price: 0.10005,
            quantity: 100.0,
        },
        event_time: 1_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    match decision.intent {
        OrderIntent::Place {
            side,
            order_type,
            rationale,
            ..
        } => {
            assert_eq!(side, trade::Side::Buy);
            assert_eq!(order_type, trade::OrderType::Maker);
            assert_eq!(rationale, "microprice_imbalance_maker_entry");
        }
        _ => panic!("expected maker buy intent"),
    }
}

#[tokio::test]
async fn exits_when_imbalance_reverses() {
    let mut strategy =
        MicropriceImbalanceMakerStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.0998,
            quantity: 500.0,
        },
        ask: BookLevel {
            price: 0.0999,
            quantity: 1500.0,
        },
        event_time: 3_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(100.0, 0.0998, 1.0));
    match decision.intent {
        OrderIntent::Place {
            side,
            order_type,
            rationale,
            ..
        } => {
            assert_eq!(side, trade::Side::Sell);
            assert_eq!(order_type, trade::OrderType::Maker);
            assert_eq!(rationale, "imbalance_reversal");
        }
        _ => panic!("expected maker sell intent"),
    }
}

#[tokio::test]
async fn emits_decision_metrics_on_entry() {
    let mut strategy =
        MicropriceImbalanceMakerStrategy::from_file("/dev/null").expect("strategy loads");
    let mut market_state = MarketState::new("DOGEUSDT", strategy.market_state_window_millis());

    market_state.apply(&MarketEvent::BookTicker(BookTicker {
        bid: BookLevel {
            price: 0.1000,
            quantity: 4000.0,
        },
        ask: BookLevel {
            price: 0.10005,
            quantity: 100.0,
        },
        event_time: 2_000,
    }));

    let decision = strategy.decide(&market_state, &context_with_position(0.0, 0.0, 0.0));
    assert!(!decision.metrics.is_empty());
}
