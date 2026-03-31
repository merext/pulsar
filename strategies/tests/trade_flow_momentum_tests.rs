use strategies::TradeFlowMomentumStrategy;
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
    let mut strategy = TradeFlowMomentumStrategy::from_file("/dev/null").expect("strategy loads");
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
