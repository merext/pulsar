use strategies::LiquiditySweepReversalStrategy;
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
async fn enters_after_local_sweep_and_reclaim() {
    let config_path = std::env::temp_dir().join("liquidity_sweep_reversal_test.toml");
    fs::write(
        &config_path,
        "trade_window_millis = 2000\nmin_trades_in_window = 18\nmin_sweep_drop_bps = 10.0\nmin_buyer_reclaim_imbalance = -0.05\nmin_reclaim_bps = 2.0\nmax_reclaim_bps = 20.0\nmax_spread_bps = 14.0\nmin_large_trade_ratio = 0.10\nstop_loss_bps = 16.0\ntake_profit_bps = 20.0\nhold_time_millis = 5000\nentry_cooldown_millis = 3000\n",
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
        OrderIntent::Place { side, rationale, .. } => {
            assert_eq!(side, trade::Side::Buy);
            assert_eq!(rationale, "liquidity_sweep_reversal_entry");
        }
        _ => panic!("expected buy intent"),
    }

    let _ = fs::remove_file(config_path);
}

#[tokio::test]
async fn exits_when_reversal_fails() {
    let mut strategy = LiquiditySweepReversalStrategy::from_file("/dev/null").expect("strategy loads");
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
        OrderIntent::Place { side, rationale, .. } => {
            assert_eq!(side, trade::Side::Sell);
            assert_eq!(rationale, "reversal_failed");
        }
        _ => panic!("expected sell intent"),
    }
}
