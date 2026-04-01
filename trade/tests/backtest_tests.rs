use trade::{BacktestEngine, MarketPrice, Side, Signal, TradeConfig};

fn load_test_config() -> TradeConfig {
    TradeConfig::from_file("../config/trading_config.toml").expect("config loads")
}

#[test]
fn simulated_buy_uses_positive_slippage() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let execution = engine.execute(Signal::Buy, 100.0, 10.0);

    assert!(execution.execution_price > 100.0);
    assert!(execution.fee_paid > 0.0);
}

#[test]
fn simulated_sell_uses_negative_slippage() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let execution = engine.execute(Signal::Sell, 100.0, 10.0);

    assert!(execution.execution_price < 100.0);
    assert!(execution.fee_paid > 0.0);
}

#[test]
fn simulated_buy_respects_available_cash() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let execution = engine.execute_with_constraints(Signal::Buy, 0.20, 500.0, 100.0);

    assert!(execution.executed_quantity > 0.0);
    assert!(execution.total_cost() <= 100.0 + 1e-9);
}

#[test]
fn simulated_buy_rejects_when_cash_below_min_notional() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let execution = engine.execute_with_constraints(Signal::Buy, 0.20, 10.0, 5.0);

    assert!(execution.is_rejected());
    assert_eq!(execution.rejected_reason, Some("min_notional"));
}

#[test]
fn quote_aware_buy_uses_ask_without_extra_synthetic_spread() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let quoted = engine.execute_with_constraints_at(
        Signal::Buy,
        MarketPrice::Quote {
            bid: 100.0,
            ask: 100.1,
        },
        1.0,
        f64::INFINITY,
    );
    let trade_only = engine.execute_with_constraints(Signal::Buy, 100.1, 1.0, f64::INFINITY);

    assert!(quoted.execution_price >= 100.1);
    assert!(quoted.execution_price < trade_only.execution_price);
}

#[test]
fn quote_aware_sell_uses_bid_without_extra_synthetic_spread() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let quoted = engine.execute_with_constraints_at(
        Signal::Sell,
        MarketPrice::Quote {
            bid: 99.9,
            ask: 100.0,
        },
        1.0,
        f64::INFINITY,
    );
    let trade_only = engine.execute_with_constraints(Signal::Sell, 99.9, 1.0, f64::INFINITY);

    assert!(quoted.execution_price <= 99.9);
    assert!(quoted.execution_price > trade_only.execution_price);
}

#[test]
fn quote_aware_execution_has_zero_synthetic_spread_bps() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let quoted = engine.execute_with_constraints_at(
        Signal::Buy,
        MarketPrice::Quote {
            bid: 100.0,
            ask: 100.1,
        },
        1.0,
        f64::INFINITY,
    );

    assert_eq!(quoted.synthetic_half_spread_rate, 0.0);
    assert!(quoted.slippage_rate > 0.0);
}

#[test]
fn trade_only_execution_applies_separate_latency_and_market_impact_components() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let execution = engine.execute_with_constraints(Signal::Buy, 100.0, 10.0, f64::INFINITY);

    assert!(execution.latency_seconds > 0.0);
    assert!(execution.latency_impact_rate > 0.0);
    assert!(execution.market_impact_rate > 0.0);
    assert!(
        execution.total_price_offset_rate()
            >= execution.latency_impact_rate + execution.market_impact_rate
    );
}

#[test]
fn market_impact_grows_with_order_size() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let small = engine.execute_with_constraints(Signal::Buy, 100.0, 1.0, f64::INFINITY);
    let large = engine.execute_with_constraints(Signal::Buy, 100.0, 200.0, f64::INFINITY);

    assert!(large.market_impact_rate >= small.market_impact_rate);
    assert!(large.execution_price >= small.execution_price);
}

#[test]
fn passive_fill_estimate_is_quote_aware_and_size_sensitive() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let small = engine.estimate_passive_fill(
        Side::Buy,
        MarketPrice::Quote {
            bid: 100.0,
            ask: 100.1,
        },
        10.0,
    );
    let large = engine.estimate_passive_fill(
        Side::Buy,
        MarketPrice::Quote {
            bid: 100.0,
            ask: 100.1,
        },
        10_000.0,
    );

    assert!(small.fill_probability >= large.fill_probability);
    assert!(small.expected_fill_ratio >= large.expected_fill_ratio);
    assert!(small.queue_ahead_quantity > 0.0);
}

#[test]
fn passive_order_reports_pending_or_partial_fill() {
    let config = load_test_config();
    let engine = BacktestEngine::new(config);

    let report = engine.simulate_passive_order(
        Side::Buy,
        MarketPrice::Quote {
            bid: 100.0,
            ask: 100.1,
        },
        1_000.0,
        5.0,
    );

    assert!(matches!(
        report.status,
        trade::ExecutionStatus::Pending
            | trade::ExecutionStatus::PartiallyFilled
            | trade::ExecutionStatus::Filled
    ));
    assert_eq!(report.order_type, Some(trade::OrderType::Maker));
}
