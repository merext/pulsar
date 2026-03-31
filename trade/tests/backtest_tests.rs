use trade::{BacktestEngine, Signal, TradeConfig};

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
