use trade::{
    ExecutionReport, ExecutionStatus, PerformanceMetrics, Signal, TradeManager, TradeRecord,
};

#[test]
fn win_rate_counts_only_closed_trades() {
    let mut metrics = PerformanceMetrics::new();

    metrics.record_trade(TradeRecord {
        timestamp: 1.0,
        price: 10.0,
        quantity: 1.0,
        signal: Signal::Buy,
        pnl: None,
    });
    metrics.record_trade(TradeRecord {
        timestamp: 2.0,
        price: 11.0,
        quantity: 1.0,
        signal: Signal::Sell,
        pnl: Some(1.0),
    });
    metrics.record_trade(TradeRecord {
        timestamp: 3.0,
        price: 10.0,
        quantity: 1.0,
        signal: Signal::Sell,
        pnl: Some(-0.5),
    });

    assert_eq!(metrics.entry_trades(), 1);
    assert_eq!(metrics.closed_trades(), 2);
    assert!((metrics.win_rate() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn trade_manager_applies_fees_to_realized_pnl() {
    let mut manager = TradeManager::new(0.001, 100.0);
    manager
        .open_position("DOGEUSDT", 0.10, 100.0, 1.0)
        .expect("position opens");

    let pnl = manager.close_position("DOGEUSDT", 0.11, 2.0);

    assert!((pnl - 0.979).abs() < 1e-9);
    assert!((manager.realized_pnl() - 0.979).abs() < 1e-9);
}

#[test]
fn trade_manager_tracks_cash_and_drawdown() {
    let mut manager = TradeManager::new(0.001, 100.0);
    manager
        .open_position("DOGEUSDT", 0.20, 100.0, 1.0)
        .expect("position opens");

    let equity = manager.mark_to_market("DOGEUSDT", 0.18);

    assert!(manager.available_cash() < 100.0);
    assert!(equity < 100.0);
    assert!(manager.max_drawdown() > 0.0);
}

#[test]
fn performance_metrics_track_execution_attribution_and_fill_quality() {
    let mut metrics = PerformanceMetrics::new();
    metrics.record_execution_report(&ExecutionReport {
        status: ExecutionStatus::Filled,
        side: None,
        order_type: None,
        requested_quantity: 10.0,
        executed_quantity: 8.0,
        execution_price: Some(100.0),
        fee_paid: 0.1,
        latency_seconds: 0.02,
        synthetic_half_spread_bps: 1.0,
        slippage_bps: 2.0,
        latency_impact_bps: 0.5,
        market_impact_bps: 1.5,
        reason: None,
        expected_edge_bps: 4.0,
    });
    metrics.record_execution_report(&ExecutionReport {
        status: ExecutionStatus::Rejected,
        side: None,
        order_type: None,
        requested_quantity: 5.0,
        executed_quantity: 0.0,
        execution_price: None,
        fee_paid: 0.0,
        latency_seconds: 0.0,
        synthetic_half_spread_bps: 0.0,
        slippage_bps: 0.0,
        latency_impact_bps: 0.0,
        market_impact_bps: 0.0,
        reason: Some("test"),
        expected_edge_bps: 0.0,
    });

    assert_eq!(metrics.filled_orders(), 1);
    assert_eq!(metrics.rejected_orders(), 1);
    assert!((metrics.fill_ratio() - (8.0 / 15.0)).abs() < 1e-9);
    assert!((metrics.rejection_rate() - 0.5).abs() < 1e-9);
    assert!((metrics.avg_slippage_bps() - 2.0).abs() < 1e-9);
    assert!((metrics.avg_market_impact_bps() - 1.5).abs() < 1e-9);
}
