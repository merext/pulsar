use trade::{
    ExecutionReport, ExecutionStatus, PerformanceMetrics, Signal, TradeManager, TradeRecord,
};

#[test]
fn win_rate_counts_only_closed_trades() {
    let mut metrics = PerformanceMetrics::new();

    metrics.record_trade(TradeRecord {
        trade_id: 1,
        symbol: "DOGEUSDT".to_string(),
        timestamp: 1.0,
        price: 10.0,
        quantity: 1.0,
        signal: Signal::Buy,
        pnl: None,
        gross_pnl: None,
        fee_paid: 0.0,
        expected_edge_bps: 0.0,
        rationale: Some("entry"),
        decision_confidence: 0.5,
        requested_quantity: 1.0,
        executed_quantity: 1.0,
        synthetic_half_spread_bps: 0.0,
        slippage_bps: 0.0,
        latency_impact_bps: 0.0,
        market_impact_bps: 0.0,
        hold_time_millis: None,
        exit_reason: None,
        entry_price: Some(10.0),
    });
    metrics.record_trade(TradeRecord {
        trade_id: 2,
        symbol: "DOGEUSDT".to_string(),
        timestamp: 2.0,
        price: 11.0,
        quantity: 1.0,
        signal: Signal::Sell,
        pnl: Some(1.0),
        gross_pnl: Some(1.1),
        fee_paid: 0.1,
        expected_edge_bps: 0.0,
        rationale: Some("tp"),
        decision_confidence: 1.0,
        requested_quantity: 1.0,
        executed_quantity: 1.0,
        synthetic_half_spread_bps: 0.0,
        slippage_bps: 0.0,
        latency_impact_bps: 0.0,
        market_impact_bps: 0.0,
        hold_time_millis: Some(1000),
        exit_reason: Some("tp"),
        entry_price: Some(10.0),
    });
    metrics.record_trade(TradeRecord {
        trade_id: 3,
        symbol: "DOGEUSDT".to_string(),
        timestamp: 3.0,
        price: 10.0,
        quantity: 1.0,
        signal: Signal::Sell,
        pnl: Some(-0.5),
        gross_pnl: Some(-0.4),
        fee_paid: 0.1,
        expected_edge_bps: 0.0,
        rationale: Some("sl"),
        decision_confidence: 1.0,
        requested_quantity: 1.0,
        executed_quantity: 1.0,
        synthetic_half_spread_bps: 0.0,
        slippage_bps: 0.0,
        latency_impact_bps: 0.0,
        market_impact_bps: 0.0,
        hold_time_millis: Some(500),
        exit_reason: Some("sl"),
        entry_price: Some(10.5),
    });

    assert_eq!(metrics.entry_trades(), 1);
    assert_eq!(metrics.closed_trades(), 2);
    assert!((metrics.win_rate() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn trade_manager_applies_fees_to_realized_pnl() {
    let mut manager = TradeManager::new(0.001, 100.0);
    manager
        .open_position("DOGEUSDT", 0.10, 100.0, 1.0, None)
        .expect("position opens");

    let pnl = manager.close_position("DOGEUSDT", 0.11, 2.0);

    assert!((pnl - 0.979).abs() < 1e-9);
    assert!((manager.realized_pnl() - 0.979).abs() < 1e-9);
}

#[test]
fn trade_manager_tracks_cash_and_drawdown() {
    let mut manager = TradeManager::new(0.001, 100.0);
    manager
        .open_position("DOGEUSDT", 0.20, 100.0, 1.0, None)
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
        symbol: Some("DOGEUSDT".to_string()),
        side: None,
        order_type: None,
        rationale: Some("entry"),
        decision_confidence: 0.7,
        decision_metrics: Vec::new(),
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
        symbol: Some("DOGEUSDT".to_string()),
        side: None,
        order_type: None,
        rationale: Some("reject"),
        decision_confidence: 0.0,
        decision_metrics: Vec::new(),
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

#[test]
fn trade_manager_records_attribution_fields_from_execution_reports() {
    let mut manager = TradeManager::new(0.001, 100.0);
    let entry_report = ExecutionReport {
        status: ExecutionStatus::Filled,
        symbol: Some("DOGEUSDT".to_string()),
        side: Some(trade::Side::Buy),
        order_type: None,
        rationale: Some("trade_flow_momentum_entry"),
        decision_confidence: 0.84,
        decision_metrics: vec![trade::DecisionMetric {
            name: "flow_imbalance",
            value: 0.31,
        }],
        requested_quantity: 100.0,
        executed_quantity: 100.0,
        execution_price: Some(0.10),
        fee_paid: 0.01,
        latency_seconds: 0.01,
        synthetic_half_spread_bps: 1.0,
        slippage_bps: 2.0,
        latency_impact_bps: 0.5,
        market_impact_bps: 1.5,
        reason: None,
        expected_edge_bps: 12.0,
    };
    manager.record_execution_report(&entry_report);
    manager
        .open_position("DOGEUSDT", 0.10, 100.0, 1.0, Some(&entry_report))
        .expect("position opens");

    let exit_report = ExecutionReport {
        status: ExecutionStatus::Filled,
        symbol: Some("DOGEUSDT".to_string()),
        side: Some(trade::Side::Sell),
        order_type: None,
        rationale: Some("take_profit"),
        decision_confidence: 1.0,
        decision_metrics: Vec::new(),
        requested_quantity: 100.0,
        executed_quantity: 100.0,
        execution_price: Some(0.11),
        fee_paid: 0.011,
        latency_seconds: 0.01,
        synthetic_half_spread_bps: 0.8,
        slippage_bps: 1.8,
        latency_impact_bps: 0.4,
        market_impact_bps: 1.1,
        reason: None,
        expected_edge_bps: 0.0,
    };
    manager.record_execution_report(&exit_report);
    let _ = manager.close_position_with_report("DOGEUSDT", 0.11, 2.0, Some(&exit_report));

    let trades = manager.get_metrics().get_trades();
    assert_eq!(trades.len(), 2);
    assert_eq!(trades[0].rationale, Some("trade_flow_momentum_entry"));
    assert!((trades[0].decision_confidence - 0.84).abs() < 1e-9);
    assert!((trades[0].expected_edge_bps - 12.0).abs() < 1e-9);
    assert_eq!(trades[1].exit_reason, Some("take_profit"));
    assert_eq!(trades[1].entry_price, Some(0.10));
    assert_eq!(trades[1].hold_time_millis, Some(1000));
}
