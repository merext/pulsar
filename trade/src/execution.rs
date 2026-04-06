use crate::trader::OrderType;

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionMetric {
    pub name: &'static str,
    pub value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeInForce {
    Gtc,
    Ioc,
    Fok,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderIntent {
    NoAction,
    Place {
        side: Side,
        order_type: OrderType,
        price: Option<f64>,
        quantity: f64,
        time_in_force: TimeInForce,
        rationale: &'static str,
        expected_edge_bps: f64,
    },
    Cancel {
        rationale: &'static str,
    },
    /// Two-sided market-making: simultaneously quote buy and sell.
    /// The trader manages two independent limit orders (one per side).
    /// Fills on each side are processed independently — a buy fill
    /// increases inventory, a sell fill decreases it.
    QuoteBothSides {
        buy_price: f64,
        buy_quantity: f64,
        sell_price: f64,
        sell_quantity: f64,
        rationale: &'static str,
        expected_edge_bps: f64,
    },
    /// Multi-level ladder quoting: place multiple orders at different price
    /// levels on each side simultaneously.
    ///
    /// Example with 3 levels:
    ///   Level 0: best_bid / best_ask       — small size, queue priority (1 bps)
    ///   Level 1: best ± 1 tick             — medium size (3 bps round trip)
    ///   Level 2: best ± 2 ticks            — larger size (5 bps round trip)
    ///
    /// Each Vec entry is (price, quantity). Empty vec = no orders on that side.
    /// The trader manages a Vec<ActiveLimitOrder> per side, diffing desired
    /// levels against existing orders: place new / cancel stale / amend changed.
    QuoteLadder {
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
        rationale: &'static str,
        expected_edge_bps: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Pending,
    Filled,
    PartiallyFilled,
    Rejected,
    Cancelled,
    Ignored,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReport {
    pub status: ExecutionStatus,
    pub symbol: Option<String>,
    pub side: Option<Side>,
    pub order_type: Option<OrderType>,
    pub rationale: Option<&'static str>,
    pub decision_confidence: f64,
    pub decision_metrics: Vec<DecisionMetric>,
    pub requested_quantity: f64,
    pub executed_quantity: f64,
    pub execution_price: Option<f64>,
    pub fee_paid: f64,
    pub latency_seconds: f64,
    pub synthetic_half_spread_bps: f64,
    pub slippage_bps: f64,
    pub latency_impact_bps: f64,
    pub market_impact_bps: f64,
    pub reason: Option<&'static str>,
    pub expected_edge_bps: f64,
}

impl ExecutionReport {
    pub fn ignored() -> Self {
        Self {
            status: ExecutionStatus::Ignored,
            symbol: None,
            side: None,
            order_type: None,
            rationale: None,
            decision_confidence: 0.0,
            decision_metrics: Vec::new(),
            requested_quantity: 0.0,
            executed_quantity: 0.0,
            execution_price: None,
            fee_paid: 0.0,
            latency_seconds: 0.0,
            synthetic_half_spread_bps: 0.0,
            slippage_bps: 0.0,
            latency_impact_bps: 0.0,
            market_impact_bps: 0.0,
            reason: None,
            expected_edge_bps: 0.0,
        }
    }
}
