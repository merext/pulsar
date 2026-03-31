use crate::trader::OrderType;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Filled,
    PartiallyFilled,
    Rejected,
    Cancelled,
    Ignored,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionReport {
    pub status: ExecutionStatus,
    pub side: Option<Side>,
    pub order_type: Option<OrderType>,
    pub requested_quantity: f64,
    pub executed_quantity: f64,
    pub execution_price: Option<f64>,
    pub fee_paid: f64,
    pub latency_seconds: f64,
    pub reason: Option<&'static str>,
    pub expected_edge_bps: f64,
}

impl ExecutionReport {
    pub fn ignored() -> Self {
        Self {
            status: ExecutionStatus::Ignored,
            side: None,
            order_type: None,
            requested_quantity: 0.0,
            executed_quantity: 0.0,
            execution_price: None,
            fee_paid: 0.0,
            latency_seconds: 0.0,
            reason: None,
            expected_edge_bps: 0.0,
        }
    }
}
