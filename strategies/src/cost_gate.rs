pub const DEFAULT_ASSUMED_ROUND_TRIP_TAKER_COST_BPS: f64 = 22.7;
pub const DEFAULT_MIN_EXPECTED_EDGE_AFTER_COST_BPS: f64 = 0.0;

pub fn expected_edge_after_cost_bps(
    expected_edge_bps: f64,
    assumed_round_trip_cost_bps: f64,
) -> f64 {
    expected_edge_bps - assumed_round_trip_cost_bps
}

pub fn clears_taker_cost_gate(
    expected_edge_bps: f64,
    assumed_round_trip_cost_bps: f64,
    min_expected_edge_after_cost_bps: f64,
) -> bool {
    expected_edge_after_cost_bps(expected_edge_bps, assumed_round_trip_cost_bps)
        >= min_expected_edge_after_cost_bps
}
