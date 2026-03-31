use serde::Serialize;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub entry_time: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimulationAccount {
    pub initial_cash: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub fees_paid: f64,
    pub equity_peak: f64,
    pub max_drawdown: f64,
}

impl SimulationAccount {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            initial_cash,
            cash: initial_cash,
            realized_pnl: 0.0,
            fees_paid: 0.0,
            equity_peak: initial_cash,
            max_drawdown: 0.0,
        }
    }

    pub fn equity(&self, mark_price: f64, position: Option<&Position>) -> f64 {
        let position_value =
            position.map_or(0.0, |open_position| open_position.quantity * mark_price);
        self.cash + position_value
    }

    pub fn record_buy(&mut self, notional_value: f64, fee_paid: f64) {
        self.cash -= notional_value + fee_paid;
        self.fees_paid += fee_paid;
    }

    pub fn record_sell(&mut self, notional_value: f64, fee_paid: f64, realized_pnl: f64) {
        self.cash += notional_value - fee_paid;
        self.fees_paid += fee_paid;
        self.realized_pnl += realized_pnl;
    }

    pub fn update_drawdown(&mut self, mark_price: f64, position: Option<&Position>) -> f64 {
        let equity = self.equity(mark_price, position);
        if equity > self.equity_peak {
            self.equity_peak = equity;
        }

        let drawdown = if self.equity_peak <= f64::EPSILON {
            0.0
        } else {
            ((self.equity_peak - equity) / self.equity_peak).max(0.0)
        };

        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }

        equity
    }
}

impl Default for SimulationAccount {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Position {{ symbol: \"{}\", quantity: {:.6}, entry_price: {:.8}, entry_time: {:.3} }}",
            self.symbol, self.quantity, self.entry_price, self.entry_time
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Trade {
    pub event_type: String,
    pub event_time: u64,
    pub symbol: String,
    pub trade_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub buyer_order_id: Option<u64>,
    pub seller_order_id: Option<u64>,
    pub trade_time: u64,
    pub is_buyer_market_maker: bool,
}

impl Trade {
    pub fn timestamp_secs(&self) -> f64 {
        self.trade_time as f64 / 1000.0
    }
}
