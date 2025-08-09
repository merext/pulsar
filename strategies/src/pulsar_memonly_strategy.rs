use std::collections::VecDeque;

use async_trait::async_trait;
use serde::Deserialize;

use crate::strategy::Strategy;
use trade::models::TradeData;
use trade::signal::Signal;
use trade::trader::Position;

// Pulsar-MemOnly: Micro-Momentum (memory-only)
// - Uses only recent trades buffered in RAM
// - Signals from short-horizon returns, aggression imbalance, event intensity
// - No offline data or heavy models

#[derive(Clone)]
pub struct PulsarMemOnlyStrategy {
    // Tunables
    decision_interval_ns: u64,
    buy_threshold: f64,
    sell_threshold: f64,
    snapshot_window_ns: u64,
    trade_buffer_cap: usize,
    snapshot_buffer_cap: usize,

    // State
    trade_buffer: VecDeque<TradeEvent>,
    snapshot_buffer: VecDeque<Snapshot>,
    last_snapshot_time: u64,
}

#[derive(Clone, Debug)]
struct TradeEvent {
    t: u64,
    p: f64,
    qty: f64,
    is_sell_aggr: bool,
}

#[derive(Clone, Debug)]
struct Snapshot {
    t: u64,
    p: f64,
    event_intensity: f64,
    aggression_ratio: f64,
}

#[derive(Debug, Deserialize)]
struct StrategySettings {
    mpc: Mpc,
    memory: Memory,
}

#[derive(Debug, Deserialize)]
struct Mpc {
    decision_interval_ms: u64,
    execution_threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct Memory {
    snapshot_buffer_size: usize,
    trade_buffer_size: usize,
}

impl PulsarMemOnlyStrategy {
    pub fn new() -> Self {
        // Strategy config
        let settings: StrategySettings = toml::from_str(
            &std::fs::read_to_string("config/pulsar_memonly_strategy.toml")
                .expect("strategy config")
        )
        .expect("parse strategy config");

        let decision_interval_ns = settings.mpc.decision_interval_ms.max(1) * 1_000_000;
        let thresh = settings.mpc.execution_threshold.unwrap_or(0.0001).abs();

        Self {
            decision_interval_ns,
            buy_threshold: thresh,
            sell_threshold: thresh,
            snapshot_window_ns: 100_000_000, // 100ms consolidation window
            trade_buffer_cap: settings.memory.trade_buffer_size.max(10_000),
            snapshot_buffer_cap: settings.memory.snapshot_buffer_size.max(2_000),
            trade_buffer: VecDeque::with_capacity(settings.memory.trade_buffer_size),
            snapshot_buffer: VecDeque::with_capacity(settings.memory.snapshot_buffer_size),
            last_snapshot_time: 0,
        }
    }

    fn maybe_snapshot(&mut self, now: u64) {
        if now - self.last_snapshot_time < self.snapshot_window_ns {
            return;
        }
        self.last_snapshot_time = now;
        if self.trade_buffer.is_empty() {
            return;
        }
        let last_price = self.trade_buffer.back().unwrap().p;

        // Compute features over the last 1s
        let one_sec_ago = now.saturating_sub(1_000_000_000);
        let mut n = 0usize;
        let mut aggr_sum = 0.0f64;
        for e in self.trade_buffer.iter().rev() {
            if e.t < one_sec_ago { break; }
            n += 1;
            aggr_sum += if e.is_sell_aggr { -1.0 } else { 1.0 };
        }
        let event_intensity = n as f64;
        let aggression_ratio = if n > 0 { aggr_sum / n as f64 } else { 0.0 };

        self.snapshot_buffer.push_back(Snapshot { t: now, p: last_price, event_intensity, aggression_ratio });
        while self.snapshot_buffer.len() > self.snapshot_buffer_cap {
            self.snapshot_buffer.pop_front();
        }
    }

    fn compute_score(&self) -> (f64, f64) {
        let len = self.snapshot_buffer.len();
        if len < 6 { return (0.0, 0.0); }

        let last = &self.snapshot_buffer[len - 1];
        let p_last = last.p;
        let p_2 = self.snapshot_buffer[len - 3].p;
        let p_5 = self.snapshot_buffer[len - 6].p;

        let r2 = (p_last - p_2) / p_2;
        let r5 = (p_last - p_5) / p_5;
        let aggr = last.aggression_ratio; // [-1,1]
        let ei = last.event_intensity; // >=0

        // Normalize event intensity over recent window
        let ei_norm = {
            let n = 10.min(len);
            let mut s = 0.0; let mut c = 0;
            for i in (len - n)..len { s += self.snapshot_buffer[i].event_intensity; c += 1; }
            if c > 0 { (ei - s / c as f64) / (s / c as f64 + 1e-6) } else { 0.0 }
        };

        // Composite score
        let score = 0.6 * r2 + 0.3 * r5 + 0.7 * aggr + 0.1 * ei_norm;
        let conf = score.abs().min(1.0);
        (score, conf)
    }
}

#[async_trait]
impl Strategy for PulsarMemOnlyStrategy {
    fn get_info(&self) -> String {
        "Pulsar-MemOnly: Micro-Momentum (memory-only)".to_string()
    }

    async fn on_trade(&mut self, trade: TradeData) {
        // append to trade buffer
        let ev = TradeEvent { t: trade.time, p: trade.price, qty: trade.qty, is_sell_aggr: trade.is_buyer_maker };
        self.trade_buffer.push_back(ev);
        while self.trade_buffer.len() > self.trade_buffer_cap {
            self.trade_buffer.pop_front();
        }
        self.maybe_snapshot(trade.time);
    }

    fn get_signal(
        &self,
        current_price: f64,
        current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // respect decision cadence
        if let Some(last) = self.snapshot_buffer.back() {
            if (current_timestamp as u64).saturating_sub(last.t) < self.decision_interval_ns {
                return (Signal::Hold, 0.0);
            }
        }

        let (score, mut conf) = self.compute_score();
        let mut signal = Signal::Hold;

        // Align entries with aggression and sufficient recent activity
        let (aggr, ei) = if let Some(last) = self.snapshot_buffer.back() {
            (last.aggression_ratio, last.event_intensity)
        } else { (0.0, 0.0) };

        let ei_ok = ei > 1.0; // at least a couple trades in last second

        if score > self.buy_threshold && aggr > 0.0 && ei_ok {
            if current_position.quantity == 0.0 {
                signal = Signal::Buy;
            }
        } else if score < -self.sell_threshold && aggr < 0.0 && ei_ok {
            if current_position.quantity > 0.0 {
                signal = Signal::Sell;
            }
        }

        // PnL-aware exit
        if matches!(signal, Signal::Hold) && current_position.quantity > 0.0 {
            let entry = current_position.entry_price;
            if entry > 0.0 {
                let pnl = (current_price - entry) / entry;
                // Let winners run more; cut losers faster
                if pnl >= 0.0003 || pnl <= -0.0002 {
                    signal = Signal::Sell;
                    conf = (conf + 0.2).min(1.0);
                }
            }
        }

        (signal, conf)
    }
}


