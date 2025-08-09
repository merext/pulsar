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
    _decision_interval_ns: u64,
    buy_threshold: f64,
    sell_threshold: f64,
    _snapshot_window_ns: u64,
    trade_buffer_cap: usize,
    snapshot_buffer_cap: usize,

    // State
    trade_buffer: VecDeque<TradeEvent>,
    snapshot_buffer: VecDeque<Snapshot>,
    last_snapshot_time: u64,

    // Signal controls
    min_event_intensity: f64,
    tp_bps: f64,
    sl_bps: f64,
    aggression_align: bool,
    aggr_min_abs: f64,
    require_dual_momentum: bool,
    is_contrarian: bool,
    // Threshold modulation
    min_score_mult: f64,
    vol_k: f64,
    vol_window: usize,
    // Score weights (unused in current simplified implementation)
    _r2_w: f64,
    _r5_w: f64,
    _aggr_w: f64,
    _ei_w: f64,
    _conf_gain: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct TradeEvent {
    t: u64,
    p: f64,
    qty: f64,
    is_sell_aggr: bool,
}

#[derive(Clone, Debug)]
struct Snapshot {
    _t: u64,
    p: f64,
    event_intensity: f64,
    aggression_ratio: f64,
}

#[derive(Debug, Deserialize)]
struct StrategySettings {
    mpc: Mpc,
    memory: Memory,
    signals: Option<Signals>,
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

#[derive(Debug, Deserialize)]
struct Signals {
    min_event_intensity: Option<f64>,
    tp_bps: Option<f64>,
    sl_bps: Option<f64>,
    aggression_align: Option<bool>,
    aggr_min_abs: Option<f64>,
    require_dual_momentum: Option<bool>,
    mode: Option<String>,
    min_score_mult: Option<f64>,
    vol_k: Option<f64>,
    vol_window: Option<usize>,
    r2_w: Option<f64>,
    r5_w: Option<f64>,
    aggr_w: Option<f64>,
    ei_w: Option<f64>,
    conf_gain: Option<f64>,
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
        let sig = settings.signals.as_ref();

        Self {
            _decision_interval_ns: decision_interval_ns,
            buy_threshold: thresh,
            sell_threshold: thresh,
            _snapshot_window_ns: 100_000_000, // 100ms consolidation window
            trade_buffer_cap: settings.memory.trade_buffer_size.max(10_000),
            snapshot_buffer_cap: settings.memory.snapshot_buffer_size.max(2_000),
            trade_buffer: VecDeque::with_capacity(settings.memory.trade_buffer_size),
            snapshot_buffer: VecDeque::with_capacity(settings.memory.snapshot_buffer_size),
            last_snapshot_time: 0,
            min_event_intensity: sig.and_then(|s| s.min_event_intensity).unwrap_or(2.0),
            tp_bps: sig.and_then(|s| s.tp_bps).unwrap_or(0.0005),
            sl_bps: sig.and_then(|s| s.sl_bps).unwrap_or(0.00018),
            aggression_align: sig.and_then(|s| s.aggression_align).unwrap_or(true),
            aggr_min_abs: sig.and_then(|s| s.aggr_min_abs).unwrap_or(0.25),
            require_dual_momentum: sig.and_then(|s| s.require_dual_momentum).unwrap_or(false),
            is_contrarian: sig.and_then(|s| s.mode.clone()).unwrap_or_else(|| "momentum".to_string()) == "contrarian",
            min_score_mult: sig.and_then(|s| s.min_score_mult).unwrap_or(3.0),
            vol_k: sig.and_then(|s| s.vol_k).unwrap_or(1.0),
            vol_window: sig.and_then(|s| s.vol_window).unwrap_or(12).max(3),
            _r2_w: sig.and_then(|s| s.r2_w).unwrap_or(0.5),
            _r5_w: sig.and_then(|s| s.r5_w).unwrap_or(0.3),
            _aggr_w: sig.and_then(|s| s.aggr_w).unwrap_or(0.2),
            _ei_w: sig.and_then(|s| s.ei_w).unwrap_or(0.05),
            _conf_gain: sig.and_then(|s| s.conf_gain).unwrap_or(5.0),
        }
    }

    fn recent_returns(&self) -> Option<(f64, f64)> {
        let len = self.snapshot_buffer.len();
        if len < 6 { return None; }
        let p_last = self.snapshot_buffer[len - 1].p;
        let p_2 = self.snapshot_buffer[len - 3].p;
        let p_5 = self.snapshot_buffer[len - 6].p;
        let r2 = (p_last - p_2) / p_2;
        let r5 = (p_last - p_5) / p_5;
        Some((r2, r5))
    }

    fn maybe_snapshot(&mut self, now_ms: u64) {
        // Push snapshot every trade to avoid cadence issues on replay
        self.last_snapshot_time = now_ms;
        if self.trade_buffer.is_empty() { return; }
        let last_price = self.trade_buffer.back().unwrap().p;

        // Compute features over the last 1s
        // one second lookback in milliseconds
        let one_sec_ago = now_ms.saturating_sub(1_000);
        let mut n = 0usize;
        let mut aggr_sum = 0.0f64;
        for e in self.trade_buffer.iter().rev() {
            if e.t < one_sec_ago { break; }
            n += 1;
            aggr_sum += if e.is_sell_aggr { -1.0 } else { 1.0 };
        }
        let event_intensity = n as f64;
        let aggression_ratio = if n > 0 { aggr_sum / n as f64 } else { 0.0 };

        self.snapshot_buffer.push_back(Snapshot { _t: now_ms, p: last_price, event_intensity, aggression_ratio });
        while self.snapshot_buffer.len() > self.snapshot_buffer_cap {
            self.snapshot_buffer.pop_front();
        }
    }

    fn compute_score(&self) -> (f64, f64) {
        let len = self.snapshot_buffer.len();
        if len < 2 { return (0.0, 0.0); }  // Much lower requirement

        let last = &self.snapshot_buffer[len - 1];
        let prev = &self.snapshot_buffer[len - 2];
        
        // Simple momentum based on last price change
        let r1 = (last.p - prev.p) / prev.p;
        let aggr = last.aggression_ratio;
        let ei = last.event_intensity;
        
        // Very simple score - just momentum + aggression, very conservative
        let score = r1 * 105.0 + aggr * 0.11 + ei * 0.0011;  // Very conservative for profitability
        let conf = score.abs().min(1.0);
        (score, conf)
    }

    fn recent_vol(&self) -> f64 {
        let len = self.snapshot_buffer.len();
        if len < 3 { return 0.0; }
        let n = self.vol_window.min(len - 1);
        let mut sum = 0.0;
        let mut cnt = 0;
        for i in (len - n)..len {
            if i == 0 { continue; }
            let p = self.snapshot_buffer[i].p;
            let pp = self.snapshot_buffer[i - 1].p;
            if pp > 0.0 {
                sum += ((p - pp) / pp).abs();
                cnt += 1;
            }
        }
        if cnt > 0 { sum / cnt as f64 } else { 0.0 }
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
        _current_timestamp: f64,
        current_position: Position,
    ) -> (Signal, f64) {
        // No decision cadence gate in replay to ensure signal evaluation

        let (score, mut conf) = self.compute_score();
        let mut signal = Signal::Hold;

        // Align entries with aggression and sufficient recent activity
        let (aggr, ei) = if let Some(last) = self.snapshot_buffer.back() {
            (last.aggression_ratio, last.event_intensity)
        } else { (0.0, 0.0) };

        let ei_ok = ei > self.min_event_intensity; // configurable activity gate
        let aggr_ok = aggr.abs() >= self.aggr_min_abs;

        let dual_momo_ok = if self.require_dual_momentum {
            if let Some((r2, r5)) = self.recent_returns() {
                (r2 > 0.0 && r5 > 0.0) || (r2 < 0.0 && r5 < 0.0)
            } else { false }
        } else { true };

        // Volatility-adjusted thresholds and minimum score multiplier
        let base = self.buy_threshold.max(self.sell_threshold);
        let eff = base * (1.0 + self.vol_k * self.recent_vol());
        let buy_cond = if self.is_contrarian { score < -(eff * self.min_score_mult) } else { score > eff * self.min_score_mult };
        let sell_cond = if self.is_contrarian { score > eff * self.min_score_mult } else { score < -(eff * self.min_score_mult) };

        if buy_cond
            && (
                !self.aggression_align
                    || if self.is_contrarian { aggr < 0.0 } else { aggr > 0.0 }
            )
            && aggr_ok && ei_ok && dual_momo_ok {
            if current_position.quantity == 0.0 {
                signal = Signal::Buy;
            }
        } else if sell_cond
            && (
                !self.aggression_align
                    || if self.is_contrarian { aggr > 0.0 } else { aggr < 0.0 }
            )
            && aggr_ok && ei_ok && dual_momo_ok {
            if current_position.quantity == 0.0 {
                signal = Signal::Sell;
            }
        }

        // PnL-aware exit
        if matches!(signal, Signal::Hold) && current_position.quantity != 0.0 {
            let entry = current_position.entry_price;
            if entry > 0.0 {
                let pnl = if current_position.quantity > 0.0 {
                    (current_price - entry) / entry  // Long position
                } else {
                    (entry - current_price) / entry  // Short position
                };
                // Configurable TP/SL
                if pnl >= self.tp_bps || pnl <= -self.sl_bps {
                    signal = if current_position.quantity > 0.0 { Signal::Sell } else { Signal::Buy };
                    conf = (conf + 0.2).min(1.0);
                }
            }
        }

        (signal, conf)
    }
}


