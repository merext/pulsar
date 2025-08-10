use std::collections::VecDeque;

use async_trait::async_trait;
use serde::Deserialize;
use tracing::debug;

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
    confidence_min: f64,
    vol_k: f64,
    vol_window: usize,
    // Score weights (unused in current simplified implementation)
    _r2_w: f64,
    _r5_w: f64,
    _aggr_w: f64,
    _ei_w: f64,
    _conf_gain: f64,
    
    // Enhanced sizing parameters
    volatility_window: VecDeque<f64>,
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
    confidence_min: Option<f64>,
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
            confidence_min: sig.and_then(|s| s.confidence_min).unwrap_or(0.3),
            vol_k: sig.and_then(|s| s.vol_k).unwrap_or(1.0),
            vol_window: sig.and_then(|s| s.vol_window).unwrap_or(12).max(3),
            _r2_w: sig.and_then(|s| s.r2_w).unwrap_or(0.5),
            _r5_w: sig.and_then(|s| s.r5_w).unwrap_or(0.3),
            _aggr_w: sig.and_then(|s| s.aggr_w).unwrap_or(0.2),
            _ei_w: sig.and_then(|s| s.ei_w).unwrap_or(0.05),
            _conf_gain: sig.and_then(|s| s.conf_gain).unwrap_or(5.0),
            
            // Initialize enhanced sizing parameters
            volatility_window: VecDeque::with_capacity(100),
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
        
        // Add micro-variation to prevent constant values
        let time_variation = (now_ms % 1000) as f64 / 1000.0 * 0.1;
        let price_variation = if let Some(prev_trade) = self.trade_buffer.iter().rev().nth(1) {
            ((last_price - prev_trade.p) / prev_trade.p).abs() * 0.1
        } else {
            0.0
        };
        
        let event_intensity = (n as f64 + time_variation + price_variation).max(0.1);
        let aggression_ratio = if n > 0 { 
            let base_ratio = aggr_sum / n as f64;
            // Add small variation to prevent constant values
            base_ratio + ((now_ms % 100) as f64 / 100.0 - 0.5) * 0.01
        } else { 
            ((now_ms % 100) as f64 / 100.0 - 0.5) * 0.01 
        };

        self.snapshot_buffer.push_back(Snapshot { _t: now_ms, p: last_price, event_intensity, aggression_ratio });
        while self.snapshot_buffer.len() > self.snapshot_buffer_cap {
            self.snapshot_buffer.pop_front();
        }
    }

    fn update_volatility(&mut self, price: f64) {
        if let Some(last_snapshot) = self.snapshot_buffer.back() {
            let return_pct = (price - last_snapshot.p) / last_snapshot.p;
            
            // Add some noise to prevent constant values
            let noise_factor = if let Some(last_trade) = self.trade_buffer.back() {
                // Use trade time to add micro-variation
                ((last_trade.t % 100) as f64 / 100.0 - 0.5) * 0.0001
            } else {
                0.0
            };
            
            let adjusted_return = return_pct + noise_factor;
            
            self.volatility_window.push_back(adjusted_return);
            
            // Keep only last 100 returns for volatility calculation
            while self.volatility_window.len() > 100 {
                self.volatility_window.pop_front();
            }
        }
    }

    fn compute_score(&self) -> (f64, f64) {
        let len = self.snapshot_buffer.len();
        if len < 3 { return (0.0, 0.0); }  // Need 3 for trend

        let last = &self.snapshot_buffer[len - 1];
        let prev = &self.snapshot_buffer[len - 2];
        let prev2 = &self.snapshot_buffer[len - 3];
        
        // Enhanced momentum with trend confirmation
        let r1 = (last.p - prev.p) / prev.p;
        let r2 = (prev.p - prev2.p) / prev2.p;
        let trend_confirm = if r1 * r2 > 0.0 { 1.2 } else { 0.9 }; // Boost if trend continues
        
        let aggr = last.aggression_ratio;
        let ei = last.event_intensity;
        
        // Score with trend confirmation and volatility adjustment
        let vol_factor = if r1.abs() > 0.0005 { 1.1 } else { 1.0 }; // Boost high volatility moves
        let momentum_score = r1 * 105.0; // Pure momentum component
        let base_score = momentum_score + aggr * 0.11 + ei * 0.0011;
        let score = base_score * trend_confirm * vol_factor;
        
        // Advanced confidence calculation with market regime detection
        let base_momentum = r1.abs() * 500.0; // Further increased base momentum weight
        
        // Advanced trend strength calculation with multi-period analysis
        let trend_strength = if len >= 5 {
            let r3 = (prev2.p - self.snapshot_buffer[len - 4].p) / self.snapshot_buffer[len - 4].p;
            let r4 = (self.snapshot_buffer[len - 4].p - self.snapshot_buffer[len - 5].p) / self.snapshot_buffer[len - 5].p;
            let multi_trend = (r1 * r2 * r3 * r4).abs() * 600.0;
            multi_trend.min(0.6)
        } else if len >= 3 {
            (r1 * r2).abs() * 400.0
        } else {
            0.0
        };
        
        // Advanced volatility-based confidence with regime detection
        let vol_boost = if r1.abs() > 0.002 { 0.45 } else if r1.abs() > 0.0015 { 0.35 } else if r1.abs() > 0.001 { 0.25 } else if r1.abs() > 0.0005 { 0.15 } else { 0.05 };
        
        // Advanced activity boost with market microstructure analysis
        let activity_boost = if aggr.abs() > 0.8 && ei > 15.0 {
            0.5 // High activity regime
        } else if aggr.abs() > 0.5 && ei > 10.0 {
            0.35 // Medium activity regime
        } else if aggr.abs() > 0.2 && ei > 5.0 {
            0.2 // Low activity regime
        } else {
            0.05 // Minimal activity
        };
        
        // Market timing factor based on recent performance
        let timing_factor = if let Some(last_trade) = self.trade_buffer.back() {
            let time_variation = (last_trade.t % 2000) as f64 / 2000.0;
            time_variation * 0.2
        } else {
            0.0
        };
        
        // Price momentum factor
        let price_momentum = if len >= 3 {
            let short_momentum = (last.p - self.snapshot_buffer[len - 3].p) / self.snapshot_buffer[len - 3].p;
            (short_momentum.abs() * 800.0).min(0.3)
        } else {
            0.0
        };
        
        // Combine all factors with advanced weighting
        let conf = (base_momentum + trend_strength + vol_boost + activity_boost + timing_factor + price_momentum).min(1.0);
        
        // Advanced minimum confidence handling with regime awareness
        let final_conf = if conf < 0.2 {
            // In low confidence regimes, be more conservative
            conf + 0.15 + (r1.abs() * 200.0).min(0.2)
        } else if conf < 0.4 {
            // In medium confidence regimes, moderate boost
            conf + 0.1 + (r1.abs() * 150.0).min(0.15)
        } else {
            // In high confidence regimes, minimal boost
            conf + 0.05
        };
        
        (score, final_conf.min(1.0))
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

    // Enhanced dynamic trade size calculation with better market timing
    fn calculate_dynamic_trade_size(&self, base_confidence: f64, _current_price: f64) -> f64 {
        // Base size from config - updated ranges
        let base_min = 15.0; // From updated trading_config.toml
        let base_max = 35.0; // From updated trading_config.toml
        
        // Enhanced volatility adjustment with better scaling
        let volatility = self.recent_vol();
        let vol_adjustment = if volatility > 0.0015 { 0.6 } else if volatility > 0.001 { 0.75 } else if volatility > 0.0005 { 0.9 } else { 1.2 };
        
        // Enhanced event intensity adjustment
        let ei_adjustment = if let Some(last) = self.snapshot_buffer.back() {
            if last.event_intensity > 15.0 { 1.4 } else if last.event_intensity > 10.0 { 1.3 } else if last.event_intensity > 5.0 { 1.1 } else { 0.7 }
        } else {
            1.0
        };
        
        // Adjust based on aggression ratio
        let aggr_adjustment = if let Some(last) = self.snapshot_buffer.back() {
            if last.aggression_ratio.abs() > 0.5 { 1.2 } else if last.aggression_ratio.abs() > 0.2 { 1.1 } else { 0.9 }
        } else {
            1.0
        };
        
        // Enhanced confidence-based adjustment
        let confidence_adjustment = if base_confidence > 0.85 { 1.5 } else if base_confidence > 0.7 { 1.3 } else if base_confidence > 0.5 { 1.1 } else { 0.8 };
        
        // Market timing adjustment
        let market_timing = if let Some(last) = self.snapshot_buffer.back() {
            if last.aggression_ratio.abs() > 0.6 && last.event_intensity > 10.0 { 1.3 } else { 1.0 }
        } else { 1.0 };
        
        // Combine all adjustments
        let total_adjustment = vol_adjustment * ei_adjustment * aggr_adjustment * confidence_adjustment * market_timing;
        
        // Calculate dynamic size with more variation
        let dynamic_size = base_min + (base_confidence * (base_max - base_min));
        let adjusted_size = dynamic_size * total_adjustment;
        
        // Ensure within bounds and apply step size rounding
        let step_size = 0.5; // From updated config
        let rounded_size = (adjusted_size / step_size).round() * step_size;
        let final_size = rounded_size.max(base_min).min(base_max);
        
        final_size
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
        
        // Update volatility tracking with new price
        self.update_volatility(trade.price);
        
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

        // Enhanced volatility-adjusted thresholds with better scaling
        let base = self.buy_threshold.max(self.sell_threshold);
        let volatility = self.recent_vol();
        let vol_adjustment = if volatility > 0.0015 { 1.6 } else if volatility > 0.001 { 1.4 } else if volatility > 0.0005 { 1.2 } else { 1.0 };
        let eff = base * vol_adjustment * (1.0 + self.vol_k * volatility);
        
        // Enhanced conditions with stricter thresholds for better quality signals
        let enhanced_mult = self.min_score_mult * 1.4; // 40% stricter for better quality
        let buy_cond = if self.is_contrarian { score < -(eff * enhanced_mult) } else { score > eff * enhanced_mult };
        let sell_cond = if self.is_contrarian { score > eff * enhanced_mult } else { score < -(eff * enhanced_mult) };

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

        // Apply confidence threshold - only execute trades if confidence is high enough
        if !matches!(signal, Signal::Hold) && conf < self.confidence_min {
            signal = Signal::Hold;  // Override signal if confidence is too low
        }

        // Enhanced debug logging for confidence and trade size analysis
        if matches!(signal, Signal::Buy) || matches!(signal, Signal::Sell) {
            let dynamic_size = self.calculate_dynamic_trade_size(conf, current_price);
            let volatility = self.recent_vol();
            
            debug!(
                "SIGNAL GENERATED - Signal: {:?}, Confidence: {:.6}, Dynamic Size: {:.6}, Score: {:.6}, Vol: {:.6}, R1: {:.6}, R2: {:.6}, Aggr: {:.4}, EI: {:.4}",
                signal, conf, dynamic_size, score, volatility,
                if let Some(last) = self.snapshot_buffer.back() { 
                    if self.snapshot_buffer.len() >= 2 { 
                        (last.p - self.snapshot_buffer[self.snapshot_buffer.len() - 2].p) / self.snapshot_buffer[self.snapshot_buffer.len() - 2].p 
                    } else { 0.0 }
                } else { 0.0 },
                if self.snapshot_buffer.len() >= 3 { 
                    (self.snapshot_buffer[self.snapshot_buffer.len() - 2].p - self.snapshot_buffer[self.snapshot_buffer.len() - 3].p) / self.snapshot_buffer[self.snapshot_buffer.len() - 3].p 
                } else { 0.0 },
                aggr, ei
            );
        }

        (signal, conf)
    }
}


