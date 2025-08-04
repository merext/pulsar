# Configuration Parameter Usage Analysis

## Overview
This document analyzes which configuration parameters from `trading_config.toml` are actually used in the code and identifies missing or unused parameters.

## 🔍 Analysis Results

### **✅ Parameters That ARE Used in Code:**

#### **Exchange Config (Used)**
- `taker_fee` ✅ - Used in `trading_engine.rs:229`
- `maker_fee` ✅ - Used in `trading_engine.rs:227`
- `maker_rebate` ✅ - Used in `trading_engine.rs:236`
- `min_slippage` ✅ - Used in `trading_engine.rs:217`
- `max_slippage` ✅ - Used in `trading_engine.rs:218`

#### **Risk Management Config (Used)**
- `max_position_size` ✅ - Used in `trading_engine.rs:28`
- `max_daily_loss` ✅ - Used in `trading_engine.rs:269`
- `max_drawdown` ✅ - Used in `trading_engine.rs:263`
- `max_consecutive_losses` ✅ - Used in `trading_engine.rs:255`

#### **Order Management Config (Used)**
- `partial_fill_probability` ✅ - Used in `trading_engine.rs:43`

#### **Performance Tracking Config (Used)**
- `track_fill_rate` ✅ - Used in `trading_engine.rs:48`
- `track_slippage` ✅ - Used in `trading_engine.rs:49`
- `track_fees` ✅ - Used in `trading_engine.rs:50`
- `track_rebates` ✅ - Used in `trading_engine.rs:51`

### **❌ Parameters That Are NOT Used in Code:**

#### **Exchange Config (Missing from Code)**
- `tick_size` ❌ - Not used anywhere
- `min_notional` ❌ - Hardcoded in `backtest.rs:36` and `trade.rs:95`
- `max_order_size` ❌ - Not used anywhere

#### **Order Management Config (Missing from Code)**
- `order_timeout` ❌ - Not used (code has `limit_order_timeout` instead)
- `max_retries` ❌ - Not used anywhere
- `retry_delay` ❌ - Not used anywhere

#### **Market Data Config (Missing from Code)**
- `min_trade_size` ❌ - Not used anywhere
- `max_price_change` ❌ - Not used anywhere
- `outlier_threshold` ❌ - Not used anywhere
- `trading_hours_start` ❌ - Not used anywhere
- `trading_hours_end` ❌ - Not used anywhere

#### **Performance Tracking Config (Missing from Code)**
- `track_order_rejections` ❌ - Not used anywhere

### **❌ Backtest Settings (Not Implemented)**

#### **All Backtest-Specific Parameters Are Missing:**
- `initial_capital` ❌ - Not used in backtest
- `commission_model` ❌ - Not used
- `slippage_model` ❌ - Not used
- `fill_model` ❌ - Not used
- `handle_data_gaps` ❌ - Not used
- `interpolate_missing_data` ❌ - Not used
- `filter_outliers` ❌ - Not used
- `prevent_lookahead_bias` ❌ - Not used
- `use_point_in_time_data` ❌ - Not used
- `simulate_market_microstructure` ❌ - Not used
- `account_for_latency_in_signals` ❌ - Not used
- `simulate_order_queue_position` ❌ - Not used
- `simulate_partial_fills` ❌ - Not used
- `simulate_order_cancellations` ❌ - Not used
- `simulate_market_impact` ❌ - Not used
- `market_impact_model` ❌ - Not used
- `impact_decay_rate` ❌ - Not used
- `simulate_order_delays` ❌ - Not used
- `simulate_exchange_errors` ❌ - Not used
- `simulate_network_timeouts` ❌ - Not used
- **All latency simulation parameters** ❌ - Not used
- **All market microstructure parameters** ❌ - Not used

## 🚨 Critical Issues Found

### **1. Configuration Struct Mismatch**
The `TradingConfig` struct in `trading_engine.rs` doesn't match our TOML configuration:

**Code has:**
```rust
pub struct ExchangeConfig {
    pub taker_fee: f64,
    pub maker_fee: f64,
    pub maker_rebate: f64,
    pub min_slippage: f64,
    pub max_slippage: f64,
    pub slippage_volatility: f64,  // ❌ Not in our TOML
}
```

**Our TOML has:**
```toml
[exchange]
taker_fee = 0.001
maker_fee = 0.0001
maker_rebate = 0.0001
tick_size = 0.00001        # ❌ Not in code struct
min_notional = 5.0         # ❌ Not in code struct
max_order_size = 1000.0    # ❌ Not in code struct
```

### **2. Missing Configuration Sections**
Our TOML has sections that don't exist in the code:
- `[slippage]` section ❌ - Not in code
- `[market_data]` section ❌ - Not in code
- `[backtest_settings]` section ❌ - Not in code

### **3. Hardcoded Values**
Some values are hardcoded instead of using config:
- `min_notional = 1.0 + 3.0 * confidence` in `backtest.rs:36`
- `min_notional = 1.0 + 3.0 * confidence` in `trade.rs:95`

### **4. Unused Parameters**
Many parameters in our TOML are not used anywhere in the codebase.

## 🔧 Required Fixes

### **1. Update Configuration Structs**
Need to update `TradingConfig` structs to match our TOML configuration.

### **2. Implement Missing Sections**
Need to add support for:
- `[slippage]` section
- `[market_data]` section
- `[backtest_settings]` section

### **3. Replace Hardcoded Values**
Replace hardcoded values with config parameters.

### **4. Implement Backtest Settings**
Actually implement the backtest-specific functionality.

### **5. Remove Unused Parameters**
Either remove unused parameters or implement their functionality.

## 📋 Action Items

1. **Fix Configuration Structs** - Update code to match TOML structure
2. **Implement Missing Functionality** - Add support for unused parameters
3. **Replace Hardcoded Values** - Use config values instead
4. **Add Backtest Implementation** - Implement backtest-specific features
5. **Test Parameter Usage** - Verify all parameters are actually used 