# Configuration Separation: Trading vs Backtesting

## Overview
The `trading_config.toml` file contains both general trading settings and backtest-specific settings. This document explains how these are separated and used by different commands.

## 📁 Configuration Structure

### **General Trading Settings** (Used by both `trade` and `backtest` commands)

These settings apply to both live trading and backtesting:

#### **`[exchange]`** - Exchange Configuration
- Fees and rebates
- Market microstructure
- Tick size and order limits
- Bid-ask spread simulation

#### **`[slippage]`** - Slippage Modeling
- Min/max slippage
- Volatility multipliers
- Size-based slippage

#### **`[order_execution]`** - Order Execution
- Fill rates and probabilities
- Order rejection simulation
- Network latency

#### **`[risk_management]`** - Risk Management
- Position sizing limits
- Circuit breakers
- Drawdown limits

#### **`[market_data]`** - Market Data Handling
- Data quality filters
- Trading hours
- Market event handling

#### **`[performance_tracking]`** - Performance Metrics
- Comprehensive tracking
- Advanced analytics
- Performance ratios

# Note: Latency simulation and market microstructure are now part of backtest_settings

### **Backtest-Specific Settings** (Only used by `backtest` command)

These settings are **ignored** by the `trade` command:

#### **`[backtest_settings]`** - Backtest-Only Parameters

**Simulation Parameters:**
- `initial_capital` - Starting capital for backtesting
- `commission_model` - Fee structure for simulation
- `slippage_model` - Slippage model for simulation
- `fill_model` - Fill model for simulation

**Data Handling (Backtest-Specific):**
- `handle_data_gaps` - Handle missing data in historical data
- `interpolate_missing_data` - Data interpolation settings
- `filter_outliers` - Filter outliers in historical data

**Bias Prevention (Backtest-Specific):**
- `prevent_lookahead_bias` - Prevent future data leakage
- `use_point_in_time_data` - Use data available at each timestamp

**Simulation-Only Features:**
- `simulate_market_microstructure` - Simulate realistic market structure
- `account_for_latency_in_signals` - Apply latency to signal generation
- `simulate_order_queue_position` - Simulate order book queue position
- `simulate_partial_fills` - Simulate realistic partial fills
- `simulate_order_cancellations` - Simulate order cancellations

**Market Impact Simulation:**
- `simulate_market_impact` - Account for order size impact
- `market_impact_model` - Market impact model type
- `impact_decay_rate` - Impact decay rate

**Realistic Order Execution:**
- `simulate_order_delays` - Simulate order processing delays
- `simulate_exchange_errors` - Simulate exchange errors
- `simulate_network_timeouts` - Simulate network timeouts

**Latency Simulation (Backtest-Only):**
- `total_latency_model` - Realistic latency model for simulation
- `propagation_delay`, `transmission_delay`, `queuing_delay` - Network latency components
- `latency_spike_probability`, `latency_spike_multiplier` - Latency variation factors
- `exchange_processing_time`, `order_matching_latency` - Exchange-specific latency
- `signal_generation_time`, `order_construction_time` - Strategy latency considerations

**Market Microstructure Simulation (Backtest-Only):**
- `order_book_simulation` - Simulate realistic order book
- `bid_ask_spread_model`, `spread_volatility_factor` - Dynamic spread modeling
- `order_book_levels`, `level_size_distribution` - Order book depth simulation
- `market_maker_presence`, `mm_spread_tightening` - Market maker behavior
- `hft_competition`, `hft_latency_advantage` - HFT competition effects

## 🔧 Command Usage

### **`make test` (Backtest Command)**
```bash
make test  # Uses ALL settings including [backtest_settings]
```

**Reads:**
- ✅ All general trading settings
- ✅ All backtest-specific settings
- ✅ Comprehensive realistic simulation

### **`make trade` (Live Trading Command)**
```bash
make trade  # Uses ONLY general trading settings
```

**Reads:**
- ✅ All general trading settings
- ❌ **Ignores** `[backtest_settings]` section
- ✅ Real-time trading with live data

## 🎯 Key Benefits

### **1. Clear Separation**
- Backtest settings don't affect live trading
- Live trading uses only real-world applicable settings
- No confusion between simulation and reality

### **2. Realistic Backtesting**
- Comprehensive simulation parameters
- Bias prevention mechanisms
- Realistic market microstructure

### **3. Safe Live Trading**
- Only real-world applicable settings
- No simulation artifacts
- Production-ready configuration

## 📋 Example Usage

### **For Backtesting:**
```toml
[backtest_settings]
initial_capital = 10000.0
simulate_market_microstructure = true
prevent_lookahead_bias = true
```

### **For Live Trading:**
```toml
[risk_management]
max_position_size = 1000.0
max_drawdown = 0.20
```

The live trading command will use the risk management settings but ignore the backtest settings, ensuring safe and realistic trading. 