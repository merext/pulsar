# Pulsar Trading Bot

A sophisticated cryptocurrency trading bot with multiple advanced strategies, optimized for both traditional and high-frequency trading (HFT).

## 🚀 Features

- **Multiple Trading Strategies**: 13 different strategies from simple RSI to advanced neural networks
- **High-Frequency Trading (HFT)**: Ultra-low latency strategies optimized for microsecond execution
- **Centralized Configuration**: Single configuration file for all strategies
- **Risk Management**: Built-in risk controls and position sizing
- **Performance Monitoring**: Real-time performance tracking and optimization
- **Binance Integration**: Direct integration with Binance exchange

## 📊 Available Strategies

### Legacy Strategies
- **RSI Strategy** - Mean reversion based on overbought/oversold conditions
- **Mean Reversion Strategy** - Statistical arbitrage using moving averages
- **Momentum Scalping Strategy** - Short-term momentum capture
- **Kalman Filter Strategy** - Noise filtering and trend following
- **Order Book Imbalance Strategy** - Order flow analysis
- **Spline Strategy** - Smooth trend following using interpolation
- **VWAP Deviation Strategy** - Volume-weighted analysis
- **Z-Score Strategy** - Statistical arbitrage using z-scores
- **Fractal Approximation Strategy** - Pattern recognition using fractals

### Advanced Strategies
- **Adaptive Multi-Factor Strategy** - Ensemble approach combining multiple indicators
- **Neural Market Microstructure Strategy** - ML-inspired pattern recognition

### HFT Strategies
- **HFT Ultra-Fast Strategy** - Ultra-low latency directional trading
- **HFT Market Maker Strategy** - Spread capture and market making

## ⚙️ Configuration

Each strategy has its own separate configuration file in the `config/` directory. All strategies can be initialized without parameters using `Strategy::new()`.

### Individual Configuration Files

Each strategy loads its configuration from its own file:

- **`config/rsi_strategy.toml`** - RSI Strategy parameters
- **`config/mean_reversion_strategy.toml`** - Mean Reversion Strategy parameters  
- **`config/momentum_scalping_strategy.toml`** - Momentum Scalping Strategy parameters
- **`config/kalman_filter_strategy.toml`** - Kalman Filter Strategy parameters
- **`config/order_book_imbalance_strategy.toml`** - Order Book Imbalance Strategy parameters
- **`config/spline_strategy.toml`** - Spline Strategy parameters
- **`config/vwap_deviation_strategy.toml`** - VWAP Deviation Strategy parameters
- **`config/zscore_strategy.toml`** - Z-Score Strategy parameters
- **`config/fractal_approximation_strategy.toml`** - Fractal Approximation Strategy parameters
- **`config/hft_ultra_fast_strategy.toml`** - HFT Ultra-Fast Strategy parameters
- **`config/hft_market_maker_strategy.toml`** - HFT Market Maker Strategy parameters
- **`config/adaptive_multi_factor_strategy.toml`** - Adaptive Multi-Factor Strategy parameters
- **`config/neural_market_microstructure_strategy.toml`** - Neural Market Microstructure Strategy parameters

### Usage

```rust
// Simple initialization - no parameters needed
let rsi_strategy = RsiStrategy::new(); // Loads from config/rsi_strategy.toml
let hft_strategy = HftUltraFastStrategy::new(); // Loads from config/hft_ultra_fast_strategy.toml
let adaptive_strategy = AdaptiveMultiFactorStrategy::new(); // Loads from config/adaptive_multi_factor_strategy.toml
```

### Configuration Loading

- **Automatic Loading**: Each strategy automatically loads its configuration file
- **Default Fallbacks**: If a config file is missing, sensible defaults are used
- **No Code Changes**: Modify parameters by editing the TOML files
- **Easy Switching**: Change strategies by simply changing the constructor call

### Signal Thresholds

All strategies include a `signal_threshold` parameter (default: 0.6) that defines the minimum confidence level required to generate trading signals.

## 🏗️ Architecture

```
pulsar/
├── bots/
│   └── binance/          # Binance trading bot
├── strategies/           # All trading strategies
├── trade/               # Core trading functionality
├── exchanges/           # Exchange integrations
└── config/              # Configuration files
```

## 🚀 Quick Start

1. **Set up environment variables**:
   ```bash
   export BINANCE_API_KEY="your_api_key"
   export BINANCE_API_SECRET="your_api_secret"
   ```

2. **Configure your strategy** by editing the appropriate config file:
   ```bash
   # For HFT Ultra-Fast Strategy
   nano config/hft_ultra_fast_strategy.toml
   
   # For RSI Strategy
   nano config/rsi_strategy.toml
   
   # For Adaptive Multi-Factor Strategy
   nano config/adaptive_multi_factor_strategy.toml
   ```

3. **Run the bot**:
   ```bash
   cd bots/binance
   cargo run -- trade
   ```

## 📈 Strategy Performance

### HFT Strategies
- **Latency**: < 1 microsecond
- **Buffer Size**: 64 ticks
- **Signal Threshold**: 60% confidence
- **Stop Loss**: 0.2%
- **Take Profit**: 0.2%

### Advanced Strategies
- **Multi-timeframe Analysis**: Short, medium, and long-term signals
- **Adaptive Parameters**: Adjusts to market conditions
- **Risk Management**: Built-in position sizing and stop losses
- **Performance Tracking**: Real-time metrics and optimization

## 🔧 Development

### Adding a New Strategy

1. Create a new strategy file in `strategies/src/`
2. Implement the `Strategy` trait
3. Add configuration section to `config/advanced_strategies_config.toml`
4. Add default values to `strategies/src/config.rs`
5. Update `strategies/src/lib.rs` to export the strategy

### Running Examples

```bash
# Strategy configuration example
cargo run --example strategy_config_example

# Advanced strategies example
cargo run --example advanced_strategies_example
```

## 📚 Documentation

- **Strategy Guide**: See `STRATEGY_GUIDE.md` for detailed strategy explanations
- **HFT Guide**: See `HFT_README.md` for HFT-specific documentation
- **Examples**: See `examples/` directory for usage examples

## ⚠️ Risk Warning

Trading cryptocurrencies involves significant risk. This bot is for educational purposes. Always:
- Start with small amounts
- Test thoroughly in simulation mode
- Monitor performance closely
- Never risk more than you can afford to lose

## 📄 License

This project is for educational purposes. Use at your own risk.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For questions or issues, please open an issue on GitHub. 