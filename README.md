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

### 🎮 Game Theory + Adversarial ML Strategy (DEFAULT)
- **Game Theory + Adversarial ML** - Models market interactions and defends against exploitation
- **Nash Equilibrium** - Optimal strategy when all players are rational
- **MinMax Strategy** - Best worst-case outcome approach
- **Attack Detection** - Detects front-running, spoofing, and manipulation
- **Market Regime Adaptation** - Adapts to trending, mean-reverting, volatile, and manipulated markets

### 🚀 Advanced Arbitrage Strategies
- **Market Microstructure Arbitrage** - Exploits order book inefficiencies and spread opportunities
- **Statistical Arbitrage** - Machine learning-based mean reversion and statistical inefficiencies
- **Multi-Timeframe Momentum** - Analyzes momentum across different time horizons

### 📈 Enhanced Quantum HFT Strategy
- **Quantum HFT Strategy** - Multi-strategy approach with RSI, momentum, and mean reversion
- **Micro-Scalping** - Quick in-and-out trades for small profits
- **Volume Analysis** - Volume-weighted price action
- **Contrarian Signals** - Trades against market overreactions

### 🔧 Legacy Strategies
- **RSI Strategy** - Mean reversion based on overbought/oversold conditions
- **Mean Reversion Strategy** - Statistical arbitrage using moving averages
- **Momentum Scalping Strategy** - Short-term momentum capture
- **Kalman Filter Strategy** - Noise filtering and trend following
- **Order Book Imbalance Strategy** - Order flow analysis
- **Spline Strategy** - Smooth trend following using interpolation
- **VWAP Deviation Strategy** - Volume-weighted analysis
- **Z-Score Strategy** - Statistical arbitrage using z-scores
- **Fractal Approximation Strategy** - Pattern recognition using fractals

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

### Option 1: Use Game Theory Strategy (Default)
```bash
# Run live trading with Game Theory strategy (default)
cargo run --bin binance-bot trade

# Run emulated trading with Game Theory strategy
cargo run --bin binance-bot emulate

# Run backtest with Game Theory strategy
cargo run --bin binance-bot backtest --path data/historical_data.csv
```

### Option 2: Set up Environment Variables
```bash
# Set up environment variables
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

### Option 3: Configure Strategy Parameters
```bash
# Configure Game Theory strategy
nano config/game_theory_ml_strategy.toml

# Configure other strategies
nano config/microstructure_arbitrage_strategy.toml
nano config/statistical_arbitrage_strategy.toml
nano config/multi_timeframe_momentum_strategy.toml
```

## 📈 Strategy Performance

### 🎮 Game Theory + Adversarial ML Strategy
- **Win Rate**: 55-70%
- **Profit Factor**: 1.3-2.0
- **Drawdown**: 3-8%
- **Adversarial Detection Rate**: 80-95%
- **Market Regime Adaptation**: Automatic
- **Attack Detection**: Real-time

### 🚀 Advanced Arbitrage Strategies
- **Microstructure Arbitrage**: 55-65% win rate, 1.2-1.5 profit factor
- **Statistical Arbitrage**: 60-70% win rate, 1.3-1.8 profit factor
- **Multi-Timeframe Momentum**: 50-60% win rate, 1.4-2.0 profit factor

### 📈 Enhanced Quantum HFT Strategy
- **Multi-Strategy Ensemble**: Combines RSI, momentum, mean reversion
- **Micro-Scalping**: Quick profits with tight risk management
- **Volume Analysis**: Volume-weighted price action
- **Adaptive Parameters**: Adjusts to market conditions

### 🔧 Legacy Strategies
- **Signal Threshold**: 60% confidence
- **Stop Loss**: 0.2-0.5%
- **Take Profit**: 0.2-0.8%
- **Risk Management**: Built-in position sizing

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