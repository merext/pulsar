.PHONY: test trade

test:
	RUST_LOG=info ./target/debug/binance-bot backtest --path ./data/DOGEUSDT-trades-2025-05-28.zip

emulate:
	RUST_LOG=info ./target/debug/binance-bot emulate

trade:
	RUST_LOG=info ./target/debug/binance-bot trade
