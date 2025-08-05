.PHONY: test trade

test:
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-05-28.zip | tail -1

emulate:
	RUST_LOG=info cargo run emulate

trade:
	RUST_LOG=info cargo run trade
