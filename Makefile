.PHONY: test trade

test:
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-05-28.zip

trade:
	RUST_LOG=info cargo run trade
