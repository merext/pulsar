.PHONY: test trade

backtest:
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-05-28.zip | tail -1
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-08-01.zip | tail -1
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-08-02.zip | tail -1
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-08-08.zip | tail -1
	RUST_LOG=info cargo run backtest --path ./data/DOGEUSDT-trades-2025-08-09.zip | tail -1

test:
	RUST_LOG=debug cargo run backtest --path ./data/test.zip | tail -1

trade:
	RUST_LOG=info cargo run trade

emulate:
	RUST_LOG=info cargo run emulate