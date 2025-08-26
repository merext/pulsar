.PHONY: test trade

backtest:
	RUST_LOG=info cargo run --bin binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-05-28.zip
	RUST_LOG=info cargo run --bin binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-06-28.zip
	RUST_LOG=info cargo run --bin binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-08-08.zip

test:
	RUST_LOG=info cargo run --bin binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-06-28.zip

trade:
	RUST_LOG=info cargo run --bin binance-bot trade

emulate:
	RUST_LOG=info cargo run --bin binance-bot emulate