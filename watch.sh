while true; do
  date
	RUST_LOG=info ./target/debug/binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-05-28.zip | grep profit | tail -1
	RUST_LOG=info ./target/debug/binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-06-28.zip | grep profit | tail -1
	RUST_LOG=info ./target/debug/binance-bot backtest --uri ./data/DOGEUSDT-trades-2025-08-08.zip | grep profit | tail -1
  sleep 10
done
