linux:
	cargo build --release --target x86_64-unknown-linux-gnu

deploy:
	scp target/x86_64-unknown-linux-gnu/release/binance-bot ec2-user@3.112.246.38:~/
