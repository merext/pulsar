# ═══════════════════════════════════════════════════════════════════════════
# Pulsar — Makefile
# ═══════════════════════════════════════════════════════════════════════════

BIN      := binance-bot
STRATEGY := market-maker
CONFIG   := config/trading_config_bnbusdt.toml
STRAT_CF := config/strategies/market_maker_bnbusdt.toml

# ─── Build ────────────────────────────────────────────────────────────────

.PHONY: build build-debug test check clean

build:                          ## Release build
	cargo build --release -p $(BIN)

build-debug:                    ## Debug build
	cargo build -p $(BIN)

test:                           ## Run all workspace tests
	cargo test --workspace

check:                          ## Lint (clippy + fmt check)
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings

clean:                          ## Remove build artefacts
	cargo clean

# ─── Live / Emulate ──────────────────────────────────────────────────────

EMULATE_DURATION ?= 180

.PHONY: trade emulate rebalance multi-trade

trade: build                    ## Live trading (default symbol from CONFIG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade

multi-trade: build              ## Live trading on TOP 5 pairs simultaneously
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		multi-trade

emulate: build                  ## Emulate with live data, simulated execution
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		--duration-secs $(EMULATE_DURATION) \
		emulate

rebalance: build                ## Sell base asset back to quote
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		rebalance

# ─── FDUSD Zero-Fee Shortcuts ────────────────────────────────────────────

.PHONY: trade-bnbfdusd trade-wiffdusd trade-trumpfdusd trade-linkfdusd trade-suifdusd trade-dogefdusd trade-dogefdusd-twosided

LIVE_LOG := /tmp/live_trade.txt

trade-bnbfdusd: build           ## Live trade BNBFDUSD (0% maker fee)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_bnbfdusd.toml \
		--strategy-config config/strategies/market_maker_bnbfdusd.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

trade-wiffdusd: build           ## Live trade WIFFDUSD (0% maker fee, ~55 bps spread)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_wiffdusd.toml \
		--strategy-config config/strategies/market_maker_wiffdusd.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

trade-trumpfdusd: build         ## Live trade TRUMPFDUSD (0% maker fee, ~17 bps spread)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_trumpfdusd.toml \
		--strategy-config config/strategies/market_maker_trumpfdusd.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

trade-linkfdusd: build          ## Live trade LINKFDUSD (0% maker fee, ~11 bps spread)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_linkfdusd.toml \
		--strategy-config config/strategies/market_maker_linkfdusd.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

trade-suifdusd: build           ## Live trade SUIFDUSD (0% maker fee, ~2.3 bps spread)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_suifdusd.toml \
		--strategy-config config/strategies/market_maker_suifdusd.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

trade-dogefdusd: build          ## Live trade DOGEFDUSD (0% maker fee, ~2.2 bps spread)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_dogefdusd.toml \
		--strategy-config config/strategies/market_maker_dogefdusd.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

trade-dogefdusd-twosided: build ## Live trade DOGEFDUSD two-sided MM (0% maker fee)
	truncate -s0 $(LIVE_LOG)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_dogefdusd.toml \
		--strategy-config config/strategies/market_maker_dogefdusd_twosided.toml \
		$(if $(TRADE_DURATION),--duration-secs $(TRADE_DURATION)) \
		trade 2>&1 | tee $(LIVE_LOG)

# ─── Backtest ─────────────────────────────────────────────────────────────

# Usage:
#   make backtest URI=data/binance/daily/trades/TRUUSDT/TRUUSDT-trades-2026-03-31.zip
#   make backtest-symbol SYMBOL=ARKMUSDT DATE=2026-03-31

URI ?= data/binance/daily/trades/BNBUSDT/BNBUSDT-trades-2026-03-31.zip

.PHONY: backtest backtest-symbol backtest-portfolio

backtest: build                 ## Single-file backtest (URI=path/to/file.zip)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		backtest --uri $(URI)

SYMBOL ?= BNBUSDT
DATE   ?= 2026-03-31
SYM_LC  = $(shell echo $(SYMBOL) | tr A-Z a-z)

backtest-symbol: build          ## Backtest specific symbol (SYMBOL=X DATE=Y)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config config/trading_config_$(SYM_LC).toml \
		--strategy-config config/strategies/market_maker_$(SYM_LC).toml \
		backtest --uri data/binance/daily/trades/$(SYMBOL)/$(SYMBOL)-trades-$(DATE).zip

backtest-portfolio:             ## Backtest all 19 portfolio symbols
	./scripts/backtest_portfolio.sh

# ─── Compare / Search / Optimize ─────────────────────────────────────────

# Usage:
#   make compare URIS="file1.zip file2.zip"
#   make search PARAMETER=cash_fraction VALUES="0.03,0.05,0.08,0.10" URIS="file1.zip file2.zip"
#   make walk-forward PARAMETER=cash_fraction VALUES="0.03,0.05,0.08" URIS="f1.zip f2.zip f3.zip f4.zip"

URIS      ?=
PARAMETER ?= cash_fraction
VALUES    ?= 0.03,0.05,0.08

.PHONY: compare search optimize walk-forward

compare: build                  ## Compare strategies across datasets (URIS="...")
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		compare --uris $(URIS)

search: build                   ## Parameter search (PARAMETER=X VALUES=X URIS="...")
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		search \
		--strategy $(STRATEGY) \
		--parameter $(PARAMETER) \
		--values $(VALUES) \
		--uris $(URIS)

optimize: build                 ## Optimize parameter (PARAMETER=X VALUES=X URIS="...")
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		optimize \
		--strategy $(STRATEGY) \
		--parameter $(PARAMETER) \
		--values $(VALUES) \
		--uris $(URIS)

walk-forward: build             ## Walk-forward validation (PARAMETER=X VALUES=X URIS="...")
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		walk-forward \
		--strategy $(STRATEGY) \
		--parameter $(PARAMETER) \
		--values $(VALUES) \
		--uris $(URIS)

# ─── Features / Diagnostics ──────────────────────────────────────────────

# Usage:
#   make features URI=data/.../file.zip OUTPUT=features.csv
#   make diagnostics URIS="file1.zip file2.zip"

OUTPUT ?= features.csv

.PHONY: features diagnostics attribution

features: build                 ## Extract ML features (URI=X OUTPUT=X)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		--strategy-config $(STRAT_CF) \
		features --uri $(URI) --output $(OUTPUT)

diagnostics: build              ## Strategy diagnostics (URIS="...")
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		strategy-diagnostics --uris $(URIS)

attribution: build              ## Trade attribution (URIS="...")
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		trade-attribution --uris $(URIS)

# ─── Capture ─────────────────────────────────────────────────────────────

# Usage:
#   make capture OUTPUT=data/capture/snap.jsonl DURATION=300 DEPTH=5
#   make capture-batch BATCH_ID=morning PARTS=6 DURATION=600

CAPTURE_OUTPUT ?= data/binance/capture/snapshot.jsonl
DURATION       ?= 300
DEPTH          ?= 5
BATCH_ID       ?= batch1
PARTS          ?= 4
GAP            ?= 0

.PHONY: capture capture-batch capture-backfill capture-index capture-compare

capture: build                  ## Capture live market data (OUTPUT=X DURATION=X DEPTH=X)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		capture \
		--output $(CAPTURE_OUTPUT) \
		--duration-secs $(DURATION) \
		--depth-levels $(DEPTH)

capture-batch: build            ## Batch capture (BATCH_ID=X PARTS=X DURATION=X)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		capture-batch \
		--batch-id $(BATCH_ID) \
		--parts $(PARTS) \
		--duration-secs $(DURATION) \
		--depth-levels $(DEPTH) \
		--gap-secs $(GAP)

capture-backfill: build         ## Backfill trade data into capture (INPUT=X)
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		--config $(CONFIG) \
		capture-backfill \
		--input $(INPUT)

capture-index:                  ## Index captured datasets
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		capture-index

capture-compare:                ## Compare captured datasets
	cargo run --release -p $(BIN) -- \
		--strategy $(STRATEGY) \
		capture-compare

# ─── Data download ───────────────────────────────────────────────────────

# Usage:
#   make pull PAIR=DOGEFDUSD DAYS=7

PAIR ?= DOGEFDUSD
DAYS ?= 7
DATA_BASE_URL := https://data.binance.vision/data/spot/daily/trades

.PHONY: pull

pull:                           ## Download trade data (PAIR=X DAYS=N)
	@mkdir -p data/binance/daily/trades/$(PAIR)
	@for i in $$(seq 1 $(DAYS)); do \
		d=$$(date -v-$${i}d +%Y-%m-%d); \
		f="$(PAIR)-trades-$$d.zip"; \
		dest="data/binance/daily/trades/$(PAIR)/$$f"; \
		if [ -f "$$dest" ]; then \
			echo "  SKIP  $$f (exists)"; \
		else \
			echo "  PULL  $$f"; \
			curl -sf -o "$$dest" "$(DATA_BASE_URL)/$(PAIR)/$$f" || \
				{ echo "  FAIL  $$f (not available)"; rm -f "$$dest"; }; \
		fi; \
	done
	@echo "Done. Files in data/binance/daily/trades/$(PAIR)/"

# ─── Analysis scripts ────────────────────────────────────────────────────

.PHONY: correlation

correlation:                    ## Compute portfolio correlation from daily_pnl.csv
	./scripts/compute_correlation.sh

# ─── Help ─────────────────────────────────────────────────────────────────

.PHONY: help
help:                           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
