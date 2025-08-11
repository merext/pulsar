use binance_exchange::client::BinanceClient;
use futures_util::StreamExt;
use strategies::strategy::Strategy;
use trade::models::Trade;
use trade::trader::Trader;
use trade::trading_engine::{TradingEngine, PerformanceMetrics, TradingConfig};

use tracing::info;

pub async fn run_backtest(
    source: &str,
    mut strategy: impl Strategy + Send,
    symbol: &str,
    config_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _binance_client = BinanceClient::new().await?;
    let mut trade_stream: Box<dyn futures_util::Stream<Item = Trade> + Unpin + Send> =
        if source.starts_with("wss://") {
            Box::new(BinanceClient::trade_data(source).await?)
        } else {
            Box::new(BinanceClient::trade_data_from_path(source).await?)
        };

    let cfg = TradingConfig::from_file(config_path)?;
    let mut trader = TradingEngine::new_with_config(symbol, cfg)?;
    
    let mut trade_count = 0;
    let mut _last_metrics_print = 0;

    // Determine if this is live emulation (WebSocket) or backtest (file)
    let is_live_emulation = source.starts_with("wss://");

    info!("Starting {} for {} with source: {}", 
        if is_live_emulation { "LIVE EMULATION" } else { "BACKTEST" }, 
        symbol, 
        source
    );

    if is_live_emulation {
        info!("🔄 Live emulation mode: Consuming real-time WebSocket stream...");
        info!("📊 Strategy will generate signals based on live market data");
        info!("💻 All trades will be simulated with realistic slippage, fees, and latency");
        info!("⏹️  Press Ctrl+C to stop emulation");
    }

    while let Some(trade) = trade_stream.next().await {
        trade_count += 1;
        
        let trade_price = trade.price;
        let trade_time = trade.event_time as f64;
        
        // Update strategy with live trade data
        strategy.on_trade(trade.clone().into()).await;
        
        // Get signal from strategy
        let (signal, confidence) = 
            strategy.get_signal(trade_price, trade_time, trader.position());
        
        // Check if we should trade
        let should_trade = trader.should_trade(&signal, confidence, trade_price, trade_time);
        
        if should_trade {
            // Use central TradingConfig-based size calculation
            let quantity_to_trade = trader.calculate_trade_size(
                symbol, 
                trade_price, 
                confidence, 
                trader.config.position_sizing.trading_size_min,
                trader.config.position_sizing.trading_size_max,
                trader.config.exchange.step_size
            );
            
            // Use TradingEngine for realistic trade execution simulation
            let trade_result = trader.on_emulate(signal, trade_price, quantity_to_trade).await;
            
            // Log executed trades with execution metrics
            if let Some((fill_price, fees, slippage, rebates, order_type)) = trade_result {
                info!(
                    "🚀 {} EXECUTED TRADE - Signal: {:?}, Confidence: {:.3}, Price: {:.6}, Fill: {:.6}, Qty: {:.2}, Order: {:?}, Fees: {:.8}, Slippage: {:.8}, Rebates: {:.8}",
                    if is_live_emulation { "LIVE EMULATION" } else { "BACKTEST" },
                    signal, confidence, trade_price, fill_price, quantity_to_trade, order_type, 
                    fees, slippage, rebates
                );
            } else {
                info!(
                    "🚀 {} EXECUTED TRADE - Signal: {:?}, Confidence: {:.3}, Price: {:.6}, Qty: {:.2}",
                    if is_live_emulation { "LIVE EMULATION" } else { "BACKTEST" },
                    signal, confidence, trade_price, quantity_to_trade
                );
            }
        }

        // Print metrics more frequently for live emulation
        let metrics_interval = if is_live_emulation { 100 } else { 1000 };
        if trade_count % metrics_interval == 0 {
            _last_metrics_print = trade_count;
            print_metrics(&trader.metrics, trade_count, is_live_emulation);
        }
    }

    // Print final metrics
    print_metrics(&trader.metrics, trade_count, is_live_emulation);
    Ok(())
}

fn print_metrics(metrics: &PerformanceMetrics, trade_count: usize, is_live_emulation: bool) {
    let mode = if is_live_emulation { "LIVE EMULATION" } else { "BACKTEST" };
    info!(
        "📊 {} - TRADES: {} | PnL: {:.6} | Sharpe: {:.3} | MaxDD: {:.3} | WinRate: {:.1}% | ProfitFactor: {:.2}",
        mode,
        trade_count,
        metrics.net_pnl_after_costs(),
        metrics.sharpe_ratio(),
        metrics.max_drawdown,
        metrics.win_rate() * 100.0,
        metrics.profit_factor()
    );
}
