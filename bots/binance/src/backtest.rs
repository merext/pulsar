use binance_exchange::client::BinanceClient;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::info;
use trade::models::Trade;
use trade::trader::Trader;
use trade::trading_engine::{TradingEngine, PerformanceMetrics};

pub async fn run_backtest(
    source: &str,
    mut strategy: impl Strategy + Send,
    symbol: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new().await?;
    let mut trade_stream: Box<dyn futures_util::Stream<Item = Trade> + Unpin> =
        if source.starts_with("http") {
            Box::new(binance_client.trade_data(source).await?)
        } else {
            Box::new(binance_client.trade_data_from_path(source).await?)
        };

    let mut trader = TradingEngine::new(symbol)?;
    let mut trade_count = 0;
    let mut last_metrics_print = 0;

    while let Some(trade) = trade_stream.next().await {
        strategy.on_trade(trade.clone().into()).await;

        let trade_price = trade.price;
        let trade_time = trade.trade_time as f64;

        let (signal, confidence) =
            strategy.get_signal(trade_price, trade_time, trader.position());

        // Check if we should trade based on signal strength and risk management
        if trader.should_trade(&signal, confidence, trade_price, trade_time) {
            // Use dynamic position sizing based on pair-specific configuration
            let available_capital = if let Some(backtest_settings) = &trader.config.backtest_settings {
                backtest_settings.initial_capital
            } else {
                10000.0 // Default fallback
            };
            
            let quantity_to_trade = trader.calculate_position_size(symbol, trade_price, confidence, available_capital);
            
            trader.on_emulate(signal, trade_price, quantity_to_trade).await;
        }

        trade_count += 1;
        
        // Print metrics every 1000 trades
        if trade_count - last_metrics_print >= 1000 {
            print_metrics(&trader.metrics, trade_count);
            last_metrics_print = trade_count;
        }
    }

    // Print final metrics
    info!("=== FINAL BACKTEST RESULTS ===");
    print_metrics(&trader.metrics, trade_count);
    
    Ok(())
}

fn print_metrics(metrics: &PerformanceMetrics, total_trades: usize) {
    info!(
        "Metrics - Trades: {}/{} ({}%), Win Rate: {:.2}%, Net PnL: {:.6}, Fees: {:.6}, Rebates: {:.6}, Slippage: {:.6}, Max Drawdown: {:.2}%",
        metrics.total_trades,
        total_trades,
        if total_trades > 0 { (metrics.total_trades * 100) / total_trades } else { 0 },
        metrics.win_rate() * 100.0,
        metrics.net_pnl_after_costs(),
        metrics.total_fees,
        metrics.total_rebates,
        metrics.total_slippage,
        metrics.max_drawdown * 100.0
    );
}
