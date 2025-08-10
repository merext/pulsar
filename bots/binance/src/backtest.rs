use binance_exchange::client::BinanceClient;
use strategies::strategy::Strategy;
use tokio_stream::StreamExt;
use tracing::info;
use trade::models::Trade;
use trade::trader::Trader;
use trade::trading_engine::{TradingEngine, PerformanceMetrics, TradingConfig};

pub async fn run_backtest(
    source: &str,
    mut strategy: impl Strategy + Send,
    symbol: &str,
    config_path: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let binance_client = BinanceClient::new().await?;
    let mut trade_stream: Box<dyn futures_util::Stream<Item = Trade> + Unpin> =
        if source.starts_with("http") {
            Box::new(binance_client.trade_data(source).await?)
        } else {
            Box::new(binance_client.trade_data_from_path(source).await?)
        };

    let cfg = TradingConfig::from_file(config_path)?;
    let mut trader = TradingEngine::new_with_config(symbol, cfg)?;
    let mut trade_count = 0;
    let mut last_metrics_print = 0;

    while let Some(trade) = trade_stream.next().await {
        strategy.on_trade(trade.clone().into()).await;

        let trade_price = trade.price;
        let trade_time = trade.trade_time as f64;

        let (signal, confidence) =
            strategy.get_signal(trade_price, trade_time, trader.position());

        // Check if we should trade based on signal strength and risk management
        let should_trade = trader.should_trade(&signal, confidence, trade_price, trade_time);
        
        if should_trade {
            // Use central TradingConfig-based size calculation
            let quantity_to_trade = trader.calculate_trade_size(symbol, trade_price, confidence, 0.0, 0.0, 0.0);
            
            // Debug logging for trade execution
            tracing::debug!(
                "TRADE EXECUTION - Signal: {:?}, Confidence: {:.6}, Price: {:.6}, Qty: {:.6}, Time: {:.3}",
                signal, confidence, trade_price, quantity_to_trade, trade_time
            );
            
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
    let profit_factor = metrics.profit_factor();
    let avg_win = metrics.average_win();
    let avg_loss = metrics.average_loss();
    let sharpe = metrics.sharpe_ratio();
    let total_costs = metrics.total_costs();
    
    info!(
        "Metrics - Trades: {}/{} ({}%), Win Rate: {:.2}%, Net PnL: {:.6}, Gross PnL: {:.6}, Total Costs: {:.6}, Profit Factor: {:.2}, Avg Win: {:.6}, Avg Loss: {:.6}, Sharpe: {:.3}, Max DD: {:.2}%",
        metrics.total_trades,
        total_trades,
        if total_trades > 0 { (metrics.total_trades * 100) / total_trades } else { 0 },
        metrics.win_rate() * 100.0,
        metrics.net_pnl_after_costs(),
        metrics.gross_pnl(),
        total_costs,
        profit_factor,
        avg_win,
        avg_loss,
        sharpe,
        metrics.max_drawdown * 100.0
    );
}
