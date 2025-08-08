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
    trading_size_min: f64,
    trading_size_max: f64,
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

        // Debug logging for signal flow
        if trade_count % 100 == 0 {
            println!("DEBUG: Backtest - Trade count: {}, Signal: {:?}, Confidence: {:.3}", 
                trade_count, signal, confidence);
        }

        // Check if we should trade based on signal strength and risk management
        let should_trade = trader.should_trade(&signal, confidence, trade_price, trade_time);
        
        // Debug logging for trade decision
        if trade_count % 100 == 0 {
            println!("DEBUG: Trade decision - should_trade: {}, signal: {:?}, confidence: {:.3}", 
                should_trade, signal, confidence);
        }
        
        if should_trade {
            // Exchange calculates exact trade size based on symbol, price, confidence, min/max trade sizes, and step size
            let quantity_to_trade = trader.calculate_trade_size(symbol, trade_price, confidence, trading_size_min, trading_size_max, 1.0);
            
            // Debug logging for trade execution
            if trade_count % 100 == 0 {
                println!("DEBUG: Executing trade - Signal: {:?}, Price: {:.6}, Quantity: {:.6}", 
                    signal, trade_price, quantity_to_trade);
            }
            
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
