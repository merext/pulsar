pub fn simple_moving_average(prices: Vec<f64>, window_size: usize) -> f64 {
    let sum: f64 = prices.iter().take(window_size).sum();
    sum / window_size as f64
}