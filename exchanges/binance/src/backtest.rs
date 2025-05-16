#[derive(Debug, serde::Deserialize)]
struct BinanceCsvKlineRow(
    (),                         // open time
    (),                         // open price
    (),                         // high price
    (),                         // low price
    (),                         // close price
    (),                         // volume
    (),                         // close time
    #[allow(dead_code)] String, // quote asset volume (ignored)
    #[allow(dead_code)] u64,    // number of trades (ignored)
    #[allow(dead_code)] String, // taker buy base volume (ignored)
    #[allow(dead_code)] String, // taker buy quote volume (ignored)
    #[allow(dead_code)] String, // ignore
);
