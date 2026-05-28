# data/downday/

Small static files (committed):
- backtest_universe.csv  - universe of symbols used in backtests
- market_data.duckdb     - lightweight market context DB

Large files (NOT committed - download first):
- daily_features.h5       (~4.0 GB)  primary feature store
- daily_features.parquet  (~2.4 GB)  parquet mirror
- prices_daily.csv        (~950 MB)  daily OHLCV bars
- prices_hourly.csv       (~2.3 GB)  hourly OHLCV bars
- prices_hourly.parquet   (~312 MB)  hourly OHLCV bars
- nasdaq_screener.csv     (~1 MB)    market cap and sector metadata

Run from repo root to download:
    python setup_backtest_data.py

Dataset owner sync from local TradingMethods/DownDay sources:
    python setup_backtest_data.py --sync

Upload only selected files:
    python setup_backtest_data.py --upload --files prices_daily.csv prices_hourly.csv nasdaq_screener.csv

Source:
    https://huggingface.co/datasets/natedoggztn/autotrade-backtest-data
