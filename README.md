# AutoTrade

AutoTrade is a local Python research and trading-system workspace focused on
small and mid-cap equities. This public repo contains the code, tests, configs,
and lightweight seed files. Large market-data files are hosted separately on
Hugging Face so the Git repository stays usable.

## Quick Start

1. Create and activate a Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Download the public backtest dataset.

```powershell
python setup_backtest_data.py
```

The script downloads these files into `data/downday/` from
`natedoggztn/autotrade-backtest-data`:

- `daily_features.h5`
- `daily_features.parquet`
- `prices_daily.csv`
- `prices_hourly.csv`
- `prices_hourly.parquet`
- `nasdaq_screener.csv`

3. Copy `.env.example` to `.env` and add any credentials you intend to use.
Do not commit `.env`.

4. Run targeted tests or backtesting tools.

```powershell
python -m pytest tests/test_data_ingestion_module.py::TestIngestionPaths -q
```

## Data Sync

Normal users only need:

```powershell
python setup_backtest_data.py
```

Dataset maintainers can upload refreshed local source files with:

```powershell
python setup_backtest_data.py --sync
```

To upload only selected files:

```powershell
python setup_backtest_data.py --upload --files prices_daily.csv prices_hourly.csv nasdaq_screener.csv
```

## Core Data Inputs

The main local data cache is `data/downday/`.

- `backtest_universe.csv` and `market_data.duckdb` are committed lightweight seed files.
- Large data files are ignored by Git and downloaded from Hugging Face.
- Config defaults resolve data paths relative to `data/downday/`.

## Project Map

- `autotrade/backtesting/` - backtest engines, validation, metrics, and protocols
- `autotrade/signals/` - scanners and signal generation
- `autotrade/feature_engineering/` - deterministic feature transforms
- `autotrade/data_ingestion/` - local data path resolution and bootstrap checks
- `autotrade/risk/` - risk policies and position controls
- `config/` - default YAML and typed config loader
- `tests/` - targeted regression tests

## Security Notes

- Real API keys should live only in `.env` or your shell environment.
- `.env` is ignored by Git.
- The public data downloader supports `HF_TOKEN` for maintainers, but no token is
  required for the public dataset download path.

## Disclaimer

This software is for research and educational purposes. Trading involves risk.
Review and test any strategy thoroughly before connecting it to a live account.
