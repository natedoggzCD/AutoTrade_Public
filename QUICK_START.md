# Quick Start

## 1. Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Download Market Data

```powershell
python setup_backtest_data.py
```

Files are downloaded into `data/downday/` from:
`https://huggingface.co/datasets/natedoggztn/autotrade-backtest-data`

## 3. Configure Environment

Copy `.env.example` to `.env` and fill in only the services you intend to use.

```powershell
Copy-Item .env.example .env
```

Backtesting and local data checks do not require live brokerage credentials.

## 4. Verify Setup

```powershell
python -m pytest tests/test_data_ingestion_module.py::TestIngestionPaths -q
python setup_backtest_data.py --help
```

## 5. Useful Paths

- `data/downday/` - local market-data cache
- `autotrade/backtesting/` - backtest engines and validation
- `autotrade/signals/` - scanner and signal logic
- `config/trading_config.yaml` - default runtime config
- `setup_backtest_data.py` - Hugging Face download/sync helper
