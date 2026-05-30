# AutoTrade

AutoTrade is a local Python research and trading-system workspace for small and
mid-cap equities. It combines large historical market datasets, technical
feature engineering, backtesting, signal generation, YouTube market-intelligence
summaries, risk controls, and optional Alpaca paper/live execution.

The public repository keeps code, tests, configs, prompt templates, and small
seed data in Git. Large backtest files are downloaded from the public Hugging
Face dataset:

`https://huggingface.co/datasets/natedoggztn/autotrade-backtest-data`

This project is research software. Do not connect it to a funded brokerage
account until you have reviewed the code, tested the strategy behavior, and
accepted the risk.

## What It Does

- Downloads a reproducible local backtest dataset from Hugging Face.
- Builds deterministic technical features for daily and hourly OHLCV data.
- Runs strategy backtests with leakage controls, metrics, and validation helpers.
- Generates candidate signals from momentum, breakout, mean reversion, PEAD,
  inverse ETF, and other strategy families.
- Uses YouTube market commentary as a contextual overlay for regime, sizing,
  sector bias, and avoid lists.
- Produces overnight, premarket, market-hours, power-hour, and post-market
  workflow artifacts.
- Supports paper/live brokerage integration through Alpaca when credentials are
  explicitly supplied.

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the public backtest data:

```powershell
python setup_backtest_data.py
```

Copy the example environment file and fill in only the services you plan to use:

```powershell
Copy-Item .env.example .env
```

Run a small verification:

```powershell
python setup_backtest_data.py --help
python setup_backtest_data.py --list
python -m pytest tests/test_data_ingestion_module.py::TestIngestionPaths -q
```

## Docker Testing

Use Docker when you want an isolated test environment that does not modify your
local Python, conda, CUDA, or package installation:

```powershell
docker compose build autotrade-test
docker compose run --rm autotrade-test
```

The default container command runs a narrow ingestion-path smoke test. For an
interactive shell:

```powershell
docker compose run --rm --profile shell autotrade-shell
```

See `DOCKER.md` for data mounts, Hugging Face downloads, and credential handling.

## Backtest Data

Large market-data files are intentionally ignored by Git and downloaded into
`data/downday/`.

Managed files:

- `daily_features.h5` - primary historical feature store.
- `daily_features.parquet` - parquet mirror for faster pandas reads.
- `prices_daily.csv` - daily OHLCV bars from TradingMethods.
- `prices_hourly.csv` - hourly OHLCV bars from TradingMethods.
- `prices_hourly.parquet` - compressed hourly OHLCV bars.
- `nasdaq_screener.csv` - symbol metadata, sector, industry, and market-cap
  fields.

Useful downloader commands:

```powershell
# Download missing files from Hugging Face
python setup_backtest_data.py

# Re-download selected files
python setup_backtest_data.py --force --files prices_daily.csv prices_hourly.csv

# Show the managed file inventory and local source paths
python setup_backtest_data.py --list
```

Dataset maintainer commands:

```powershell
# Verify local upload sources before touching Hugging Face
python setup_backtest_data.py --check-sources

# Upload fresh local files, then verify/download into this repo
python setup_backtest_data.py --sync

# Upload only selected files
python setup_backtest_data.py --upload --files prices_daily.csv prices_hourly.csv nasdaq_screener.csv
```

Upload sources default to:

- `%USERPROFILE%\Desktop\TradingMethods`
- `%USERPROFILE%\Desktop\DownDay`

Override them when needed:

```powershell
python setup_backtest_data.py --sync `
  --tradingmethods-dir D:\TradingMethods `
  --downday-dir D:\DownDay
```

Parquet files are much smaller than CSV because parquet is columnar, typed, and
compressed. CSV repeats text delimiters and column text row-by-row; parquet
stores typed columns in compressed blocks.

## YouTube Intelligence

AutoTrade includes a YouTube scanning pipeline for market-context extraction.
The scanner checks configured trading channels, downloads or reuses transcripts,
runs producer-specific extraction prompts, and stores daily intelligence reports
for the agent to consume.

Main files:

- `config/youtube_channels.yaml` - channel list, scan cadence, model/provider
  settings, output directories, and readiness rules.
- `tools/youtube_daily_scanner.py` - daily scanner and report generator.
- `tools/youtube_transcriber.py` - yt-dlp plus faster-whisper transcription.
- `tools/youtube_weekend_scanner.py` - longer weekend/deep-dive workflow.
- `tools/check_youtube_coverage.py` - coverage diagnostics.
- `tools/export_youtube_cookies.py` - optional browser-cookie export for videos
  that require an authenticated YouTube session.
- `prompts/llm/youtube_*.md` - producer-specific extraction and synthesis
  prompts.
- `autotrade/utils/youtube_readiness.py` - readiness checks used by the
  autonomous workflow.

Run the scanner:

```powershell
# Check for new videos, transcribe, extract, and build the daily report
python tools/youtube_daily_scanner.py

# Process only one configured channel
python tools/youtube_daily_scanner.py --channel trade_brigade

# Show unprocessed videos without running the full workflow
python tools/youtube_daily_scanner.py --list-pending

# Rebuild the daily report from existing extractions
python tools/youtube_daily_scanner.py --report-only
```

Transcribe a single video:

```powershell
python tools/youtube_transcriber.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

If YouTube blocks a video behind bot checks or login state, export local browser
cookies. The generated cookie file is ignored by Git:

```powershell
python tools/export_youtube_cookies.py
```

The default config uses OpenAI/OpenRouter-compatible providers for extraction
and summary synthesis. Set keys in `.env` or your shell:

```powershell
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
```

YouTube artifacts are written under `data/youtube/` and are ignored by Git.

## Runtime Workflow

The autonomous workflow is organized around market phases in Eastern Time:

| Phase | Window | Purpose |
| --- | --- | --- |
| Overnight | 8:00 PM - 4:00 AM | Research, YouTube refresh, broad candidate discovery |
| Early Premarket | 4:00 AM - 7:00 AM | Data checks, plan validation, market context |
| Premarket | 7:00 AM - 9:30 AM | Gap analysis, entry adjustment, watchlist ranking |
| Market Open | 9:30 AM - 10:00 AM | Opening execution and observation |
| Market Hours | 10:00 AM - 3:30 PM | Day Manager position/risk management |
| Power Hour | 3:30 PM - 4:00 PM | Final entry/exit adjustments |
| Post Market | 4:00 PM - 6:30 PM | Daily review and outcome capture |
| PM Workflow | 6:30 PM - 8:00 PM | Tomorrow's plan and overnight handoff |

The workflow is designed to fail closed. Backtests and local diagnostics can run
without brokerage credentials. Live trading requires explicit credentials and
configuration.

Launchers:

```powershell
.\autonomous_agent_main.bat
.\start_autonomous_agent.bat
```

## Project Map

- `autotrade/backtesting/` - backtest engines, protocols, metrics, leakage
  guards, walk-forward validation, and statistical controls.
- `autotrade/feature_engineering/` - deterministic feature transforms shared by
  scanners, signals, and backtests.
- `autotrade/signals/` - alpha families, signal pipeline, screeners, regime
  routing, and runtime signal contracts.
- `autotrade/risk/` - risk policy, sizing, exposure controls, inverse ETF
  handling, and fail-safe logic.
- `autotrade/execution/` - simulation and broker execution contracts.
- `autotrade/core/` - autonomous workflow, Day Manager, PM workflow, premarket
  manager, daily review, and orchestration.
- `autotrade/analysis/` - market regime, news, ranking, post-market, and
  performance analysis.
- `autotrade/data_ingestion/` - path resolution, startup checks, and data
  schemas.
- `autotrade/monitoring/` - alerting, dashboard, liquidity, and halt monitors.
- `tools/` - operational scripts, YouTube scanner, data utilities, and
  diagnostics.
- `config/` - runtime config, YouTube channel config, prompt profiles, and
  service settings.
- `tests/` - regression tests for data, signals, risk, YouTube, replay, and
  workflow behavior.

## Common Commands

```powershell
# Run a narrow smoke test
python -m pytest tests/test_data_ingestion_module.py::TestIngestionPaths -q

# Run YouTube readiness tests
python -m pytest tests/test_youtube_readiness.py tests/test_youtube_speedup.py -q

# Compile changed Python files
python -m py_compile setup_backtest_data.py tools/youtube_daily_scanner.py tools/youtube_transcriber.py

# Inspect data helper options
python setup_backtest_data.py --help
```

## Credentials And Private Files

Do not commit real credentials. Keep secrets in `.env` or your shell
environment.

Private/local files ignored by Git include:

- `.env`
- `config/youtube_cookies.txt`
- `data/youtube/`
- large backtest files in `data/downday/`
- generated logs, reports, and runtime artifacts

Credentials used by optional integrations:

- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` for Alpaca.
- `OPENAI_API_KEY` for OpenAI extraction/synthesis.
- `OPENROUTER_API_KEY` for OpenRouter-compatible models.
- `HF_TOKEN` only for maintainers uploading to Hugging Face. Public downloads
  do not require a token.

## Publishing Notes

The public repo is intended to be useful after a normal clone:

```powershell
git clone https://github.com/natedoggzCD/AutoTrade_Public.git
cd AutoTrade_Public
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_backtest_data.py
```

After that, users can run tests, backtests, and YouTube diagnostics locally.
Brokerage execution and paid model providers remain opt-in.
