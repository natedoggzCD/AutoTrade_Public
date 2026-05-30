# Docker Test Environment

Docker lets you run AutoTrade tests without changing your local Python, conda,
CUDA, or package installation.

## Build

```powershell
docker compose build autotrade-test
```

## Run The Default Smoke Test

```powershell
docker compose run --rm autotrade-test
```

The default command runs:

```bash
python -m pytest tests/test_data_ingestion_module.py::TestIngestionPaths -q
```

## Open A Container Shell

```powershell
docker compose run --rm --profile shell autotrade-shell
```

From inside the shell:

```bash
python setup_backtest_data.py --help
python setup_backtest_data.py --list
python -m pytest tests/test_youtube_readiness.py -q
```

## Data Files

The compose file mounts your local `data/downday` folder into the container at:

```text
/workspace/data/downday
```

To download the public Hugging Face dataset into the mounted folder:

```powershell
docker compose run --rm autotrade-test python setup_backtest_data.py
```

Large files remain ignored by Git. If you do not mount `data/downday`, downloads
stay inside that container run and disappear when the container is removed.

## Credentials

Do not bake real credentials into the image. Pass optional credentials only at
runtime:

```powershell
docker compose run --rm `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e OPENROUTER_API_KEY=$env:OPENROUTER_API_KEY `
  autotrade-test python tools/youtube_daily_scanner.py --list-pending
```

Hugging Face downloads do not need a token for the public dataset. Maintainer
uploads can pass `HF_TOKEN` at runtime:

```powershell
docker compose run --rm -e HF_TOKEN=$env:HF_TOKEN autotrade-test python setup_backtest_data.py --check-sources
```

## Notes

- The image is CPU-oriented. GPU transcription for YouTube can be configured
  later with an NVIDIA runtime image if needed.
- The Docker build installs dependencies inside the image only. It does not
  modify your host Python or conda environments.
- `.dockerignore` excludes `.env`, cookies, logs, generated reports, and large
  market-data files from the build context.
