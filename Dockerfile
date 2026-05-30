FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AUTOTRADE_ROOT=/workspace \
    DOWNDAY_ROOT=/workspace/data/downday \
    ALPACA_PAPER=true \
    ALPACA_BASE_URL=https://paper-api.alpaca.markets

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p data/downday data/youtube logs reports plans \
    && chmod +x /workspace/docker/entrypoint.sh

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["python", "-m", "pytest", "tests/test_data_ingestion_module.py::TestIngestionPaths", "-q"]
