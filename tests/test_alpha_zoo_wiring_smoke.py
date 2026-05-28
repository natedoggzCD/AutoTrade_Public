"""
End-to-end smoke test for modular alpha-zoo wiring.

Validates that the core stage modules can be initialized and executed together
with mocked data/bootstrap inputs.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd

from autotrade.data_ingestion.schemas import (
    DataFreshnessLevel,
    DataFreshnessStatus,
    IngestionHealthReport,
)
from autotrade.feature_engineering.pipeline import FeaturePipeline
from autotrade.feature_engineering.schemas import FeatureRequest
from autotrade.signals.pipeline import SignalGenerationPipeline
from autotrade.signals.regime_router import RegimeRouter
from autotrade.signals.registry import register_signal_zoo_from_config


def _feature_input() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for symbol_idx, symbol in enumerate(["AAA", "BBB"]):
        base = 50.0 + symbol_idx * 30.0
        noise = rng.normal(0.0, 0.6, size=280).cumsum()
        closes = np.maximum(1.0, base + np.linspace(0, 22, 280) + noise)
        highs = closes * (1.01 + rng.uniform(0, 0.01, size=280))
        lows = closes * (0.99 - rng.uniform(0, 0.01, size=280))
        opens = closes * (1.0 + rng.uniform(-0.004, 0.004, size=280))
        vols = rng.integers(250_000, 2_500_000, size=280)
        for i in range(280):
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                    "volume": int(vols[i]),
                }
            )
    return pd.DataFrame(rows)


def _signal_price_input() -> pd.DataFrame:
    rng = np.random.default_rng(456)
    rows = []
    for ticker_idx, ticker in enumerate(["AAA", "BBB", "CCC"]):
        base = 35.0 + ticker_idx * 15.0
        noise = rng.normal(0.0, 0.4, size=300).cumsum()
        close = np.maximum(1.0, base + np.linspace(0, 12, 300) + noise)
        high = close * (1.01 + rng.uniform(0, 0.01, size=300))
        low = close * (0.99 - rng.uniform(0, 0.01, size=300))
        volume = rng.integers(300_000, 2_000_000, size=300)
        for i in range(300):
            rows.append(
                {
                    "ticker": ticker,
                    "close": float(close[i]),
                    "high": float(high[i]),
                    "low": float(low[i]),
                    "volume": int(volume[i]),
                }
            )
    return pd.DataFrame(rows)


def test_full_wired_pipeline_smoke(monkeypatch):
    # Mock startup data-ingestion bootstrap readiness.
    import autotrade.data_ingestion.bootstrap as ingestion_bootstrap

    freshness = DataFreshnessStatus(
        level=DataFreshnessLevel.FRESH,
        latest_date=date.today(),
        expected_date=date.today(),
        staleness_days=0,
    )
    mocked_report = IngestionHealthReport(
        is_healthy=True,
        can_trade=True,
        primary_source_ready=True,
        freshness=freshness,
    )
    monkeypatch.setattr(
        ingestion_bootstrap,
        "ensure_core_market_data_ready",
        lambda **kwargs: mocked_report,
    )

    report = ingestion_bootstrap.ensure_core_market_data_ready(fail_fast=False)
    assert report.can_trade is True
    assert report.freshness is not None

    # Feature engineering pipeline execution.
    feature_data = _feature_input()
    feature_pipeline = FeaturePipeline(
        config={
            "alpha_families_enabled": [
                "trend",
                "ts_momentum",
                "mean_reversion",
                "breakout",
                "pairs",
            ],
            "cache_ttl_minutes": 60,
        }
    )
    req = FeatureRequest(
        symbols=["AAA", "BBB"],
        start_date=str(feature_data["date"].min().date()),
        end_date=str(feature_data["date"].max().date()),
        families=feature_pipeline.families_enabled,
        as_of_date=str(feature_data["date"].max().date()),
    )
    feature_frame = feature_pipeline.execute(request=req, data=feature_data)
    assert feature_frame.df is not None
    assert not feature_frame.df.empty

    # Signal registry + signal generation pipeline execution.
    registry = register_signal_zoo_from_config()
    assert registry.get_enabled_models()

    signal_pipeline = SignalGenerationPipeline()
    legacy_candidates = [
        {"ticker": "AAA", "price": 62.0, "score": 66.0, "action": "buy_open"},
        {"ticker": "BBB", "price": 88.0, "score": 61.0, "action": "watch"},
        {"ticker": "CCC", "price": 45.0, "score": 59.0, "action": "buy_open"},
    ]
    context = signal_pipeline.build_context(
        tickers=["AAA", "BBB", "CCC"],
        price_data=_signal_price_input(),
        feature_data=feature_frame.df,
    )
    result = signal_pipeline.run(
        context=context,
        legacy_candidates=legacy_candidates,
        include_baselines=False,
    )
    assert result.legacy_candidates

    # Regime router should execute with provided feature/price context.
    router = RegimeRouter()
    regime = router.get_current_regime(
        price_data=_signal_price_input(),
        feature_data=feature_frame.df,
    )
    assert regime.regime is not None
    assert regime.timestamp <= datetime.now()
