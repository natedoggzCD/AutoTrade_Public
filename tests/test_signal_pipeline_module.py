import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from autotrade.signals.contracts import SignalAction, SignalDecision, SignalFamily
from autotrade.signals.pipeline import SignalGenerationPipeline


def _build_price_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for ticker_idx, ticker in enumerate(["XLF", "XLK", "XLE", "XOP"]):
        base = 40 + ticker_idx * 15
        drift = 0.1 + ticker_idx * 0.03
        noise = rng.normal(0.0, 0.5, size=280).cumsum()
        close = np.maximum(1.0, base + np.linspace(0, drift * 280, 280) + noise)
        high = close * (1.01 + rng.uniform(0, 0.01, size=280))
        low = close * (0.99 - rng.uniform(0, 0.01, size=280))
        volume = rng.integers(500_000, 2_000_000, size=280)
        for i in range(280):
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


def test_signal_pipeline_outputs_legacy_compatible_candidates():
    pipeline = SignalGenerationPipeline()
    context = pipeline.build_context(
        tickers=["XLF", "XLK", "XLE", "XOP"],
        price_data=_build_price_data(),
    )
    legacy_candidates = [
        {"ticker": "XLF", "price": 55.0, "score": 70.0, "action": "buy_open"},
        {"ticker": "XLK", "price": 75.0, "score": 66.0, "action": "watch"},
    ]
    result = pipeline.run(
        context=context,
        legacy_candidates=legacy_candidates,
        include_baselines=False,
    )

    assert result.legacy_candidates
    first = result.legacy_candidates[0]
    assert "ticker" in first
    assert "action" in first
    assert "score" in first
    assert "entry_price" in first
    assert "stop_price" in first
    assert "target_price" in first
    assert "validation" in first


def test_signal_pipeline_is_deterministic_for_same_input():
    pipeline = SignalGenerationPipeline()
    context = pipeline.build_context(
        tickers=["XLF", "XLK", "XLE", "XOP"],
        price_data=_build_price_data(),
    )
    legacy_candidates = [
        {"ticker": "XLF", "price": 55.0, "score": 70.0, "action": "buy_open"},
        {"ticker": "XLK", "price": 75.0, "score": 66.0, "action": "watch"},
    ]
    r1 = pipeline.run(
        context=context, legacy_candidates=legacy_candidates, include_baselines=False
    )
    r2 = pipeline.run(
        context=context, legacy_candidates=legacy_candidates, include_baselines=False
    )

    s1 = [(c["ticker"], round(float(c["score"]), 6)) for c in r1.legacy_candidates]
    s2 = [(c["ticker"], round(float(c["score"]), 6)) for c in r2.legacy_candidates]
    assert s1 == s2


def test_signal_pipeline_captures_arrival_price():
    pipeline = SignalGenerationPipeline()
    # Create multi-index price data for realism
    rng = np.random.default_rng(42)
    tickers = ["AAPL", "TSLA"]
    dates = pd.date_range("2026-03-01 09:30:00", periods=5, freq="1min")
    index = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "timestamp"])

    data = {
        "open": rng.uniform(100, 200, size=len(index)),
        "high": rng.uniform(101, 201, size=len(index)),
        "low": rng.uniform(99, 199, size=len(index)),
        "close": rng.uniform(100, 200, size=len(index)),
        "volume": rng.integers(1000, 5000, size=len(index)),
    }
    df = pd.DataFrame(data, index=index)

    context = pipeline.build_context(
        tickers=tickers,
        price_data=df,
    )

    legacy_candidates = [
        {"ticker": "AAPL", "price": 150.0, "score": 70.0, "action": "buy_open"},
        {"ticker": "TSLA", "price": 250.0, "score": 60.0, "action": "buy_open"},
    ]

    result = pipeline.run(
        context=context,
        legacy_candidates=legacy_candidates,
        include_baselines=False,
    )

    assert len(result.batch.signals) == 2
    for s in result.batch.signals:
        expected_close = float(df.xs(s.ticker, level=0)["close"].iloc[-1])
        assert s.arrival_price == expected_close

    for c in result.legacy_candidates:
        expected_close = float(df.xs(c["ticker"], level=0)["close"].iloc[-1])
        assert c["arrival_price"] == expected_close


class _FakeAlphaModel:
    def __init__(self, name: str, ticker: str, score: float):
        self._name = name
        self._ticker = ticker
        self._score = score

    @property
    def name(self):
        return self._name

    @property
    def family(self):
        return SignalFamily.TS_MOMENTUM

    @property
    def version(self):
        return "test"

    def validate_input(self, context):
        return []

    def generate(self, context):
        return [
            SignalDecision(
                ticker=self._ticker,
                action=SignalAction.BUY_OPEN,
                signal_strength=(self._score - 50.0) / 50.0,
                score=self._score,
                family=SignalFamily.TS_MOMENTUM,
                source=self._name,
            )
        ]


class _FakeRegistry:
    def __init__(self, models):
        self._models = models

    def get_enabled_models(self):
        return list(self._models)


def _pipeline_with_fake_models(max_workers: int) -> SignalGenerationPipeline:
    pipeline = SignalGenerationPipeline.__new__(SignalGenerationPipeline)
    pipeline._cfg = SimpleNamespace(alpha_generation_max_workers=max_workers)
    pipeline._logger = logging.getLogger("test_signal_pipeline_parallel")
    pipeline._registry = _FakeRegistry(
        [
            _FakeAlphaModel("alpha_a", "AAA", 61.0),
            _FakeAlphaModel("alpha_b", "BBB", 72.0),
            _FakeAlphaModel("alpha_c", "CCC", 83.0),
        ]
    )
    pipeline._alpha_min_win_rate = 0.0
    pipeline._alpha_family_win_rate = {}
    return pipeline


def test_parallel_alpha_generation_matches_serial_order_and_output():
    context = _pipeline_with_fake_models(1).build_context(
        tickers=["AAA", "BBB", "CCC"],
        price_data=pd.DataFrame(),
    )
    serial = _pipeline_with_fake_models(1).stage_alpha_generation(context)
    parallel = _pipeline_with_fake_models(4).stage_alpha_generation(context)

    assert [(s.ticker, s.source, s.score) for s in parallel] == [
        (s.ticker, s.source, s.score) for s in serial
    ]
