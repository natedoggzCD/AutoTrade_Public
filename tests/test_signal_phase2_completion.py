import numpy as np
import pandas as pd

from autotrade.signals.contracts import SignalContext, SignalFamily
from autotrade.signals.registry import register_signal_zoo


def _build_price_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tickers = ["XLF", "XLK", "XLE", "XOP", "XLP", "XLY", "SMH", "SOXX"]
    rows = []

    for idx, ticker in enumerate(tickers):
        base = 50.0 + idx * 10.0
        trend = 0.12 + idx * 0.01
        noise = rng.normal(0.0, 0.8, size=300).cumsum()
        close = base + np.linspace(0.0, trend * 300, 300) + noise
        close = np.maximum(close, 1.0)

        high = close * (1.0 + 0.01 + rng.uniform(0.0, 0.01, size=300))
        low = close * (1.0 - 0.01 - rng.uniform(0.0, 0.01, size=300))
        volume = rng.integers(800_000, 3_000_000, size=300)

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


def test_phase2_signal_zoo_has_required_coverage():
    registry = register_signal_zoo(include_baselines=False)
    models = registry.get_all_models()

    assert len(models) >= 10

    families = {m.family for m in models}
    assert SignalFamily.TS_MOMENTUM in families
    assert SignalFamily.XS_MOMENTUM in families
    assert SignalFamily.MEAN_REVERSION in families
    assert SignalFamily.BREAKOUT in families
    assert SignalFamily.PULLBACK in families
    assert SignalFamily.PAIRS in families


def test_phase2_registry_respects_family_enablement():
    registry = register_signal_zoo(
        families_enabled=["ts_momentum", "pullback"],
        include_baselines=False,
    )
    enabled = registry.get_enabled_models()

    assert enabled
    alpha_families = {"ts_momentum", "xs_momentum", "mean_reversion", "breakout", "pullback"}
    enabled_alpha = [m for m in enabled if m.family.value in alpha_families]
    assert enabled_alpha
    assert all(m.family.value in {"ts_momentum", "pullback"} for m in enabled_alpha)


def test_phase2_generated_signals_respect_contracts():
    registry = register_signal_zoo(
        families_enabled=["ts_momentum", "xs_momentum", "mean_reversion", "pullback"],
        include_baselines=False,
    )
    context = SignalContext(
        tickers=["XLF", "XLK", "XLE", "XOP", "XLP", "XLY", "SMH", "SOXX"],
        price_data=_build_price_data(),
    )

    produced = []
    for model in registry.get_enabled_models():
        produced.extend(model.generate(context))

    assert produced, "Expected at least one signal from enabled phase 2 models"

    for signal in produced:
        assert -1.0 <= signal.signal_strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.metadata.expected_holding_period_bars > 0
        assert signal.metadata.cost_sensitivity >= 0.0
        assert signal.diagnostics.generation_time_ms >= 0.0
