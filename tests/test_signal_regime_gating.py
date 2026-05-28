import pandas as pd

from autotrade.signals.contracts import (
    RegimeLabel,
    SignalAction,
    SignalDecision,
    SignalFamily,
)
from autotrade.signals.pipeline import SignalGenerationPipeline
from autotrade.signals.regime_router import RegimeRouter


def _mk_signal(ticker: str, family: SignalFamily) -> SignalDecision:
    return SignalDecision(
        ticker=ticker,
        action=SignalAction.BUY,
        signal_strength=0.6,
        score=70.0,
        confidence=0.7,
        family=family,
        source=f"{family.value}_test",
    )


def test_regime_router_uses_stage2_feature_label():
    router = RegimeRouter()
    feature_df = pd.DataFrame(
        [
            {"regime_label": "chop", "trend_strength": 0.01},
            {"regime_label": "trend", "trend_strength": 0.2},
        ]
    )
    result = router.get_current_regime(feature_data=feature_df)
    assert result.regime == RegimeLabel.TREND
    assert result.method == "feature_label"


def test_regime_filter_keeps_baseline_when_alpha_filtered():
    router = RegimeRouter()
    signals = [
        _mk_signal("AAA", SignalFamily.BASELINE_A),
        _mk_signal("BBB", SignalFamily.TS_MOMENTUM),
    ]
    accepted, filtered = router.filter_signals_by_regime(signals, regime=RegimeLabel.CRISIS)

    accepted_families = {s.family for s in accepted}
    filtered_families = {s.family for s in filtered}
    assert SignalFamily.BASELINE_A in accepted_families
    assert SignalFamily.TS_MOMENTUM in filtered_families


def test_pipeline_regime_routing_failure_preserves_baselines():
    pipeline = SignalGenerationPipeline()
    pipeline._cfg.enable_regime_router = True

    class _BrokenRouter:
        def get_current_regime(self, *args, **kwargs):
            return type("R", (), {"regime": RegimeLabel.TREND, "confidence": 0.5})()

        def filter_signals_by_regime(self, *args, **kwargs):
            raise RuntimeError("router failure")

    pipeline._regime_router = _BrokenRouter()
    baseline = _mk_signal("AAA", SignalFamily.BASELINE_A)
    alpha = _mk_signal("BBB", SignalFamily.TS_MOMENTUM)
    output = pipeline.stage_regime_routing([baseline, alpha], pipeline.build_context())
    families = {s.family for s in output}
    assert SignalFamily.BASELINE_A in families
    assert SignalFamily.TS_MOMENTUM in families
