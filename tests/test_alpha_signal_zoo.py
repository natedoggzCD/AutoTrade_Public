"""
Alpha Signal Zoo Tests
======================

Family-level tests for at least one canonical setup per alpha family:
- ts_momentum: Time-series momentum
- xs_momentum: Cross-sectional momentum
- mean_reversion: RSI and down-day exhaustion
- breakout: Squeeze expansion and Donchian
- pullback: Pullback continuation
- pairs: Stat-arb/pairs

Phase 6: Determinism + Golden Regression Tests
"""

import numpy as np
import pandas as pd
import pytest

from autotrade.signals.contracts import (
    SignalContext,
    SignalDecision,
    SignalFamily,
    RegimeLabel,
)
from autotrade.signals.interfaces import get_signal_registry, reset_signal_registry
from autotrade.signals.alpha import (
    TSMomentum12_1,
    TSMomentum6_1,
    MATrendSignal,
    XSMomentumSignal,
    RSISignal,
    DownDayExhaustionSignal,
    SqueezeExpansionSignal,
    DonchianBreakoutSignal,
    PullbackContinuationSignal,
    PairsSignal,
    register_all_alpha_signals,
)


def _make_price_data(
    tickers: list,
    n_bars: int = 300,
    seed: int = 42,
    trend: float = 0.0,
) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    rows = []

    for ticker_idx, ticker in enumerate(tickers):
        base = 50.0 + ticker_idx * 10
        prices = base + rng.normal(trend, 1.0, size=n_bars).cumsum()
        prices = np.maximum(prices, 1.0)

        high = prices * (1 + rng.uniform(0.001, 0.02, size=n_bars))
        low = prices * (1 - rng.uniform(0.001, 0.02, size=n_bars))
        close = prices
        open_price = prices + rng.normal(0, 0.1, size=n_bars)
        volume = rng.integers(500_000, 5_000_000, size=n_bars)

        for i in range(n_bars):
            rows.append(
                {
                    "ticker": ticker,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "open": float(open_price[i]),
                    "high": float(high[i]),
                    "low": float(low[i]),
                    "close": float(close[i]),
                    "volume": int(volume[i]),
                }
            )

    return pd.DataFrame(rows)


class TestTSMomentumFamily:
    """Tests for Time-Series Momentum alpha family."""

    def test_ts_momentum_12_1_registers(self):
        """TSMomentum12_1 should register successfully."""
        reset_signal_registry()
        from autotrade.signals.alpha.ts_momentum import register_ts_momentum_signals

        register_ts_momentum_signals()
        registry = get_signal_registry()
        models = registry.get_all_models()

        names = [m.name for m in models]
        assert any("TSMomentum12_1" in name for name in names)

    def test_ts_momentum_12_1_family_is_ts_momentum(self):
        """TSMomentum12_1 should be in TS_MOMENTUM family."""
        signal = TSMomentum12_1()
        assert signal.family == SignalFamily.TS_MOMENTUM

    def test_ts_momentum_12_1_generates_valid_signals(self):
        """TSMomentum12_1 should generate valid SignalDecision objects."""
        signal = TSMomentum12_1()
        tickers = ["AAA", "BBB", "CCC"]
        price_data = _make_price_data(tickers, n_bars=300, trend=0.05)

        context = SignalContext(
            tickers=tickers,
            price_data=price_data,
            regime=RegimeLabel.TREND,
        )

        decisions = signal.generate(context)

        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.TS_MOMENTUM
            assert decision.signal_strength >= -1.0 and decision.signal_strength <= 1.0
            assert decision.score >= 0 and decision.score <= 100

    def test_ts_momentum_6_1_generates_valid_signals(self):
        """TSMomentum6_1 should generate valid signals."""
        signal = TSMomentum6_1()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for d in decisions:
            assert d.family == SignalFamily.TS_MOMENTUM

    def test_ts_momentum_deterministic(self):
        """TS Momentum signals should be deterministic."""
        signal = TSMomentum12_1()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)

        result1 = signal.generate(context)
        result2 = signal.generate(context)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1.ticker == r2.ticker
            assert r1.score == r2.score


class TestXSMomentumFamily:
    """Tests for Cross-Sectional Momentum alpha family."""

    def test_xs_momentum_registers(self):
        """XSMomentumSignal should register successfully."""
        reset_signal_registry()
        from autotrade.signals.alpha.xs_momentum import register_xs_momentum_signals

        register_xs_momentum_signals()
        registry = get_signal_registry()
        models = registry.get_all_models()

        names = [m.name for m in models]
        assert any("XSMomentum" in name for name in names)

    def test_xs_momentum_family_is_xs_momentum(self):
        """XSMomentumSignal should be in XS_MOMENTUM family."""
        signal = XSMomentumSignal()
        assert signal.family == SignalFamily.XS_MOMENTUM

    def test_xs_momentum_generates_valid_signals(self):
        """XSMomentumSignal should generate valid signals."""
        signal = XSMomentumSignal()
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.XS_MOMENTUM

    def test_xs_momentum_rank_column_survives_ranking(self):
        signal = XSMomentumSignal()
        tickers = ["AAA", "BBB", "CCC"]
        price_data = _make_price_data(tickers, n_bars=90)

        ranked = signal._rank_by_momentum(price_data)

        assert "rank" in ranked.columns
        assert ranked["rank"].between(0.0, 1.0).all()


class TestMeanReversionFamily:
    """Tests for Mean Reversion alpha family."""

    def test_rsi_signal_registers(self):
        """RSISignal should register successfully."""
        reset_signal_registry()
        from autotrade.signals.alpha.mean_reversion import (
            register_mean_reversion_signals,
        )

        register_mean_reversion_signals()
        registry = get_signal_registry()
        models = registry.get_all_models()

        names = [m.name for m in models]
        assert any("RSI" in name for name in names)

    def test_rsi_signal_family_is_mean_reversion(self):
        """RSISignal should be in MEAN_REVERSION family."""
        signal = RSISignal()
        assert signal.family == SignalFamily.MEAN_REVERSION

    def test_rsi_signal_generates_valid_signals(self):
        """RSISignal should generate valid signals."""
        signal = RSISignal()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=100)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.MEAN_REVERSION

    def test_down_day_exhaustion_signal(self):
        """DownDayExhaustionSignal should work."""
        signal = DownDayExhaustionSignal()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=100)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for d in decisions:
            assert d.family == SignalFamily.MEAN_REVERSION


class TestBreakoutFamily:
    """Tests for Breakout alpha family."""

    def test_squeeze_expansion_registers(self):
        """SqueezeExpansionSignal should register successfully."""
        reset_signal_registry()
        from autotrade.signals.alpha.breakout import register_breakout_signals

        register_breakout_signals()
        registry = get_signal_registry()
        models = registry.get_all_models()

        names = [m.name for m in models]
        assert any("Squeeze" in name for name in names)

    def test_squeeze_expansion_family_is_breakout(self):
        """SqueezeExpansionSignal should be in BREAKOUT family."""
        signal = SqueezeExpansionSignal()
        assert signal.family == SignalFamily.BREAKOUT

    def test_squeeze_expansion_generates_valid_signals(self):
        """SqueezeExpansionSignal should generate valid signals."""
        signal = SqueezeExpansionSignal()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=200)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.BREAKOUT

    def test_donchian_breakout_signal(self):
        """DonchianBreakoutSignal should work."""
        signal = DonchianBreakoutSignal()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=200)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for d in decisions:
            assert d.family == SignalFamily.BREAKOUT


class TestPullbackFamily:
    """Tests for Pullback Continuation alpha family."""

    def test_pullback_continuation_registers(self):
        """PullbackContinuationSignal should register successfully."""
        reset_signal_registry()
        from autotrade.signals.alpha.pullback import register_pullback_signals

        register_pullback_signals()
        registry = get_signal_registry()
        models = registry.get_all_models()

        names = [m.name for m in models]
        assert any("Pullback" in name for name in names)

    def test_pullback_continuation_family_is_pullback(self):
        """PullbackContinuationSignal should be in PULLBACK family."""
        signal = PullbackContinuationSignal()
        assert signal.family == SignalFamily.PULLBACK

    def test_pullback_continuation_generates_valid_signals(self):
        """PullbackContinuationSignal should generate valid signals."""
        signal = PullbackContinuationSignal()
        tickers = ["AAA", "BBB", "CCC"]
        price_data = _make_price_data(tickers, n_bars=300, trend=0.02)

        context = SignalContext(
            tickers=tickers, price_data=price_data, regime=RegimeLabel.TREND
        )
        decisions = signal.generate(context)

        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.PULLBACK


class TestPairsFamily:
    """Tests for Pairs/Stat-Arb alpha family."""

    def test_pairs_registers(self):
        """PairsSignal should register successfully."""
        reset_signal_registry()
        from autotrade.signals.alpha.pairs import register_pairs_signals

        register_pairs_signals()
        registry = get_signal_registry()
        models = registry.get_all_models()

        names = [m.name for m in models]
        assert any("Pairs" in name for name in names)

    def test_pairs_family_is_pairs(self):
        """PairsSignal should be in PAIRS family."""
        signal = PairsSignal()
        assert signal.family == SignalFamily.PAIRS

    def test_pairs_generates_valid_signals(self):
        """PairsSignal should generate valid signals."""
        signal = PairsSignal()
        tickers = ["AAA", "BBB", "CCC"]
        price_data = _make_price_data(tickers, n_bars=200)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        # Pairs may return empty if no pairs found
        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.PAIRS


class TestAlphaSignalMetadata:
    """Tests for alpha signal metadata requirements."""

    def test_alpha_signals_have_metadata(self):
        """Alpha signals should include metadata."""
        signal = TSMomentum12_1()
        tickers = ["AAA"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for d in decisions:
            assert hasattr(d, "metadata")
            assert d.metadata is not None
            assert d.metadata.expected_holding_period_bars > 0

    def test_alpha_signals_have_diagnostics(self):
        """Alpha signals should include diagnostics."""
        signal = TSMomentum12_1()
        tickers = ["AAA"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for d in decisions:
            assert hasattr(d, "diagnostics")
            assert d.diagnostics is not None


class TestAlphaSignalDeterminism:
    """Tests for deterministic alpha signal generation."""

    @pytest.mark.parametrize(
        "signal_class",
        [
            TSMomentum12_1,
            TSMomentum6_1,
            MATrendSignal,
            RSISignal,
            DownDayExhaustionSignal,
            SqueezeExpansionSignal,
            DonchianBreakoutSignal,
            PullbackContinuationSignal,
        ],
    )
    def test_signal_deterministic(self, signal_class):
        """Each alpha signal should produce deterministic output."""
        signal = signal_class()
        tickers = ["AAA", "BBB", "CCC"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)

        result1 = signal.generate(context)
        result2 = signal.generate(context)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1.ticker == r2.ticker
            assert r1.score == r2.score


class TestAlphaSignalSchemaCompliance:
    """Tests for schema compliance across alpha families."""

    @pytest.mark.parametrize(
        "signal_class",
        [
            TSMomentum12_1,
            TSMomentum6_1,
            MATrendSignal,
            XSMomentumSignal,
            RSISignal,
            DownDayExhaustionSignal,
            SqueezeExpansionSignal,
            DonchianBreakoutSignal,
            PullbackContinuationSignal,
            PairsSignal,
        ],
    )
    def test_signal_output_schema_compliant(self, signal_class):
        """Each alpha signal should output schema-compliant decisions."""
        signal = signal_class()
        tickers = ["AAA", "BBB"]
        price_data = _make_price_data(tickers, n_bars=300)

        context = SignalContext(tickers=tickers, price_data=price_data)
        decisions = signal.generate(context)

        for d in decisions:
            assert d.ticker
            assert hasattr(d, "action")
            assert d.signal_strength >= -1.0 and d.signal_strength <= 1.0
            assert d.score >= 0 and d.score <= 100
            assert d.entry_price > 0
            assert d.stop_price > 0
            assert d.target_price > 0


class TestAlphaRegistryIntegration:
    """Tests for alpha signal registration and registry."""

    def test_register_all_alpha_signals(self):
        """register_all_alpha_signals should register all families."""
        reset_signal_registry()
        register_all_alpha_signals()

        registry = get_signal_registry()
        models = registry.get_all_models()

        families = {m.family for m in models}
        expected_families = {
            SignalFamily.TS_MOMENTUM,
            SignalFamily.XS_MOMENTUM,
            SignalFamily.MEAN_REVERSION,
            SignalFamily.BREAKOUT,
            SignalFamily.PULLBACK,
            SignalFamily.PAIRS,
        }

        assert expected_families.issubset(families)

    def test_registry_keeps_baseline_disabled_when_not_requested(self):
        """Registry should not include baselines unless explicitly enabled."""
        reset_signal_registry()
        register_all_alpha_signals()

        registry = get_signal_registry()
        models = registry.get_all_models()

        baseline_families = {
            SignalFamily.BASELINE_A,
            SignalFamily.BASELINE_B,
        }

        # After alpha-only registration, baselines should not be present
        families = {m.family for m in models}
        # This is expected to pass since we only register alpha
        for bf in baseline_families:
            # Either not present OR present but disabled
            if bf in families:
                assert not registry.is_enabled(bf.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
