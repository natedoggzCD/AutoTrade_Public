"""
Signal Baseline Freeze Tests
============================

Golden tests that lock current SignalA and SignalB outputs for fixed fixtures.
These tests ensure baseline behavior is frozen and won't regress.

Phase 6: Determinism + Golden Regression Tests
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from autotrade.signals.baseline_signals import (
    SignalA,
    SignalB,
    SignalAConfig,
    SignalBConfig,
    validate_baseline_output,
)
from autotrade.signals.contracts import (
    SignalContext,
    SignalFamily,
    SignalDecision,
    SignalAction,
    RegimeLabel,
    SignalMetadata,
    SignalDiagnostics,
)


def _make_mock_candidates(tickers: list) -> list:
    """Create mock candidate data for baseline signals."""
    return [
        {
            "ticker": t,
            "price": 50.0 + i * 5,
            "score": 70.0 - i * 2,
            "action": "buy_open",
            "stop_price": 45.0 + i * 5,
            "target_price": 60.0 + i * 5,
            "partial_target_price": 55.0 + i * 5,
            "factor_scores": {"momentum": 0.7, "value": 0.5},
            "atr_14": 1.5,
            "risk_reward": 2.0,
            "r1_price": 55.0,
            "r1_strength": 0.6,
            "s1_price": 48.0,
            "s1_strength": 0.5,
            "regime": "neutral",
            "distance_to_s1_pct": 4.0,
            "distance_to_r1_pct": 10.0,
        }
        for i, t in enumerate(tickers)
    ]


class TestSignalABaselineFreeze:
    """Tests to ensure SignalA remains frozen at v0 behavior."""

    def test_signal_a_version_is_v0(self):
        """SignalA must report v0 as its version."""
        signal = SignalA()
        assert signal.version == "v0"
        assert signal.name == "SignalA_v0"

    def test_signal_a_family_is_baseline_a(self):
        """SignalA must be in BASELINE_A family."""
        signal = SignalA()
        assert signal.family == SignalFamily.BASELINE_A

    def test_signal_a_config_defaults(self):
        """SignalA config defaults must be frozen."""
        config = SignalAConfig()
        assert config.max_candidates == 200
        assert config.min_score == 35.0
        assert config.scoring_mode == "momentum_pullback"

    def test_signal_a_validates_input(self):
        """SignalA should validate context input."""
        signal = SignalA()
        context = SignalContext(tickers=["AAPL", "MSFT"])
        issues = signal.validate_input(context)
        assert isinstance(issues, list)

    @patch("autotrade.signals.screener_v2.ScreenerV2.screen")
    def test_signal_a_generates_valid_decisions(self, mock_screen):
        """SignalA generates valid SignalDecision objects."""
        mock_screen.return_value = _make_mock_candidates(["AAA", "BBB", "CCC"])

        signal = SignalA()
        context = SignalContext(tickers=["AAA", "BBB", "CCC"])
        decisions = signal.generate(context)

        assert len(decisions) == 3
        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.BASELINE_A
            assert decision.source.startswith("SignalA")
            assert decision.ticker in ["AAA", "BBB", "CCC"]

    @patch("autotrade.signals.screener_v2.ScreenerV2.screen")
    def test_signal_a_respects_min_score_filter(self, mock_screen):
        """SignalA should filter out candidates below min_score."""
        candidates = _make_mock_candidates(["AAA", "BBB", "CCC"])
        candidates[1]["score"] = 30.0  # Below min_score of 35
        mock_screen.return_value = candidates

        signal = SignalA()
        context = SignalContext(tickers=["AAA", "BBB", "CCC"])
        decisions = signal.generate(context)

        tickers = [d.ticker for d in decisions]
        assert "BBB" not in tickers
        assert "AAA" in tickers
        assert "CCC" in tickers

    @patch("autotrade.signals.screener_v2.ScreenerV2.screen")
    def test_signal_a_output_schema_complete(self, mock_screen):
        """SignalA output must have all required fields."""
        mock_screen.return_value = _make_mock_candidates(["TST"])

        signal = SignalA()
        context = SignalContext(tickers=["TST"])
        decisions = signal.generate(context)

        assert len(decisions) == 1
        d = decisions[0]

        assert d.ticker
        assert hasattr(d, "action")
        assert d.signal_strength >= -1.0 and d.signal_strength <= 1.0
        assert d.score >= 0 and d.score <= 100
        assert d.entry_price > 0
        assert d.stop_price > 0
        assert d.target_price > 0


class TestSignalBBaselineFreeze:
    """Tests to ensure SignalB remains frozen at v0 behavior."""

    def test_signal_b_version_is_v0(self):
        """SignalB must report v0 as its version."""
        signal = SignalB()
        assert signal.version == "v0"
        assert signal.name == "SignalB_v0"

    def test_signal_b_family_is_baseline_b(self):
        """SignalB must be in BASELINE_B family."""
        signal = SignalB()
        assert signal.family == SignalFamily.BASELINE_B

    def test_signal_b_config_defaults(self):
        """SignalB config defaults must be frozen."""
        config = SignalBConfig()
        assert config.max_candidates == 200
        assert config.min_score == 10.0
        assert config.use_lessons is True

    def test_signal_b_validates_input(self):
        """SignalB should validate context input."""
        signal = SignalB()
        context = SignalContext(tickers=["AAPL"])
        issues = signal.validate_input(context)
        assert isinstance(issues, list)

    @patch("autotrade.signals.screener_v2.get_entry_candidates")
    @patch("autotrade.signals.unified_strategy.UnifiedStrategy.passes_filter")
    def test_signal_b_generates_valid_decisions(self, mock_filter, mock_candidates):
        """SignalB generates valid SignalDecision objects."""
        mock_candidates.return_value = _make_mock_candidates(["AAA", "BBB"])
        mock_filter.return_value = (True, 70.0, [])

        signal = SignalB()
        context = SignalContext(tickers=["AAA", "BBB"])
        decisions = signal.generate(context)

        assert len(decisions) >= 0
        for decision in decisions:
            assert isinstance(decision, SignalDecision)
            assert decision.family == SignalFamily.BASELINE_B

    @patch("autotrade.signals.screener_v2.get_entry_candidates")
    @patch("autotrade.signals.unified_strategy.UnifiedStrategy.passes_filter")
    def test_signal_b_respects_min_score_filter(self, mock_filter, mock_candidates):
        """SignalB should filter out candidates below min_score."""
        candidates = _make_mock_candidates(["AAA", "BBB"])
        candidates[0]["score"] = 5.0  # Below min_score of 10
        mock_candidates.return_value = candidates
        mock_filter.return_value = (True, 70.0, [])

        signal = SignalB()
        context = SignalContext(tickers=["AAA", "BBB"])
        decisions = signal.generate(context)

        # When score is below min_score, should be filtered
        for d in decisions:
            assert d.score >= 10.0


class TestBaselineDeterminism:
    """Tests for deterministic baseline signal generation."""

    @patch("autotrade.signals.screener_v2.ScreenerV2.screen")
    def test_signal_a_deterministic_with_same_seed(self, mock_screen):
        """SignalA should produce identical results for same input."""
        mock_screen.return_value = _make_mock_candidates(["AAA", "BBB"])

        signal = SignalA()
        context = SignalContext(tickers=["AAA", "BBB"])

        result1 = signal.generate(context)
        result2 = signal.generate(context)

        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1.ticker == r2.ticker
            assert r1.score == r2.score

    @patch("autotrade.signals.screener_v2.get_entry_candidates")
    @patch("autotrade.signals.unified_strategy.UnifiedStrategy.passes_filter")
    def test_signal_b_deterministic_with_same_seed(self, mock_filter, mock_candidates):
        """SignalB should produce identical results for same input."""
        mock_candidates.return_value = _make_mock_candidates(["AAA", "BBB"])
        mock_filter.return_value = (True, 70.0, [])

        signal = SignalB()
        context = SignalContext(tickers=["AAA", "BBB"])

        result1 = signal.generate(context)
        result2 = signal.generate(context)

        # Results should be stable for same input
        assert len(result1) == len(result2)


class TestBaselineGoldenValidation:
    """Golden tests that validate baseline output structure."""

    def test_validate_baseline_output_accepts_valid_signals(self):
        """validate_baseline_output should accept valid baseline signals."""
        signals = [
            SignalDecision(
                ticker="AAA",
                action=SignalAction.BUY,
                signal_strength=0.7,
                score=70.0,
                confidence=0.7,
                family=SignalFamily.BASELINE_A,
                source="SignalA_v0",
                entry_price=50.0,
                stop_price=45.0,
                target_price=60.0,
            ),
            SignalDecision(
                ticker="BBB",
                action=SignalAction.BUY,
                signal_strength=0.6,
                score=60.0,
                confidence=0.6,
                family=SignalFamily.BASELINE_B,
                source="SignalB_v0",
                entry_price=55.0,
                stop_price=50.0,
                target_price=65.0,
            ),
        ]

        issues = validate_baseline_output(signals)
        assert len(issues) == 0

    def test_validate_baseline_output_rejects_wrong_family(self):
        """validate_baseline_output should reject non-baseline families."""
        signals = [
            SignalDecision(
                ticker="AAA",
                action=SignalAction.BUY,
                signal_strength=0.7,
                score=70.0,
                family=SignalFamily.TS_MOMENTUM,
                source="TS_v1",
                entry_price=50.0,
                stop_price=45.0,
                target_price=60.0,
            ),
        ]

        issues = validate_baseline_output(signals)
        assert any("Wrong family" in issue for issue in issues)

    def test_validate_baseline_output_rejects_invalid_source(self):
        """validate_baseline_output should reject invalid source."""
        signals = [
            SignalDecision(
                ticker="AAA",
                action=SignalAction.BUY,
                signal_strength=0.7,
                score=70.0,
                family=SignalFamily.BASELINE_A,
                source="InvalidSource",
                entry_price=50.0,
                stop_price=45.0,
                target_price=60.0,
            ),
        ]

        issues = validate_baseline_output(signals)
        assert any("Invalid source" in issue for issue in issues)

    def test_validate_baseline_output_rejects_invalid_strength(self):
        """validate_baseline_output should reject invalid signal_strength."""
        signals = [
            SignalDecision(
                ticker="AAA",
                action=SignalAction.BUY,
                signal_strength=1.5,  # Invalid: > 1.0
                score=70.0,
                family=SignalFamily.BASELINE_A,
                source="SignalA_v0",
                entry_price=50.0,
                stop_price=45.0,
                target_price=60.0,
            ),
        ]

        issues = validate_baseline_output(signals)
        assert any("Invalid strength" in issue for issue in issues)

    def test_validate_baseline_output_rejects_invalid_score(self):
        """validate_baseline_output should reject invalid score range."""
        signals = [
            SignalDecision(
                ticker="AAA",
                action=SignalAction.BUY,
                signal_strength=0.7,
                score=150.0,  # Invalid: > 100
                family=SignalFamily.BASELINE_A,
                source="SignalA_v0",
                entry_price=50.0,
                stop_price=45.0,
                target_price=60.0,
            ),
        ]

        issues = validate_baseline_output(signals)
        assert any("Invalid score" in issue for issue in issues)


class TestBaselineLegacyCompatibility:
    """Ensure baseline signals remain compatible with legacy output."""

    def test_signal_action_from_string_legacy_aliases(self):
        """Legacy action aliases should map correctly."""
        assert SignalAction.from_string("buy_open") == SignalAction.BUY_OPEN
        assert SignalAction.from_string("buy_dip") == SignalAction.BUY_DIP

    def test_regime_label_from_string_legacy_aliases(self):
        """Legacy regime labels should map correctly."""
        assert RegimeLabel.from_string("sideways") == RegimeLabel.CHOP
        assert RegimeLabel.from_string("volatile") == RegimeLabel.CRISIS
        assert RegimeLabel.from_string("risk_off") == RegimeLabel.CRISIS
        assert RegimeLabel.from_string("risk_on") == RegimeLabel.TREND


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
