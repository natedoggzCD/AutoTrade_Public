"""
Signal Schema Contract Tests
=============================

Alpha-zoo schema/invariant tests:
- No missing required fields
- Output range respected
- Action normalization valid

Phase 6: Determinism + Golden Regression Tests
"""

import pytest
from datetime import datetime

from autotrade.signals.contracts import (
    SignalDecision,
    SignalBatch,
    SignalContext,
    SignalAction,
    SignalFamily,
    RegimeLabel,
    SignalMetadata,
    SignalDiagnostics,
    validate_signal_dict,
    REQUIRED_SIGNAL_FIELDS,
    OPTIONAL_SIGNAL_FIELDS,
)


class TestSignalSchemaRequiredFields:
    """Tests for required signal fields."""

    def test_required_fields_defined(self):
        """Required fields must be defined."""
        assert "ticker" in REQUIRED_SIGNAL_FIELDS
        assert "action" in REQUIRED_SIGNAL_FIELDS
        assert "signal_strength" in REQUIRED_SIGNAL_FIELDS
        assert "score" in REQUIRED_SIGNAL_FIELDS
        assert "entry_price" in REQUIRED_SIGNAL_FIELDS
        assert "stop_price" in REQUIRED_SIGNAL_FIELDS
        assert "target_price" in REQUIRED_SIGNAL_FIELDS

    def test_signal_decision_has_required_fields(self):
        """SignalDecision must have all required fields."""
        decision = SignalDecision(
            ticker="AAPL",
            action=SignalAction.BUY,
            signal_strength=0.75,
            score=75.0,
            entry_price=150.0,
            stop_price=140.0,
            target_price=170.0,
        )

        d = decision.to_dict()
        for field in REQUIRED_SIGNAL_FIELDS:
            assert field in d, f"Missing required field: {field}"

    def test_validate_signal_dict_accepts_valid_signal(self):
        """validate_signal_dict should accept valid signal."""
        valid_signal = {
            "ticker": "AAPL",
            "action": "buy",
            "signal_strength": 0.8,
            "score": 80.0,
            "entry_price": 150.0,
            "stop_price": 140.0,
            "target_price": 170.0,
        }

        issues = validate_signal_dict(valid_signal)
        assert len(issues) == 0

    def test_validate_signal_dict_rejects_missing_required(self):
        """validate_signal_dict should reject missing required fields."""
        invalid_signal = {
            "ticker": "AAPL",
            "action": "buy",
            # Missing signal_strength, score, entry_price, etc.
        }

        issues = validate_signal_dict(invalid_signal)
        assert len(issues) > 0
        assert any("Missing required field" in issue for issue in issues)

    def test_validate_signal_dict_rejects_none_value(self):
        """validate_signal_dict should reject None values for required fields."""
        invalid_signal = {
            "ticker": "AAPL",
            "action": "buy",
            "signal_strength": 0.8,
            "score": 80.0,
            "entry_price": None,  # None should fail
            "stop_price": 140.0,
            "target_price": 170.0,
        }

        issues = validate_signal_dict(invalid_signal)
        assert any("Missing required field" in issue for issue in issues)


class TestSignalStrengthRange:
    """Tests for signal_strength output range validation."""

    def test_signal_strength_must_be_in_range(self):
        """signal_strength must be in [-1, 1]."""
        # Valid ranges
        for strength in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            signal = {
                "ticker": "AAPL",
                "action": "buy",
                "signal_strength": strength,
                "score": 50.0,
                "entry_price": 100.0,
                "stop_price": 90.0,
                "target_price": 120.0,
            }
            issues = validate_signal_dict(signal)
            assert len(issues) == 0, f"Expected {strength} to be valid"

    def test_signal_strength_rejects_above_1(self):
        """signal_strength > 1.0 should be rejected."""
        signal = {
            "ticker": "AAPL",
            "action": "buy",
            "signal_strength": 1.5,
            "score": 50.0,
            "entry_price": 100.0,
            "stop_price": 90.0,
            "target_price": 120.0,
        }
        issues = validate_signal_dict(signal)
        assert any("signal_strength must be in [-1, 1]" in issue for issue in issues)

    def test_signal_strength_rejects_below_minus_1(self):
        """signal_strength < -1.0 should be rejected."""
        signal = {
            "ticker": "AAPL",
            "action": "buy",
            "signal_strength": -1.5,
            "score": 50.0,
            "entry_price": 100.0,
            "stop_price": 90.0,
            "target_price": 120.0,
        }
        issues = validate_signal_dict(signal)
        assert any("signal_strength must be in [-1, 1]" in issue for issue in issues)

    def test_signal_strength_type_validation(self):
        """signal_strength must be numeric."""
        for invalid in ["0.5", "high", None]:
            signal = {
                "ticker": "AAPL",
                "action": "buy",
                "signal_strength": invalid,
                "score": 50.0,
                "entry_price": 100.0,
                "stop_price": 90.0,
                "target_price": 120.0,
            }
            issues = validate_signal_dict(signal)
            assert len(issues) > 0


class TestScoreRange:
    """Tests for score output range validation."""

    def test_score_must_be_in_range(self):
        """score must be in [0, 100]."""
        for score in [0.0, 50.0, 100.0]:
            signal = {
                "ticker": "AAPL",
                "action": "buy",
                "signal_strength": 0.5,
                "score": score,
                "entry_price": 100.0,
                "stop_price": 90.0,
                "target_price": 120.0,
            }
            issues = validate_signal_dict(signal)
            assert len(issues) == 0, f"Expected score {score} to be valid"

    def test_score_rejects_above_100(self):
        """score > 100 should be rejected."""
        signal = {
            "ticker": "AAPL",
            "action": "buy",
            "signal_strength": 0.5,
            "score": 150.0,
            "entry_price": 100.0,
            "stop_price": 90.0,
            "target_price": 120.0,
        }
        issues = validate_signal_dict(signal)
        assert any("score must be in [0, 100]" in issue for issue in issues)

    def test_score_rejects_negative(self):
        """score < 0 should be rejected."""
        signal = {
            "ticker": "AAPL",
            "action": "buy",
            "signal_strength": 0.5,
            "score": -10.0,
            "entry_price": 100.0,
            "stop_price": 90.0,
            "target_price": 120.0,
        }
        issues = validate_signal_dict(signal)
        assert any("score must be in [0, 100]" in issue for issue in issues)


class TestActionNormalization:
    """Tests for action normalization."""

    def test_canonical_actions_defined(self):
        """Canonical actions must be defined."""
        assert SignalAction.BUY is not None
        assert SignalAction.SELL is not None
        assert SignalAction.WATCH is not None
        assert SignalAction.AVOID is not None
        assert SignalAction.HOLD is not None
        assert SignalAction.EXIT is not None

    def test_legacy_aliases_work(self):
        """Legacy action aliases should map to canonical."""
        assert SignalAction.from_string("buy_open") == SignalAction.BUY_OPEN
        assert SignalAction.from_string("buy_dip") == SignalAction.BUY_DIP

    def test_unknown_action_defaults_to_watch(self):
        """Unknown action should default to WATCH."""
        result = SignalAction.from_string("unknown_action_xyz")
        assert result == SignalAction.WATCH

    def test_signal_action_value_is_string(self):
        """SignalAction value must be a string."""
        for action in SignalAction:
            assert isinstance(action.value, str)


class TestSignalFamilyClassification:
    """Tests for signal family classification."""

    def test_all_families_defined(self):
        """All required families must be defined."""
        families = [
            SignalFamily.TS_MOMENTUM,
            SignalFamily.XS_MOMENTUM,
            SignalFamily.MEAN_REVERSION,
            SignalFamily.BREAKOUT,
            SignalFamily.PULLBACK,
            SignalFamily.PAIRS,
            SignalFamily.BASELINE_A,
            SignalFamily.BASELINE_B,
        ]
        for family in families:
            assert family is not None
            assert isinstance(family.value, str)

    def test_family_value_is_string(self):
        """Family value must be a string."""
        for family in SignalFamily:
            assert isinstance(family.value, str)


class TestRegimeLabelClassification:
    """Tests for regime label classification."""

    def test_all_regimes_defined(self):
        """All required regimes must be defined."""
        regimes = [
            RegimeLabel.TREND,
            RegimeLabel.CHOP,
            RegimeLabel.CRISIS,
            RegimeLabel.NEUTRAL,
            RegimeLabel.BULL,
            RegimeLabel.BEAR,
        ]
        for regime in regimes:
            assert regime is not None

    def test_regime_from_string_aliases(self):
        """Regime aliases should map correctly."""
        assert RegimeLabel.from_string("sideways") == RegimeLabel.CHOP
        assert RegimeLabel.from_string("volatile") == RegimeLabel.CRISIS
        assert RegimeLabel.from_string("risk_off") == RegimeLabel.CRISIS
        assert RegimeLabel.from_string("") == RegimeLabel.NEUTRAL


class TestSignalBatchSchema:
    """Tests for SignalBatch schema."""

    def test_signal_batch_to_dict(self):
        """SignalBatch should serialize correctly."""
        batch = SignalBatch(
            batch_id="test-123",
            signals=[
                SignalDecision(
                    ticker="AAPL",
                    action=SignalAction.BUY,
                    signal_strength=0.8,
                    score=80.0,
                    entry_price=150.0,
                    stop_price=140.0,
                    target_price=170.0,
                )
            ],
        )

        d = batch.to_dict()
        assert d["batch_id"] == "test-123"
        assert len(d["signals"]) == 1
        assert d["signals"][0]["ticker"] == "AAPL"

    def test_signal_batch_family_counts(self):
        """SignalBatch should track family counts."""
        batch = SignalBatch()
        batch.family_counts[SignalFamily.BASELINE_A] = 5
        batch.family_counts[SignalFamily.TS_MOMENTUM] = 3

        assert batch.family_counts[SignalFamily.BASELINE_A] == 5
        assert batch.family_counts[SignalFamily.TS_MOMENTUM] == 3


class TestSignalContextSchema:
    """Tests for SignalContext schema."""

    def test_signal_context_defaults(self):
        """SignalContext should have sensible defaults."""
        context = SignalContext()

        assert context.tickers == []
        assert context.max_signals == 200
        assert context.min_score == 35.0
        assert context.regime == RegimeLabel.NEUTRAL
        assert isinstance(context.timestamp, datetime)

    def test_signal_context_to_dict(self):
        """SignalContext should serialize correctly."""
        context = SignalContext(
            tickers=["AAPL", "MSFT"],
            max_signals=100,
            min_score=50.0,
        )

        d = context.to_dict()
        assert d["tickers"] == ["AAPL", "MSFT"]
        assert d["max_signals"] == 100
        assert d["min_score"] == 50.0


class TestSignalMetadataSchema:
    """Tests for SignalMetadata schema."""

    def test_signal_metadata_defaults(self):
        """SignalMetadata should have sensible defaults."""
        metadata = SignalMetadata()

        assert metadata.expected_holding_period_bars == 20
        assert metadata.cost_sensitivity == 0.5
        assert metadata.regime_preference == RegimeLabel.NEUTRAL
        assert metadata.min_confidence == 0.5
        assert metadata.max_position_size_pct == 5.0

    def test_signal_metadata_to_dict(self):
        """SignalMetadata should serialize correctly."""
        metadata = SignalMetadata(
            expected_holding_period_bars=15,
            cost_sensitivity=0.7,
            regime_preference=RegimeLabel.TREND,
        )

        d = metadata.to_dict()
        assert d["expected_holding_period_bars"] == 15
        assert d["cost_sensitivity"] == 0.7
        assert d["regime_preference"] == "trend"


class TestSignalDiagnosticsSchema:
    """Tests for SignalDiagnostics schema."""

    def test_signal_diagnostics_defaults(self):
        """SignalDiagnostics should have sensible defaults."""
        diagnostics = SignalDiagnostics()

        assert diagnostics.generation_time_ms == 0.0
        assert diagnostics.feature_count == 0
        assert diagnostics.missing_features == []
        assert diagnostics.warnings == []

    def test_signal_diagnostics_to_dict(self):
        """SignalDiagnostics should serialize correctly."""
        diagnostics = SignalDiagnostics(
            generation_time_ms=150.5,
            feature_count=10,
            warnings=["test warning"],
        )

        d = diagnostics.to_dict()
        assert d["generation_time_ms"] == 150.5
        assert d["feature_count"] == 10
        assert "test warning" in d["warnings"]


class TestDecisionToLegacyDict:
    """Test SignalDecision.to_dict() backward compatibility."""

    def test_legacy_fields_present(self):
        """Legacy fields must be present in output."""
        decision = SignalDecision(
            ticker="AAPL",
            action=SignalAction.BUY,
            signal_strength=0.75,
            score=75.0,
            entry_price=150.0,
            stop_price=140.0,
            target_price=170.0,
            partial_target_price=160.0,
            confidence=0.8,
            family=SignalFamily.BASELINE_A,
            source="SignalA_v0",
            reason="Strong momentum",
            atr_14=2.5,
            risk_reward=2.5,
        )

        d = decision.to_dict()

        assert "ticker" in d
        assert "action" in d
        assert "score" in d
        assert "entry_price" in d
        assert "stop_price" in d
        assert "target_price" in d
        assert "confidence" in d
        assert "family" in d
        assert "source" in d
        assert "reason" in d
        assert "atr_14" in d
        assert "risk_reward" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
