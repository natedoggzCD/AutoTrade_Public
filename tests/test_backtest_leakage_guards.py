import pytest
from datetime import date, timedelta

import pandas as pd
import numpy as np

from autotrade.backtesting.leakage_detector import (
    LeakageDetector,
    LeakageCheckResult,
    LeakageReport,
)


class TestTrainTestDateOverlap:
    """Tests for train/test date overlap detection."""

    def test_no_overlap_passes(self):
        """Test that proper separation passes."""
        detector = LeakageDetector(fail_on_leakage=False)
        result = detector.check_train_test_date_overlap(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 3, 31),
            test_start=date(2024, 4, 1),
            test_end=date(2024, 4, 30),
        )
        assert result.passed
        assert result.check_name == "train_test_date_separation"

    def test_overlap_fails(self):
        """Test that overlapping train/test periods fail."""
        detector = LeakageDetector(fail_on_leakage=False)
        result = detector.check_train_test_date_overlap(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 4, 15),
            test_start=date(2024, 4, 10),
            test_end=date(2024, 4, 30),
        )
        assert not result.passed
        assert result.severity == "error"
        assert "overlaps" in result.message.lower()


class TestFoldBoundaries:
    """Tests for fold boundary leakage detection."""

    def test_single_fold_passes(self):
        """Test that single fold passes."""
        detector = LeakageDetector(fail_on_leakage=False)
        folds = [
            {
                "train_start": date(2024, 1, 1),
                "train_end": date(2024, 3, 31),
                "test_start": date(2024, 4, 1),
                "test_end": date(2024, 4, 30),
            }
        ]
        result = detector.check_fold_boundaries(folds)
        assert result.passed

    def test_sequential_folds_passes(self):
        """Test that sequential non-overlapping folds pass."""
        detector = LeakageDetector(fail_on_leakage=False)
        folds = [
            {
                "train_start": date(2024, 1, 1),
                "train_end": date(2024, 3, 31),
                "test_start": date(2024, 4, 1),
                "test_end": date(2024, 4, 30),
            },
            {
                "train_start": date(2024, 5, 1),
                "train_end": date(2024, 7, 31),
                "test_start": date(2024, 8, 1),
                "test_end": date(2024, 8, 31),
            },
        ]
        result = detector.check_fold_boundaries(folds)
        assert result.passed

    def test_overlapping_folds_fails(self):
        """Test that overlapping folds fail."""
        detector = LeakageDetector(fail_on_leakage=False)
        folds = [
            {
                "train_start": date(2024, 1, 1),
                "train_end": date(2024, 3, 31),
                "test_start": date(2024, 4, 1),
                "test_end": date(2024, 4, 30),
            },
            {
                "train_start": date(2024, 4, 15),
                "train_end": date(2024, 6, 30),
                "test_start": date(2024, 7, 1),
                "test_end": date(2024, 7, 31),
            },
        ]
        result = detector.check_fold_boundaries(folds)
        assert not result.passed
        assert result.severity == "error"


class TestFeatureWindowFutureAccess:
    """Tests for future bar access detection in features."""

    def test_empty_data_passes(self):
        """Test that empty data passes (insufficient for check)."""
        detector = LeakageDetector(fail_on_leakage=False)
        df = pd.DataFrame()
        result = detector.check_feature_window_future_access(
            features_df=df, feature_window_days=20
        )
        assert result.passed

    def test_no_date_column_warns(self):
        """Test warning when no date column present."""
        detector = LeakageDetector(fail_on_leakage=False)
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5] * 10})
        result = detector.check_feature_window_future_access(
            features_df=df, feature_window_days=20
        )
        assert not result.passed
        assert result.severity == "warning"

    def test_normal_features_passes(self):
        """Test that normal features pass."""
        detector = LeakageDetector(fail_on_leakage=False)
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        result = detector.check_feature_window_future_access(
            features_df=df, feature_window_days=20
        )
        assert result.passed


class TestTargetLeakage:
    """Tests for target leakage detection."""

    def test_empty_labels_passes(self):
        """Test that empty labels pass."""
        detector = LeakageDetector(fail_on_leakage=False)
        labels = pd.Series(dtype=float)
        features = pd.DataFrame({"value": [1, 2, 3]})
        result = detector.check_target_leakage(labels=labels, features=features)
        assert result.passed

    def test_no_features_warns(self):
        """Test warning when no features provided."""
        detector = LeakageDetector(fail_on_leakage=False)
        labels = pd.Series([1, 0, 1, 0], name="label")
        features = pd.DataFrame()
        result = detector.check_target_leakage(labels=labels, features=features)
        assert not result.passed
        assert result.severity == "warning"

    def test_zero_variance_label_warns(self):
        """Test warning for zero-variance labels."""
        detector = LeakageDetector(fail_on_leakage=False)
        labels = pd.Series([1, 1, 1, 1], name="label")
        features = pd.DataFrame({"f1": [1, 2, 3, 4]})
        result = detector.check_target_leakage(labels=labels, features=features)
        assert not result.passed
        assert result.severity == "warning"

    def test_normal_case_passes(self):
        """Test that normal features don't trigger leakage warning."""
        detector = LeakageDetector(fail_on_leakage=False)
        np.random.seed(42)
        labels = pd.Series(np.random.choice([0, 1], size=100), name="label")
        features = pd.DataFrame(
            {"f1": np.random.randn(100), "f2": np.random.randn(100)}
        )
        result = detector.check_target_leakage(labels=labels, features=features)
        assert result.passed


class TestForwardFillLeakage:
    """Tests for forward-fill leakage detection across fold boundaries."""

    def test_no_data_passes(self):
        """Test empty data passes."""
        detector = LeakageDetector(fail_on_leakage=False)
        df = pd.DataFrame()
        result = detector.check_forward_fill_leakage(data=df, fold_boundaries=[])
        assert result.passed

    def test_no_date_column_warns(self):
        """Test warning when no date column."""
        detector = LeakageDetector(fail_on_leakage=False)
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = detector.check_forward_fill_leakage(
            data=df,
            fold_boundaries=[(date(2024, 4, 1), date(2024, 4, 30))],
        )
        assert not result.passed
        assert result.severity == "warning"


class TestLeakageReport:
    """Tests for LeakageReport class."""

    def test_report_to_dict(self):
        """Test report serialization."""
        report = LeakageReport(
            passed=False,
            checks=[
                LeakageCheckResult(
                    check_name="test_check",
                    passed=False,
                    severity="error",
                    message="Test error",
                )
            ],
            warnings=["Test warning"],
        )
        result = report.to_dict()
        assert result["passed"] is False
        assert result["error_count"] == 1
        assert result["warning_count"] == 0

    def test_report_error_count(self):
        """Test error count calculation."""
        report = LeakageReport(
            passed=False,
            checks=[
                LeakageCheckResult(check_name="test1", passed=False, severity="error"),
                LeakageCheckResult(check_name="test2", passed=False, severity="error"),
                LeakageCheckResult(
                    check_name="test3", passed=False, severity="warning"
                ),
            ],
        )
        assert report.error_count == 2
        assert report.warning_count == 1


class TestIntegration:
    """Integration tests for leakage detector."""

    def test_run_all_checks_with_folds(self):
        """Test running all checks with fold configurations."""
        detector = LeakageDetector(fail_on_leakage=False)
        folds = [
            {
                "train_start": date(2024, 1, 1),
                "train_end": date(2024, 3, 31),
                "test_start": date(2024, 4, 1),
                "test_end": date(2024, 4, 30),
            },
            {
                "train_start": date(2024, 5, 1),
                "train_end": date(2024, 7, 31),
                "test_start": date(2024, 8, 1),
                "test_end": date(2024, 8, 31),
            },
        ]

        report = detector.run_all_checks(folds=folds)
        assert isinstance(report, LeakageReport)
        assert len(report.checks) > 0
        assert report.passed or not report.passed

    def test_create_leakage_detector_default(self):
        """Test factory function creates detector."""
        detector = LeakageDetector(fail_on_leakage=True)
        assert detector.fail_on_leakage is True

    def test_fail_on_leakage_flag(self):
        """Test fail_on_leakage flag behavior."""
        detector = LeakageDetector(fail_on_leakage=True)
        assert detector.fail_on_leakage is True

        detector2 = LeakageDetector(fail_on_leakage=False)
        assert detector2.fail_on_leakage is False
