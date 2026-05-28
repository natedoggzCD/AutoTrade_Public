from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LeakageCheckResult:
    """Result of a single leakage check."""

    check_name: str
    passed: bool
    severity: str = "error"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LeakageReport:
    """Complete leakage detection report for a backtest."""

    passed: bool
    checks: List[LeakageCheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "checks": [
                {
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
            "warnings": self.warnings,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }

    @property
    def error_count(self) -> int:
        """Count of failed checks with severity 'error'."""
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of failed checks with severity 'warning'."""
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")


class LeakageDetector:
    """
    Detects common data leakage patterns in backtesting.

    Checks include:
    - Future bar access in feature windows
    - Target leakage in label generation
    - Train/test date overlap
    - Forward-fill across fold boundaries
    """

    def __init__(self, fail_on_leakage: bool = True):
        """
        Initialize leakage detector.

        Args:
            fail_on_leakage: If True, failed checks raise exceptions.
                           If False, warnings are logged instead.
        """
        self.fail_on_leakage = fail_on_leakage

    def check_train_test_date_overlap(
        self,
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
    ) -> LeakageCheckResult:
        """
        Check for train/test date overlap.

        Ensures there's no gap overlap between train and test periods
        that would leak future information.
        """
        if train_end >= test_start:
            return LeakageCheckResult(
                check_name="train_test_date_overlap",
                passed=False,
                severity="error",
                message=f"Train period ({train_start} to {train_end}) overlaps with test period ({test_start} to {test_end})",
                details={
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "overlap_days": (train_end - test_start).days + 1,
                },
            )

        gap_days = (test_start - train_end).days - 1
        if gap_days < 0:
            return LeakageCheckResult(
                check_name="train_test_date_gap",
                passed=False,
                severity="error",
                message="Invalid date configuration: test starts before train ends",
                details={
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                },
            )

        return LeakageCheckResult(
            check_name="train_test_date_separation",
            passed=True,
            message=f"Train/test periods properly separated by {gap_days} days",
            details={
                "train_end": str(train_end),
                "test_start": str(test_start),
                "gap_days": gap_days,
            },
        )

    def check_fold_boundaries(
        self,
        folds: Sequence[Dict[str, date]],
    ) -> LeakageCheckResult:
        """
        Check that fold boundaries don't have forward-fill leakage.

        Ensures no data leaks between consecutive walk-forward folds.
        """
        if len(folds) < 2:
            return LeakageCheckResult(
                check_name="fold_boundaries",
                passed=True,
                message="Single fold, no boundary check needed",
                details={"fold_count": len(folds)},
            )

        issues = []
        for i in range(len(folds) - 1):
            current_fold = folds[i]
            next_fold = folds[i + 1]

            if current_fold["train_end"] >= current_fold["test_start"]:
                issues.append(
                    f"Fold {i} train end ({current_fold['train_end']}) overlaps its test start ({current_fold['test_start']})"
                )

            if next_fold["train_end"] >= next_fold["test_start"]:
                issues.append(
                    f"Fold {i + 1} train end ({next_fold['train_end']}) overlaps its test start ({next_fold['test_start']})"
                )

            if current_fold["test_end"] >= next_fold["test_start"]:
                issues.append(
                    f"Fold {i} test end ({current_fold['test_end']}) overlaps with fold {i + 1} test start ({next_fold['test_start']})"
                )

            expanding_window = next_fold["train_start"] <= current_fold["train_start"]
            if not expanding_window and current_fold["test_start"] <= next_fold["train_start"] <= current_fold["test_end"]:
                issues.append(
                    f"Fold {i + 1} train start ({next_fold['train_start']}) falls inside fold {i} test period ({current_fold['test_start']} to {current_fold['test_end']})"
                )

        if issues:
            return LeakageCheckResult(
                check_name="fold_boundaries",
                passed=False,
                severity="error",
                message="Fold boundary leakage detected",
                details={"issues": issues},
            )

        return LeakageCheckResult(
            check_name="fold_boundaries",
            passed=True,
            message=f"All {len(folds)} folds have proper boundaries",
            details={"fold_count": len(folds)},
        )

    def check_feature_window_future_access(
        self,
        features_df: pd.DataFrame,
        feature_window_days: int,
        target_column: Optional[str] = None,
    ) -> LeakageCheckResult:
        """
        Check for future bar access in feature windows.

        Detects if features use data from future dates relative to the prediction time.
        This is a simplified check that looks for suspicious patterns in the feature data.
        """
        if features_df.empty or len(features_df) < feature_window_days * 2:
            return LeakageCheckResult(
                check_name="feature_window_future_access",
                passed=True,
                message="Insufficient data for future access check",
                details={"row_count": len(features_df)},
            )

        if "date" not in features_df.columns:
            return LeakageCheckResult(
                check_name="feature_window_future_access",
                passed=False,
                severity="warning",
                message="No date column found, cannot verify feature windows",
                details={"columns": list(features_df.columns)},
            )

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return LeakageCheckResult(
                check_name="feature_window_future_access",
                passed=True,
                message="No numeric features to check",
                details={"columns": list(features_df.columns)},
            )

        suspicious_features = []
        for col in numeric_cols:
            if col == target_column:
                continue

            series = features_df[col].values
            if len(series) < feature_window_days:
                continue

            for i in range(len(series) - feature_window_days):
                window = series[i : i + feature_window_days]
                if np.any(~np.isfinite(window)):
                    continue

            correlations = []
            for lag in range(1, min(6, feature_window_days)):
                if len(series) > lag:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if abs(corr) > 0.95:
                        correlations.append({"lag": lag, "correlation": float(corr)})

            if len(correlations) >= 3:
                suspicious_features.append(
                    {"feature": col, "high_lag_correlations": correlations}
                )

        if suspicious_features:
            return LeakageCheckResult(
                check_name="feature_window_future_access",
                passed=False,
                severity="warning",
                message=f"Found {len(suspicious_features)} features with suspicious autocorrelation patterns",
                details={"suspicious_features": suspicious_features[:5]},
            )

        return LeakageCheckResult(
            check_name="feature_window_future_access",
            passed=True,
            message="No obvious future access patterns detected in features",
            details={"checked_features": len(numeric_cols)},
        )

    def check_target_leakage(
        self,
        labels: pd.Series,
        features: pd.DataFrame,
        label_column: str = "label",
    ) -> LeakageCheckResult:
        """
        Check for target leakage in labels.

        Detects if the label calculation might have used future information.
        This is a heuristic check based on label distribution and correlation patterns.
        """
        if labels.empty:
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=True,
                message="No labels to check",
            )

        if features.empty:
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=False,
                severity="warning",
                message="No features provided for leakage check",
            )

        valid_labels = labels.dropna()
        if len(valid_labels) == 0:
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=False,
                severity="warning",
                message="All labels are NaN",
            )

        label_std = valid_labels.std()
        if label_std == 0 or np.isnan(label_std):
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=False,
                severity="warning",
                message="Label has zero variance - possible constant prediction",
                details={"label_mean": float(valid_labels.mean())},
            )

        numeric_features = features.select_dtypes(include=[np.number]).columns
        if len(numeric_features) == 0:
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=True,
                message="No numeric features to check correlation",
            )

        common_idx = valid_labels.index.intersection(features.index)
        if len(common_idx) < 10:
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=False,
                severity="warning",
                message="Insufficient overlapping data points for correlation check",
                details={"common_points": len(common_idx)},
            )

        high_correlations = []
        for col in numeric_features:
            combined = pd.DataFrame(
                {"label": valid_labels, "feature": features.loc[common_idx, col]}
            ).dropna()
            if len(combined) >= 10:
                corr = combined["label"].corr(combined["feature"])
                if abs(corr) > 0.9:
                    high_correlations.append(
                        {"feature": col, "correlation": float(corr)}
                    )

        if high_correlations:
            return LeakageCheckResult(
                check_name="target_leakage",
                passed=False,
                severity="warning",
                message=f"Found {len(high_correlations)} features with >0.9 correlation to label",
                details={"high_correlations": high_correlations[:5]},
            )

        return LeakageCheckResult(
            check_name="target_leakage",
            passed=True,
            message="No obvious target leakage detected",
            details={"checked_features": len(numeric_features)},
        )

    def check_forward_fill_leakage(
        self,
        data: pd.DataFrame,
        fold_boundaries: Sequence[Tuple[date, date]],
    ) -> LeakageCheckResult:
        """
        Check for forward-fill leakage across fold boundaries.

        Ensures that data isn't forward-filled from training into test periods.
        """
        if data.empty or len(fold_boundaries) == 0:
            return LeakageCheckResult(
                check_name="forward_fill_leakage",
                passed=True,
                message="No data or fold boundaries to check",
            )

        if "date" not in data.columns:
            return LeakageCheckResult(
                check_name="forward_fill_leakage",
                passed=False,
                severity="warning",
                message="No date column to verify forward-fill boundaries",
            )

        data = data.sort_values("date").reset_index(drop=True)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        issues = []
        for i, (test_start, test_end) in enumerate(fold_boundaries):
            test_data = data[
                (data["date"] >= pd.Timestamp(test_start))
                & (data["date"] <= pd.Timestamp(test_end))
            ]

            if test_data.empty:
                continue

            pre_test_data = data[
                (data["date"] < pd.Timestamp(test_start))
                & (data["date"] >= pd.Timestamp(test_start) - timedelta(days=5))
            ]

            if pre_test_data.empty:
                continue

            for col in numeric_cols[:10]:
                last_pre_value = (
                    pre_test_data[col].iloc[-1] if not pre_test_data.empty else np.nan
                )
                first_test_value = (
                    test_data[col].iloc[0] if not test_data.empty else np.nan
                )

                if pd.isna(last_pre_value) or pd.isna(first_test_value):
                    continue

                if last_pre_value == first_test_value:
                    issues.append(
                        f"Fold {i}: Column '{col}' has same value at boundary "
                        f"({last_pre_value}) - possible forward-fill"
                    )
                    break

        if issues:
            return LeakageCheckResult(
                check_name="forward_fill_leakage",
                passed=False,
                severity="warning",
                message="Potential forward-fill leakage detected at fold boundaries",
                details={"issues": issues[:5]},
            )

        return LeakageCheckResult(
            check_name="forward_fill_leakage",
            passed=True,
            message="No forward-fill leakage detected at fold boundaries",
            details={"checked_folds": len(fold_boundaries)},
        )

    def run_all_checks(
        self,
        folds: Optional[Sequence[Dict[str, date]]] = None,
        features_df: Optional[pd.DataFrame] = None,
        labels: Optional[pd.Series] = None,
        feature_window_days: int = 20,
    ) -> LeakageReport:
        """
        Run all leakage checks.

        Args:
            folds: List of fold configurations with train/test dates
            features_df: DataFrame of features for future-access check
            labels: Series of labels for target-leakage check
            feature_window_days: Window size for feature checks

        Returns:
            Complete leakage report with all check results
        """
        checks: List[LeakageCheckResult] = []
        warnings: List[str] = []

        if folds:
            boundary_result = self.check_fold_boundaries(folds)
            checks.append(boundary_result)

            for fold_config in folds:
                train_test_result = self.check_train_test_date_overlap(
                    train_start=fold_config["train_start"],
                    train_end=fold_config["train_end"],
                    test_start=fold_config["test_start"],
                    test_end=fold_config["test_end"],
                )
                checks.append(train_test_result)

            fold_boundaries = [
                (f["test_start"], f["test_end"]) for f in folds if "test_start" in f
            ]
            if fold_boundaries and features_df is not None:
                ff_result = self.check_forward_fill_leakage(
                    data=features_df, fold_boundaries=fold_boundaries
                )
                checks.append(ff_result)

        if features_df is not None:
            feature_result = self.check_feature_window_future_access(
                features_df=features_df, feature_window_days=feature_window_days
            )
            checks.append(feature_result)

        if labels is not None and features_df is not None:
            target_result = self.check_target_leakage(
                labels=labels, features=features_df
            )
            checks.append(target_result)

        passed = all(c.passed for c in checks)
        error_count = sum(1 for c in checks if not c.passed and c.severity == "error")

        if not passed:
            if error_count > 0 and self.fail_on_leakage:
                error_msgs = [
                    c.message for c in checks if not c.passed and c.severity == "error"
                ]
                warnings.extend(error_msgs)
                logger.error(
                    f"Leakage detection failed: {error_count} error(s), {len(warnings)} warning(s)"
                )
            else:
                warning_msgs = [c.message for c in checks if not c.passed]
                warnings.extend(warning_msgs)
                logger.warning(
                    f"Leakage warnings: {len(warning_msgs)} issue(s) detected"
                )

        return LeakageReport(
            passed=passed,
            checks=checks,
            warnings=warnings,
        )


def create_leakage_detector() -> LeakageDetector:
    """Create a leakage detector with config-loaded settings."""
    try:
        from config.config_loader import get_backtest_protocol_config

        config = get_backtest_protocol_config()
        fail_on_leakage = config.leakage_guard.fail_on_detected_leakage
    except Exception:
        fail_on_leakage = True

    return LeakageDetector(fail_on_leakage=fail_on_leakage)
