from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from autotrade.backtesting.contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
    MetricBundle,
)


def compute_config_hash(config: Dict[str, Any], seed: Optional[int] = None) -> str:
    """Generate deterministic hash from configuration dictionary.

    Args:
        config: Configuration dictionary to hash
        seed: Optional seed to include in hash

    Returns:
        16-character hex hash for run identification
    """
    config_copy = dict(config)
    if seed is not None:
        config_copy["_seed"] = seed

    json_str = json.dumps(config_copy, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def generate_run_id(strategy_id: str, config_hash: str, timestamp: str) -> str:
    """Generate unique run identifier.

    Args:
        strategy_id: Strategy identifier
        config_hash: Configuration hash
        timestamp: Run timestamp

    Returns:
        Unique run ID in format: {strategy_id}_{config_hash}_{timestamp}
    """
    return f"{strategy_id}_{config_hash}_{timestamp}"


@runtime_checkable
class BacktestRunner(Protocol):
    """Protocol for backtest execution engines."""

    def run(self, request: BacktestRequest) -> BacktestResultArtifact:
        """Execute a backtest run.

        Args:
            request: Backtest configuration request

        Returns:
            Complete backtest result artifact
        """
        ...

    def run_walk_forward(
        self, request: BacktestRequest
    ) -> List[FoldResult]:
        """Execute walk-forward backtesting.

        Args:
            request: Backtest configuration request

        Returns:
            List of fold results
        """
        ...

    def validate_request(self, request: BacktestRequest) -> List[str]:
        """Validate backtest request configuration.

        Args:
            request: Backtest configuration request

        Returns:
            List of validation error messages (empty if valid)
        """
        ...


@runtime_checkable
class EvaluationSuite(Protocol):
    """Protocol for evaluation and metric computation."""

    def compute_metrics(
        self, trades: List[Dict[str, Any]], equity_curve: Any
    ) -> MetricBundle:
        """Compute evaluation metrics from trades and equity curve.

        Args:
            trades: List of trade records
            equity_curve: DataFrame with equity over time

        Returns:
            Bundle of computed metrics
        """
        ...

    def compare_strategies(
        self, results: List[BacktestResultArtifact]
    ) -> Dict[str, Any]:
        """Compare multiple strategy results.

        Args:
            results: List of backtest artifacts to compare

        Returns:
            Comparison summary with rankings
        """
        ...

    def assess_promotion_eligibility(
        self, artifact: BacktestResultArtifact
    ) -> tuple[bool, Optional[str]]:
        """Assess if a strategy is eligible for promotion.

        Args:
            artifact: Backtest result artifact

        Returns:
            Tuple of (eligible, reason_code)
        """
        ...


@runtime_checkable
class ModelSelector(Protocol):
    """Protocol for model/strategy selection with bias controls."""

    def select_best(
        self, candidates: List[BacktestResultArtifact]
    ) -> Optional[BacktestResultArtifact]:
        """Select best model from candidates.

        Args:
            candidates: List of candidate backtest artifacts

        Returns:
            Best candidate or None if no suitable model
        """
        ...

    def apply_multiple_testing_correction(
        self, candidates: List[BacktestResultArtifact]
    ) -> List[BacktestResultArtifact]:
        """Apply multiple testing correction to candidate rankings.

        Args:
            candidates: List of candidate backtest artifacts

        Returns:
            Candidates with adjusted rankings/p-values
        """
        ...

    def estimate_pbo(
        self, fold_sharpes: List[float], n_folds: int
    ) -> float:
        """Estimate probability of backtest overfitting.

        Args:
            fold_sharpes: List of Sharpe ratios from different folds
            n_folds: Number of folds

        Returns:
            PBO estimate (0-1, lower is better)
        """
        ...

    def compute_deflated_sharpe(
        self,
        sharpe: float,
        n_trials: int,
        n_observations: int,
        correlation: float = 0.0,
    ) -> float:
        """Compute deflated Sharpe ratio.

        Args:
            sharpe: Observed Sharpe ratio
            n_trials: Number of strategy variants tested
            n_observations: Number of return observations
            correlation: Average correlation between strategies

        Returns:
            Deflated Sharpe ratio (adjusted for selection bias)
        """
        ...
