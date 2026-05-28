"""Backtesting package (local-data, no external APIs)."""

from .engine import BacktestEngine
from .signal_validator import SignalValidator, ValidationResult
from .strategy_backtester import StrategyBacktester, BacktestResult
from .strategy_config import LabStrategyConfig

# Stage 6 — protocolized backtest contracts, evaluation, and promotion
from .contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
    MetricBundle,
)
from .interfaces import BacktestRunner, EvaluationSuite as EvaluationSuiteProtocol, ModelSelector
from .protocol import BacktestProtocol, ProtocolAdapter
from .evaluation import EvaluationSuite, StatisticalControlReport
from .results import (
    HyperparameterSearchRunner,
    PromotionGate,
    PromotionRule,
    RunRegistry,
    SearchResult,
    SearchRun,
    apply_promotion_rules,
    create_default_promotion_gate,
    create_strict_promotion_gate,
    create_search_runner_from_config,
)
from .leakage_detector import LeakageDetector, LeakageReport
from .artifact_persistence import ArtifactPersister

__all__ = [
    # Legacy
    "BacktestEngine",
    "SignalValidator",
    "ValidationResult",
    "StrategyBacktester",
    "BacktestResult",
    "LabStrategyConfig",
    # Stage 6 — contracts
    "BacktestRequest",
    "BacktestResultArtifact",
    "FoldResult",
    "MetricBundle",
    # Stage 6 — interfaces
    "BacktestRunner",
    "EvaluationSuiteProtocol",
    "ModelSelector",
    # Stage 6 — protocol
    "BacktestProtocol",
    "ProtocolAdapter",
    # Stage 6 — evaluation
    "EvaluationSuite",
    "StatisticalControlReport",
    # Stage 6 — results / promotion / registry
    "HyperparameterSearchRunner",
    "PromotionGate",
    "PromotionRule",
    "RunRegistry",
    "SearchResult",
    "SearchRun",
    "apply_promotion_rules",
    "create_default_promotion_gate",
    "create_strict_promotion_gate",
    "create_search_runner_from_config",
    # Stage 6 — leakage + persistence
    "LeakageDetector",
    "LeakageReport",
    "ArtifactPersister",
]
