"""
Signal Contracts
================
Standardized dataclasses for signal generation pipeline.

These define the contract between signal models and downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class SignalAction(str, Enum):
    """Canonical signal actions."""

    BUY = "buy"
    SELL = "sell"
    WATCH = "watch"
    AVOID = "avoid"
    HOLD = "hold"
    EXIT = "exit"

    # Legacy aliases
    BUY_OPEN = "buy_open"
    BUY_DIP = "buy_dip"

    @classmethod
    def from_string(cls, action: str) -> "SignalAction":
        """Convert legacy action strings to canonical form."""
        # Defensive: handle float/numeric values from DataFrames
        if not isinstance(action, str):
            action = str(action)
        
        action_lower = action.lower().strip()

        # Legacy mappings
        legacy_map = {
            "buy_open": cls.BUY_OPEN,
            "buy_dip": cls.BUY_DIP,
            "long": cls.BUY,
            "short": cls.SELL,
            "close": cls.EXIT,
            "sell_close": cls.EXIT,
        }

        if action_lower in legacy_map:
            return legacy_map[action_lower]

        try:
            return cls(action_lower)
        except ValueError:
            return cls.WATCH


class SignalFamily(str, Enum):
    """Alpha family classifications."""

    TS_MOMENTUM = "ts_momentum"
    XS_MOMENTUM = "xs_momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    PULLBACK = "pullback"
    PAIRS = "pairs"
    BASELINE_A = "baseline_a"
    BASELINE_B = "baseline_b"


class RegimeLabel(str, Enum):
    """Market regime labels."""

    TREND = "trend"
    CHOP = "chop"
    CRISIS = "crisis"
    NEUTRAL = "neutral"
    BULL = "bull"
    BEAR = "bear"
    BEARISH = "bearish"

    @classmethod
    def from_string(cls, regime: str) -> "RegimeLabel":
        """Convert regime strings to a supported enum value."""
        if not regime:
            return cls.NEUTRAL

        # Defensive: handle float/numeric values from DataFrames
        if not isinstance(regime, str):
            regime = str(regime)

        normalized = regime.lower().strip()
        alias_map = {
            "sideways": cls.CHOP,
            "volatile": cls.CRISIS,
            "risk_off": cls.CRISIS,
            "risk_on": cls.TREND,
            "bear_market": cls.BEARISH,
            "selloff": cls.BEARISH,
            "sell_off": cls.BEARISH,
        }

        if normalized in alias_map:
            return alias_map[normalized]

        try:
            return cls(normalized)
        except ValueError:
            return cls.NEUTRAL


@dataclass
class SignalMetadata:
    """Metadata about a signal's expected behavior and constraints."""

    expected_holding_period_bars: int = 20
    cost_sensitivity: float = 0.5
    regime_preference: RegimeLabel = RegimeLabel.NEUTRAL
    min_confidence: float = 0.5
    max_position_size_pct: float = 5.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_holding_period_bars": self.expected_holding_period_bars,
            "cost_sensitivity": self.cost_sensitivity,
            "regime_preference": self.regime_preference.value,
            "min_confidence": self.min_confidence,
            "max_position_size_pct": self.max_position_size_pct,
            "tags": self.tags,
        }


@dataclass
class SignalDiagnostics:
    """Diagnostic information for signal debugging and analysis."""

    generation_time_ms: float = 0.0
    data_freshness_minutes: float = 0.0
    feature_count: int = 0
    missing_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_time_ms": self.generation_time_ms,
            "data_freshness_minutes": self.data_freshness_minutes,
            "feature_count": self.feature_count,
            "missing_features": self.missing_features,
            "warnings": self.warnings,
            "raw_scores": self.raw_scores,
        }


@dataclass
class StrategyProfile:
    """A robust strategy profile for signal generation."""
    strategy_id: str
    setup_type: str
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]
    risk_params: Dict[str, Any]
    validation_stats: Dict[str, Any]
    last_validated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "setup_type": self.setup_type,
            "entry_rules": self.entry_rules,
            "exit_rules": self.exit_rules,
            "risk_params": self.risk_params,
            "validation_stats": self.validation_stats,
            "last_validated": self.last_validated.isoformat()
        }


@dataclass
class SignalDecision:
    """
    Complete signal decision for a single ticker.

    This is the primary output of signal models.
    """

    ticker: str
    action: SignalAction
    signal_strength: float

    entry_price: float = 0.0
    arrival_price: float = 0.0
    nbbo: Optional[Tuple[float, float]] = None
    stop_price: float = 0.0
    target_price: float = 0.0
    partial_target_price: float = 0.0

    score: float = 50.0
    confidence: float = 0.5

    family: SignalFamily = SignalFamily.TS_MOMENTUM
    source: str = ""

    r1_price: float = 0.0
    r1_strength: float = 0.0
    s1_price: float = 0.0
    s1_strength: float = 0.0

    atr_14: float = 0.0
    risk_reward: float = 0.0

    reason: str = ""

    metadata: SignalMetadata = field(default_factory=SignalMetadata)
    diagnostics: SignalDiagnostics = field(default_factory=SignalDiagnostics)

    # Factor scores for analysis
    factor_scores: Dict[str, float] = field(default_factory=dict)

    # Timestamps
    generated_at: datetime = field(default_factory=datetime.now)

    # Regime context
    regime: RegimeLabel = RegimeLabel.NEUTRAL

    # Strategy metadata (Phase 1 alpha overhaul)
    strategy_id: str = ""
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    backtest_win_rate: float = 0.0
    backtest_profit_factor: float = 0.0
    walk_forward_validated: bool = False

    # Additional fields for compatibility
    s1_dist_pct: float = 0.0
    r1_dist_pct: float = 0.0
    r2_in_range: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to legacy dict format for compatibility."""
        return {
            "ticker": self.ticker,
            "action": self.action.value,
            "signal_strength": self.signal_strength,
            "entry_price": self.entry_price,
            "arrival_price": self.arrival_price,
            "nbbo": self.nbbo,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "partial_target_price": self.partial_target_price,
            "score": self.score,
            "confidence": self.confidence,
            "family": self.family.value,
            "source": self.source,
            "r1_price": self.r1_price,
            "r1_strength": self.r1_strength,
            "s1_price": self.s1_price,
            "s1_strength": self.s1_strength,
            "atr_14": self.atr_14,
            "risk_reward": self.risk_reward,
            "reason": self.reason,
            "metadata": self.metadata.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "factor_scores": self.factor_scores,
            "generated_at": self.generated_at.isoformat()
            if self.generated_at
            else None,
            "regime": self.regime.value,
            "strategy_id": self.strategy_id,
            "strategy_params": self.strategy_params,
            "backtest_win_rate": self.backtest_win_rate,
            "backtest_profit_factor": self.backtest_profit_factor,
            "walk_forward_validated": self.walk_forward_validated,
            "s1_dist_pct": self.s1_dist_pct,
            "r1_dist_pct": self.r1_dist_pct,
            "r2_in_range": self.r2_in_range,
        }

    @classmethod
    def from_candidate_dict(
        cls, data: Dict[str, Any], family: SignalFamily = SignalFamily.BASELINE_A
    ) -> "SignalDecision":
        """Create from legacy candidate dict format."""
        action_str = data.get("action", data.get("action_plan", "watch"))
        action = SignalAction.from_string(action_str)

        return cls(
            ticker=data.get("ticker", ""),
            action=action,
            signal_strength=data.get("score", 50.0) / 100.0,
            entry_price=data.get("price", 0.0),
            arrival_price=data.get("arrival_price", 0.0),
            nbbo=data.get("nbbo"),
            stop_price=data.get("stop_price", 0.0),
            target_price=data.get("target_price", 0.0),
            partial_target_price=data.get("partial_target_price", 0.0),
            score=data.get("score", data.get("normalized_score", 50.0)),
            confidence=data.get("normalized_score", 50.0) / 100.0,
            family=family,
            source=f"{family.value}_v0",
            r1_price=data.get("r1_price", 0.0),
            r1_strength=data.get("r1_strength", 0.0),
            s1_price=data.get("s1_price", 0.0),
            s1_strength=data.get("s1_strength", 0.0),
            atr_14=data.get("atr_14", 0.0),
            risk_reward=data.get("risk_reward", 0.0),
            reason=data.get("reason", data.get("action_plan", "")),
            factor_scores=data.get("factor_scores", {}),
            regime=RegimeLabel.from_string(data.get("regime", "neutral")),
            strategy_id=data.get("strategy_id", ""),
            strategy_params=data.get("strategy_params", {}),
            backtest_win_rate=float(data.get("backtest_win_rate", 0.0)),
            backtest_profit_factor=float(data.get("backtest_profit_factor", 0.0)),
            walk_forward_validated=bool(data.get("walk_forward_validated", False)),
            s1_dist_pct=data.get("distance_to_s1_pct", 0.0),
            r1_dist_pct=data.get("distance_to_r1_pct", 0.0),
        )


@dataclass
class SignalBatch:
    """
    Collection of signal decisions with batch-level metadata.

    Represents a complete signal generation run.
    """

    signals: List[SignalDecision] = field(default_factory=list)
    batch_id: str = ""
    generated_at: datetime = field(default_factory=datetime.now)

    regime: RegimeLabel = RegimeLabel.NEUTRAL
    regime_confidence: float = 0.5

    family_counts: Dict[SignalFamily, int] = field(default_factory=dict)

    # Pipeline metadata
    pipeline_version: str = "1.0.0"
    config_hash: str = ""

    # Filtering stats
    total_generated: int = 0
    total_filtered: int = 0
    filter_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "batch_id": self.batch_id,
            "generated_at": self.generated_at.isoformat()
            if self.generated_at
            else None,
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "family_counts": {k.value: v for k, v in self.family_counts.items()},
            "pipeline_version": self.pipeline_version,
            "config_hash": self.config_hash,
            "total_generated": self.total_generated,
            "total_filtered": self.total_filtered,
            "filter_reasons": self.filter_reasons,
        }

    def to_legacy_list(self) -> List[Dict[str, Any]]:
        """Convert to legacy list format for compatibility."""
        return [s.to_dict() for s in self.signals]

    def add_filter_reason(self, reason: str) -> None:
        """Record a filter reason."""
        self.filter_reasons[reason] = self.filter_reasons.get(reason, 0) + 1
        self.total_filtered += 1


@dataclass
class SignalContext:
    """
    Input context for signal generation.

    Contains all data needed to generate signals.
    """

    tickers: List[str] = field(default_factory=list)

    price_data: Any = None
    sr_data: Any = None
    feature_data: Any = None
    nbbo_data: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    regime: RegimeLabel = RegimeLabel.NEUTRAL
    regime_confidence: float = 0.5

    timestamp: datetime = field(default_factory=datetime.now)

    config_override: Dict[str, Any] = field(default_factory=dict)

    # Execution hints
    max_signals: int = 200
    min_score: float = 35.0

    # Universe constraints
    exclude_tickers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tickers": self.tickers,
            "regime": self.regime.value,
            "regime_confidence": self.regime_confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "config_override": self.config_override,
            "max_signals": self.max_signals,
            "min_score": self.min_score,
            "exclude_tickers": self.exclude_tickers,
        }


# Registry of signal contracts for validation
SIGNAL_CONTRACT_VERSION = "1.0.0"

REQUIRED_SIGNAL_FIELDS = [
    "ticker",
    "action",
    "signal_strength",
    "score",
    "entry_price",
    "stop_price",
    "target_price",
]

OPTIONAL_SIGNAL_FIELDS = [
    "partial_target_price",
    "confidence",
    "family",
    "source",
    "r1_price",
    "r1_strength",
    "s1_price",
    "s1_strength",
    "atr_14",
    "risk_reward",
    "reason",
    "metadata",
    "diagnostics",
    "factor_scores",
    "generated_at",
    "regime",
    "strategy_id",
    "strategy_params",
    "backtest_win_rate",
    "backtest_profit_factor",
    "walk_forward_validated",
]


def validate_signal_dict(data: Dict[str, Any]) -> List[str]:
    """Validate a signal dict and return list of issues."""
    issues = []

    for field_name in REQUIRED_SIGNAL_FIELDS:
        if field_name not in data or data[field_name] is None:
            issues.append(f"Missing required field: {field_name}")

    if "signal_strength" in data:
        strength = data["signal_strength"]
        if not isinstance(strength, (int, float)) or strength < -1.0 or strength > 1.0:
            issues.append(f"signal_strength must be in [-1, 1], got {strength}")

    if "score" in data:
        score = data["score"]
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            issues.append(f"score must be in [0, 100], got {score}")

    return issues
