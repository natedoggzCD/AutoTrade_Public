"""
Baseline Signals - Frozen SignalA and SignalB
===============================================
Versioned wrappers around the baseline signal generation logic.

SignalA (v0): ScreenerV2 with momentum_pullback scoring
SignalB (v0): UnifiedStrategy-based signal generation

These are frozen and should NOT be modified. Any changes must
create new versions (SignalA_v1, SignalB_v1, etc.).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from autotrade.signals.contracts import (
    SignalContext,
    SignalDecision,
    SignalBatch,
    SignalDiagnostics,
    SignalMetadata,
    SignalFamily,
    RegimeLabel,
    SignalAction,
    validate_signal_dict,
)
from autotrade.signals.interfaces import SignalModel, get_signal_registry

logger = logging.getLogger(__name__)


@dataclass
class SignalAConfig:
    """Configuration for SignalA (frozen v0)."""

    max_candidates: int = 200
    min_score: float = 35.0
    scoring_mode: str = "momentum_pullback"
    exclude_symbols: List[str] = None

    def __post_init__(self):
        if self.exclude_symbols is None:
            self.exclude_symbols = []


@dataclass
class SignalBConfig:
    """Configuration for SignalB (frozen v0)."""

    max_candidates: int = 200
    min_score: float = 10.0
    use_lessons: bool = True

    def __post_init__(self):
        pass


class SignalA:
    """
    SignalA - Frozen v0 baseline.

    Wraps ScreenerV2 with momentum_pullback scoring.
    This is the primary overnight screening signal.

    NOT MODIFIABLE - Create SignalA_v1 for changes.
    """

    VERSION = "v0"
    FAMILY = SignalFamily.BASELINE_A

    def __init__(self, config: Optional[SignalAConfig] = None):
        self.config = config or SignalAConfig()
        self._screener = None

    @property
    def name(self) -> str:
        return f"SignalA_{self.VERSION}"

    @property
    def family(self) -> SignalFamily:
        return self.FAMILY

    @property
    def version(self) -> str:
        return self.VERSION

    def _get_screener(self):
        """Lazy load screener to avoid import overhead."""
        if self._screener is None:
            from autotrade.signals.screener_v2 import ScreenerV2

            self._screener = ScreenerV2(scoring_mode=self.config.scoring_mode)
        return self._screener

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        """Generate signals using frozen ScreenerV2 logic."""
        start_time = time.perf_counter()

        try:
            screener = self._get_screener()
            candidates = screener.screen(
                symbols=context.tickers or None,
                max_candidates=self.config.max_candidates,
                exclude_symbols=context.exclude_tickers or self.config.exclude_symbols,
                log_samples=False,
            )

            signals = []
            for c in candidates:
                score = c.get("score", c.get("normalized_score", 50.0))
                if score < self.config.min_score:
                    continue

                signal = SignalDecision(
                    ticker=c.get("ticker", ""),
                    action=SignalAction.from_string(
                        c.get("action", c.get("action_plan", "watch"))
                    ),
                    signal_strength=score / 100.0,
                    entry_price=c.get("price", 0.0),
                    stop_price=c.get("stop_price", 0.0),
                    target_price=c.get("target_price", 0.0),
                    partial_target_price=c.get("partial_target_price", 0.0),
                    score=score,
                    confidence=score / 100.0,
                    family=self.FAMILY,
                    source=self.name,
                    r1_price=c.get("r1_price", 0.0),
                    r1_strength=c.get("r1_strength", 0.0),
                    s1_price=c.get("s1_price", 0.0),
                    s1_strength=c.get("s1_strength", 0.0),
                    atr_14=c.get("atr_14", 0.0),
                    risk_reward=c.get("risk_reward", 0.0),
                    reason=c.get("reason", ""),
                    factor_scores=c.get("factor_scores", {}),
                    regime=RegimeLabel.from_string(c.get("regime", "neutral")),
                    s1_dist_pct=c.get("distance_to_s1_pct", 0.0),
                    r1_dist_pct=c.get("distance_to_r1_pct", 0.0),
                    diagnostics=SignalDiagnostics(
                        generation_time_ms=(time.perf_counter() - start_time) * 1000,
                        feature_count=len(c.get("factor_scores", {})),
                    ),
                    metadata=SignalMetadata(
                        expected_holding_period_bars=20,
                        cost_sensitivity=0.5,
                        regime_preference=RegimeLabel.NEUTRAL,
                    ),
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"[SignalA] Generation failed: {e}")
            return []

    def validate_input(self, context: SignalContext) -> List[str]:
        """Validate input context."""
        issues = []
        if not context.tickers and not hasattr(self, "_screener"):
            issues.append("No tickers provided and screener unavailable")
        return issues


class SignalB:
    """
    SignalB - Frozen v0 baseline.

    Wraps UnifiedStrategy-based signal generation.
    Uses lesson-based filtering with S/R levels.

    NOT MODIFIABLE - Create SignalB_v1 for changes.
    """

    VERSION = "v0"
    FAMILY = SignalFamily.BASELINE_B

    def __init__(self, config: Optional[SignalBConfig] = None):
        self.config = config or SignalBConfig()
        self._strategy = None

    @property
    def name(self) -> str:
        return f"SignalB_{self.VERSION}"

    @property
    def family(self) -> SignalFamily:
        return self.FAMILY

    @property
    def version(self) -> str:
        return self.VERSION

    def _get_strategy(self):
        """Lazy load strategy."""
        if self._strategy is None:
            from autotrade.signals.unified_strategy import UnifiedStrategy

            self._strategy = UnifiedStrategy(min_score=self.config.min_score)
        return self._strategy

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        """Generate signals using frozen UnifiedStrategy logic."""
        start_time = time.perf_counter()

        try:
            strategy = self._get_strategy()
            screener_results = self._get_screener_candidates(context)

            signals = []
            for c in screener_results:
                if self.config.use_lessons:
                    passes, score, rules = strategy.passes_filter(c)
                    if not passes:
                        continue

                score = c.get("score", c.get("normalized_score", 50.0))
                if score < self.config.min_score:
                    continue

                signal = SignalDecision(
                    ticker=c.get("ticker", ""),
                    action=SignalAction.BUY,
                    signal_strength=score / 100.0,
                    entry_price=c.get("price", 0.0),
                    stop_price=c.get("stop_price", 0.0),
                    target_price=c.get("target_price", 0.0),
                    partial_target_price=c.get("partial_target_price", 0.0),
                    score=score,
                    confidence=score / 100.0,
                    family=self.FAMILY,
                    source=self.name,
                    r1_price=c.get("r1_price", 0.0),
                    r1_strength=c.get("r1_strength", 0.0),
                    s1_price=c.get("s1_price", 0.0),
                    s1_strength=c.get("s1_strength", 0.0),
                    atr_14=c.get("atr_14", 0.0),
                    risk_reward=c.get("risk_reward", 0.0),
                    reason=c.get("reason", ""),
                    factor_scores=c.get("factor_scores", {}),
                    regime=RegimeLabel.from_string(c.get("regime", "neutral")),
                    s1_dist_pct=c.get("distance_to_s1_pct", 0.0),
                    r1_dist_pct=c.get("distance_to_r1_pct", 0.0),
                    diagnostics=SignalDiagnostics(
                        generation_time_ms=(time.perf_counter() - start_time) * 1000,
                        feature_count=len(c.get("factor_scores", {})),
                    ),
                    metadata=SignalMetadata(
                        expected_holding_period_bars=15,
                        cost_sensitivity=0.6,
                        regime_preference=RegimeLabel.BULL,
                    ),
                )
                signals.append(signal)

            return signals

        except Exception as e:
            logger.error(f"[SignalB] Generation failed: {e}")
            return []

    def _get_screener_candidates(self, context: SignalContext) -> List[Dict]:
        """Get candidates from screener for lesson filtering."""
        try:
            from autotrade.signals.screener_v2 import get_entry_candidates

            return get_entry_candidates(
                max_candidates=self.config.max_candidates * 2,
                symbols=context.tickers or None,
                exclude_symbols=context.exclude_tickers,
                log_samples=False,
            )
        except Exception as e:
            logger.warning(f"[SignalB] Could not get screener candidates: {e}")
            return []

    def validate_input(self, context: SignalContext) -> List[str]:
        """Validate input context."""
        issues = []
        if not context.tickers and not hasattr(self, "_screener"):
            issues.append("No tickers provided")
        return issues


def get_baseline_signal_a(config: Optional[SignalAConfig] = None) -> SignalA:
    """Get SignalA instance."""
    return SignalA(config)


def get_baseline_signal_b(config: Optional[SignalBConfig] = None) -> SignalB:
    """Get SignalB instance."""
    return SignalB(config)


def register_baseline_signals() -> None:
    """Register both baseline signals with the global registry."""
    registry = get_signal_registry()

    signal_a = SignalA()
    signal_b = SignalB()

    registry.register_model(signal_a, enabled=True)
    registry.register_model(signal_b, enabled=True)

    logger.info(
        f"[Signal Registry] Registered baselines: {signal_a.name}, {signal_b.name}"
    )


def get_baseline_signals() -> List[SignalModel]:
    """Get list of all baseline signal models."""
    return [SignalA(), SignalB()]


def generate_baseline_batch(
    context: SignalContext,
    include_a: bool = True,
    include_b: bool = True,
) -> SignalBatch:
    """
    Generate signals from baseline models.

    Args:
        context: Signal context
        include_a: Include SignalA
        include_b: Include SignalB

    Returns:
        Combined signal batch
    """
    from datetime import datetime
    import uuid

    batch = SignalBatch(
        batch_id=str(uuid.uuid4()),
        generated_at=datetime.now(),
    )

    all_signals = []

    if include_a:
        signal_a = SignalA()
        signals_a = signal_a.generate(context)
        all_signals.extend(signals_a)
        batch.family_counts[SignalFamily.BASELINE_A] = len(signals_a)

    if include_b:
        signal_b = SignalB()
        signals_b = signal_b.generate(context)
        all_signals.extend(signals_b)
        batch.family_counts[SignalFamily.BASELINE_B] = len(signals_b)

    batch.signals = all_signals
    batch.total_generated = len(all_signals)

    return batch


def validate_baseline_output(signals: List[SignalDecision]) -> List[str]:
    """
    Validate baseline signal outputs.

    This is the golden test for baseline freeze.
    """
    issues = []

    for i, signal in enumerate(signals):
        if not signal.ticker:
            issues.append(f"Signal {i}: Missing ticker")

        if signal.family not in (SignalFamily.BASELINE_A, SignalFamily.BASELINE_B):
            issues.append(f"Signal {i}: Wrong family {signal.family}")

        if not signal.source.startswith("Signal"):
            issues.append(f"Signal {i}: Invalid source {signal.source}")

        if signal.signal_strength < -1.0 or signal.signal_strength > 1.0:
            issues.append(f"Signal {i}: Invalid strength {signal.signal_strength}")

        if signal.score < 0 or signal.score > 100:
            issues.append(f"Signal {i}: Invalid score {signal.score}")

    return issues
