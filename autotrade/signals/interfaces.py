"""
Signal Interfaces
==================
Protocol definitions for signal generation components.

These define the contracts that signal models, post-processors,
and routers must implement.
"""

from __future__ import annotations

from typing import Protocol, List, Dict, Any, Optional, runtime_checkable
from datetime import datetime

from autotrade.signals.contracts import (
    SignalContext,
    SignalDecision,
    SignalBatch,
    SignalFamily,
    RegimeLabel,
)


@runtime_checkable
class SignalModel(Protocol):
    """
    Protocol for signal generation models.

    Any signal model must implement this interface.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this model."""
        ...

    @property
    def family(self) -> SignalFamily:
        """Alpha family this model belongs to."""
        ...

    @property
    def version(self) -> str:
        """Model version for tracking."""
        ...

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        """
        Generate signals for the given context.

        Args:
            context: Input context with tickers and market data

        Returns:
            List of signal decisions
        """
        ...

    def validate_input(self, context: SignalContext) -> List[str]:
        """
        Validate input context before generation.

        Returns:
            List of validation issues (empty if valid)
        """
        ...


@runtime_checkable
class SignalPostProcessor(Protocol):
    """
    Protocol for signal post-processing.

    Post-processors can filter, transform, or enhance signals.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this processor."""
        ...

    def process(
        self, signals: List[SignalDecision], context: SignalContext
    ) -> List[SignalDecision]:
        """
        Process signals.

        Args:
            signals: Input signals
            context: Original context

        Returns:
            Processed signals
        """
        ...

    def filter_reason(self, signal: SignalDecision) -> Optional[str]:
        """
        If signal is filtered, return reason.
        If signal passes, return None.
        """
        ...


@runtime_checkable
class SignalRouter(Protocol):
    """
    Protocol for regime/trend-aware signal routing.

    Routers can direct signals to different strategies based on market conditions.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this router."""
        ...

    @property
    def default_regime(self) -> RegimeLabel:
        """Default regime when routing fails."""
        ...

    def route(
        self,
        signals: List[SignalDecision],
        regime: RegimeLabel,
        confidence: float = 0.5,
    ) -> Dict[RegimeLabel, List[SignalDecision]]:
        """
        Route signals based on regime.

        Args:
            signals: Input signals
            regime: Current market regime
            confidence: Confidence in regime estimate

        Returns:
            Dict mapping regimes to signal lists
        """
        ...

    def get_regime_for_ticker(self, ticker: str, context: SignalContext) -> RegimeLabel:
        """
        Determine regime for a specific ticker.

        Args:
            ticker: Stock symbol
            context: Market context

        Returns:
            Estimated regime for this ticker
        """
        ...


@runtime_checkable
class SignalCalibrator(Protocol):
    """
    Protocol for expected-value calibration.

    Calibrators adjust signal scores based on historical performance.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this calibrator."""
        ...

    def calibrate(
        self, signals: List[SignalDecision], lookback_bars: int = 252
    ) -> List[SignalDecision]:
        """
        Calibrate signal scores using historical data.

        Args:
            signals: Input signals
            lookback_bars: Number of bars for calibration

        Returns:
            Signals with adjusted scores
        """
        ...

    def get_ev_score(
        self, signal: SignalDecision, historical_returns: List[float]
    ) -> float:
        """
        Calculate expected value score.

        Args:
            signal: Signal to evaluate
            historical_returns: Historical return series

        Returns:
            Expected value score
        """
        ...


class SignalRegistry:
    """
    Central registry for signal models.

    Provides lookup and config-driven enable/disable.
    """

    def __init__(self):
        self._models: Dict[str, SignalModel] = {}
        self._postprocessors: Dict[str, SignalPostProcessor] = {}
        self._routers: Dict[str, SignalRouter] = {}
        self._calibrators: Dict[str, SignalCalibrator] = {}

        self._enabled_models: Dict[str, bool] = {}

    def register_model(self, model: SignalModel, enabled: bool = True) -> None:
        """Register a signal model."""
        self._models[model.name] = model
        self._enabled_models[model.name] = enabled

    def register_postprocessor(self, processor: SignalPostProcessor) -> None:
        """Register a post-processor."""
        self._postprocessors[processor.name] = processor

    def register_router(self, router: SignalRouter) -> None:
        """Register a router."""
        self._routers[router.name] = router

    def register_calibrator(self, calibrator: SignalCalibrator) -> None:
        """Register a calibrator."""
        self._calibrators[calibrator.name] = calibrator

    def get_model(self, name: str) -> Optional[SignalModel]:
        """Get a model by name."""
        return self._models.get(name)

    def get_enabled_models(self) -> List[SignalModel]:
        """Get all enabled models."""
        return [
            m for m in self._models.values() if self._enabled_models.get(m.name, False)
        ]

    def get_all_models(self) -> List[SignalModel]:
        """Get all registered models."""
        return list(self._models.values())

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a model."""
        if name in self._models:
            self._enabled_models[name] = enabled

    def is_enabled(self, name: str) -> bool:
        """Check if a model is enabled."""
        return self._enabled_models.get(name, False)

    def get_models_by_family(self, family: SignalFamily) -> List[SignalModel]:
        """Get all models in a family."""
        return [m for m in self._models.values() if m.family == family]

    def get_postprocessor(self, name: str) -> Optional[SignalPostProcessor]:
        """Get a post-processor by name."""
        return self._postprocessors.get(name)

    def get_all_postprocessors(self) -> List[SignalPostProcessor]:
        """Get all registered post-processors."""
        return list(self._postprocessors.values())

    def get_router(self, name: str) -> Optional[SignalRouter]:
        """Get a router by name."""
        return self._routers.get(name)

    def get_default_router(self) -> Optional[SignalRouter]:
        """Get the default router (first registered)."""
        if self._routers:
            return next(iter(self._routers.values()))
        return None

    def get_calibrator(self, name: str) -> Optional[SignalCalibrator]:
        """Get a calibrator by name."""
        return self._calibrators.get(name)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_models": len(self._models),
            "enabled_models": sum(1 for v in self._enabled_models.values() if v),
            "families": list(set(m.family.value for m in self._models.values())),
            "postprocessors": list(self._postprocessors.keys()),
            "routers": list(self._routers.keys()),
            "calibrators": list(self._calibrators.keys()),
        }


# Global registry instance
_global_registry: Optional[SignalRegistry] = None


def get_signal_registry() -> SignalRegistry:
    """Get the global signal registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = SignalRegistry()
    return _global_registry


def reset_signal_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
