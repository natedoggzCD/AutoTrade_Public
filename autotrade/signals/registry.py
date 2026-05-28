"""
Signal registry wiring helpers.

Provides config-driven registration and enable/disable controls
for baseline and alpha signal models.
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

from autotrade.signals.alpha import register_all_alpha_signals
from autotrade.signals.baseline_signals import register_baseline_signals
from autotrade.signals.contracts import SignalFamily
from autotrade.signals.interfaces import (
    SignalModel,
    SignalRegistry,
    get_signal_registry,
    reset_signal_registry,
)
from autotrade.signals.regime_router import create_regime_router

ALPHA_FAMILIES = {
    SignalFamily.TS_MOMENTUM,
    SignalFamily.XS_MOMENTUM,
    SignalFamily.MEAN_REVERSION,
    SignalFamily.BREAKOUT,
    SignalFamily.PULLBACK,
}


def _normalize_enabled_families(families_enabled: Optional[Iterable[str]]) -> Set[str]:
    if not families_enabled:
        return set()
    return {str(f).strip().lower() for f in families_enabled if str(f).strip()}


def register_signal_zoo(
    families_enabled: Optional[Iterable[str]] = None,
    include_baselines: bool = True,
    include_router: bool = True,
) -> SignalRegistry:
    """
    Register baseline and alpha models into the global registry.

    Args:
        families_enabled: Optional family names (e.g. ["ts_momentum", "breakout"]).
            When provided, alpha models not in this list are disabled.
        include_baselines: Whether to register SignalA_v0 and SignalB_v0.
        include_router: Whether to register the regime router.

    Returns:
        Populated global SignalRegistry.
    """
    reset_signal_registry()
    registry = get_signal_registry()

    if include_baselines:
        register_baseline_signals()
    register_all_alpha_signals()

    enabled_families = _normalize_enabled_families(families_enabled)
    if enabled_families:
        for model in registry.get_all_models():
            if model.family in ALPHA_FAMILIES:
                registry.set_enabled(model.name, model.family.value in enabled_families)

    if include_router:
        router = create_regime_router()
        registry.register_router(router)

    return registry


def register_signal_zoo_from_config() -> SignalRegistry:
    """Register signal zoo using `config.signal_generation.families_enabled`."""
    from config.config_loader import get_config

    config = get_config()
    families_enabled = getattr(config.signal_generation, "families_enabled", [])
    return register_signal_zoo(families_enabled=families_enabled)


def get_enabled_signal_models(
    families_enabled: Optional[Iterable[str]] = None,
    include_baselines: bool = True,
) -> list[SignalModel]:
    """Build registry and return enabled models only."""
    registry = register_signal_zoo(
        families_enabled=families_enabled,
        include_baselines=include_baselines,
    )
    return registry.get_enabled_models()
