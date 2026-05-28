"""Theory-driven alpha catalog (replaces grid-sweep strategy factory).

Use `iter_alphas()` to enumerate registered variants. Use `get_alpha(alpha_id)`
to fetch a specific variant. New alpha modules register themselves on import
by appending to `_REGISTRY` (see existing modules for the pattern).
"""

from __future__ import annotations

from typing import Iterator, List

from autotrade.backtesting.alpha_catalog.base import (
    AlphaContext,
    AlphaDefinition,
    AlphaSignalRow,
    empty_signal_frame,
    signal_frame_from_rows,
)

_REGISTRY: List[AlphaDefinition] = []


def register(alpha: AlphaDefinition) -> AlphaDefinition:
    existing = {a.alpha_id for a in _REGISTRY}
    if alpha.alpha_id in existing:
        raise ValueError(f"duplicate alpha_id: {alpha.alpha_id}")
    _REGISTRY.append(alpha)
    return alpha


def iter_alphas() -> Iterator[AlphaDefinition]:
    return iter(_REGISTRY)


def get_alpha(alpha_id: str) -> AlphaDefinition:
    for a in _REGISTRY:
        if a.alpha_id == alpha_id:
            return a
    raise KeyError(alpha_id)


def list_alpha_ids() -> List[str]:
    return [a.alpha_id for a in _REGISTRY]


def alphas_for_regime(regime: str) -> List[AlphaDefinition]:
    return [a for a in _REGISTRY if regime in a.regime_compatibility]


def alphas_for_family(family: str) -> List[AlphaDefinition]:
    return [a for a in _REGISTRY if a.family == family]


# Eagerly import every alpha module so they self-register on package import.
from autotrade.backtesting.alpha_catalog import momentum  # noqa: F401,E402
from autotrade.backtesting.alpha_catalog import trend  # noqa: F401,E402
from autotrade.backtesting.alpha_catalog import mean_reversion  # noqa: F401,E402
from autotrade.backtesting.alpha_catalog import volatility  # noqa: F401,E402
from autotrade.backtesting.alpha_catalog import patterns  # noqa: F401,E402
from autotrade.backtesting.alpha_catalog import cross_sectional  # noqa: F401,E402
from autotrade.backtesting.alpha_catalog import catalysts  # noqa: F401,E402

__all__ = [
    "AlphaContext",
    "AlphaDefinition",
    "AlphaSignalRow",
    "empty_signal_frame",
    "signal_frame_from_rows",
    "register",
    "iter_alphas",
    "get_alpha",
    "list_alpha_ids",
    "alphas_for_regime",
    "alphas_for_family",
]
