"""
Legacy compatibility wrapper for squeeze alpha imports.

Older runtime paths import `autotrade.signals.alpha_squeeze` and expect:
- `SqueezeAlphaSource(parent_logger=...)`
- `generate_signals()` with no explicit SignalContext

The canonical implementation now lives in `autotrade.signals.alpha.fundamental`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from autotrade.signals.alpha.fundamental import SqueezeAlphaSource as _SqueezeAlphaSource
from autotrade.signals.contracts import RegimeLabel, SignalContext, SignalDecision

logger = logging.getLogger(__name__)


class SqueezeAlphaSource(_SqueezeAlphaSource):
    """Backward-compatible squeeze source wrapper."""

    def __init__(
        self,
        min_short_pct: float = 15.0,
        parent_logger: Optional[logging.Logger] = None,
    ):
        super().__init__(min_short_pct=min_short_pct)
        self.logger = parent_logger or logger

    def generate_signals(
        self,
        tickers: Optional[List[str]] = None,
        regime: RegimeLabel = RegimeLabel.NEUTRAL,
    ) -> List[SignalDecision]:
        context = SignalContext(
            tickers=list(tickers or []),
            price_data=pd.DataFrame(),
            regime=regime,
        )
        return self.generate(context)


__all__ = ["SqueezeAlphaSource"]
