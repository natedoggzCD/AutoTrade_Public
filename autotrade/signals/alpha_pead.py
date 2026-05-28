"""
Legacy compatibility wrapper for PEAD alpha imports.

Older runtime paths import `autotrade.signals.alpha_pead` and expect:
- `PEADAlphaSource(parent_logger=...)`
- `generate_signals()` with no explicit SignalContext

The canonical implementation now lives in `autotrade.signals.alpha.pead`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from autotrade.signals.alpha.pead import PEADAlphaSource as _PEADAlphaSource
from autotrade.signals.contracts import RegimeLabel, SignalContext, SignalDecision

logger = logging.getLogger(__name__)


class PEADAlphaSource(_PEADAlphaSource):
    """Backward-compatible PEAD source wrapper."""

    def __init__(
        self,
        lookback_days: int = 3,
        min_surprise: float = 5.0,
        parent_logger: Optional[logging.Logger] = None,
    ):
        super().__init__(lookback_days=lookback_days, min_surprise=min_surprise)
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


__all__ = ["PEADAlphaSource"]
