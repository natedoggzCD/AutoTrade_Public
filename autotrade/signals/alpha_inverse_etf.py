"""Legacy compatibility wrapper for inverse ETF alpha imports.

Older runtime paths import `autotrade.signals.alpha_inverse_etf` and expect:
- `InverseETFAlphaSource(parent_logger=...)`
- `generate_signals(...)` instead of the canonical `generate(SignalContext)`

The canonical implementation lives in `autotrade.signals.alpha.inverse_etf`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from autotrade.signals.alpha.inverse_etf import (
    InverseETFAlphaSource as _InverseETFAlphaSource,
)
from autotrade.signals.contracts import RegimeLabel, SignalContext, SignalDecision

logger = logging.getLogger(__name__)


class InverseETFAlphaSource(_InverseETFAlphaSource):
    """Backward-compatible inverse ETF source wrapper."""

    def __init__(self, parent_logger: Optional[logging.Logger] = None):
        super().__init__()
        self.logger = parent_logger or logger

    def generate_signals(
        self,
        regime: RegimeLabel | str = RegimeLabel.NEUTRAL,
        tickers: Optional[List[str]] = None,
    ) -> List[SignalDecision]:
        if isinstance(regime, str):
            try:
                regime = RegimeLabel(regime.lower())
            except ValueError:
                regime = RegimeLabel.NEUTRAL
        context = SignalContext(
            tickers=list(tickers or []),
            price_data=pd.DataFrame(),
            regime=regime,
        )
        return self.generate(context)


__all__ = ["InverseETFAlphaSource"]
