
"""
Fundamental and Event-Driven Alpha Signals
==========================================

Alpha family: TS_MOMENTUM (Primary classification)

Implements:
- SqueezeAlphaSource: Short Squeeze Detection

These signals use fundamental and event data from FinancialDB.
"""

from __future__ import annotations

import logging
import time
from typing import List

from autotrade.signals.alpha.pead import PEADAlphaSource as _PEADAlphaSource
from autotrade.signals.contracts import (
    SignalContext,
    SignalDecision,
    SignalDiagnostics,
    SignalMetadata,
    SignalFamily,
    RegimeLabel,
    SignalAction,
)
from autotrade.signals.interfaces import SignalModel, get_signal_registry
from autotrade.utils.financial_db import FinancialDB

logger = logging.getLogger(__name__)


class PEADAlphaSource(_PEADAlphaSource):
    """
    Backward-compatible export for callers still importing PEAD from
    `autotrade.signals.alpha.fundamental`.
    """

    def __init__(self, lookback_days: int = 3, min_surprise: float = 5.0):
        super().__init__(lookback_days=lookback_days, min_surprise=min_surprise)
        self.db = FinancialDB()


class SqueezeAlphaSource(SignalModel):
    """
    Short Squeeze Detection.
    Identifies stocks with high short interest.
    """

    VERSION = "v1"

    def __init__(self, min_short_pct: float = 15.0):
        self.min_short_pct = min_short_pct
        self.db = FinancialDB()

    @property
    def name(self) -> str:
        return "SqueezeAlphaSource_v1"

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.TS_MOMENTUM

    @property
    def version(self) -> str:
        return self.VERSION

    def validate_input(self, context: SignalContext) -> List[str]:
        return []

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        self.logger = logger
        self.logger.info(f"[{self.name}] Scanning for high short interest (min={self.min_short_pct}%)")

        candidates = self.db.get_high_short_interest_tickers(min_short_pct=self.min_short_pct)
        if not candidates:
            self.logger.info(f"[{self.name}] No high short interest candidates found.")
            return []

        signals = []
        for row in candidates:
            ticker = row["ticker"]
            
            # Filter by tickers in context if provided
            if context.tickers and ticker not in context.tickers:
                continue
                
            short_pct = row["short_pct"]

            decision = SignalDecision(
                ticker=ticker,
                action=SignalAction.BUY,
                signal_strength=min(1.0, 0.4 + (short_pct / 100.0)),
                score=min(100.0, 50.0 + short_pct),
                family=self.family,
                source=self.name,
                reason=f"SQUEEZE: High short interest of {short_pct:.1f}%",
                strategy_id="short_squeeze_v1",
                strategy_params={
                    "stop_atr": 1.5,
                    "target_atr": 5.0,
                    "max_hold_days": 3
                },
                diagnostics=SignalDiagnostics(
                    generation_time_ms=(time.perf_counter() - start_time) * 1000,
                ),
                metadata=SignalMetadata(
                    expected_holding_period_bars=3,
                    cost_sensitivity=0.6,
                    regime_preference=RegimeLabel.CRISIS,
                    tags=["squeeze", "short_interest"],
                )
            )
            signals.append(decision)

        self.logger.info(f"[{self.name}] Generated {len(signals)} potential signals")
        return signals


def register_fundamental_signals() -> None:
    """Register all fundamental signals with the global registry."""
    registry = get_signal_registry()

    models = [
        SqueezeAlphaSource(),
    ]

    for model in models:
        registry.register_model(model, enabled=True)
        logger.info(f"[Signal Registry] Registered: {model.name}")
