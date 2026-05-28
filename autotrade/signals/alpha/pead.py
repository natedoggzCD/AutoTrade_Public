
"""
Post-Earnings Announcement Drift (PEAD) Alpha
=============================================

Alpha family: TS_MOMENTUM

Implements:
- PEADAlphaSource: Identifies stocks with recent positive earnings surprises.
"""

from __future__ import annotations

import logging
import time
from typing import List

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

class PEADAlphaSource(SignalModel):
    """
    Post-Earnings Announcement Drift (PEAD).
    Identifies stocks with recent positive earnings surprises.
    """

    VERSION = "v1"

    def __init__(self, lookback_days: int = 3, min_surprise: float = 5.0):
        self.lookback_days = lookback_days
        self.min_surprise = min_surprise
        self.db = FinancialDB()

    @property
    def name(self) -> str:
        return "PEADAlphaSource_v1"

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
        self.logger.info(f"[{self.name}] Scanning for earnings surprises in last {self.lookback_days} days (min={self.min_surprise}%)")    

        surprises = self.db.get_recent_surprises(days=self.lookback_days, min_surprise=self.min_surprise)
        if not surprises:
            self.logger.info(f"[{self.name}] No recent surprises found.")
            return []

        signals = []
        for row in surprises:
            ticker = row["ticker"]
            
            # Filter by tickers in context if provided
            if context.tickers and ticker not in context.tickers:
                continue
                
            surprise = row["surprise_pct"]

            decision = SignalDecision(
                ticker=ticker,
                action=SignalAction.BUY,
                signal_strength=min(1.0, 0.5 + (surprise / 100.0)),
                score=min(100.0, 60.0 + surprise),
                family=self.family,
                source=self.name,
                reason=f"PEAD: Positive earnings surprise of {surprise:.1f}% on {row['earnings_date']}",
                strategy_id="pead_momentum_v1",
                strategy_params={
                    "stop_atr": 2.0,
                    "target_atr": 4.0,
                    "max_hold_days": 5
                },
                diagnostics=SignalDiagnostics(
                    generation_time_ms=(time.perf_counter() - start_time) * 1000,
                ),
                metadata=SignalMetadata(
                    expected_holding_period_bars=5,
                    cost_sensitivity=0.4,
                    regime_preference=RegimeLabel.TREND,
                    tags=["pead", "earnings"],
                )
            )
            signals.append(decision)

        self.logger.info(f"[{self.name}] Generated {len(signals)} potential signals")
        return signals

def register_pead_signals() -> None:
    """Register PEAD signals with the global registry."""
    registry = get_signal_registry()
    model = PEADAlphaSource()
    registry.register_model(model, enabled=True)
    logger.info(f"[Signal Registry] Registered: {model.name}")
