
"""
Alpha Source: Inverse ETF Hedging.
Generates entry signals for inverse ETFs during market crisis/risk-off regimes.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any

from config.config_loader import get_config
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

logger = logging.getLogger(__name__)

class InverseETFAlphaSource(SignalModel):
    """
    Inverse ETF Hedging signal generator.
    Activated during high-volatility or crisis regimes.
    """

    VERSION = "v1"

    def __init__(self):
        self.config = get_config()

    @property
    def name(self) -> str:
        return "InverseETFAlphaSource_v1"

    @property
    def family(self) -> SignalFamily:
        # We classify as TS_MOMENTUM for general pipeline purposes, 
        # but reason/source indicates it's a hedge.
        return SignalFamily.TS_MOMENTUM

    @property
    def version(self) -> str:
        return self.VERSION

    def validate_input(self, context: SignalContext) -> List[str]:
        return []

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        
        hedge_cfg = getattr(self.config, "inverse_etf_hedging", None)
        if not hedge_cfg or not getattr(hedge_cfg, "enabled", False):
            return []

        current_regime = context.regime or RegimeLabel.NEUTRAL
        allowed = [str(r).lower() for r in getattr(hedge_cfg, "allowed_regimes", [])]
        
        if str(current_regime.value).lower() not in allowed:
            return []

        logger.info(f"[{self.name}] Regime '{current_regime.value}' permits inverse ETF hedging.")

        symbols = getattr(hedge_cfg, "symbols", [])
        if not symbols:
            return []

        signals = []
        for ticker in symbols:
            decision = SignalDecision(
                ticker=ticker,
                action=SignalAction.BUY,
                signal_strength=0.8,
                score=75.0,
                family=self.family,
                source=self.name,
                reason=f"HEDGE: Market regime '{current_regime.value}' triggers inverse ETF exposure.",
                strategy_id="inverse_etf_hedge_v1",
                strategy_params={
                    "stop_atr": 3.0,
                    "target_atr": 5.0,
                    "max_hold_days": getattr(hedge_cfg, "max_hold_days", 3)
                },
                diagnostics=SignalDiagnostics(
                    generation_time_ms=(time.perf_counter() - start_time) * 1000,
                ),
                metadata=SignalMetadata(
                    expected_holding_period_bars=getattr(hedge_cfg, "max_hold_days", 3),
                    cost_sensitivity=0.2,
                    regime_preference=current_regime,
                    tags=["hedge", "inverse_etf"],
                )
            )
            signals.append(decision)

        logger.info(f"[{self.name}] Generated {len(signals)} hedge signals")
        return signals


def register_inverse_etf_signals() -> None:
    """Register inverse ETF hedging signals with the global registry."""
    registry = get_signal_registry()

    model = InverseETFAlphaSource()
    registry.register_model(model, enabled=True)
    logger.info(f"[Signal Registry] Registered: {model.name}")
