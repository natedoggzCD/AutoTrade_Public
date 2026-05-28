"""
Pullback Continuation Signals
=============================

Alpha family: pullback

Implements:
- PullbackContinuationSignal: Daily trend context + hourly pullback reclaim trigger

Pullback continuation signals identify entries in the direction of the
daily trend when price pulls back and reclaims a key level.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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


@dataclass
class PullbackConfig:
    """Configuration for pullback continuation signals."""

    trend_lookback: int = 50
    pullback_lookback: int = 10
    reclaim_threshold: float = 0.02
    min_trend_strength: float = 0.03
    max_pullback_depth: float = 0.08
    use_hourly: bool = True
    min_confidence: float = 0.3
    signal_threshold: float = 0.1
    max_signals: int = 100


class PullbackContinuationSignal(SignalModel):
    """
    Pullback continuation signal.

    Identifies when:
    1. Daily trend is strongly upward (>3% over lookback)
    2. Price has pulled back (5-8%)
    3. Price reclaims moving average or prior high

    This is a trend-following strategy that buys dips in uptrends.
    """

    VERSION = "v1"

    def __init__(self, config: Optional[PullbackConfig] = None):
        self.config = config or PullbackConfig()

    @property
    def name(self) -> str:
        return "PullbackContinuation_v1"

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.PULLBACK

    @property
    def version(self) -> str:
        return self.VERSION

    def _calculate_trend_strength(self, close: pd.Series, lookback: int) -> float:
        """Calculate trend strength as percent change."""
        if len(close) < lookback:
            return 0.0

        current = close.iloc[-1]
        past = close.iloc[-lookback]

        if past == 0:
            return 0.0

        return (current - past) / past

    def _calculate_pullback_depth(
        self, close: pd.Series, high: pd.Series, lookback: int
    ) -> float:
        """Calculate how deep the current pullback is."""
        if len(close) < lookback:
            return 0.0

        period_high = high.iloc[-lookback:].max()
        current_close = close.iloc[-1]

        if period_high == 0:
            return 0.0

        return (period_high - current_close) / period_high

    def _calculate_ma_reclaim(
        self, close: pd.Series, ma_period: int = 20
    ) -> Dict[str, float]:
        """Calculate if price is reclaiming moving average."""
        if len(close) < ma_period:
            return {"reclaiming": False, "distance_pct": 0.0}

        ma = close.rolling(ma_period).mean()
        current_ma = ma.iloc[-1]
        current_close = close.iloc[-1]

        if current_ma == 0:
            return {"reclaiming": False, "distance_pct": 0.0}

        distance_pct = (current_close - current_ma) / current_ma

        recent_below = (close.iloc[-5:-1] < ma.iloc[-5:-1]).all()
        now_above = current_close > current_ma

        reclaiming = recent_below and now_above and distance_pct > 0

        return {
            "reclaiming": reclaiming,
            "distance_pct": distance_pct,
            "ma_value": current_ma,
        }

    def _calculate_support_test(
        self, low: pd.Series, lookback: int = 20
    ) -> Dict[str, float]:
        """Calculate if price is testing support."""
        if len(low) < lookback:
            return {"at_support": False, "support_level": 0.0}

        recent_low = low.iloc[-lookback:].min()
        current_low = low.iloc[-1]

        tolerance = recent_low * 0.01

        at_support = abs(current_low - recent_low) <= tolerance

        return {
            "at_support": at_support,
            "support_level": recent_low,
            "distance_to_support": (current_low - recent_low) / recent_low
            if recent_low > 0
            else 0,
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(df) < period:
            return 0.0

        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]

        return atr if not np.isnan(atr) else 0.0

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            required = ["close", "high", "low"]
            if not all(c in price_data.columns for c in required):
                logger.warning(f"[{self.name}] Missing required columns")
                return signals

            if context.tickers:
                tickers_to_process = context.tickers
            else:
                tickers_to_process = (
                    price_data["ticker"].unique()
                    if "ticker" in price_data.columns
                    else [context.config_override.get("default_ticker", "UNKNOWN")]
                )

            for ticker in tickers_to_process:
                try:
                    ticker_data = price_data
                    if "ticker" in price_data.columns:
                        ticker_data = price_data[price_data["ticker"] == ticker]

                    if len(ticker_data) < self.config.trend_lookback + 10:
                        continue

                    close = ticker_data["close"]
                    high = ticker_data["high"]
                    low = ticker_data["low"]

                    trend = self._calculate_trend_strength(
                        close, self.config.trend_lookback
                    )

                    if trend < self.config.min_trend_strength:
                        continue

                    pullback = self._calculate_pullback_depth(
                        close, high, self.config.pullback_lookback
                    )

                    if (
                        pullback < self.config.reclaim_threshold
                        or pullback > self.config.max_pullback_depth
                    ):
                        continue

                    ma_reclaim = self._calculate_ma_reclaim(close, 20)
                    support_test = self._calculate_support_test(low, 20)

                    valid_entry = ma_reclaim["reclaiming"] or support_test["at_support"]

                    if not valid_entry:
                        continue

                    signal_strength = pullback / self.config.max_pullback_depth
                    signal_strength = np.clip(signal_strength, 0.0, 1.0)

                    confidence = min(1.0, trend * 10 + signal_strength * 0.5)

                    if signal_strength < self.config.signal_threshold:
                        continue

                    atr_14 = self._calculate_atr(ticker_data, 14)
                    entry_price = close.iloc[-1]
                    stop_price = (
                        entry_price - atr_14 * 2 if atr_14 > 0 else entry_price * 0.97
                    )
                    target_price = (
                        entry_price + atr_14 * 4 if atr_14 > 0 else entry_price * 1.06
                    )

                    risk_reward = (
                        (target_price - entry_price) / (entry_price - stop_price)
                        if entry_price > stop_price
                        else 0.0
                    )

                    metadata = SignalMetadata(
                        expected_holding_period_bars=15,
                        cost_sensitivity=0.5,
                        regime_preference=RegimeLabel.TREND,
                        tags=["pullback", "trend_continuation"],
                    )

                    signal = SignalDecision(
                        ticker=ticker,
                        action=SignalAction.BUY,
                        signal_strength=signal_strength,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        score=min(100, max(0, 50 + signal_strength * 50)),
                        confidence=confidence,
                        family=self.family,
                        source=self.name,
                        atr_14=atr_14,
                        risk_reward=risk_reward,
                        reason=f"Pullback continuation: trend={trend:.2%}, pullback={pullback:.2%}",
                        factor_scores={
                            "trend_strength": trend,
                            "pullback_depth": pullback,
                            "ma_reclaiming": ma_reclaim["reclaiming"],
                            "at_support": support_test["at_support"],
                        },
                        diagnostics=SignalDiagnostics(
                            generation_time_ms=(time.perf_counter() - start_time)
                            * 1000,
                        ),
                        metadata=metadata,
                    )
                    signals.append(signal)

                except Exception as e:
                    logger.debug(f"[{self.name}] Error processing {ticker}: {e}")
                    continue

        except Exception as e:
            logger.error(f"[{self.name}] Generation failed: {e}")

        logger.info(
            f"[{self.name}] Generated {len(signals)} signals in "
            f"{(time.perf_counter() - start_time) * 1000:.1f}ms"
        )
        return signals

    def validate_input(self, context: SignalContext) -> List[str]:
        issues = []
        if not context.tickers and context.price_data is None:
            issues.append("No tickers or price data provided")
        return issues


def register_pullback_signals() -> None:
    """Register pullback continuation signals with the global registry."""
    registry = get_signal_registry()

    model = PullbackContinuationSignal()
    registry.register_model(model, enabled=True)
    logger.info(f"[Signal Registry] Registered: {model.name}")
