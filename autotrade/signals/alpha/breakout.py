"""
Breakout / Volatility Expansion Signals
========================================

Alpha family: breakout

Implements:
- SqueezeExpansionSignal: Squeeze to expansion breakout
- DonchianBreakoutSignal: Donchian channel breakout

Breakout signals identify volatility expansion after consolidation.
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
class BreakoutConfig:
    """Configuration for breakout signals."""

    donchian_window: int = 20
    squeeze_window: int = 120
    squeeze_percentile_threshold: float = 10.0
    breakout_threshold: float = 0.5
    min_confidence: float = 0.3
    signal_threshold: float = 0.1
    max_signals: int = 100


class BaseBreakout(SignalModel):
    """Base class for breakout signals."""

    VERSION = "v1"

    def __init__(self, config: Optional[BreakoutConfig] = None):
        self.config = config or BreakoutConfig()

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.BREAKOUT

    @property
    def version(self) -> str:
        return self.VERSION

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

    def _create_signal(
        self,
        ticker: str,
        signal_strength: float,
        confidence: float,
        price_data: pd.DataFrame,
        metadata: Optional[SignalMetadata] = None,
    ) -> SignalDecision:
        start_time = time.perf_counter()

        action = SignalAction.BUY if signal_strength > 0 else SignalAction.WATCH

        entry_price = price_data["close"].iloc[-1] if len(price_data) > 0 else 0.0

        atr_14 = self._calculate_atr(price_data, 14)

        stop_pct = 0.02
        target_pct = 0.05
        stop_price = (
            entry_price - atr_14 * 2 if atr_14 > 0 else entry_price * (1 - stop_pct)
        )
        target_price = (
            entry_price + atr_14 * 4 if atr_14 > 0 else entry_price * (1 + target_pct)
        )

        risk_reward = (
            (target_price - entry_price) / (entry_price - stop_price)
            if entry_price > stop_price
            else 0.0
        )

        return SignalDecision(
            ticker=ticker,
            action=action,
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
            reason=f"{self.name}: strength={signal_strength:.3f}, conf={confidence:.2f}",
            diagnostics=SignalDiagnostics(
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
            ),
            metadata=metadata
            or SignalMetadata(
                expected_holding_period_bars=10,
                cost_sensitivity=0.7,
                regime_preference=RegimeLabel.TREND,
                tags=["breakout"],
            ),
        )

    def validate_input(self, context: SignalContext) -> List[str]:
        issues = []
        if not context.tickers and context.price_data is None:
            issues.append("No tickers or price data provided")
        return issues


class SqueezeExpansionSignal(BaseBreakout):
    """
    Squeeze to expansion breakout signal.

    Identifies when volatility contracts (squeeze) and then expands,
    generating directional momentum signals.
    """

    @property
    def name(self) -> str:
        return "SqueezeExpansion_v1"

    def _calculate_squeeze(
        self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
    ) -> Dict[str, float]:
        """Calculate squeeze metrics."""
        if len(high) < window or len(low) < window:
            return {"squeeze_ratio": 1.0, "atr_ratio": 1.0}

        donchian_upper = high.rolling(window).max().iloc[-1]
        donchian_lower = low.rolling(window).min().iloc[-1]
        donchian_mid = (donchian_upper + donchian_lower) / 2

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        current_atr = tr.iloc[-1]
        avg_atr = tr.rolling(window).mean().iloc[-1]

        squeeze_ratio = (
            (donchian_upper - donchian_lower) / current_atr if current_atr > 0 else 1.0
        )
        atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        return {
            "squeeze_ratio": squeeze_ratio,
            "atr_ratio": atr_ratio,
            "position": (close.iloc[-1] - donchian_lower)
            / (donchian_upper - donchian_lower)
            if donchian_upper > donchian_lower
            else 0.5,
        }

    def _calculate_percentile_rank(self, values: pd.Series, window: int) -> float:
        """Calculate percentile rank of current value."""
        if len(values) < window:
            return 50.0

        current = values.iloc[-1]
        history = values.iloc[-window:-1]

        if len(history) == 0:
            return 50.0

        percentile = (history < current).sum() / len(history) * 100
        return percentile

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            required = ["high", "low", "close"]
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

                    if len(ticker_data) < self.config.squeeze_window:
                        continue

                    high = ticker_data["high"]
                    low = ticker_data["low"]
                    close = ticker_data["close"]

                    squeeze_metrics = self._calculate_squeeze(
                        high, low, close, self.config.donchian_window
                    )

                    squeeze_ratio = squeeze_metrics["squeeze_ratio"]
                    atr_ratio = squeeze_metrics["atr_ratio"]

                    atr_series = abs(high - low)
                    squeeze_percentile = self._calculate_percentile_rank(
                        atr_series, self.config.squeeze_window
                    )

                    in_squeeze = (
                        squeeze_percentile < self.config.squeeze_percentile_threshold
                    )
                    expanding = atr_ratio > 1.5

                    if in_squeeze:
                        continue

                    if not expanding:
                        continue

                    position = squeeze_metrics.get("position", 0.5)
                    signal_strength = (position - 0.5) * 2

                    confidence = min(1.0, atr_ratio / 3)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=10,
                        cost_sensitivity=0.7,
                        regime_preference=RegimeLabel.TREND,
                        tags=["squeeze_expansion", "volatility_breakout"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
                    )
                    signal.factor_scores["squeeze_ratio"] = squeeze_ratio
                    signal.factor_scores["atr_ratio"] = atr_ratio
                    signal.factor_scores["squeeze_percentile"] = squeeze_percentile
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


class DonchianBreakoutSignal(BaseBreakout):
    """
    Donchian channel breakout signal.

    Identifies breakouts above/below Donchian channel boundaries.
    """

    @property
    def name(self) -> str:
        return f"DonchianBreakout_{self.config.donchian_window}_v1"

    def _calculate_donchian(
        self, high: pd.Series, low: pd.Series, window: int
    ) -> Dict[str, float]:
        """Calculate Donchian channel metrics."""
        if len(high) < window or len(low) < window:
            return {"upper": 0, "lower": 0, "mid": 0, "position": 0.5}

        upper = high.rolling(window).max().iloc[-1]
        lower = low.rolling(window).min().iloc[-1]
        mid = (upper + lower) / 2

        current_close = close.iloc[-1]
        position = (current_close - lower) / (upper - lower) if upper > lower else 0.5

        return {
            "upper": upper,
            "lower": lower,
            "mid": mid,
            "position": position,
            "range": upper - lower,
        }

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            required = ["high", "low", "close"]
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

                    if len(ticker_data) < self.config.donchian_window + 10:
                        continue

                    high = ticker_data["high"]
                    low = ticker_data["low"]
                    close = ticker_data["close"]

                    donchian = self._calculate_donchian(
                        high, low, self.config.donchian_window
                    )

                    position = donchian["position"]

                    above_upper = position > (1 - self.config.breakout_threshold)
                    below_lower = position < self.config.breakout_threshold

                    if not above_upper and not below_lower:
                        continue

                    signal_strength = (position - 0.5) * 2
                    signal_strength = np.clip(signal_strength, -1.0, 1.0)

                    range_normalized = (
                        donchian.get("range", 0) / close.iloc[-1]
                        if close.iloc[-1] > 0
                        else 0
                    )
                    confidence = min(1.0, range_normalized * 10)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=self.config.donchian_window,
                        cost_sensitivity=0.6,
                        regime_preference=RegimeLabel.TREND,
                        tags=["donchian_breakout", "channel_breakout"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
                    )
                    signal.factor_scores["donchian_position"] = position
                    signal.factor_scores["donchian_range"] = donchian.get("range", 0)
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


def register_breakout_signals() -> None:
    """Register all breakout signals with the global registry."""
    registry = get_signal_registry()

    models = [
        SqueezeExpansionSignal(),
        DonchianBreakoutSignal(BreakoutConfig(donchian_window=20)),
        DonchianBreakoutSignal(BreakoutConfig(donchian_window=30)),
    ]

    for model in models:
        registry.register_model(model, enabled=True)
        logger.info(f"[Signal Registry] Registered: {model.name}")
