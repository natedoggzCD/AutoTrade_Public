"""
Time-Series Momentum Signals
=============================

Alpha family: ts_momentum

Implements:
- TSMomentum12_1: 12-bar vs 1-bar momentum
- TSMomentum6_1: 6-bar vs 1-bar momentum
- MATrendSignal: MA trend with breakout confirmation

These signals use daily and hourly OHLCV data to generate directional
time-series momentum signals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

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
class TSMomentumConfig:
    """Configuration for time-series momentum signals."""

    lookback_long_bars: int = 252
    lookback_short_bars: int = 126
    exclude_recent_bars: int = 21
    min_confidence: float = 0.3
    signal_threshold: float = 0.1
    max_signals: int = 200


class BaseTSMomentum(SignalModel):
    """Base class for time-series momentum signals."""

    VERSION = "v1"

    def __init__(self, config: Optional[TSMomentumConfig] = None):
        self.config = config or TSMomentumConfig()

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.TS_MOMENTUM

    @property
    def version(self) -> str:
        return self.VERSION

    def _calculate_momentum(
        self, prices: pd.Series, lookback: int, exclude: int
    ) -> float:
        """Calculate momentum as percent change over lookback period."""
        if len(prices) < lookback + exclude:
            return 0.0

        start_idx = exclude
        end_idx = lookback + exclude

        if end_idx > len(prices):
            return 0.0

        current = prices.iloc[-1]
        past = prices.iloc[-end_idx]

        if past == 0:
            return 0.0

        return (current - past) / past

    def _calculate_ma_trend(
        self, prices: pd.Series, fast_period: int = 20, slow_period: int = 50
    ) -> float:
        """Calculate MA trend strength (-1 to 1)."""
        if len(prices) < slow_period:
            return 0.0

        fast_ma = prices.rolling(fast_period).mean().iloc[-1]
        slow_ma = prices.rolling(slow_period).mean().iloc[-1]

        if slow_ma == 0:
            return 0.0

        return (fast_ma - slow_ma) / slow_ma

    def _calculate_breakout_confirmation(
        self, highs: pd.Series, lows: pd.Series, close: pd.Series, window: int = 20
    ) -> float:
        """Calculate breakout confirmation score."""
        if len(highs) < window or len(lows) < window:
            return 0.0

        recent_high = highs.iloc[-window:].max()
        recent_low = lows.iloc[-window:].min()
        current_close = close.iloc[-1]

        range_size = recent_high - recent_low
        if range_size == 0:
            return 0.0

        position = (current_close - recent_low) / range_size

        return (position - 0.5) * 2

    def _create_signal(
        self,
        ticker: str,
        signal_strength: float,
        confidence: float,
        price_data: pd.DataFrame,
        metadata: Optional[SignalMetadata] = None,
    ) -> SignalDecision:
        """Create a SignalDecision from computed values."""
        start_time = time.perf_counter()

        action = SignalAction.BUY if signal_strength > 0 else SignalAction.SELL
        if abs(signal_strength) < self.config.signal_threshold:
            action = SignalAction.WATCH

        entry_price = price_data["close"].iloc[-1] if len(price_data) > 0 else 0.0

        atr_14 = self._calculate_atr(price_data, 14) if len(price_data) >= 14 else 0.0

        stop_pct = 0.02
        target_pct = 0.04
        stop_price = (
            entry_price * (1 - stop_pct)
            if signal_strength > 0
            else entry_price * (1 + stop_pct)
        )
        target_price = (
            entry_price * (1 + target_pct)
            if signal_strength > 0
            else entry_price * (1 - target_pct)
        )

        risk_reward = target_pct / stop_pct if stop_pct > 0 else 0.0

        factor_scores = {
            "momentum_raw": signal_strength,
            "confidence": confidence,
        }

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
            reason=f"{self.name}: momentum={signal_strength:.3f}, confidence={confidence:.2f}",
            factor_scores=factor_scores,
            diagnostics=SignalDiagnostics(
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
            ),
            metadata=metadata
            or SignalMetadata(
                expected_holding_period_bars=20,
                cost_sensitivity=0.5,
                regime_preference=RegimeLabel.TREND,
            ),
        )

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

    def validate_input(self, context: SignalContext) -> List[str]:
        issues = []
        if not context.tickers:
            issues.append("No tickers provided")
        if context.price_data is None:
            issues.append("No price data provided")
        return issues


class TSMomentum12_1(BaseTSMomentum):
    """
    12-bar vs 1-bar time-series momentum.

    Compares 12-bar return to 1-bar return to identify
    sustained directional momentum.
    """

    LOOKBACK_LONG = 12
    LOOKBACK_SHORT = 1

    @property
    def name(self) -> str:
        return "TSMomentum12_1_v1"

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            if "close" not in price_data.columns:
                logger.warning(f"[{self.name}] Missing close column")
                return signals

            if context.tickers:
                tickers_to_process = context.tickers
            else:
                tickers_to_process = [
                    context.config_override.get("default_ticker", "UNKNOWN")
                ]

            for ticker in tickers_to_process:
                try:
                    ticker_data = price_data
                    if "ticker" in price_data.columns:
                        ticker_data = price_data[price_data["ticker"] == ticker]

                    if (
                        len(ticker_data)
                        < self.LOOKBACK_LONG + self.config.exclude_recent_bars
                    ):
                        continue

                    close = ticker_data["close"]

                    momentum_12 = self._calculate_momentum(
                        close, self.LOOKBACK_LONG, self.config.exclude_recent_bars
                    )
                    momentum_1 = self._calculate_momentum(close, self.LOOKBACK_SHORT, 0)

                    signal_strength = momentum_12 - momentum_1 * 0.5
                    signal_strength = np.clip(signal_strength, -1.0, 1.0)

                    confidence = min(1.0, abs(signal_strength) * 2)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=self.LOOKBACK_LONG,
                        cost_sensitivity=0.4,
                        regime_preference=RegimeLabel.TREND,
                        tags=["ts_momentum", "12_1"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
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


class TSMomentum6_1(BaseTSMomentum):
    """
    6-bar vs 1-bar time-series momentum.

    Shorter-term momentum for more responsive signals.
    """

    LOOKBACK_LONG = 6
    LOOKBACK_SHORT = 1

    @property
    def name(self) -> str:
        return "TSMomentum6_1_v1"

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            if "close" not in price_data.columns:
                logger.warning(f"[{self.name}] Missing close column")
                return signals

            if context.tickers:
                tickers_to_process = context.tickers
            else:
                tickers_to_process = [
                    context.config_override.get("default_ticker", "UNKNOWN")
                ]

            for ticker in tickers_to_process:
                try:
                    ticker_data = price_data
                    if "ticker" in price_data.columns:
                        ticker_data = price_data[price_data["ticker"] == ticker]

                    if len(ticker_data) < self.LOOKBACK_LONG + 5:
                        continue

                    close = ticker_data["close"]

                    momentum_6 = self._calculate_momentum(close, self.LOOKBACK_LONG, 0)
                    momentum_1 = self._calculate_momentum(close, self.LOOKBACK_SHORT, 0)

                    signal_strength = momentum_6 - momentum_1 * 0.3
                    signal_strength = np.clip(signal_strength, -1.0, 1.0)

                    confidence = min(1.0, abs(signal_strength) * 2)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=self.LOOKBACK_LONG,
                        cost_sensitivity=0.3,
                        regime_preference=RegimeLabel.TREND,
                        tags=["ts_momentum", "6_1"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
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


class MATrendSignal(BaseTSMomentum):
    """
    Moving Average trend signal with breakout confirmation.

    Uses fast vs slow MA crossover and breakout confirmation
    to generate trend-following signals.
    """

    FAST_PERIOD = 20
    SLOW_PERIOD = 50

    @property
    def name(self) -> str:
        return "MATrendSignal_v1"

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            required_cols = ["close", "high", "low"]
            if not all(c in price_data.columns for c in required_cols):
                logger.warning(f"[{self.name}] Missing required columns")
                return signals

            if context.tickers:
                tickers_to_process = context.tickers
            else:
                tickers_to_process = [
                    context.config_override.get("default_ticker", "UNKNOWN")
                ]

            for ticker in tickers_to_process:
                try:
                    ticker_data = price_data
                    if "ticker" in price_data.columns:
                        ticker_data = price_data[price_data["ticker"] == ticker]

                    if len(ticker_data) < self.SLOW_PERIOD + 20:
                        continue

                    close = ticker_data["close"]
                    high = ticker_data["high"]
                    low = ticker_data["low"]

                    ma_trend = self._calculate_ma_trend(
                        close, self.FAST_PERIOD, self.SLOW_PERIOD
                    )
                    breakout = self._calculate_breakout_confirmation(
                        high, low, close, window=20
                    )

                    signal_strength = ma_trend * 0.7 + breakout * 0.3
                    signal_strength = np.clip(signal_strength, -1.0, 1.0)

                    confidence = min(1.0, abs(signal_strength) * 1.5)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=30,
                        cost_sensitivity=0.5,
                        regime_preference=RegimeLabel.TREND,
                        tags=["ma_trend", "breakout"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
                    )
                    signal.factor_scores["ma_trend"] = ma_trend
                    signal.factor_scores["breakout"] = breakout
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


def register_ts_momentum_signals() -> None:
    """Register all time-series momentum signals with the global registry."""
    registry = get_signal_registry()

    models = [
        TSMomentum12_1(),
        TSMomentum6_1(),
        MATrendSignal(),
    ]

    for model in models:
        registry.register_model(model, enabled=True)
        logger.info(f"[Signal Registry] Registered: {model.name}")
