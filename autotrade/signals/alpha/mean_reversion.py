"""
Mean Reversion Signals
=======================

Alpha family: mean_reversion

Implements:
- RSISignal: RSI(2/3) stretch/snapback signals
- DownDayExhaustionSignal: Down-day exhaustion with volume confirmation

Mean reversion signals identify oversold conditions expecting price to revert.
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
class MeanReversionConfig:
    """Configuration for mean reversion signals."""

    rsi_window: int = 2
    rsi_entry_threshold: float = 10.0
    rsi_exit_threshold: float = 50.0
    volume_confirmation: bool = True
    volume_ma_period: int = 20
    min_confidence: float = 0.3
    signal_threshold: float = 0.1
    max_signals: int = 100


class BaseMeanReversion(SignalModel):
    """Base class for mean reversion signals."""

    VERSION = "v1"

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        self.config = config or MeanReversionConfig()

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.MEAN_REVERSION

    @property
    def version(self) -> str:
        return self.VERSION

    def _calculate_rsi(self, prices: pd.Series, period: int = 2) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0

        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        avg_gain = gains.rolling(period).mean().iloc[-1]
        avg_loss = losses.rolling(period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi if not np.isnan(rsi) else 50.0

    def _calculate_volume_ratio(self, volume: pd.Series, period: int = 20) -> float:
        """Calculate volume ratio vs moving average."""
        if len(volume) < period:
            return 1.0

        current_vol = volume.iloc[-1]
        avg_vol = volume.rolling(period).mean().iloc[-1]

        if avg_vol == 0:
            return 1.0

        return current_vol / avg_vol

    def _create_signal(
        self,
        ticker: str,
        signal_strength: float,
        confidence: float,
        price_data: pd.DataFrame,
        metadata: Optional[SignalMetadata] = None,
    ) -> SignalDecision:
        start_time = time.perf_counter()

        action = SignalAction.BUY if signal_strength < 0 else SignalAction.WATCH

        entry_price = price_data["close"].iloc[-1] if len(price_data) > 0 else 0.0

        atr_14 = 0.0
        if all(c in price_data.columns for c in ["high", "low", "close"]):
            high = price_data["high"]
            low = price_data["low"]
            close = price_data["close"]
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else 0.0

        stop_pct = 0.015
        target_pct = 0.03
        stop_price = entry_price * (1 - stop_pct)
        target_price = entry_price * (1 + target_pct)

        risk_reward = target_pct / stop_pct if stop_pct > 0 else 0.0

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
                expected_holding_period_bars=5,
                cost_sensitivity=0.6,
                regime_preference=RegimeLabel.CHOP,
                tags=["mean_reversion"],
            ),
        )

    def validate_input(self, context: SignalContext) -> List[str]:
        issues = []
        if not context.tickers and context.price_data is None:
            issues.append("No tickers or price data provided")
        return issues


class RSISignal(BaseMeanReversion):
    """
    RSI-based mean reversion signal.

    Identifies oversold (RSI < threshold) as buying opportunities
    and overbought (RSI > 100 - threshold) as selling opportunities.
    """

    @property
    def name(self) -> str:
        return f"RSI_{self.config.rsi_window}_Signal_v1"

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

                    if len(ticker_data) < self.config.rsi_window + 10:
                        continue

                    close = ticker_data["close"]
                    rsi = self._calculate_rsi(close, self.config.rsi_window)

                    oversold = rsi < self.config.rsi_entry_threshold
                    overbought = rsi > (100 - self.config.rsi_entry_threshold)

                    if not oversold and not overbought:
                        continue

                    signal_strength = (50 - rsi) / 50
                    signal_strength = np.clip(signal_strength, -1.0, 1.0)

                    raw_confidence = (
                        self.config.rsi_entry_threshold - abs(rsi - 50)
                    ) / self.config.rsi_entry_threshold
                    confidence = float(np.clip(raw_confidence, 0.0, 1.0))

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=self.config.rsi_window * 3,
                        cost_sensitivity=0.6,
                        regime_preference=RegimeLabel.CHOP,
                        tags=["rsi", f"rsi_{self.config.rsi_window}"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
                    )
                    signal.factor_scores["rsi"] = rsi
                    signal.factor_scores["rsi_window"] = self.config.rsi_window
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


class DownDayExhaustionSignal(BaseMeanReversion):
    """
    Down-day exhaustion signal with volume confirmation.

    Identifies when a stock has declined for multiple consecutive days
    with high volume, suggesting exhaustion and potential reversal.
    """

    CONSECUTIVE_DAYS = 3

    @property
    def name(self) -> str:
        return "DownDayExhaustion_v1"

    def _calculate_consecutive_down_days(
        self, close: pd.Series, window: int = 5
    ) -> int:
        """Calculate consecutive down days."""
        if len(close) < window:
            return 0

        returns = close.pct_change()
        down_days = (returns < 0).astype(int)

        consecutive = 0
        for i in range(len(down_days) - 1, -1, -1):
            if down_days.iloc[i] == 1:
                consecutive += 1
            else:
                break

        return consecutive

    def generate(self, context: SignalContext) -> List[SignalDecision]:
        start_time = time.perf_counter()
        signals = []

        try:
            price_data = context.price_data

            if price_data is None or not isinstance(price_data, pd.DataFrame):
                logger.warning(f"[{self.name}] No price data available")
                return signals

            required = ["close", "volume"]
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

                    if len(ticker_data) < 10:
                        continue

                    close = ticker_data["close"]
                    volume = ticker_data["volume"]

                    consecutive_down = self._calculate_consecutive_down_days(close)

                    if consecutive_down < self.CONSECUTIVE_DAYS:
                        continue

                    volume_ratio = self._calculate_volume_ratio(
                        volume, self.config.volume_ma_period
                    )

                    if self.config.volume_confirmation and volume_ratio < 1.0:
                        continue

                    recent_return = (
                        close.iloc[-1] - close.iloc[-consecutive_down - 1]
                    ) / close.iloc[-consecutive_down - 1]

                    signal_strength = np.clip(recent_return * 5, -1.0, 0.0)

                    confidence = min(1.0, consecutive_down / 5 * volume_ratio)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    metadata = SignalMetadata(
                        expected_holding_period_bars=5,
                        cost_sensitivity=0.7,
                        regime_preference=RegimeLabel.CHOP,
                        tags=["down_day_exhaustion", "volume_confirmed"],
                    )

                    signal = self._create_signal(
                        ticker,
                        signal_strength,
                        confidence,
                        ticker_data,
                        metadata,
                    )
                    signal.factor_scores["consecutive_down_days"] = consecutive_down
                    signal.factor_scores["volume_ratio"] = volume_ratio
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


def register_mean_reversion_signals() -> None:
    """Register all mean reversion signals with the global registry."""
    registry = get_signal_registry()

    models = [
        RSISignal(MeanReversionConfig(rsi_window=2)),
        RSISignal(MeanReversionConfig(rsi_window=3)),
        DownDayExhaustionSignal(),
    ]

    for model in models:
        registry.register_model(model, enabled=True)
        logger.info(f"[Signal Registry] Registered: {model.name}")
