"""
Pairs / Stat-Arb Signals
========================

Alpha family: pairs

Implements:
- PairsSignal: Spread z-score reversion with pair-selection interface hooks

Pairs trading identifies two correlated securities and trades
the spread when it deviates from its historical relationship.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
class PairsConfig:
    """Configuration for pairs trading signals."""

    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.5
    lookback_period: int = 60
    min_correlation: float = 0.7
    pair_selection_method: str = "sector"
    known_pairs: List[Tuple[str, str]] = field(default_factory=list)
    min_confidence: float = 0.3
    signal_threshold: float = 0.3
    max_signals: int = 50


class PairsSignal(SignalModel):
    """
    Pairs trading signal using spread z-score reversion.

    Identifies when the spread between two correlated securities
    deviates significantly from its historical mean, expecting reversion.
    """

    VERSION = "v1"

    DEFAULT_PAIRS = [
        ("XLF", "XLK"),
        ("XLE", "XOP"),
        ("XLP", "XLY"),
        ("XLB", "XLI"),
        ("XLV", "XHI"),
        ("IWM", "QQQ"),
        ("SMH", "SOXX"),
    ]

    def __init__(self, config: Optional[PairsConfig] = None):
        self.config = config or PairsConfig()
        self._cached_pairs: Dict[Tuple[str, str], Dict] = {}

    @property
    def name(self) -> str:
        return "PairsSignal_v1"

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.PAIRS

    @property
    def version(self) -> str:
        return self.VERSION

    def _calculate_spread(self, price_a: pd.Series, price_b: pd.Series) -> pd.Series:
        """Calculate spread between two price series."""
        return price_a - price_b

    def _calculate_zscore(self, spread: pd.Series, lookback: int = 60) -> float:
        """Calculate z-score of current spread."""
        if len(spread) < lookback:
            return 0.0

        recent_spread = spread.iloc[-lookback:]
        mean = recent_spread.mean()
        std = recent_spread.std()

        if std == 0:
            return 0.0

        current = spread.iloc[-1]
        zscore = (current - mean) / std

        return zscore

    def _calculate_correlation(
        self, price_a: pd.Series, price_b: pd.Series, lookback: int = 60
    ) -> float:
        """Calculate correlation between two price series."""
        if len(price_a) < lookback or len(price_b) < lookback:
            return 0.0

        a = price_a.iloc[-lookback:]
        b = price_b.iloc[-lookback:]

        corr = a.corr(b)
        return corr if not np.isnan(corr) else 0.0

    def _calculate_hedge_ratio(
        self, price_a: pd.Series, price_b: pd.Series, lookback: int = 60
    ) -> float:
        """Calculate optimal hedge ratio using OLS."""
        if len(price_a) < lookback or len(price_b) < lookback:
            return 1.0

        a = price_a.iloc[-lookback:].values
        b = price_b.iloc[-lookback:].values

        try:
            b_reshaped = b.reshape(-1, 1)
            coeffs = np.linalg.lstsq(b_reshaped, a, rcond=None)[0]
            return coeffs[0] if len(coeffs) > 0 else 1.0
        except Exception:
            return 1.0

    def _get_pairs_to_analyze(self, price_data: pd.DataFrame) -> List[Tuple[str, str]]:
        """Get list of pairs to analyze."""
        if self.config.known_pairs:
            return self.config.known_pairs

        return self.DEFAULT_PAIRS

    def _get_ticker_data(
        self, price_data: pd.DataFrame, ticker: str
    ) -> Optional[pd.DataFrame]:
        """Get price data for a specific ticker."""
        if price_data is None or len(price_data) == 0:
            return None

        if "ticker" in price_data.columns:
            return price_data[price_data["ticker"] == ticker]

        if len(price_data) > 1 and "close" in price_data.columns:
            return price_data

        return None

    def _create_signal(
        self,
        ticker: str,
        signal_strength: float,
        confidence: float,
        price_data: pd.DataFrame,
        pair_info: Dict,
    ) -> SignalDecision:
        start_time = time.perf_counter()

        # Phase 2 pairs generator emits long-reversion legs only.
        action = SignalAction.BUY

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
        target_pct = 0.025

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
            score=min(100, max(0, 50 + abs(signal_strength) * 50)),
            confidence=confidence,
            family=self.family,
            source=self.name,
            atr_14=atr_14,
            risk_reward=risk_reward,
            reason=f"Pairs: zscore={pair_info.get('zscore', 0):.2f}, corr={pair_info.get('correlation', 0):.2f}",
            factor_scores={
                "zscore": pair_info.get("zscore", 0),
                "correlation": pair_info.get("correlation", 0),
                "spread": pair_info.get("spread", 0),
            },
            diagnostics=SignalDiagnostics(
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
            ),
            metadata=SignalMetadata(
                expected_holding_period_bars=10,
                cost_sensitivity=0.3,
                regime_preference=RegimeLabel.CHOP,
                tags=["pairs", "stat_arb"],
            ),
        )

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

            pairs = self._get_pairs_to_analyze(price_data)

            for pair in pairs:
                if len(pair) != 2:
                    continue

                ticker_a, ticker_b = pair

                try:
                    data_a = self._get_ticker_data(price_data, ticker_a)
                    data_b = self._get_ticker_data(price_data, ticker_b)

                    if data_a is None or data_b is None:
                        continue

                    if (
                        len(data_a) < self.config.lookback_period
                        or len(data_b) < self.config.lookback_period
                    ):
                        continue

                    close_a = data_a["close"]
                    close_b = data_b["close"]

                    correlation = self._calculate_correlation(
                        close_a, close_b, self.config.lookback_period
                    )

                    if correlation < self.config.min_correlation:
                        continue

                    hedge_ratio = self._calculate_hedge_ratio(
                        close_a, close_b, self.config.lookback_period
                    )

                    spread = self._calculate_spread(close_a, close_b * hedge_ratio)
                    zscore = self._calculate_zscore(spread, self.config.lookback_period)

                    if abs(zscore) < self.config.zscore_entry_threshold:
                        continue

                    spread_direction = -1 if zscore > 0 else 1
                    signal_strength = spread_direction * min(1.0, abs(zscore) / 3.0)

                    confidence = min(1.0, correlation * abs(zscore) / 2)

                    if abs(signal_strength) < self.config.signal_threshold:
                        continue

                    pair_info = {
                        "zscore": zscore,
                        "correlation": correlation,
                        "spread": spread.iloc[-1],
                        "hedge_ratio": hedge_ratio,
                        "ticker_a": ticker_a,
                        "ticker_b": ticker_b,
                    }

                    if zscore > 0:
                        signal = self._create_signal(
                            ticker_b,
                            abs(signal_strength),
                            confidence,
                            data_b,
                            pair_info,
                        )
                        signal.reason = (
                            f"Pairs: Long {ticker_b} vs {ticker_a} (spread overvalued)"
                        )
                    else:
                        signal = self._create_signal(
                            ticker_a,
                            abs(signal_strength),
                            confidence,
                            data_a,
                            pair_info,
                        )
                        signal.reason = (
                            f"Pairs: Long {ticker_a} vs {ticker_b} (spread undervalued)"
                        )

                    signals.append(signal)

                except Exception as e:
                    logger.debug(f"[{self.name}] Error processing pair {pair}: {e}")
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
        if context.price_data is None:
            issues.append("No price data provided")
        return issues


def register_pairs_signals() -> None:
    """Register pairs trading signals with the global registry."""
    registry = get_signal_registry()

    model = PairsSignal()
    registry.register_model(model, enabled=True)
    logger.info(f"[Signal Registry] Registered: {model.name}")
