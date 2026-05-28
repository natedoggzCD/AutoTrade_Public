"""
Cross-Sectional Momentum Signals
================================

Alpha family: xs_momentum

Implements:
- XSMomentumSignal: Rank-and-rebalance signal over universe slices

Cross-sectional momentum ranks stocks by recent returns and generates
signals based on relative strength within a sector or universe.
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
class XSMomentumConfig:
    """Configuration for cross-sectional momentum."""

    lookback_periods: List[int] = None
    min_confidence: float = 0.3
    signal_threshold: float = 0.15
    top_pct: float = 0.2
    max_signals: int = 50
    rebalance_frequency: str = "monthly"

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [20, 60]


class XSMomentumSignal(SignalModel):
    """
    Cross-sectional momentum using rank-and-rebalance.

    Ranks stocks by momentum within universe slices and generates
    signals for top performers.
    """

    VERSION = "v1"

    def __init__(self, config: Optional[XSMomentumConfig] = None):
        self.config = config or XSMomentumConfig()

    @property
    def name(self) -> str:
        return "XSMomentum_v1"

    @property
    def family(self) -> SignalFamily:
        return SignalFamily.XS_MOMENTUM

    @property
    def version(self) -> str:
        return self.VERSION

    def _calculate_returns(
        self, prices: pd.Series, periods: List[int]
    ) -> Dict[int, float]:
        """Calculate returns for multiple lookback periods."""
        returns = {}
        for period in periods:
            if len(prices) >= period:
                current = prices.iloc[-1]
                past = prices.iloc[-period]
                if past != 0:
                    returns[period] = (current - past) / past
        return returns

    def _rank_by_momentum(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Rank stocks by cross-sectional momentum."""
        if price_data is None or len(price_data) == 0:
            return pd.DataFrame()

        results = []

        tickers = price_data.get("ticker", pd.Series(price_data.index)).unique()

        for ticker in tickers:
            try:
                ticker_data = price_data
                if "ticker" in price_data.columns:
                    ticker_data = price_data[price_data["ticker"] == ticker]

                if len(ticker_data) < max(self.config.lookback_periods):
                    continue

                close = ticker_data["close"]
                returns = self._calculate_returns(close, self.config.lookback_periods)

                if not returns:
                    continue

                avg_return = np.mean(list(returns.values()))

                results.append(
                    {
                        "ticker": ticker,
                        "momentum_score": avg_return,
                        "returns_20d": returns.get(20, 0),
                        "returns_60d": returns.get(60, 0),
                        "close": close.iloc[-1],
                    }
                )
            except Exception:
                continue

        if not results:
            return pd.DataFrame()

        momentum_scores = pd.Series(
            [float(row["momentum_score"]) for row in results], dtype=float
        )
        ranked_results = []
        for row, rank in zip(results, momentum_scores.rank(pct=True).tolist()):
            updated = dict(row)
            updated["rank"] = float(rank)
            ranked_results.append(updated)

        return pd.DataFrame(ranked_results)

    def _create_signal(
        self,
        ticker: str,
        rank: float,
        momentum_score: float,
        price_data: pd.DataFrame,
    ) -> SignalDecision:
        start_time = time.perf_counter()

        signal_strength = (rank - 0.5) * 2
        confidence = min(1.0, abs(momentum_score) * 10)

        action = SignalAction.BUY if signal_strength > 0 else SignalAction.WATCH

        entry_price = price_data["close"].iloc[-1] if len(price_data) > 0 else 0.0

        atr_14 = 0.0
        if "high" in price_data.columns and "low" in price_data.columns:
            high = price_data["high"]
            low = price_data["low"]
            close = price_data["close"]
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else 0.0

        stop_pct = 0.025
        target_pct = 0.05
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
            score=rank * 100,
            confidence=confidence,
            family=self.family,
            source=self.name,
            atr_14=atr_14,
            risk_reward=risk_reward,
            reason=f"XS momentum: rank={rank:.2%}, momentum={momentum_score:.2%}",
            factor_scores={
                "momentum_score": momentum_score,
                "cross_sectional_rank": rank,
            },
            diagnostics=SignalDiagnostics(
                generation_time_ms=(time.perf_counter() - start_time) * 1000,
            ),
            metadata=SignalMetadata(
                expected_holding_period_bars=20,
                cost_sensitivity=0.4,
                regime_preference=RegimeLabel.TREND,
                tags=["xs_momentum", "rank_rebalance"],
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

            ranked = self._rank_by_momentum(price_data)

            if ranked.empty:
                logger.warning(f"[{self.name}] No ranked stocks available")
                return signals

            top_threshold = 1.0 - self.config.top_pct
            top_stocks = ranked[ranked["rank"] >= top_threshold].sort_values(
                "rank", ascending=False
            )

            top_stocks = top_stocks.head(self.config.max_signals)

            for _, row in top_stocks.iterrows():
                try:
                    ticker = row["ticker"]
                    ticker_data = price_data
                    if "ticker" in price_data.columns:
                        ticker_data = price_data[price_data["ticker"] == ticker]

                    if len(ticker_data) == 0:
                        continue

                    signal = self._create_signal(
                        ticker=ticker,
                        rank=row["rank"],
                        momentum_score=row["momentum_score"],
                        price_data=ticker_data,
                    )
                    signals.append(signal)

                except Exception as e:
                    logger.debug(f"[{self.name}] Error creating signal: {e}")
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


def register_xs_momentum_signals() -> None:
    """Register cross-sectional momentum signals with the global registry."""
    registry = get_signal_registry()

    model = XSMomentumSignal()
    registry.register_model(model, enabled=True)
    logger.info(f"[Signal Registry] Registered: {model.name}")
