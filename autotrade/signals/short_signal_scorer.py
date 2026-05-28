"""Dry-run stock short signal scoring.

The scorer mirrors long-side risk math: a valid short has entry below stop,
target below entry, risk_per_share = stop - entry, and target_per_share =
entry - target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ShortSignal:
    ticker: str
    score: float
    entry: float
    stop: float
    target: float
    risk_per_share: float
    target_per_share: float
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "risk_per_share": self.risk_per_share,
            "target_per_share": self.target_per_share,
            "reasons": list(self.reasons),
            "side": "short",
        }


class ShortSignalScorer:
    """Score bearish-symmetric setups from row-like market features."""

    def __init__(self, min_score: float = 55.0):
        self.min_score = float(min_score)

    @staticmethod
    def _float(row: Dict[str, Any], *names: str, default: float = 0.0) -> float:
        for name in names:
            value = row.get(name)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default

    def score_row(self, row: Dict[str, Any]) -> Optional[ShortSignal]:
        ticker = str(row.get("ticker") or row.get("symbol") or "").upper().strip()
        if not ticker:
            return None
        close = self._float(row, "close", "price", "entry")
        if close <= 0:
            return None
        atr_14 = max(
            self._float(row, "atr_14", "atr", default=close * 0.02), close * 0.01
        )
        ema20 = self._float(row, "ema_20", "ema20")
        ema50 = self._float(row, "ema_50", "ema50")
        vwap = self._float(row, "vwap")
        rsi = self._float(row, "rsi_14", "rsi", default=50.0)
        volume_ratio = self._float(row, "volume_ratio", "relative_volume", default=1.0)
        day_change_pct = self._float(row, "day_change_pct", "pct_change", default=0.0)
        news_sentiment = str(row.get("news_sentiment") or "").lower()

        score = 0.0
        reasons: List[str] = []
        if ema20 > 0 and close < ema20:
            score += 18.0
            reasons.append("ema20_loss")
        if ema50 > 0 and close < ema50:
            score += 16.0
            reasons.append("ema50_loss")
        if vwap > 0 and close < vwap:
            score += 12.0
            reasons.append("below_vwap")
        if volume_ratio >= 1.5 and day_change_pct <= -1.0:
            score += 18.0
            reasons.append("breakdown_volume")
        if 25.0 <= rsi <= 50.0:
            score += 10.0
            reasons.append("bearish_momentum_rsi")
        elif rsi < 25.0:
            score -= 6.0
            reasons.append("oversold_bounce_risk")
        if any(
            token in news_sentiment for token in ("offering", "dilution", "downgrade")
        ):
            score += 14.0
            reasons.append("negative_catalyst")

        score = max(0.0, min(100.0, score))
        if score < self.min_score:
            return None

        entry = close
        stop = round(entry + max(atr_14 * 1.5, entry * 0.02), 2)
        target = round(max(0.01, entry - max(atr_14 * 3.0, entry * 0.04)), 2)
        risk_per_share = round(stop - entry, 4)
        target_per_share = round(entry - target, 4)
        if risk_per_share <= 0 or target_per_share <= 0:
            return None
        return ShortSignal(
            ticker=ticker,
            score=round(score, 1),
            entry=round(entry, 2),
            stop=stop,
            target=target,
            risk_per_share=risk_per_share,
            target_per_share=target_per_share,
            reasons=reasons,
        )

    def score_universe(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        signals = [sig for row in rows if (sig := self.score_row(dict(row or {})))]
        signals.sort(key=lambda sig: sig.score, reverse=True)
        return [sig.to_dict() for sig in signals]
