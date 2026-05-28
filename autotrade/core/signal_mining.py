from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List


MARKET_CONTEXT_SYMBOLS = {"SPY", "QQQ"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class SignalMiningEngine:
    gap_threshold_pct: float = 5.0
    breadth_threshold_pct: float = 65.0
    volatility_atr_threshold: float = 6.0
    volatility_range_threshold: float = 8.0

    def mine(self, universe_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        discoveries: List[Dict[str, Any]] = []

        for row in universe_rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol or symbol in MARKET_CONTEXT_SYMBOLS:
                continue

            sector = str(row.get("sector", "Unknown") or "Unknown")
            gap_pct = abs(_safe_float(row.get("gap_pct")))
            atr_percent = _safe_float(row.get("atr_percent"))
            intraday_range = _safe_float(row.get("intraday_range"))
            breadth_pct = _safe_float(row.get("sector_breadth_pct"))
            if breadth_pct <= 0.0:
                breadth_score = _safe_float(row.get("breadth_score"))
                if 0.0 < breadth_score <= 1.0:
                    breadth_pct = breadth_score * 100.0
                else:
                    breadth_pct = breadth_score

            if gap_pct >= self.gap_threshold_pct:
                discoveries.append(
                    {
                        "symbol": symbol,
                        "family": "gap",
                        "sector": sector,
                        "score": round(min(100.0, 60.0 + gap_pct * 4.0), 2),
                        "reason": f"Gap {gap_pct:.2f}% exceeded {self.gap_threshold_pct:.2f}%",
                        "metrics": {"gap_pct": gap_pct},
                    }
                )

            if breadth_pct >= self.breadth_threshold_pct:
                discoveries.append(
                    {
                        "symbol": symbol,
                        "family": "breadth",
                        "sector": sector,
                        "score": round(min(100.0, 50.0 + breadth_pct * 0.6), 2),
                        "reason": (
                            f"Sector breadth {breadth_pct:.1f}% exceeded "
                            f"{self.breadth_threshold_pct:.1f}%"
                        ),
                        "metrics": {"sector_breadth_pct": breadth_pct},
                    }
                )

            if (
                atr_percent >= self.volatility_atr_threshold
                or intraday_range >= self.volatility_range_threshold
            ):
                volatility_strength = max(
                    atr_percent / max(self.volatility_atr_threshold, 1.0),
                    intraday_range / max(self.volatility_range_threshold, 1.0),
                )
                discoveries.append(
                    {
                        "symbol": symbol,
                        "family": "volatility",
                        "sector": sector,
                        "score": round(min(100.0, 55.0 + volatility_strength * 15.0), 2),
                        "reason": (
                            "Volatility expansion detected "
                            f"(ATR%={atr_percent:.2f}, range={intraday_range:.2f})"
                        ),
                        "metrics": {
                            "atr_percent": atr_percent,
                            "intraday_range": intraday_range,
                        },
                    }
                )

        discoveries.sort(
            key=lambda item: (float(item.get("score", 0.0)), item.get("symbol", "")),
            reverse=True,
        )
        return discoveries


class DiscoveryRegistry:
    def __init__(self, items: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.items: Dict[str, Dict[str, Any]] = dict(items or {})

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "DiscoveryRegistry":
        registry_state = state.get("discovery_registry", {}) if isinstance(state, dict) else {}
        items = registry_state.get("items", {}) if isinstance(registry_state, dict) else {}
        if not isinstance(items, dict):
            items = {}
        return cls(items=items)

    def record_many(
        self, discoveries: Iterable[Dict[str, Any]], detected_at: str | None = None
    ) -> List[Dict[str, Any]]:
        ts = detected_at or datetime.now().isoformat()
        recorded: List[Dict[str, Any]] = []

        for discovery in discoveries:
            if not isinstance(discovery, dict):
                continue
            symbol = str(discovery.get("symbol", "")).strip().upper()
            family = str(discovery.get("family", "")).strip().lower()
            if not symbol or not family:
                continue

            key = f"{symbol}:{family}"
            existing = self.items.get(key, {})
            occurrences = int(existing.get("occurrences", 0) or 0) + 1
            merged = {
                "symbol": symbol,
                "family": family,
                "sector": discovery.get("sector", existing.get("sector", "Unknown")),
                "score": max(
                    _safe_float(existing.get("score")),
                    _safe_float(discovery.get("score")),
                ),
                "reason": str(discovery.get("reason") or existing.get("reason") or ""),
                "metrics": discovery.get("metrics", existing.get("metrics", {})),
                "validation_status": str(
                    existing.get("validation_status") or "candidate"
                ),
                "source": "signal_mining",
                "first_detected_at": str(existing.get("first_detected_at") or ts),
                "last_detected_at": str(ts),
                "occurrences": occurrences,
            }
            self.items[key] = merged
            recorded.append(merged)

        return recorded

    def to_state(self) -> Dict[str, Any]:
        by_symbol: Dict[str, List[str]] = {}
        for key, item in sorted(self.items.items()):
            symbol = str(item.get("symbol", "")).strip().upper()
            family = str(item.get("family", "")).strip().lower()
            if not symbol or not family:
                continue
            by_symbol.setdefault(symbol, []).append(family)

        for families in by_symbol.values():
            families.sort()

        return {
            "items": dict(self.items),
            "by_symbol": by_symbol,
            "updated_at": datetime.now().isoformat(),
        }
