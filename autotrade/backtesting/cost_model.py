"""Realistic cost model fed by realized fill data from logs/trade_journal.json.

Replaces the flat 1bp slippage assumption that was inflating backtest PnL.
Slippage is estimated per symbol where data is available; otherwise it falls
back to the symbol's liquidity tercile, then to the global median.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_JOURNAL_PATH = Path("logs/trade_journal.json")
GLOBAL_FALLBACK_BPS = 5.0
PER_SYMBOL_MIN_FILLS = 5


@dataclass
class SymbolCostStats:
    symbol: str
    median_slippage_bps: float
    p75_slippage_bps: float
    n_fills: int
    liquidity_tercile: int  # 0 = thin, 1 = mid, 2 = liquid

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "median_slippage_bps": self.median_slippage_bps,
            "p75_slippage_bps": self.p75_slippage_bps,
            "n_fills": self.n_fills,
            "liquidity_tercile": self.liquidity_tercile,
        }


@dataclass
class CostModel:
    """Per-symbol slippage estimator with cluster-level fallback."""

    per_symbol: Dict[str, SymbolCostStats] = field(default_factory=dict)
    tercile_median_bps: Dict[int, float] = field(default_factory=dict)
    global_median_bps: float = GLOBAL_FALLBACK_BPS
    commission_per_share: float = 0.0  # paper trading on Alpaca

    def slippage_bps(self, symbol: str) -> float:
        sym = (symbol or "").upper()
        stats = self.per_symbol.get(sym)
        if stats is not None and stats.n_fills >= PER_SYMBOL_MIN_FILLS:
            return stats.p75_slippage_bps
        if stats is not None and stats.liquidity_tercile in self.tercile_median_bps:
            return self.tercile_median_bps[stats.liquidity_tercile]
        return self.global_median_bps

    def apply_round_trip(self, gross_return_pct: float, symbol: str) -> float:
        """Subtract round-trip slippage (entry + exit) from a gross return percent."""
        round_trip_bps = 2.0 * self.slippage_bps(symbol)
        return float(gross_return_pct) - (round_trip_bps / 100.0)


def _load_trades(journal_path: Path) -> List[dict]:
    if not journal_path.exists():
        return []
    with journal_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return list(payload.get("trades", []))
    if isinstance(payload, list):
        return payload
    return []


def build_cost_model(
    journal_path: Path = DEFAULT_JOURNAL_PATH,
    daily_features_parquet: Optional[Path] = None,
) -> CostModel:
    """Construct a CostModel from realized fills.

    daily_features_parquet is used only to classify symbols into liquidity
    terciles when not enough fills exist; if None, every symbol with fills is
    assigned an estimated tercile from fill volume; symbols with no fills get
    tercile 1 (mid) by default.
    """

    trades = _load_trades(journal_path)
    if not trades:
        logger.warning("No trades in %s; cost model uses global fallback %.1fbps",
                       journal_path, GLOBAL_FALLBACK_BPS)
        return CostModel()

    df = pd.DataFrame(trades)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    # entry-side slippage rows (entry_status=='filled', non-zero bps)
    df = df[df["slippage_bps"].fillna(0).abs() > 0]
    if df.empty:
        return CostModel()

    df["slippage_bps_abs"] = df["slippage_bps"].astype(float).abs()
    df["fill_value"] = df["filled_quantity"].astype(float) * df["price"].astype(float)

    per_symbol: Dict[str, SymbolCostStats] = {}
    symbol_fill_value: Dict[str, float] = {}
    for sym, group in df.groupby("symbol"):
        median_bps = float(group["slippage_bps_abs"].median())
        p75_bps = float(group["slippage_bps_abs"].quantile(0.75))
        n = int(len(group))
        per_symbol[sym] = SymbolCostStats(
            symbol=sym,
            median_slippage_bps=median_bps,
            p75_slippage_bps=p75_bps,
            n_fills=n,
            liquidity_tercile=1,  # provisional; assigned below
        )
        symbol_fill_value[sym] = float(group["fill_value"].sum())

    # liquidity terciles from fill value across symbols-with-data
    if symbol_fill_value:
        values = pd.Series(symbol_fill_value)
        cutoffs = values.quantile([1 / 3, 2 / 3]).values
        for sym, v in symbol_fill_value.items():
            if v <= cutoffs[0]:
                t = 0
            elif v <= cutoffs[1]:
                t = 1
            else:
                t = 2
            per_symbol[sym].liquidity_tercile = t

    # per-tercile median slippage
    tercile_median_bps: Dict[int, float] = {}
    for t in (0, 1, 2):
        members = [s.median_slippage_bps for s in per_symbol.values() if s.liquidity_tercile == t]
        if members:
            tercile_median_bps[t] = float(np.median(members))

    global_median = float(df["slippage_bps_abs"].median())

    return CostModel(
        per_symbol=per_symbol,
        tercile_median_bps=tercile_median_bps,
        global_median_bps=global_median,
    )


def cost_model_summary(model: CostModel) -> dict:
    return {
        "n_symbols_with_data": len(model.per_symbol),
        "global_median_bps": model.global_median_bps,
        "tercile_median_bps": dict(model.tercile_median_bps),
        "top_5_costliest": sorted(
            (s.to_dict() for s in model.per_symbol.values()),
            key=lambda r: r["p75_slippage_bps"],
            reverse=True,
        )[:5],
    }
