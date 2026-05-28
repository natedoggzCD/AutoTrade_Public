"""Live signal generation for alpha-catalog strategy rows.

The strategy lab promotes per-symbol best-alpha picks as rows like:

    {
      "strategy_name": "alpha_catalog__gap_continuation_5pct",
      "setup_type": "gap_continuation_5pct",
      "alpha_metadata": {
        "alpha_id": "gap_continuation_5pct",
        "family": "catalyst",
        "params": {"gap": 0.05, "hold": 3},
        "regime_compatibility": ["TREND", "VOLATILE"],
        ...
      },
      "metrics": {...},
      ...
    }

Legacy consumers expected a ``strategy_definition`` dict with explicit
``entry``/``exit`` rules. Alpha-catalog rows do not carry one — their logic
lives in the generator callable registered in
``autotrade.backtesting.alpha_catalog.*``. This module:

- detects the alpha-catalog row shape,
- dispatches on ``alpha_id`` to the registered generator at live signal time,
- enforces regime compatibility,
- and returns a legacy-shape signal dict the rest of the system can consume.

The module is intentionally additive: legacy rows continue to flow through
the unchanged path in ``autotrade.signals.strategy_pool``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from autotrade.backtesting.alpha_catalog import (
    AlphaContext,
    get_alpha,
    list_alpha_ids,
)

logger = logging.getLogger(__name__)


# Map live RegimeLabel values (lowercase, see autotrade.signals.contracts) to
# the uppercase tags the alpha catalog declares in regime_compatibility.
_LIVE_TO_CATALOG_REGIME = {
    "trend": "TREND",
    "chop": "CHOP",
    "crisis": "VOLATILE",
    "risk_off": "RISK_OFF_OK",
    "neutral": "NEUTRAL",
    "volatile": "VOLATILE",
    "rotation": "ROTATION",
    "risk_on": "TREND",
    "sideways": "CHOP",
}


def _normalize_regime(regime: Optional[str]) -> str:
    """Translate a live regime label to a catalog regime tag.

    Unknown labels fall through as upper-cased so callers that already use the
    catalog vocabulary work directly.
    """
    if not regime:
        return "NEUTRAL"
    key = str(regime).strip().lower()
    return _LIVE_TO_CATALOG_REGIME.get(key, str(regime).strip().upper())


def is_alpha_catalog_row(row: Dict[str, Any]) -> bool:
    """Detect the alpha-catalog row shape.

    True iff the row carries an ``alpha_metadata.alpha_id`` and does NOT have a
    populated ``strategy_definition`` (legacy rows may carry both shapes during
    transition; the legacy path takes precedence when both exist).
    """
    if not isinstance(row, dict):
        return False
    meta = row.get("alpha_metadata")
    if not isinstance(meta, dict) or not meta.get("alpha_id"):
        return False
    defn = row.get("strategy_definition")
    if isinstance(defn, dict) and defn:
        return False
    return True


def regime_compatible(row: Dict[str, Any], current_regime: Optional[str]) -> bool:
    """Return True when the row's regime_compatibility includes current_regime.

    Empty/missing regime_compatibility is treated as "all regimes" — a defensive
    fallback for older rows that pre-date the regime tag.
    """
    meta = row.get("alpha_metadata") or {}
    compat = meta.get("regime_compatibility") or []
    if not compat:
        return True
    target = _normalize_regime(current_regime)
    allowed = {str(r).strip().upper() for r in compat if str(r).strip()}
    return target in allowed


def _build_signal_dict(
    row: Dict[str, Any],
    symbol: str,
    last_signal_row: pd.Series,
    last_bar: pd.Series,
) -> Dict[str, Any]:
    """Assemble the legacy-shape signal dict expected by downstream consumers."""
    meta = row.get("alpha_metadata") or {}
    alpha_id = str(meta.get("alpha_id") or row.get("setup_type") or "")
    params = meta.get("params") or {}
    hold = int(
        params.get("hold")
        or params.get("max_hold")
        or params.get("max_hold_days")
        or 5
    )
    metrics = row.get("metrics") or {}

    return {
        "symbol": str(symbol).upper(),
        "ticker": str(symbol).upper(),
        "strategy_id": str(row.get("strategy_id") or row.get("strategy_name") or alpha_id),
        "strategy_name": str(row.get("strategy_name") or f"alpha_catalog__{alpha_id}"),
        "setup_type": alpha_id,
        "alpha_id": alpha_id,
        "alpha_family": str(meta.get("family") or ""),
        "side": str(last_signal_row.get("side") or "long"),
        "score": float(last_signal_row.get("score") or 0.0),
        "confidence": float(last_signal_row.get("score") or 0.0),
        "note": str(last_signal_row.get("note") or ""),
        "signal_date": last_signal_row.get("date"),
        "bar_date": last_bar.get("date"),
        "close": float(last_bar.get("close") or 0.0),
        "expected_hold_days": hold,
        "backtest_profit_factor": float(
            row.get("backtest_profit_factor")
            or metrics.get("profit_factor")
            or 0.0
        ),
        "backtest_win_rate": float(
            row.get("backtest_win_rate") or metrics.get("win_rate") or 0.0
        ),
        "walk_forward_validated": bool(row.get("walk_forward_validated", False)),
        "source": "alpha_catalog_runtime",
    }


def generate_alpha_catalog_signal(
    row: Dict[str, Any],
    symbol: str,
    daily_bars: pd.DataFrame,
    alpha_ctx: Optional[AlphaContext],
    current_regime: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Run the row's alpha generator and return a live signal dict (or None).

    Returns None when:
    - the row is not alpha-catalog shaped,
    - ``alpha_id`` is unknown (catalog evolved, row stale),
    - the row's regime_compatibility excludes ``current_regime``,
    - bars are insufficient (the alpha returns no rows),
    - the alpha returns no entry rows.

    Notably, this only fires when the alpha emits an entry on the LAST bar of
    ``daily_bars``. Callers should pass bars truncated to "today" so that
    historical entries do not retroactively produce live signals.
    """
    if not is_alpha_catalog_row(row):
        return None

    meta = row.get("alpha_metadata") or {}
    alpha_id = str(meta.get("alpha_id") or "")
    if not alpha_id:
        return None

    if alpha_id not in set(list_alpha_ids()):
        logger.debug("alpha_catalog_runtime: unknown alpha_id=%s for %s", alpha_id, symbol)
        return None

    if not regime_compatible(row, current_regime):
        return None

    if daily_bars is None or not isinstance(daily_bars, pd.DataFrame) or daily_bars.empty:
        return None

    try:
        alpha = get_alpha(alpha_id)
        signal_frame = alpha.generate(daily_bars, alpha_ctx)
    except Exception as exc:
        logger.warning(
            "alpha_catalog_runtime: %s.generate failed for %s: %s",
            alpha_id,
            symbol,
            exc,
        )
        return None

    if signal_frame is None or signal_frame.empty:
        return None

    entries = signal_frame[signal_frame["entry"].fillna(False).astype(bool)]
    if entries.empty:
        return None

    last_bar = daily_bars.sort_values("date").iloc[-1]
    last_signal = entries.sort_values("date").iloc[-1]

    # Only fire if the latest entry date matches the most recent bar — the
    # caller is asking "is this alpha firing right now?" not "has it ever fired
    # in this window?". A loose-equality compare handles date vs Timestamp.
    if pd.to_datetime(last_signal["date"]) != pd.to_datetime(last_bar["date"]):
        return None

    return _build_signal_dict(row, symbol, last_signal, last_bar)


__all__ = [
    "is_alpha_catalog_row",
    "regime_compatible",
    "generate_alpha_catalog_signal",
]
