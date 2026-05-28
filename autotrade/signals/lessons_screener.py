"""
Compatibility Lessons Screener (Parquet-Only)
=============================================

Legacy DB-backed lessons screener has been retired.
This module now delegates to ScreenerV2 so callers keep working without
`predictions_db.sqlite`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Backward-compatible symbol kept for old imports/tests.
# The legacy SQLite flow is retired, but downstream checks still validate this
# path to confirm we point at the canonical DownDay database location.
def _resolve_db_path() -> Path:
    env_path = os.getenv("AUTOTRADE_PREDICTIONS_DB_PATH", "").strip()
    if env_path:
        return Path(env_path)

    return Path(__file__).resolve().parents[2] / "data" / "downday" / "predictions_db.sqlite"


DB_PATH: Optional[Path] = _resolve_db_path()


def get_entry_candidates(
    max_candidates: int = 50,
    exclude_symbols: Optional[List[str]] = None,
    parent_logger: Optional[logging.Logger] = None,
    symbols: Optional[List[str]] = None,
    **kwargs,
) -> List[Dict]:
    """
    Return entry candidates using ScreenerV2.

    Args:
        max_candidates: Max rows to return.
        exclude_symbols: Optional ticker blacklist.
        parent_logger: Optional logger.
        symbols: Optional ticker allow-list.
        **kwargs: Passed through to ScreenerV2 helper.
    """
    log = parent_logger or logger
    from autotrade.signals.screener_v2 import get_entry_candidates as get_entry_candidates_v2

    candidates = get_entry_candidates_v2(
        max_candidates=max_candidates,
        symbols=symbols,
        parent_logger=log,
        **kwargs,
    )

    if exclude_symbols:
        blocked = {str(s).upper().strip() for s in exclude_symbols if str(s).strip()}
        candidates = [
            c
            for c in candidates
            if str(c.get("ticker", c.get("symbol", ""))).upper() not in blocked
        ]

    return candidates[:max_candidates]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("lessons_screener.py shim active (ScreenerV2 parquet-only mode).")
