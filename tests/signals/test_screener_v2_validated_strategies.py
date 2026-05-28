from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.signals.screener_v2 import get_entry_candidates


def _load_universe_slice(limit: int = 50) -> list[str]:
    from autotrade.signals.screener_v2 import ScreenerV2

    parquet_path = ScreenerV2().daily_features_path
    if not parquet_path.exists():
        pytest.skip(f"daily features parquet missing: {parquet_path}")

    frame = pd.read_parquet(parquet_path, columns=["ticker"])
    tickers = (
        frame["ticker"].dropna().astype(str).str.strip().str.upper().drop_duplicates().tolist()
    )
    if not tickers:
        pytest.skip("no tickers available in daily features parquet")
    return tickers[:limit]


def test_all_validated_strategies_compile_and_screen(monkeypatch):
    strategies_path = Path("data/strategy_lab/validated_strategies.json")
    if not strategies_path.exists():
        pytest.skip("validated strategies artifact missing")

    rows = json.loads(strategies_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        pytest.skip("validated strategies artifact is empty")

    symbols = _load_universe_slice(limit=50)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda force_refresh=False: {
            "is_fresh": True,
            "primary_date": "2026-04-14",
            "expected_date": "2026-04-14",
            "blocking_reasons": [],
        },
    )

    failures: list[str] = []
    for idx, strat in enumerate(rows):
        strat_name = str(
            strat.get("strategy_name")
            or (strat.get("strategy_definition") or {}).get("name")
            or f"row_{idx}"
        )
        screener_patch = dict((strat.get("config_patch") or {}).get("screener_v2") or {})
        try:
            get_entry_candidates(
                max_candidates=20,
                symbols=symbols,
                config_override=screener_patch,
                log_samples=False,
            )
        except Exception as exc:  # pragma: no cover - failure capture path
            failures.append(f"{strat_name}: {type(exc).__name__}: {exc}")

    assert not failures, "validated strategy screening failures:\n" + "\n".join(failures)
