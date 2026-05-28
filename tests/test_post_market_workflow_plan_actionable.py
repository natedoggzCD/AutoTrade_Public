import json
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from autotrade.execution import post_market_workflow as post_market_workflow_mod
from autotrade.execution.post_market_workflow import PostMarketWorkflow


def test_pm_save_path_enriches_hollow_watch_rows_from_daily_features(
    tmp_path, monkeypatch
):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    parquet_path = data_dir / "daily_features.parquet"

    feature_rows = []
    for idx in range(50):
        symbol = f"T{idx:02d}"
        feature_rows.append(
            {
                "ticker": symbol,
                "Date": "2026-04-21",
                "Close": 10.0 + idx,
                "Volume": 100_000 + idx,
                "RSI_14": 48.0,
                "atr_14": 0.4,
                "volume_ratio": 1.0,
                "entry_score": 55.0,
            }
        )
        feature_rows.append(
            {
                "ticker": symbol,
                "Date": "2026-04-22",
                "Close": 11.0 + idx,
                "Volume": 150_000 + idx,
                "RSI_14": 42.0 + (idx % 7),
                "atr_14": 0.55 + idx * 0.01,
                "volume_ratio": 1.35 + idx * 0.01,
                "entry_score": 70.0 + (idx % 10),
            }
        )
    pd.DataFrame(feature_rows).to_parquet(parquet_path, index=False)

    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )

    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 4, 23).date(),
    )
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_config",
        lambda: SimpleNamespace(
            data=SimpleNamespace(
                downday_root=str(tmp_path),
                daily_features_parquet="data/daily_features.parquet",
            )
        ),
    )

    hollow_signals = [
        {
            "symbol": f"T{idx:02d}",
            "ticker": f"T{idx:02d}",
            "recommendation": "WATCH",
            "score": 90.0 - idx,
            "final_score": 90.0 - idx,
            "confidence": 90.0 - idx,
            "entry_score": 0.0,
            "rsi": 50.0,
            "volume_ratio": 0.0,
            "momentum": 0.0,
        }
        for idx in range(10)
    ]

    result = workflow._save_plan(
        {
            "generated_at": "2026-04-23T00:15:47",
            "signals": hollow_signals,
            "entry_candidates": list(hollow_signals),
            "summary": {},
            "positions": [],
            "core_data_readiness": {
                "is_fresh": True,
                "primary_date": "2026-04-22",
                "expected_date": "2026-04-22",
                "blocking_reasons": [],
            },
        }
    )

    saved = json.loads(result.read_text(encoding="utf-8"))
    rows = saved["signals"]

    assert len(rows) == 10
    assert sum(1 for row in rows if row["recommendation"] != "WATCH") >= 1
    assert sum(1 for row in rows if row["entry_score"] > 0) >= 1
    assert sum(1 for row in rows if row["rsi"] != 50.0) >= 1
    assert sum(1 for row in rows if row["volume_ratio"] > 0) >= 1
    assert saved["entry_candidates"] == rows
