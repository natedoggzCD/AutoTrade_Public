from __future__ import annotations

import json
from types import SimpleNamespace

from autotrade.signals.strategy_pool import (
    get_top_strategies_for_symbol,
    load_validated_strategies_by_symbol,
)


def test_load_validated_strategies_by_symbol_valid_artifact(tmp_path):
    path = tmp_path / "validated_strategies_by_symbol.json"
    payload = {
        "generated_at": "2026-02-22T12:00:00",
        "source_run_id": "factory_20260222_120000",
        "top_k": 5,
        "symbols": {
            "aapl": [
                {
                    "strategy_name": "strat_b",
                    "setup_type": "reversion",
                    "symbol_score": 88.5,
                    "symbol_metrics": {"trades": 12, "win_rate": 0.55, "profit_factor": 1.4, "total_pnl": 520.0},
                },
                {
                    "strategy_name": "strat_a",
                    "setup_type": "momentum",
                    "symbol_score": 92.0,
                    "symbol_metrics": {"trades": 14, "win_rate": 0.57, "profit_factor": 1.6, "total_pnl": 700.0},
                },
            ],
            "MSFT": [
                {
                    "strategy_name": "strat_c",
                    "setup_type": "breakout",
                    "symbol_score": 77.0,
                    "symbol_metrics": {"trades": 9, "win_rate": 0.5, "profit_factor": 1.2, "total_pnl": 260.0},
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_validated_strategies_by_symbol(path=path, fallback_to_global=False)

    assert sorted(loaded.keys()) == ["AAPL", "MSFT"]
    assert [row["strategy_name"] for row in loaded["AAPL"]] == ["strat_a", "strat_b"]
    top_aapl = get_top_strategies_for_symbol(
        "aapl", symbol_map=loaded, fallback_to_global=False
    )
    assert [row["strategy_name"] for row in top_aapl] == ["strat_a", "strat_b"]


def test_load_validated_strategies_by_symbol_missing_file_falls_back_to_global(
    tmp_path, monkeypatch
):
    missing_symbol_path = tmp_path / "missing_validated_by_symbol.json"
    global_path = tmp_path / "validated_strategies.json"
    global_payload = [
        {"strategy_name": "s1", "setup_type": "momentum", "metrics": {"profit_factor": 1.20}},
        {"strategy_name": "s2", "setup_type": "reversion", "metrics": {"profit_factor": 1.10}},
        {"strategy_name": "s3", "setup_type": "breakout", "metrics": {"profit_factor": 0.95}},
    ]
    global_path.write_text(json.dumps(global_payload), encoding="utf-8")

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                validated_strategy_min_profit_factor=1.05,
                validated_strategy_pool_top_n=2,
                per_symbol_fallback_to_global=True,
            )
        ),
    )

    symbol_map = load_validated_strategies_by_symbol(
        path=missing_symbol_path,
        fallback_to_global=True,
        global_path=global_path,
    )
    assert [row["strategy_name"] for row in symbol_map["*"]] == ["s1", "s2"]

    tsla_rows = get_top_strategies_for_symbol(
        "TSLA",
        symbol_map=symbol_map,
        fallback_to_global=True,
        global_path=global_path,
    )
    assert [row["strategy_name"] for row in tsla_rows] == ["s1", "s2"]


def test_get_top_strategies_for_symbol_order_is_deterministic(tmp_path):
    path = tmp_path / "validated_strategies_by_symbol.json"
    payload = {
        "generated_at": "2026-02-22T12:00:00",
        "source_run_id": "factory_20260222_120000",
        "top_k": 5,
        "symbols": {
            "NVDA": [
                {
                    "strategy_name": "gamma",
                    "setup_type": "x",
                    "symbol_score": 80.0,
                    "symbol_metrics": {"trades": 10, "win_rate": 0.5},
                },
                {
                    "strategy_name": "alpha",
                    "setup_type": "x",
                    "symbol_score": 95.0,
                    "symbol_metrics": {"trades": 9, "win_rate": 0.6},
                },
                {
                    "strategy_name": "beta",
                    "setup_type": "x",
                    "symbol_score": 87.0,
                    "symbol_metrics": {"trades": 11, "win_rate": 0.58},
                },
            ]
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_validated_strategies_by_symbol(path=path, fallback_to_global=False)
    first = [row["strategy_name"] for row in get_top_strategies_for_symbol("NVDA", symbol_map=loaded)]
    second = [row["strategy_name"] for row in get_top_strategies_for_symbol("NVDA", symbol_map=loaded)]

    assert first == ["alpha", "beta", "gamma"]
    assert second == ["alpha", "beta", "gamma"]

