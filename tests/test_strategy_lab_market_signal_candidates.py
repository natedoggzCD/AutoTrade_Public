from types import SimpleNamespace

import pandas as pd

from tools import strategy_lab
from autotrade.backtesting.strategy_config import LabStrategyConfig


class _FakeQuery:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def df(self) -> pd.DataFrame:
        return self._frame.copy()


class _FakeConn:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def execute(self, query: str):
        return _FakeQuery(self._frame)

    def close(self) -> None:
        pass


def test_load_market_signal_candidates_coerces_missing_daily_rank(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        [
            {
                "ticker": "corz",
                "signal_date": "2026-05-04",
                "entry_date": "2026-05-05",
                "entry_open": "12.50",
                "atr_14": "0.8",
                "signal_score": 91.0,
                "daily_rank": None,
            }
        ]
    )
    monkeypatch.setattr(
        strategy_lab.duckdb,
        "connect",
        lambda database=":memory:": _FakeConn(frame),
    )
    parquet_path = tmp_path / "daily_features.parquet"
    parquet_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        strategy_lab, "_get_backtest_parquet_path", lambda: str(parquet_path)
    )

    result = strategy_lab._load_market_signal_candidates(
        strategy=SimpleNamespace(
            name="test",
            entry=SimpleNamespace(
                rsi_min=0.0,
                rsi_max=100.0,
                min_atr_pct=0.0,
                max_atr_pct=100.0,
                min_volume_ratio=0.0,
                require_above_sma20=False,
                require_sma5_curl_positive=False,
            ),
        ),
        start_date="2026-05-05",
        end_date="2026-05-05",
        symbols=["CORZ"],
        top_n=10,
    )

    assert result["ticker"].tolist() == ["CORZ"]
    assert result["daily_rank"].tolist() == [99999]
    assert str(result["daily_rank"].dtype).startswith("int")


def test_strategy_lab_includes_volatility_compression_entrant():
    candidates = strategy_lab._grid_search_candidates(
        LabStrategyConfig(name="base", description="base"),
        entry_combo_cap=1,
        exit_combo_cap=1,
    )

    compression = [
        c for c in candidates if c.entry.setup_type == "volatility_compression"
    ]

    assert compression
    candidate = compression[0]
    assert candidate.entry.require_bb_width_compression is True
    assert candidate.entry.max_bb_width_percentile == 0.10
    assert candidate.entry.require_nr7 is True
    assert candidate.entry.volume_avg_deviation_pct == 30.0
    assert candidate.exit.max_hold_days == 10
    assert candidate.exit.target_entry_range_mult == 1.5
