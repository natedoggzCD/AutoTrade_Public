from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import autotrade.utils.momentum_scanner as momentum_scanner_mod
from autotrade.utils.momentum_scanner import ET, MomentumScanner, load_momentum_watchlist


class _StubNews:
    def collect(self, symbol: str):
        return {
            "symbol": symbol,
            "sentiment_score": 0.25,
            "catalyst_score": 0.8,
            "has_catalyst": True,
            "catalyst_tags": ["earnings"],
            "catalyst_note": f"{symbol} catalyst",
        }


def _bars(prices, volumes):
    index = pd.date_range("2026-03-17 08:00:00", periods=len(prices), freq="1min", tz=ET)
    return pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.995 for p in prices],
            "close": prices,
            "volume": volumes,
        },
        index=index,
    )


def test_scan_once_writes_live_artifact(tmp_path: Path):
    universe_rows = [
        {
            "ticker": "LMND",
            "prev_close": 60.0,
            "avg_volume": 2_000_000.0,
            "atr_14": 3.0,
            "atr_percent": 5.0,
            "weekly_return": 9.0,
        },
        {
            "ticker": "SLOW",
            "prev_close": 12.0,
            "avg_volume": 300_000.0,
            "atr_14": 0.3,
            "atr_percent": 2.5,
            "weekly_return": 1.0,
        },
    ]
    bars_map = {
        "LMND": _bars(
            [61.0, 61.8, 62.4, 63.1, 63.8, 64.4, 64.9, 65.5, 66.0, 66.4],
            [20_000, 22_000, 24_000, 25_000, 26_000, 28_000, 30_000, 31_000, 32_000, 34_000],
        ),
        "SLOW": _bars(
            [12.0, 11.9, 12.0, 11.95, 11.9, 11.88, 11.87, 11.85, 11.84, 11.83],
            [2_000] * 10,
        ),
    }

    def _fetch(symbols, *_args, **_kwargs):
        return {symbol: bars_map[symbol] for symbol in symbols if symbol in bars_map}

    output_path = tmp_path / "momentum_watchlist_live.json"
    scanner = MomentumScanner(
        output_path=output_path,
        news_aggregator=_StubNews(),
        intraday_fetcher=_fetch,
        universe_rows=universe_rows,
        sleep_fn=lambda _seconds: None,
    )

    payload = scanner.scan_once(now_et=datetime(2026, 3, 17, 8, 20, tzinfo=ET))

    assert payload["status"] == "ok"
    assert payload["candidate_count"] == 1
    assert payload["generated_at_ct"].endswith("-05:00")
    assert payload["scanner_mode"] == "daemon"
    assert payload["universe_size_considered"] == 2
    assert "next_scan_due_ct" in payload
    assert payload["symbols"][0]["ticker"] == "LMND"
    assert payload["symbols"][0]["entry_source"] == "momentum_scanner"
    assert payload["symbols"][0]["has_catalyst"] is True
    assert output_path.exists()

    loaded = load_momentum_watchlist(
        path=output_path,
        now_et=datetime(2026, 3, 17, 8, 21, tzinfo=ET),
    )
    assert loaded["loaded"] is True
    assert loaded["status"] == "ok"
    assert loaded["symbols"][0]["ticker"] == "LMND"


def test_load_momentum_watchlist_marks_stale_artifact(tmp_path: Path):
    artifact_path = tmp_path / "momentum_watchlist_live.json"
    artifact_path.write_text(
        json.dumps(
            {
                "generated_at_et": "2026-03-17T08:00:00-04:00",
                "session": "premarket",
                "scan_count": 1,
                "symbols": [{"ticker": "LMND", "score": 80.0}],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_momentum_watchlist(
        path=artifact_path,
        now_et=datetime(2026, 3, 17, 8, 20, tzinfo=ET),
    )

    assert loaded["loaded"] is False
    assert loaded["stale"] is True
    assert loaded["symbols"] == []


def test_scan_once_waits_until_configured_ct_start(tmp_path: Path):
    output_path = tmp_path / "momentum_watchlist_live.json"
    scanner = MomentumScanner(
        output_path=output_path,
        news_aggregator=_StubNews(),
        intraday_fetcher=lambda *_args, **_kwargs: {},
        universe_rows=[],
        sleep_fn=lambda _seconds: None,
    )

    payload = scanner.scan_once(now_et=datetime(2026, 3, 17, 6, 30, tzinfo=ET))

    assert payload["status"] == "before_start_window"
    assert payload["session"] == "before_start_window"
    loaded = load_momentum_watchlist(
        path=output_path,
        now_et=datetime(2026, 3, 17, 6, 31, tzinfo=ET),
    )
    assert loaded["loaded"] is False
    assert loaded["reason"] == "before_start_window"


def test_load_universe_rows_returns_without_sorting_when_selection_score_missing(monkeypatch, tmp_path: Path):
    parquet_path = tmp_path / "daily_features.parquet"
    parquet_path.write_text("stub", encoding="utf-8")

    class _FakeResult:
        def fetchdf(self):
            return pd.DataFrame(
                [
                    {
                        "ticker": "SLOW",
                        "trade_date": "2026-04-09",
                        "prev_close": 10.0,
                        "avg_volume": 1_000_000.0,
                        "atr_14": 0.5,
                        "weekly_return": 2.0,
                        "atr_percent": 5.0,
                    },
                    {
                        "ticker": "FAST",
                        "trade_date": "2026-04-09",
                        "prev_close": 10.0,
                        "avg_volume": 10_000_000.0,
                        "atr_14": 1.0,
                        "weekly_return": 10.0,
                        "atr_percent": 15.0,
                    },
                ]
            )

    class _FakeConnection:
        def execute(self, *_args, **_kwargs):
            return _FakeResult()

        def close(self):
            return None

    fake_config = SimpleNamespace(
        momentum_scanner=SimpleNamespace(
            artifact_path=tmp_path / "momentum_watchlist_live.json",
            min_price=1.0,
            max_price=100.0,
            min_avg_volume=100_000.0,
        ),
        data=SimpleNamespace(
            daily_features_parquet=str(parquet_path),
            downday_root=str(tmp_path),
        ),
    )

    monkeypatch.setattr(momentum_scanner_mod, "duckdb", SimpleNamespace(connect=lambda: _FakeConnection()))

    scanner = MomentumScanner(
        config=fake_config,
        news_aggregator=_StubNews(),
        intraday_fetcher=lambda *_args, **_kwargs: {},
        sleep_fn=lambda _seconds: None,
    )

    rows = scanner._load_universe_rows()

    assert [row["ticker"] for row in rows] == ["FAST", "SLOW"]
    assert all("ticker" in row for row in rows)
