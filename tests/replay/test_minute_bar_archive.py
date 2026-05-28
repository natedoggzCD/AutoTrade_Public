from datetime import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from autotrade.replay.minute_bar_archive import (
    archive_session_minute_bars,
    load_session_archive,
)


def _bars(start_ts: str, prices: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        start_ts,
        periods=len(prices),
        freq="min",
        tz="America/New_York",
    )
    return pd.DataFrame(
        {
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1_000_000] * len(prices),
            "trade_count": [100] * len(prices),
            "vwap": prices,
        },
        index=index,
    )


def _bars_with_index_name(start_ts: str, prices: list[float], index_name: str | None) -> pd.DataFrame:
    df = _bars(start_ts, prices)
    df.index.name = index_name
    return df


class _ArchiveClient:
    def __init__(self, preferred: dict[str, pd.DataFrame], regular_only: dict[str, pd.DataFrame]):
        self.preferred = preferred
        self.regular_only = regular_only
        self.calls: dict[str, int] = {}

    def get_stock_bars(self, request):
        symbol = request.symbol_or_symbols
        call_number = self.calls.get(symbol, 0)
        self.calls[symbol] = call_number + 1
        if call_number == 0:
            df = self.preferred.get(symbol, pd.DataFrame())
        else:
            df = self.regular_only.get(symbol, pd.DataFrame())
        return SimpleNamespace(df=df)


def test_archive_session_minute_bars_roundtrip_with_regular_session_fallback(
    tmp_path: Path,
):
    db_path = tmp_path / "replay_minute_bars.duckdb"
    client = _ArchiveClient(
        preferred={"AAA": _bars("2026-03-27 04:00", [10.0, 10.3, 10.6])},
        regular_only={"BBB": _bars("2026-03-27 09:30", [20.0, 20.2, 20.4])},
    )

    result = archive_session_minute_bars(
        db_path=db_path,
        session_date="2026-03-27",
        client=client,
        symbol_sources={"AAA": {"watchlist"}, "BBB": {"screener_add"}},
        preferred_start_et=time(4, 0),
        preferred_end_et=time(16, 0),
        fallback_regular_session_only=True,
    )

    assert result["archived_symbols"] == 2
    assert result["premarket_covered_symbols"] == 1
    assert result["regular_session_only_symbols"] == 1

    loaded = load_session_archive(
        db_path=db_path,
        session_date="2026-03-27",
        symbols=["AAA", "BBB"],
    )

    assert loaded.archive_exists is True
    assert sorted(loaded.bars_by_symbol.keys()) == ["AAA", "BBB"]
    assert loaded.manifest_by_symbol["AAA"]["has_premarket_data"] is True
    assert loaded.manifest_by_symbol["BBB"]["used_regular_session_only"] is True
    assert loaded.bars_by_symbol["AAA"].index.tz is not None
    assert loaded.bars_by_symbol["BBB"].iloc[-1]["close"] == 20.4


def test_archive_session_minute_bars_replace_existing_symbols_only_preserves_other_symbols(
    tmp_path: Path,
):
    db_path = tmp_path / "replay_minute_bars.duckdb"
    initial_client = _ArchiveClient(
        preferred={
            "AAA": _bars("2026-03-27 04:00", [10.0, 10.3, 10.6]),
            "CCC": _bars("2026-03-27 04:00", [30.0, 30.2, 30.4]),
        },
        regular_only={},
    )
    archive_session_minute_bars(
        db_path=db_path,
        session_date="2026-03-27",
        client=initial_client,
        symbol_sources={"AAA": {"watchlist"}, "CCC": {"watchlist"}},
        preferred_start_et=time(4, 0),
        preferred_end_et=time(16, 0),
        fallback_regular_session_only=True,
    )

    replacement_client = _ArchiveClient(
        preferred={"AAA": _bars("2026-03-27 04:00", [11.0, 11.2, 11.4])},
        regular_only={},
    )
    archive_session_minute_bars(
        db_path=db_path,
        session_date="2026-03-27",
        client=replacement_client,
        symbol_sources={"AAA": {"weekly_backfill"}},
        preferred_start_et=time(4, 0),
        preferred_end_et=time(16, 0),
        fallback_regular_session_only=True,
        replace_existing_symbols_only=True,
    )

    loaded = load_session_archive(
        db_path=db_path,
        session_date="2026-03-27",
        symbols=["AAA", "CCC"],
    )

    assert sorted(loaded.bars_by_symbol.keys()) == ["AAA", "CCC"]
    assert loaded.bars_by_symbol["AAA"].iloc[-1]["close"] == 11.4
    assert loaded.bars_by_symbol["CCC"].iloc[-1]["close"] == 30.4


def test_archive_session_minute_bars_normalizes_named_datetime_index(
    tmp_path: Path,
):
    db_path = tmp_path / "replay_minute_bars.duckdb"
    client = _ArchiveClient(
        preferred={
            "AAA": _bars_with_index_name("2026-03-27 04:00", [10.0, 10.3], "timestamp"),
            "BBB": _bars_with_index_name("2026-03-27 04:00", [20.0, 20.3], "bar_time"),
        },
        regular_only={},
    )

    result = archive_session_minute_bars(
        db_path=db_path,
        session_date="2026-03-27",
        client=client,
        symbol_sources={"AAA": {"watchlist"}, "BBB": {"watchlist"}},
        preferred_start_et=time(4, 0),
        preferred_end_et=time(16, 0),
        fallback_regular_session_only=True,
    )

    assert result["archived_symbols"] == 2

    loaded = load_session_archive(
        db_path=db_path,
        session_date="2026-03-27",
        symbols=["AAA", "BBB"],
    )

    assert loaded.bars_by_symbol["AAA"].iloc[-1]["close"] == 10.3
    assert loaded.bars_by_symbol["BBB"].iloc[-1]["close"] == 20.3
    assert loaded.bars_by_symbol["AAA"].index.name == "timestamp_utc"
    assert loaded.bars_by_symbol["BBB"].index.name == "timestamp_utc"
