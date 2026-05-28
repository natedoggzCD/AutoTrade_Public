from autotrade.core.autonomous_agent import AutonomousAgent


def _row(symbol: str, recommendation: str = "BUY") -> dict:
    return {"symbol": symbol, "recommendation": recommendation}


def test_build_premarket_actionable_pool_uses_top50_and_full_watchlist_fallback():
    primary = [_row(f"S{i:02d}") for i in range(1, 21)]
    actionable_top50 = [_row(f"S{i:02d}") for i in range(1, 41)]
    full_watchlist = [_row(f"S{i:02d}") for i in range(1, 47)]
    plan = {
        "actionable_top50": actionable_top50,
        "overflow_signals": [],
        "full_watchlist": full_watchlist,
    }

    pool, meta = AutonomousAgent._build_premarket_actionable_pool(plan, primary)

    assert len(pool) == 46
    assert meta["primary_count"] == 20
    assert meta["overflow_count"] == 26
    assert meta["from_actionable_top50"] == 20
    assert meta["from_overflow_signals"] == 0
    assert meta["from_full_watchlist"] == 6


def test_build_premarket_actionable_pool_filters_non_actionable_full_watchlist_rows():
    primary = [_row("AAA"), _row("BBB")]
    plan = {
        "actionable_top50": [],
        "overflow_signals": [_row("CCC"), _row("DDD")],
        "full_watchlist": [
            _row("EEE", "WATCH"),
            _row("FFF", "BUY"),
            _row("CCC", "BUY"),
        ],
    }

    pool, meta = AutonomousAgent._build_premarket_actionable_pool(plan, primary)
    symbols = [row.get("symbol") for row in pool if isinstance(row, dict)]

    assert symbols == ["AAA", "BBB", "CCC", "DDD", "FFF"]
    assert meta["from_actionable_top50"] == 0
    assert meta["from_overflow_signals"] == 2
    assert meta["from_full_watchlist"] == 1
