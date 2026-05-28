import json
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace

from autotrade.core.day_manager import DayManager
from autotrade.core.position_thesis import PositionThesis, PositionThesisCache
from autotrade.risk.portfolio_exposure import calculate_portfolio_exposure
from autotrade.risk.short_engine_failsafe import SideFailsafeState
from autotrade.risk.sizing import RUnitSizer
from autotrade.signals.inverse_etf_screener import InverseETFScreener
from autotrade.signals.short_risk_gates import passes_short_gates
from autotrade.signals.short_signal_scorer import ShortSignalScorer
from autotrade.signals.shortability_gate import is_shortable
from autotrade.utils.kill_switch import entry_scope_disabled, is_disabled


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows

    def get_all_inverse_etfs(self, active_only=True):
        return list(self.rows)

    def upsert_screen_result(self, payload):
        return None


def test_runit_sizer_caps_multiplier_for_long_and_short():
    sizer = RUnitSizer(total_equity=100_000, risk_pct=0.01, max_notional_pct=0.05)

    long_qty = sizer.calculate_quantity(
        entry_price=10.0,
        stop_price=9.0,
        side="long",
        size_multiplier=2.5,
    )
    short_qty = sizer.calculate_quantity(
        entry_price=10.0,
        stop_price=11.0,
        side="short",
        size_multiplier=2.5,
    )

    assert long_qty == 500
    assert short_qty == 500
    assert long_qty * 10.0 == 5_000
    assert short_qty * 10.0 == 5_000


def test_inverse_etf_tier_filter_routes_by_breadth_and_regime(monkeypatch):
    screener = InverseETFScreener(
        _FakeDB(
            [
                {"ticker": "SH", "category": "index", "leverage": 1},
                {"ticker": "SDS", "category": "index", "leverage": 2},
                {"ticker": "SQQQ", "category": "index", "leverage": 3},
            ]
        )
    )
    monkeypatch.setattr(
        screener,
        "_score_candidate",
        lambda etf, **kwargs: {
            "ticker": etf["ticker"],
            "signal": "ENTRY",
            "composite_score": 80,
        },
    )

    mild = screener.screen_universe("selloff", breadth_pct_positive=37.0)
    severe = screener.screen_universe("capitulation", breadth_pct_positive=28.0)
    crash = screener.screen_universe("crash", breadth_pct_positive=24.0)

    assert [row["ticker"] for row in mild] == ["SH"]
    assert {row["ticker"] for row in severe} == {"SH", "SDS"}
    assert {row["ticker"] for row in crash} == {"SH", "SDS", "SQQQ"}


def test_inverse_etf_decay_exit_triggers_on_breadth_recovery():
    dm = DayManager.__new__(DayManager)
    dm._load_resolved_regime_context = lambda: {"breadth_pct_positive": 47.0}

    result = DayManager._check_inverse_etf_decay_exit(
        dm,
        SimpleNamespace(symbol="SQQQ", entry_at=None),
    )

    assert result["should_exit"] is True
    assert result["reason"].startswith("breadth_recovery")


def test_inverse_etf_pair_rebalance_computes_correlated_long_trim():
    dm = DayManager.__new__(DayManager)
    positions = [
        SimpleNamespace(symbol="LONGA", market_value=8_000, side="long", beta=1.0),
        SimpleNamespace(symbol="LONGB", market_value=500, side="long", beta=0.5),
    ]

    plan = DayManager._compute_inverse_etf_long_rebalance(
        dm,
        inverse_symbol="SH",
        inverse_notional=3_000,
        positions=positions,
    )

    assert plan["should_trim"] is True
    assert plan["symbol"] == "LONGA"
    assert plan["trim_notional"] == 2_250


def test_short_signal_scorer_outputs_short_geometry():
    rows = [
        {
            "ticker": "BEAR",
            "close": 10.0,
            "ema_20": 10.5,
            "ema_50": 11.0,
            "vwap": 10.2,
            "volume_ratio": 2.0,
            "day_change_pct": -3.0,
            "rsi_14": 35.0,
            "atr_14": 0.5,
            "news_sentiment": "downgrade",
        }
    ]

    [signal] = ShortSignalScorer(min_score=40).score_universe(rows)

    assert signal["side"] == "short"
    assert signal["stop"] > signal["entry"] > signal["target"]
    assert signal["risk_per_share"] == signal["stop"] - signal["entry"]
    assert signal["target_per_share"] == signal["entry"] - signal["target"]


def test_shortability_gate_blocks_hard_to_borrow(tmp_path):
    client = SimpleNamespace(
        get_asset=lambda symbol: SimpleNamespace(
            shortable=True,
            easy_to_borrow=False,
            fractionable=True,
        )
    )

    allowed, reason = is_shortable(
        "HTB",
        trading_client=client,
        session_date="2026-05-14",
        data_dir=tmp_path,
    )

    assert allowed is False
    assert reason == "hard_to_borrow_blocked"
    cache = json.loads((tmp_path / "shortability_cache_2026-05-14.json").read_text())
    assert cache["HTB"]["reason"] == "hard_to_borrow_blocked"


def test_short_risk_gates_block_earnings_and_price_floor(tmp_path):
    db_path = tmp_path / "financial.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE earnings_calendar(symbol TEXT, earnings_date TEXT)")
        conn.execute(
            "INSERT INTO earnings_calendar VALUES (?, ?)",
            ("EARN", (date(2026, 5, 14) + timedelta(days=3)).isoformat()),
        )

    allowed, reason = passes_short_gates(
        "EARN",
        {"entry": 10.0},
        financial_db_path=db_path,
        as_of_date=date(2026, 5, 14),
    )
    low_allowed, low_reason = passes_short_gates(
        "LOW",
        {"entry": 4.50},
        financial_db_path=db_path,
        as_of_date=date(2026, 5, 14),
    )

    assert allowed is False
    assert reason == "earnings_block"
    assert low_allowed is False
    assert low_reason == "price_floor"


def test_position_thesis_side_aware_pnl_and_stop(tmp_path):
    thesis = PositionThesis(
        symbol="DROP",
        side="short",
        key_levels={"stop": 11.0},
    )

    assert thesis.unrealized_pnl(entry_price=10.0, current_price=9.5, qty=100) == 50
    assert thesis.stop_triggered(11.25) is True
    assert "Side: short" in thesis.to_prompt_context()

    cache = PositionThesisCache(
        persist_path=str(tmp_path / "theses.json"),
        archive_path=str(tmp_path / "archive.jsonl"),
    )
    cache.seed("DROP", side="short")
    assert cache.get("DROP").side == "short"


def test_portfolio_exposure_and_side_failsafe_lanes():
    exposure = calculate_portfolio_exposure(
        [
            {"symbol": "AAA", "market_value": 5_000, "side": "long"},
            {"symbol": "BBB", "market_value": 3_000, "side": "short"},
        ],
        equity=10_000,
    )
    state = SideFailsafeState.from_payload(
        {
            "short": {"pf": 1.05, "wr": 0.55, "sample_count": 30, "level": "normal"},
            "long": {"pf": 1.3, "wr": 0.62, "sample_count": 30, "level": "normal"},
        }
    )

    assert exposure["gross_exposure_pct"] == 80.0
    assert exposure["net_exposure_pct"] == 20.0
    assert state.allows_new_entries("short") is False
    assert state.allows_new_entries("long") is True


def test_file_kill_switch_blocks_scoped_entries(tmp_path):
    (tmp_path / "disable_shorts.flag").write_text("", encoding="utf-8")

    assert is_disabled("disable_shorts", flags_dir=tmp_path) is True
    assert entry_scope_disabled("short", flags_dir=tmp_path) is True
    assert entry_scope_disabled("long", flags_dir=tmp_path) is False
