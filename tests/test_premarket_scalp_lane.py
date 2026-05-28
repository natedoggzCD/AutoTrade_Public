"""Tests for the premarket scalp lane."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Dict

import pytest

from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.execution.premarket_scalp_lane import PremarketScalpLane


class FakeOrder:
    def __init__(
        self,
        oid: str,
        status: str = "accepted",
        filled_qty: int = 0,
        filled_avg_price: float = 0.0,
    ):
        self.id = oid
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price


class FakeClient:
    """Records every order for assertion + lets tests script fill state."""

    def __init__(self):
        self.submitted = []
        self.cancels = []
        self.next_order_id = 1
        self.order_state: Dict[str, FakeOrder] = {}
        # default: orders auto-fill
        self.auto_fill = True

    def submit_order(self, request):
        oid = f"ord-{self.next_order_id}"
        self.next_order_id += 1
        self.submitted.append(request)
        # Capture relevant request attributes for assertions
        order = FakeOrder(
            oid,
            status="filled" if self.auto_fill else "accepted",
            filled_qty=int(request.qty) if self.auto_fill else 0,
            filled_avg_price=float(request.limit_price) if self.auto_fill else 0.0,
        )
        self.order_state[oid] = order
        return order

    def get_order_by_id(self, oid: str):
        return self.order_state[oid]

    def cancel_order_by_id(self, oid: str):
        self.cancels.append(oid)
        if oid in self.order_state:
            self.order_state[oid].status = "canceled"


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class FakeSessionState:
    def __init__(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def update_many(self, values):
        self.data.update(values)


def _make_cfg(**over):
    base = dict(
        enabled=True,
        paper_only=True,
        max_concurrent_positions=5,
        max_orders_per_day=10,
        size_usd=1500.0,
        min_gap_pct=5.0,
        min_volume_ratio=3.0,
        min_liquidity_score=70.0,
        min_entry_minutes_before_open=60,
        max_spread_pct=1.0,
        trailing_stop_pct=1.5,
        profit_take_pct=5.0,
        require_sector_alignment=True,
        force_exit_minutes_before_open=15,
        aggressive_exit_minutes_before_open=10,
        emergency_exit_minutes_before_open=1,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _quote(**over):
    # day-manager 2026-05-19 (H1): premarket_bars + premarket_vwap added to
    # default quote so legacy tests pass through the new momentum gate.
    # Tests that need the gate to fail should pass overrides.
    base = dict(
        has_data=True,
        pre_price=12.50,
        last_price=12.50,
        gap_pct=11.0,
        prev_close=11.26,
        bid=12.45,
        ask=12.55,
        volume_ratio=3.0,
        liquidity_score=80.0,
        premarket_vwap=12.40,
        premarket_bars=[
            {"open": 12.30, "close": 12.35, "volume": 1000},
            {"open": 12.35, "close": 12.40, "volume": 1100},
            {"open": 12.40, "close": 12.48, "volume": 1200},
            {"open": 12.48, "close": 12.50, "volume": 2000},  # current
        ],
    )
    base.update(over)
    return base


def _candidate(symbol="ABC", **over):
    base = {
        "symbol": symbol,
        "premarket_volatility_watch": True,
        "premarket_volatility_pct": 11.0,
        "sector_alignment": True,
    }
    base.update(over)
    return base


def _at(hour: int, minute: int) -> datetime:
    # Pick a weekday
    return datetime(2026, 5, 6, hour, minute, 0)


def test_disabled_lane_is_noop():
    cfg = _make_cfg(enabled=False)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),
    )
    assert lane.enter_eligible([_candidate()]) == []
    assert client.submitted == []


def test_paper_only_guard_blocks_nonpaper_client():
    cfg = _make_cfg(paper_only=True)
    client = FakeClient()
    client.paper = False
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),
    )
    assert lane.enter_eligible([_candidate()]) == []
    assert client.submitted == []
    assert lane.diagnostics()["skipped_reasons"].get("paper_only_account_guard") == 1


def test_enters_eligible_gap_up_and_caps_concurrency():
    cfg = _make_cfg(max_concurrent_positions=2)
    client = FakeClient()
    client.auto_fill = False  # leave open
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),
    )
    cands = [_candidate("AAA"), _candidate("BBB"), _candidate("CCC")]
    submitted = lane.enter_eligible(cands)
    assert submitted == ["AAA", "BBB"]
    assert len(client.submitted) == 2
    # daily counter ticks
    assert lane.diagnostics()["buys_submitted_today"] == 2


def test_skips_when_gap_below_threshold():
    cfg = _make_cfg(min_gap_pct=10.0)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(gap_pct=4.0),
        clock_et=lambda: _at(8, 0),
    )
    submitted = lane.enter_eligible([_candidate(premarket_volatility_pct=4.0)])
    assert submitted == []
    assert lane.diagnostics()["skipped_reasons"].get("gap_below_threshold") == 1


def test_extended_hours_flag_set_on_buy():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),
    )
    lane.enter_eligible([_candidate()])
    assert client.submitted, "buy should have been submitted"
    request = client.submitted[0]
    assert getattr(request, "extended_hours", False) is True
    assert getattr(request, "limit_price", 0) > 0
    assert str(getattr(request, "client_order_id", "")).startswith("scalp_buy_")


def test_force_exit_window_inactive_far_from_open():
    cfg = _make_cfg()
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),  # 90 minutes before open, well outside T-15
    )
    lane.enter_eligible([_candidate()])
    assert lane.force_exit_if_due() == []


def test_new_entries_blocked_once_force_exit_window_starts():
    cfg = _make_cfg(force_exit_minutes_before_open=15)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(9, 18),
    )
    assert lane.enter_eligible([_candidate()]) == []
    assert client.submitted == []
    assert lane.diagnostics()["skipped_reasons"].get("entry_window_closed") == 1


def test_new_entries_require_sixty_minute_runway():
    cfg = _make_cfg(min_entry_minutes_before_open=60)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 45),
    )
    assert lane.enter_eligible([_candidate()]) == []
    assert client.submitted == []
    assert lane.diagnostics()["skipped_reasons"].get("entry_window_closed") == 1


def test_spread_above_one_percent_skips_entry():
    cfg = _make_cfg(max_spread_pct=1.0)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(bid=12.00, ask=12.30),
        clock_et=lambda: _at(8, 0),
    )
    assert lane.enter_eligible([_candidate()]) == []
    assert lane.diagnostics()["skipped_reasons"].get("spread_above_threshold") == 1


def test_missing_volume_ratio_skips_entry():
    cfg = _make_cfg(min_volume_ratio=3.0)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(volume_ratio=None),
        clock_et=lambda: _at(8, 0),
    )
    assert lane.enter_eligible([_candidate()]) == []
    assert lane.diagnostics()["skipped_reasons"].get("volume_below_threshold") == 1


def test_sector_alignment_false_skips_entry():
    cfg = _make_cfg(require_sector_alignment=True)
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),
    )
    assert lane.enter_eligible([_candidate(sector_alignment=False)]) == []
    assert lane.diagnostics()["skipped_reasons"].get("sector_alignment_failed") == 1


def test_force_tier_uses_bid_limit():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = True
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    lane.monitor()
    # T-12 minutes — inside force (15) but outside aggressive (10)
    clock["now"] = _at(9, 18)
    exited = lane.force_exit_if_due()
    assert exited == ["ABC"]
    last = client.submitted[-1]
    assert getattr(last, "extended_hours", False) is True
    assert str(getattr(last, "client_order_id", "")).startswith("scalp_exit_")
    # bid touch = $12.45
    assert pytest.approx(last.limit_price, abs=0.01) == 12.45


def test_aggressive_tier_at_t10_uses_bid_times_0p95():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = True
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    lane.monitor()
    # T-8 minutes — inside aggressive (10) but outside emergency (1)
    clock["now"] = _at(9, 22)
    exited = lane.force_exit_if_due()
    assert exited == ["ABC"]
    last = client.submitted[-1]
    assert str(getattr(last, "client_order_id", "")).startswith("scalp_exit_agg_")
    # bid * 0.95 = 12.45 * 0.95 ~= 11.83
    assert pytest.approx(last.limit_price, abs=0.01) == 11.83


def test_emergency_tier_uses_one_cent_limit():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = True
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    lane.monitor()
    # T-30 seconds — inside emergency window
    clock["now"] = _at(9, 29) + timedelta(seconds=30)
    lane.force_exit_if_due()
    last = client.submitted[-1]
    assert str(getattr(last, "client_order_id", "")).startswith("scalp_exit_emerg_")
    assert pytest.approx(last.limit_price, abs=0.001) == 0.01


def test_dynamic_profit_take_full_exit():
    cfg = _make_cfg(profit_take_pct=5.0)
    client = FakeClient()
    client.auto_fill = False
    quote = {"value": _quote(pre_price=12.50, bid=12.45, ask=12.55)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: quote["value"],
        clock_et=lambda: _at(8, 0),
    )
    lane.enter_eligible([_candidate()])
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = 10.00

    quote["value"] = _quote(pre_price=10.55, bid=10.54, ask=10.56)
    lane.monitor()

    last = client.submitted[-1]
    assert str(getattr(last, "client_order_id", "")).startswith("scalp_exit_profit_")
    assert int(last.qty) == pos.qty


def test_dynamic_trailing_stop_full_exit_from_peak():
    cfg = _make_cfg(trailing_stop_pct=1.5)
    client = FakeClient()
    client.auto_fill = False
    quote = {"value": _quote(pre_price=12.50, bid=12.45, ask=12.55)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: quote["value"],
        clock_et=lambda: _at(8, 0),
    )
    lane.enter_eligible([_candidate()])
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = 10.00

    quote["value"] = _quote(pre_price=10.20, bid=10.19, ask=10.21)
    lane.monitor()
    assert lane.open_positions()["ABC"].last_exit_tier == ""

    quote["value"] = _quote(pre_price=10.03, bid=10.02, ask=10.04)
    lane.monitor()

    last = client.submitted[-1]
    assert str(getattr(last, "client_order_id", "")).startswith("scalp_exit_trail_")
    assert int(last.qty) == pos.qty


def test_tiers_escalate_in_sequence_only_resubmitting_on_change():
    """Walk a position through force -> aggressive -> emergency tiers.
    Each tier should produce exactly one new exit order; staying in the
    same tier between cycles must not spam re-submits.
    """
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False  # buy fills via monitor below
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    # simulate buy fill
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = pos.limit_price
    lane.monitor()

    # T-12: force tier kicks in
    clock["now"] = _at(9, 18)
    lane.force_exit_if_due()
    n_after_force = len(client.submitted)
    # Same cycle re-call shouldn't re-submit force at the same tier
    lane.force_exit_if_due()
    assert len(client.submitted) == n_after_force

    # T-8: aggressive tier — one new sell submit
    clock["now"] = _at(9, 22)
    lane.force_exit_if_due()
    assert len(client.submitted) == n_after_force + 1
    assert "scalp_exit_agg_" in client.submitted[-1].client_order_id

    # T-0:30: emergency tier — one more submit
    clock["now"] = _at(9, 29) + timedelta(seconds=30)
    lane.force_exit_if_due()
    assert len(client.submitted) == n_after_force + 2
    assert "scalp_exit_emerg_" in client.submitted[-1].client_order_id


def test_unfilled_buy_cancelled_in_force_window():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False  # buy stays accepted, never fills
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    # T-12, inside force window
    clock["now"] = _at(9, 18)
    lane.force_exit_if_due()
    # the buy order id should have been cancelled
    assert client.cancels, "force-exit should cancel unfilled buy"
    # no sell submitted because nothing filled
    sell_orders = [
        r
        for r in client.submitted
        if str(getattr(r, "side", "")).lower().endswith("sell")
    ]
    assert sell_orders == []
    # position marked flat
    assert "ABC" not in lane.open_positions()


def test_loser_blocks_reentry_when_exit_below_entry():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    pos = lane.open_positions()["ABC"]
    # buy fills at 12.55
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = pos.limit_price
    lane.monitor()

    # T-12: force-tier sell at bid (12.45). Make it fill at a loss.
    clock["now"] = _at(9, 18)
    lane.force_exit_if_due()
    # find the latest exit order id
    pos = lane.open_positions().get("ABC") or list(lane._positions.values())[0]
    client.order_state[pos.exit_order_id].status = "filled"
    client.order_state[pos.exit_order_id].filled_qty = pos.qty
    client.order_state[pos.exit_order_id].filled_avg_price = 11.00  # loss
    lane.monitor()

    assert lane.is_blocked_for_reentry("ABC")
    assert "ABC" in lane.loser_symbols_today()


def test_state_persists_orders_positions_and_losers_across_restart():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False
    state = FakeSessionState()
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
        session_state=state,
    )
    lane.enter_eligible([_candidate()])
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = pos.limit_price
    lane.monitor()
    clock["now"] = _at(9, 18)
    lane.force_exit_if_due()
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.exit_order_id].status = "filled"
    client.order_state[pos.exit_order_id].filled_qty = pos.qty
    client.order_state[pos.exit_order_id].filled_avg_price = pos.limit_price - 0.50
    lane.monitor()

    restarted = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
        session_state=state,
    )
    restarted.reset_daily_state_if_new_session()

    assert restarted.diagnostics()["buys_submitted_today"] == 1
    assert restarted.is_blocked_for_reentry("ABC")
    assert state.data["scalp_positions"]["ABC"]["status"] == "flat"


def test_winner_does_not_block_reentry():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = pos.limit_price
    lane.monitor()

    clock["now"] = _at(9, 18)
    lane.force_exit_if_due()
    pos = lane.open_positions().get("ABC") or list(lane._positions.values())[0]
    # exit fills above entry — profit
    client.order_state[pos.exit_order_id].status = "filled"
    client.order_state[pos.exit_order_id].filled_qty = pos.qty
    client.order_state[pos.exit_order_id].filled_avg_price = pos.limit_price + 0.10
    lane.monitor()

    assert not lane.is_blocked_for_reentry("ABC")


def test_profitable_reentry_requires_support_bounce_confirmation():
    cfg = _make_cfg(max_orders_per_day=3)
    client = FakeClient()
    client.auto_fill = False
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    pos = lane.open_positions()["ABC"]
    pos.status = "flat"
    pos.exit_filled_qty = pos.qty
    pos.filled_avg_price = 10.00
    pos.exit_filled_avg_price = 10.50

    assert lane.enter_eligible([_candidate()]) == []
    assert lane.diagnostics()["skipped_reasons"].get("support_bounce_missing") == 1

    submitted = lane.enter_eligible([_candidate(support_bounce_confirmed=True)])
    assert submitted == ["ABC"]


def test_finalize_at_open_blocks_stuck_position():
    cfg = _make_cfg()
    client = FakeClient()
    client.auto_fill = False
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate()])
    # buy fills, but we never exit
    pos = lane.open_positions()["ABC"]
    client.order_state[pos.buy_order_id].status = "filled"
    client.order_state[pos.buy_order_id].filled_qty = pos.qty
    client.order_state[pos.buy_order_id].filled_avg_price = pos.limit_price
    lane.monitor()

    # Cross 09:30 — auto-finalize via monitor() should flag the symbol
    clock["now"] = _at(9, 31)
    lane.monitor()
    assert lane.is_blocked_for_reentry("ABC")


def test_daily_order_cap_blocks_further_entries():
    cfg = _make_cfg(max_concurrent_positions=20, max_orders_per_day=2)
    client = FakeClient()
    client.auto_fill = False
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(8, 0),
    )
    lane.enter_eligible([_candidate("A"), _candidate("B")])
    lane.enter_eligible([_candidate("C")])  # should be capped
    assert lane.diagnostics()["buys_submitted_today"] == 2
    assert "C" not in lane.open_positions()
    assert lane.diagnostics()["skipped_reasons"].get("daily_order_cap_reached", 0) >= 1


def test_enter_eligible_writes_skipped_and_cycle_summary_telemetry(tmp_path):
    cfg = _make_cfg(telemetry_path=str(tmp_path / "scalp_telemetry.jsonl"))
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: {"has_data": False},
        clock_et=lambda: _at(8, 0),
    )

    assert lane.enter_eligible([_candidate("MISS")]) == []

    telemetry_path = tmp_path / f"scalp_telemetry_{_at(8, 0):%Y-%m-%d}.jsonl"
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(row.get("event") == "skipped" for row in rows)
    summary = [row for row in rows if row.get("event") == "cycle_summary"]
    assert len(summary) == 1
    assert summary[0]["candidates"] == 1
    assert summary[0]["skipped"] == 1


def test_after_market_open_no_force_exit_no_entries():
    cfg = _make_cfg()
    client = FakeClient()
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: _at(9, 30),  # exactly at open
    )
    # At/after open, force_exit_if_due returns [] (the lane stops acting)
    assert lane.force_exit_if_due() == []
    assert lane.enter_eligible([_candidate()]) == []
    assert client.submitted == []


def test_session_reset_clears_counters_on_new_day():
    cfg = _make_cfg(max_orders_per_day=1)
    client = FakeClient()
    client.auto_fill = False
    clock = {"now": _at(8, 0)}
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=client,
        quote_provider=lambda s: _quote(),
        clock_et=lambda: clock["now"],
    )
    lane.enter_eligible([_candidate("A")])
    assert lane.diagnostics()["buys_submitted_today"] == 1
    # Advance to next day
    clock["now"] = clock["now"] + timedelta(days=1)
    lane.enter_eligible([_candidate("B")])
    assert lane.diagnostics()["buys_submitted_today"] == 1  # reset, then new buy
    assert "A" not in lane.open_positions()
    assert "B" in lane.open_positions()


def test_agent_scalp_maintenance_only_skips_new_entries():
    class _Lane:
        def __init__(self):
            self.calls = []

        def monitor(self):
            self.calls.append("monitor")

        def force_exit_if_due(self):
            self.calls.append("force_exit")
            return ["ABC"]

        def enter_eligible(self, candidates):
            self.calls.append(("enter", candidates))
            return ["ABC"]

    lane = _Lane()
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent._get_premarket_scalp_lane = lambda: lane

    agent._run_premarket_scalp_cycle([_candidate()], allow_entries=False)

    assert lane.calls == ["monitor", "force_exit"]


def test_agent_paper_only_guard_fails_closed_when_credentials_live(monkeypatch):
    from autotrade.utils import alpaca_client_factory

    monkeypatch.setattr(
        alpaca_client_factory,
        "resolve_alpaca_credentials",
        lambda require=False: SimpleNamespace(paper=False),
    )
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.alpaca_client = SimpleNamespace()
    agent._premarket_scalp_lane = None
    agent.premarket_scalp_cfg = _make_cfg(paper_only=True)

    assert agent._get_premarket_scalp_lane() is None
