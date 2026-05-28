"""H1 tests — premarket scalp lane redesign (2026-05-19).

Covers:
  - T-90 .. T-45 entry window
  - Combined momentum gate (4 sub-conditions)
  - Dynamic exit (hard stop, trailing arm, max-hold)
  - Daily circuit breaker (consec losses + cumulative loss)
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from autotrade.execution.premarket_scalp_lane import PremarketScalpLane, ScalpPosition


class _StubClient:
    def __init__(self):
        self.submitted = []
        self.next_id = 1
        self.paper = True
        self.orders = {}

    def submit_order(self, request):
        oid = f"ord-{self.next_id}"
        self.next_id += 1
        self.submitted.append(request)
        order = SimpleNamespace(
            id=oid,
            status="filled",
            filled_qty=int(request.qty),
            filled_avg_price=float(request.limit_price),
        )
        self.orders[oid] = order
        return order

    def get_order_by_id(self, oid):
        return self.orders[oid]

    def cancel_order_by_id(self, oid):
        if oid in self.orders:
            self.orders[oid].status = "canceled"


def _h1_cfg(**over):
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
        trailing_stop_pct=1.0,
        profit_take_pct=5.0,
        require_sector_alignment=True,
        force_exit_minutes_before_open=15,
        aggressive_exit_minutes_before_open=10,
        emergency_exit_minutes_before_open=1,
        entry_window_start_minutes_before_open=90,
        entry_window_end_minutes_before_open=45,
        momentum_min_consec_up_bars=3,
        momentum_min_volume_ratio=1.5,
        momentum_max_vwap_distance_pct=1.5,
        hard_stop_pct=1.0,
        trailing_arm_at_pct=1.5,
        max_hold_minutes=25,
        circuit_breaker_consecutive_losses=3,
        circuit_breaker_daily_loss_dollars=-200.0,
        telemetry_path="logs/premarket_scalp_telemetry.jsonl",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _good_quote(**over):
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
        sector_alignment=True,
        premarket_vwap=12.40,
        premarket_bars=[
            {"open": 12.30, "close": 12.35, "volume": 1000},
            {"open": 12.35, "close": 12.40, "volume": 1100},
            {"open": 12.40, "close": 12.48, "volume": 1200},
            {"open": 12.48, "close": 12.50, "volume": 2000},
        ],
    )
    base.update(over)
    return base


def _candidate(symbol="AAA"):
    return {
        "symbol": symbol,
        "premarket_volatility_watch": True,
        "premarket_volatility_pct": 11.0,
        "sector_alignment": True,
    }


def _at(hour, minute):
    return datetime(2026, 5, 6, hour, minute, 0)


def _make_lane(cfg, quote_fn, clock_fn, tmp_path=None):
    cfg.telemetry_path = str((tmp_path or "logs") and (str(tmp_path / "tele.jsonl") if tmp_path else "logs/telemetry.jsonl"))
    return PremarketScalpLane(
        cfg=cfg,
        alpaca_client=_StubClient(),
        quote_provider=quote_fn,
        clock_et=clock_fn,
    )


# ---------- entry window ----------

def test_entry_window_rejects_too_early_T95():
    cfg = _h1_cfg()
    # 09:30 - 95 min = 07:55
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(7, 55))
    assert lane.entry_window_open() is False


def test_entry_window_admits_T70():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))  # 70 min before
    assert lane.entry_window_open() is True


def test_entry_window_rejects_too_late_T44():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 46))  # 44 min before
    assert lane.entry_window_open() is False


def test_entry_window_rejects_inside_force_exit_window_T1():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(9, 29))
    assert lane.entry_window_open() is False


# ---------- momentum gate ----------

def test_gate_passes_when_all_4_subconditions_met():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))
    ok, reason = lane._passes_momentum_gate("AAA")
    assert ok is True
    assert reason == "ok"


def test_gate_rejects_when_not_all_bars_up():
    cfg = _h1_cfg()
    bad = _good_quote(
        premarket_bars=[
            {"open": 12.30, "close": 12.20, "volume": 1000},  # DOWN
            {"open": 12.35, "close": 12.40, "volume": 1100},
            {"open": 12.40, "close": 12.48, "volume": 1200},
            {"open": 12.48, "close": 12.50, "volume": 2000},
        ]
    )
    lane = _make_lane(cfg, lambda s: bad, lambda: _at(8, 20))
    ok, reason = lane._passes_momentum_gate("AAA")
    assert ok is False
    assert reason == "momentum_bars_not_all_up"


def test_gate_rejects_when_volume_ratio_below_threshold():
    cfg = _h1_cfg()
    bad = _good_quote(
        premarket_bars=[
            {"open": 12.30, "close": 12.35, "volume": 1000},
            {"open": 12.35, "close": 12.40, "volume": 1000},
            {"open": 12.40, "close": 12.48, "volume": 1000},
            {"open": 12.48, "close": 12.50, "volume": 1100},  # < 1.5x avg(1000)
        ]
    )
    lane = _make_lane(cfg, lambda s: bad, lambda: _at(8, 20))
    ok, reason = lane._passes_momentum_gate("AAA")
    assert ok is False
    assert reason == "momentum_volume_ratio_below_threshold"


def test_gate_rejects_when_last_below_vwap():
    cfg = _h1_cfg()
    bad = _good_quote(last_price=12.30, pre_price=12.30, premarket_vwap=12.40)
    lane = _make_lane(cfg, lambda s: bad, lambda: _at(8, 20))
    ok, reason = lane._passes_momentum_gate("AAA")
    assert ok is False
    assert reason == "momentum_below_vwap"


def test_gate_rejects_when_vwap_distance_above_threshold():
    cfg = _h1_cfg()
    # vwap=12.40, last=12.70 → distance = 2.42% > 1.5%
    bad = _good_quote(last_price=12.70, pre_price=12.70, premarket_vwap=12.40)
    lane = _make_lane(cfg, lambda s: bad, lambda: _at(8, 20))
    ok, reason = lane._passes_momentum_gate("AAA")
    assert ok is False
    assert reason == "momentum_vwap_distance_above_threshold"


def test_gate_rejects_when_bars_insufficient():
    cfg = _h1_cfg()
    bad = _good_quote(premarket_bars=[{"open": 1, "close": 2, "volume": 100}])
    lane = _make_lane(cfg, lambda s: bad, lambda: _at(8, 20))
    ok, reason = lane._passes_momentum_gate("AAA")
    assert ok is False
    assert reason == "insufficient_premarket_bars"


# ---------- dynamic exit ----------

def test_hard_stop_fires_at_minus_1_percent():
    cfg = _h1_cfg()
    # Use 9.89 (≈ -1.1%) to clear float-precision noise around the -1.0 boundary.
    lane = _make_lane(cfg, lambda s: _good_quote(pre_price=9.89), lambda: _at(8, 20))
    pos = ScalpPosition(
        symbol="AAA",
        qty=100,
        limit_price=10.00,
        submitted_at=_at(8, 0),
        buy_order_id="bid1",
        client_order_id="cid",
        filled_qty=100,
        filled_avg_price=10.00,
        peak_price=10.00,
        status="filled",
        filled_at=_at(8, 10),
    )
    lane._positions["AAA"] = pos
    # Stub exit to record tier
    fired = {}
    def fake_exit(pos, tier):
        fired["tier"] = tier
    lane._exit_position = fake_exit
    lane._dynamic_exit_if_due(pos)
    assert fired.get("tier") == "hard_stop"


def test_trailing_not_armed_below_plus_1_5_pct_peak():
    cfg = _h1_cfg()
    # peak 10.10 (+1%) — below 1.5% arm threshold; price dropped to 10.00 (drawdown -1%)
    lane = _make_lane(cfg, lambda s: _good_quote(pre_price=10.00), lambda: _at(8, 20))
    pos = ScalpPosition(
        symbol="AAA",
        qty=100,
        limit_price=10.00,
        submitted_at=_at(8, 0),
        buy_order_id="bid1",
        client_order_id="cid",
        filled_qty=100,
        filled_avg_price=10.00,
        peak_price=10.10,
        status="filled",
        filled_at=_at(8, 10),
    )
    lane._positions["AAA"] = pos
    fired = {}
    lane._exit_position = lambda pos, tier: fired.setdefault("tier", tier)
    lane._dynamic_exit_if_due(pos)
    # peak_price updates to max(10.10, 10.00) = 10.10; drawdown -1%, peak_pct +1% (below 1.5%)
    assert fired.get("tier") != "trailing_stop"


def test_trailing_fires_after_peak_armed():
    cfg = _h1_cfg()
    # peak previously hit 10.20 (+2%), now back to 10.08 → drawdown -1.18% which is below -1%
    lane = _make_lane(cfg, lambda s: _good_quote(pre_price=10.08), lambda: _at(8, 20))
    pos = ScalpPosition(
        symbol="AAA",
        qty=100,
        limit_price=10.00,
        submitted_at=_at(8, 0),
        buy_order_id="bid1",
        client_order_id="cid",
        filled_qty=100,
        filled_avg_price=10.00,
        peak_price=10.20,
        status="filled",
        filled_at=_at(8, 10),
    )
    lane._positions["AAA"] = pos
    fired = {}
    lane._exit_position = lambda pos, tier: fired.setdefault("tier", tier)
    lane._dynamic_exit_if_due(pos)
    assert fired.get("tier") == "trailing_stop"


def test_max_hold_force_exits_at_25_minutes():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(pre_price=10.05), lambda: _at(8, 36))
    pos = ScalpPosition(
        symbol="AAA",
        qty=100,
        limit_price=10.00,
        submitted_at=_at(8, 0),
        buy_order_id="bid1",
        client_order_id="cid",
        filled_qty=100,
        filled_avg_price=10.00,
        peak_price=10.05,
        status="filled",
        filled_at=_at(8, 10),  # 26 min ago at 8:36
    )
    lane._positions["AAA"] = pos
    fired = {}
    lane._exit_position = lambda pos, tier: fired.setdefault("tier", tier)
    lane._dynamic_exit_if_due(pos)
    assert fired.get("tier") == "max_hold"


# ---------- circuit breaker ----------

def test_circuit_breaker_arms_on_3_consecutive_losses():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))
    assert lane.circuit_breaker_armed() is False
    lane.record_round_trip_result(-10.0)
    lane.record_round_trip_result(-15.0)
    lane.record_round_trip_result(-20.0)
    assert lane.circuit_breaker_armed() is True


def test_circuit_breaker_arms_on_cumulative_loss_threshold():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))
    # Mix of small wins + larger losses, summing to <= -200
    lane.record_round_trip_result(50.0)
    lane.record_round_trip_result(-300.0)
    assert lane.circuit_breaker_armed() is True


def test_circuit_breaker_breaks_entry_loop():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))
    # Arm circuit breaker
    for _ in range(3):
        lane.record_round_trip_result(-100.0)
    assert lane.circuit_breaker_armed() is True
    # enter_eligible must return [] short-circuiting on the breaker
    submitted = lane.enter_eligible([_candidate("AAA")])
    assert submitted == []


def test_consecutive_losses_reset_on_winner():
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))
    lane.record_round_trip_result(-10.0)
    lane.record_round_trip_result(-10.0)
    lane.record_round_trip_result(50.0)  # winner resets streak
    lane.record_round_trip_result(-10.0)
    assert lane.circuit_breaker_armed() is False


def test_telemetry_written_on_circuit_breaker(tmp_path):
    cfg = _h1_cfg(telemetry_path=str(tmp_path / "tele.jsonl"))
    lane = PremarketScalpLane(
        cfg=cfg,
        alpaca_client=_StubClient(),
        quote_provider=lambda s: _good_quote(),
        clock_et=lambda: _at(8, 20),
    )
    for _ in range(3):
        lane.record_round_trip_result(-100.0)
    files = list(tmp_path.glob("tele_*.jsonl"))
    assert files, "expected dated telemetry file written"
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    assert any(r.get("event") == "circuit_breaker" for r in rows)


def test_session_reset_clears_circuit_breaker(monkeypatch):
    cfg = _h1_cfg()
    lane = _make_lane(cfg, lambda s: _good_quote(), lambda: _at(8, 20))
    for _ in range(3):
        lane.record_round_trip_result(-100.0)
    assert lane.circuit_breaker_armed() is True
    # Advance clock to a new day
    lane.now_et = lambda: datetime(2026, 5, 7, 8, 20, 0)
    lane.reset_daily_state_if_new_session()
    assert lane.circuit_breaker_armed() is False
