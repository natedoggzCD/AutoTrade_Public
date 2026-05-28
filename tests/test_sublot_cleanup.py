from types import SimpleNamespace

from tests.test_day_manager_execution_policy import _new_dm_stub


def _base_dm():
    dm = _new_dm_stub()
    dm._position_qty = lambda pos: int(float(getattr(pos, "qty", 0) or 0))
    dm._position_float = lambda pos, field, default=0.0, absolute=False: (
        abs(float(getattr(pos, field, default) or default))
        if absolute
        else float(getattr(pos, field, default) or default)
    )
    dm._allow_short_selling = lambda: False
    dm._has_open_buy_order = lambda symbol: (False, "")
    dm.cycle_count = 10
    return dm


def test_validate_positions_flags_proportional_sublot_for_nbis_like_case():
    dm = _base_dm()
    pos = SimpleNamespace(
        symbol="NBIS",
        qty=4,
        current_price=40.0,
        market_value=160.0,
        unrealized_plpc=0.02,
    )

    stats = dm._validate_positions([pos])

    assert stats["sub_lot_flags"] == 1
    assert stats["sub_lot_positions"][0]["symbol"] == "NBIS"
    assert stats["sub_lot_positions"][0]["reason"] == "qty_below_half_intended_lot"


def test_build_stub_cleanup_targets_prefers_sublot_reason_over_static_floor():
    dm = _base_dm()
    pos = SimpleNamespace(symbol="ZLAB", qty=3, current_price=250.0, market_value=750.0)
    validation = {
        "sub_lot_positions": [
            {
                "symbol": "ZLAB",
                "qty": 3.0,
                "intended_lot": 12,
            }
        ]
    }

    targets = dm._build_stub_cleanup_targets([pos], validation)

    assert len(targets) == 1
    assert targets[0]["symbol"] == "ZLAB"
    assert targets[0]["reason"] == "sublot_flat_close"


def test_build_stub_cleanup_targets_emits_one_target_per_symbol():
    dm = _base_dm()
    pos_a = SimpleNamespace(
        symbol="NBIS", qty=2, current_price=200.0, market_value=400.0
    )
    pos_b = SimpleNamespace(
        symbol="NBIS", qty=2, current_price=200.0, market_value=400.0
    )
    validation = {
        "sub_lot_positions": [
            {"symbol": "NBIS", "qty": 2.0, "intended_lot": 15},
        ]
    }

    targets = dm._build_stub_cleanup_targets([pos_a, pos_b], validation)

    assert len(targets) == 1
    assert targets[0]["symbol"] == "NBIS"


def test_validate_order_allows_dynamic_replacement_buy_below_static_odd_lot_floor():
    dm = _base_dm()
    dm.get_current_price = lambda symbol: 1200.0

    valid_replacement, reason_replacement = dm._validate_order(
        "PTRN",
        2,
        "buy",
        replacement_for_symbol="NBIS",
        intended_lot=2,
    )
    valid_regular, reason_regular = dm._validate_order("PTRN", 2, "buy")

    assert valid_replacement is True
    assert reason_replacement == ""
    assert valid_regular is False
    assert reason_regular == "odd_lot_buy_qty"


def test_process_sublot_replacement_attempts_buy_within_two_cycles():
    dm = _base_dm()
    dm._sublot_replacement_pending = {"NBIS": {"exit_cycle": 9, "status": "pending"}}
    calls = []

    def _fake_execute_entry(
        ticker,
        reason,
        candidate_data=None,
        entry_wave=None,
        preflight_only=False,
        replacement_for_symbol=None,
    ):
        calls.append((ticker, reason, replacement_for_symbol))
        return False

    dm.execute_entry = _fake_execute_entry
    candidates = [{"ticker": "ZLAB", "score": 88.0}]
    held_tickers = ["NBIS"]
    stats = {"entries": 0, "replacements": 0}

    dm._process_sublot_replacement_pending(candidates, held_tickers, stats)

    assert calls == [("ZLAB", "sublot_replacement_for_NBIS", "NBIS")]
    assert dm._sublot_replacement_pending == {}


def test_process_sublot_replacement_logs_no_candidate_when_none_available(caplog):
    dm = _base_dm()
    dm._sublot_replacement_pending = {"NBIS": {"exit_cycle": 8, "status": "pending"}}
    dm.execute_entry = lambda *args, **kwargs: True
    stats = {"entries": 0, "replacements": 0}

    dm._process_sublot_replacement_pending([], ["NBIS"], stats)

    assert dm._sublot_replacement_pending == {}
    assert "no_replacement_candidate" in caplog.text
