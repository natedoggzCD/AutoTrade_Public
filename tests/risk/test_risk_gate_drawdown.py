from autotrade.risk.risk_gate import RiskGate, RiskGateConfig, RiskAction


def _make_gate(drawdown_levels, atr_k=0.0):
    cfg = RiskGateConfig(
        drawdown_trim_levels=drawdown_levels,
        drawdown_trail_atr_k=atr_k,
        hard_stop_pct=-50.0,
        trim_threshold_pct=-20.0,
        atr_k=2.0,
        atr_trail_k=1.5,
    )
    return RiskGate(cfg)


def test_drawdown_trim_first_level():
    gate = _make_gate(
        [
            {"drawdown_pct": 3.0, "trim": 0.33},
            {"drawdown_pct": 6.0, "trim": 0.45},
            {"drawdown_pct": 10.0, "trim": 1.0},
        ]
    )
    plan = gate.evaluate(
        symbol="TEST",
        entry_price=100.0,
        current_price=97.0,
        qty=100,
        atr_pct=0.0,
        time_in_trade_minutes=30,
        high_since_entry=100.0,
    )
    assert plan.action == RiskAction.TRIM
    assert plan.size_delta == -33
    assert "drawdown_trim_3" in plan.triggered_rules


def test_drawdown_exit_full_level():
    gate = _make_gate(
        [
            {"drawdown_pct": 3.0, "trim": 0.33},
            {"drawdown_pct": 6.0, "trim": 0.45},
            {"drawdown_pct": 10.0, "trim": 1.0},
        ]
    )
    plan = gate.evaluate(
        symbol="TEST",
        entry_price=100.0,
        current_price=90.0,
        qty=100,
        atr_pct=0.0,
        time_in_trade_minutes=30,
        high_since_entry=100.0,
    )
    assert plan.action == RiskAction.EXIT
    assert plan.size_delta == -100
    assert "drawdown_trim_10" in plan.triggered_rules


def test_drawdown_atr_adjustment_delays_trigger():
    gate = _make_gate(
        [
            {"drawdown_pct": 3.0, "trim": 0.33},
            {"drawdown_pct": 6.0, "trim": 0.45},
            {"drawdown_pct": 10.0, "trim": 1.0},
        ],
        atr_k=0.5,
    )
    plan = gate.evaluate(
        symbol="TEST",
        entry_price=100.0,
        current_price=96.5,  # 3.5% drawdown
        qty=100,
        atr_pct=2.0,  # +1.0% adjustment => trigger at 4.0%
        time_in_trade_minutes=30,
        high_since_entry=100.0,
    )
    assert plan.action == RiskAction.NONE
