from langgraph_workflow.nodes import node_risk_gate


def _make_state(entry_price: float, current_price: float, hold_minutes: float = 0.0):
    return {
        "position": {
            "symbol": "CRML",
            "entry_price": entry_price,
            "current_price": current_price,
            "hold_duration_minutes": hold_minutes,
        },
        "market_data": {
            "current_price": current_price,
            "atr_pct": 3.0,
        },
        "risk_limits": {},
    }


def test_risk_gate_node_clamps_conviction_score_for_big_winner():
    result = node_risk_gate(_make_state(entry_price=100.0, current_price=130.0))

    score = result["risk_gate_result"]["conviction_score"]
    assert score == 100.0
    assert 0.0 <= score <= 100.0


def test_risk_gate_node_lowers_conviction_score_for_big_loser():
    result = node_risk_gate(_make_state(entry_price=100.0, current_price=70.0))

    score = result["risk_gate_result"]["conviction_score"]
    assert score == 0.0
    assert 0.0 <= score <= 100.0
