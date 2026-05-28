from types import SimpleNamespace
from datetime import datetime

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import TomorrowsPlanGenerator
from autotrade.signals import trade_learner


class _LoggerStub:
    def __init__(self):
        self.messages = {"info": [], "warning": [], "error": [], "debug": []}

    def info(self, *args, **kwargs):
        self.messages["info"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def warning(self, *args, **kwargs):
        self.messages["warning"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def error(self, *args, **kwargs):
        self.messages["error"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def debug(self, *args, **kwargs):
        self.messages["debug"].append(args[0] % args[1:] if len(args) > 1 else args[0])


class _FilledOrderClient:
    def __init__(self):
        self.orders = []

    def submit_order(self, request):
        self.orders.append(request)
        return SimpleNamespace(
            id=f"buy-{request.symbol.lower()}",
            filled_qty=str(request.qty),
            filled_avg_price="10.25",
        )


def _entry_order(symbol: str, entry_source: str) -> dict:
    return {
        "symbol": symbol,
        "qty": 10,
        "entry_price": 10.0,
        "score": 82.0,
        "setup_type": "momentum_breakout",
        "strategy_name": "unit_test_strategy",
        "entry_source": entry_source,
        "plan_score_source": "pm_plan_2026-04-23.json",
        "source_bucket": "watchlist",
    }


def test_execute_plan_records_filled_buy_entries_with_order_ids(monkeypatch):
    recorded_entries = []

    def fake_record_entry(**kwargs):
        recorded_entries.append(kwargs)
        return f"trade-{kwargs['symbol']}"

    monkeypatch.setattr(trade_learner, "record_entry", fake_record_entry)
    monkeypatch.setattr(autonomous_agent_mod.time, "sleep", lambda seconds: None)

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _LoggerStub()
    generator.alpaca_client = _FilledOrderClient()
    generator.get_current_positions = lambda: []
    generator._get_market_context = lambda: {}
    generator._get_current_price_quick = lambda symbol: 10.0
    generator._entry_guard_reasons = lambda symbol, entry_score: []
    generator._evaluate_trade_with_llm = lambda order, current_price, market_context: {
        "execute": True,
        "adjusted_qty": order["qty"],
        "confidence": 84.0,
        "reasoning": "unit test approval",
        "gap_pct": 0.0,
    }
    generator._log_trade_decision = lambda decision, order, executed: None

    result = generator.execute_plan(
        {
            "cleanup_orders": [],
            "entry_orders": [
                _entry_order("PLAN", "overnight_plan"),
                _entry_order("MOMO", "intraday_momentum"),
            ],
        },
        dry_run=False,
    )

    assert result["entries"] == 2
    assert [entry["symbol"] for entry in recorded_entries] == ["PLAN", "MOMO"]
    assert all(entry["entry_status"] == "filled" for entry in recorded_entries)
    assert all(entry["order_id"] for entry in recorded_entries)
    assert all(entry["filled_qty"] == 10 for entry in recorded_entries)
    assert {entry["signal_context"]["entry_source"] for entry in recorded_entries} == {
        "overnight_plan",
        "intraday_momentum",
    }


def test_execute_plan_blocks_new_entries_during_market_fade(monkeypatch, tmp_path):
    monkeypatch.setattr(
        autonomous_agent_mod,
        "get_market_now",
        lambda: datetime(2026, 4, 24, 15, 35),
    )
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _LoggerStub()
    generator.alpaca_client = _FilledOrderClient()
    generator.get_current_positions = lambda: []
    generator._get_market_context = lambda: {}
    generator._entry_guard_reasons = lambda symbol, entry_score: []
    generator._get_current_price_quick = lambda symbol: 10.0
    generator._evaluate_trade_with_llm = lambda order, current_price, market_context: {
        "execute": True,
        "adjusted_qty": order["qty"],
        "confidence": 84.0,
        "reasoning": "unit test approval",
    }

    result = generator.execute_plan(
        {
            "cleanup_orders": [],
            "entry_orders": [_entry_order("FADE", "overnight_plan")],
        },
        dry_run=False,
    )

    assert result["entries"] == 0
    assert result["skipped"] == 1
    assert result["skipped_symbols"] == ["FADE"]
    assert result["block_reason"] == "market_fade_lockout:after_1530_et"
    assert generator.alpaca_client.orders == []


def test_execute_plan_blocks_prior_close_entry_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)

    generator = TomorrowsPlanGenerator.__new__(TomorrowsPlanGenerator)
    generator.logger = _LoggerStub()
    generator.alpaca_client = _FilledOrderClient()
    generator.get_current_positions = lambda: []
    generator._get_market_context = lambda: {}
    generator._entry_guard_reasons = lambda symbol, entry_score: []
    generator._get_current_price_quick = lambda symbol: 10.75
    generator._evaluate_trade_with_llm = lambda order, current_price, market_context: {
        "execute": True,
        "adjusted_qty": order["qty"],
        "confidence": 84.0,
        "reasoning": "unit test approval",
    }

    order = _entry_order("DRIFT", "overnight_plan")
    order["entry_price"] = 11.50
    order["prev_close"] = 10.00

    result = generator.execute_plan(
        {
            "cleanup_orders": [],
            "entry_orders": [order],
        },
        dry_run=False,
    )

    assert result["entries"] == 0
    assert result["skipped"] == 1
    assert result["skipped_symbols"] == ["DRIFT"]
    assert result["block_reason"] == "prev_close_entry_drift"
    assert generator.alpaca_client.orders == []
