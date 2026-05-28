import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent


class _LoggerStub:
    def __init__(self):
        self.messages = {"info": [], "warning": [], "error": [], "critical": []}

    def info(self, *args, **kwargs):
        self.messages["info"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def warning(self, *args, **kwargs):
        self.messages["warning"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def error(self, *args, **kwargs):
        self.messages["error"].append(args[0] % args[1:] if len(args) > 1 else args[0])

    def critical(self, *args, **kwargs):
        self.messages["critical"].append(args[0] % args[1:] if len(args) > 1 else args[0])


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 4, 23, 10, 16, 0)
        return value if tz is None else value.replace(tzinfo=tz)


class _PlanGeneratorStub:
    def __init__(self, plan):
        self.plan = plan
        self.logged_decisions = []

    def _load_latest_plan(self):
        return self.plan

    def _coerce_float(self, value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    def _coerce_int(self, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _get_current_price_quick(self, symbol):
        return 10.0

    def get_current_positions(self):
        return []

    def get_account_info(self):
        return {"equity": 100_000.0, "cash": 80_000.0}

    def _log_trade_decision(self, decision, order, executed):
        self.logged_decisions.append((order["symbol"], bool(executed)))


class _DayManagerStub:
    def __init__(self):
        self.signal_status = {}
        self.preflighted = []
        self.submitted = []

    def _effective_max_positions(self):
        return 20

    def execute_entry(
        self,
        symbol,
        reason,
        candidate_data=None,
        entry_wave=None,
        preflight_only=False,
    ):
        if preflight_only:
            self.preflighted.append(symbol)
            if symbol == "FAIL":
                self.signal_status[symbol] = {
                    "reason": "buy_guard:cash_floor_breached"
                }
                return False
            return True
        self.submitted.append(symbol)
        return True


def _candidate(symbol: str, score: float) -> dict:
    return {
        "symbol": symbol,
        "ticker": symbol,
        "entry_price": 10.0,
        "score": score,
        "confidence": score,
        "entry_source": "overnight_plan",
        "plan_score_source": "pm_plan_2026-04-23.json",
    }


def test_pm_plan_entry_candidates_are_evaluated_independently(tmp_path, monkeypatch):
    plans_dir = Path(tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)

    candidates = [_candidate("FAIL", 95.0)] + [
        _candidate(f"PASS{i}", 90.0 - i) for i in range(1, 10)
    ]
    execution_state_path = plans_dir / ".execution_state_20260423.json"
    execution_state_path.write_text(
        json.dumps(
            {
                "opening_snapshot": {
                    row["symbol"]: {"open_price": 10.0, "planned_entry": 10.0}
                    for row in candidates
                },
                "executed_symbols": [],
                "skipped_symbols": [],
            }
        ),
        encoding="utf-8",
    )

    dm = _DayManagerStub()
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.plan_generator = _PlanGeneratorStub({"entry_candidates": candidates})
    agent.entry_quality_cfg = SimpleNamespace()
    agent.core_max_positions = 20
    agent.reserve_max_positions = 0
    agent.max_positions = 20
    agent._position_slot_class_by_symbol = {}
    agent._refresh_strategy_failsafe = lambda source=None: SimpleNamespace(
        halt_new_entries=False
    )
    agent._apply_plan_entry_constraints = lambda plan, source="plan": {
        "max_positions": 20,
        "core_max_positions": 20,
        "reserve_max_positions": 0,
        "source": source,
    }
    agent._get_day_manager = lambda dry_run=True: dm
    agent._get_current_position_symbols = lambda: set()
    agent._decision_claw_capital_snapshot = lambda current_positions, account: {
        "deployed_pct": 0.0,
        "total_equity": 100_000.0,
    }
    agent._wave_entry_regime_gate = lambda plan, current_positions=None: (True, "")
    agent._slot_class_for_order = lambda order: "core"
    agent._slot_class_for_symbol = lambda symbol: "core"
    agent._wave_hard_reject_gap_pct = lambda: 10.0
    agent._is_wave_breakout_rescue_candidate = lambda **kwargs: False
    agent._wave_max_chase_pct = lambda score, wave: 10.0
    agent._compute_wave_limit_price = lambda **kwargs: 10.01
    agent._upsert_signals_into_dm = lambda *args, **kwargs: None

    agent._execute_entry_waves(execute=False)

    saved = json.loads(execution_state_path.read_text(encoding="utf-8"))
    assert dm.preflighted == [row["symbol"] for row in candidates]
    assert dm.submitted == [row["symbol"] for row in candidates[1:]]
    assert saved["pm_candidates_evaluated"] == 10
    assert saved["pm_candidates_submitted"] == 9
    assert saved["pm_gate_reason_counts"]["buy_guard:cash_floor_breached"] == 1
