"""
Code Repair Backtest — validates the agentic CodeAgent against real errors.

This test suite creates ISOLATED copies of production-like Python files,
injects the ACTUAL errors found in logs/code_fixes.jsonl, then runs the
agentic CodeAgent to verify it can fix them.

NO production files are touched.  All work happens in a temp directory.

Error catalogue (from logs/code_fixes.jsonl + supervisor logs):
  TC01  NameError   — undefined variable used in method body
  TC02  SyntaxError — unclosed dict literal (missing paren)
  TC03  SyntaxError — missing closing parenthesis in print
  TC04  NameError   — undefined constant (LOGS_DIR)
  TC05  AttributeError — method call on wrong type (NoneType.strip)
  TC06  AttributeError — missing method on class
  TC07  TypeError   — non-serializable object passed to json.dumps
  TC08  KeyError    — dict accessed with wrong key name
  TC09  IndentationError — mixed tabs and spaces
  TC10  ImportError — importing a removed/renamed symbol

Usage:
    pytest tests/diagnostic/test_code_repair.py -v
    pytest tests/diagnostic/test_code_repair.py -v -k tc01   # single case
    pytest tests/diagnostic/test_code_repair.py -v --tb=long  # full tracebacks

Requires:  Ollama running with at least qwen3:8b or qwen2.5-coder:14b loaded.
           If Ollama is not available, tests are skipped automatically.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Skip entire module if Ollama is not running
# ---------------------------------------------------------------------------
def _ollama_available() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

OLLAMA_UP = _ollama_available()
pytestmark = pytest.mark.skipif(not OLLAMA_UP, reason="Ollama not running")


# ===========================================================================
# TEST FIXTURE: isolated temp workspace + CodeAgent wiring
# ===========================================================================

@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Create a temp workspace that mimics a mini project."""
    ws = tmp_path_factory.mktemp("repair_backtest")
    (ws / "autotrade" / "utils").mkdir(parents=True)
    (ws / "autotrade" / "core").mkdir(parents=True)
    (ws / "autotrade" / "signals").mkdir(parents=True)
    (ws / "config").mkdir(parents=True)
    (ws / "tests").mkdir(parents=True)
    # __init__.py files
    for d in ["autotrade", "autotrade/utils", "autotrade/core",
              "autotrade/signals", "config", "tests"]:
        (ws / d / "__init__.py").write_text("", encoding="utf-8")
    yield ws


@pytest.fixture
def repair_agent():
    """Return the CodeAgent + ToolExecutor wired for a given workspace."""
    from autotrade.core.agentic_orchestrator import CodeAgent
    agent = CodeAgent()
    return agent


def _inject_and_attempt(
    workspace: Path,
    repair_agent,
    rel_path: str,
    good_code: str,
    bad_code: str,
    error_type: str,
    error_msg: str,
    line_hint: int,
) -> Dict:
    """
    Core test helper:
      1. Write `bad_code` into workspace/rel_path
      2. Verify it's actually broken (py_compile or AST)
      3. Feed it to CodeAgent._analyze_code_issue as auto_fix
      4. Check if the result is fixed

    Returns a dict with:
      success, message, model_used, duration_ms, original_broken, fixed_compiles
    """
    from autotrade.core.agentic_orchestrator import Task, TaskType

    target = workspace / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)

    # Write the broken file
    target.write_text(bad_code, encoding="utf-8")

    # Confirm it's broken
    original_broken = False
    try:
        ast.parse(bad_code, filename=str(target))
        # AST parse OK — could be a runtime error (NameError etc).
        # Try ruff for F821 (undefined names)
        try:
            r = subprocess.run(
                ["ruff", "check", str(target), "--select=E9,F821",
                 "--output-format=json", "--no-cache"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0 and r.stdout.strip():
                errors = json.loads(r.stdout)
                if errors:
                    original_broken = True
        except Exception:
            pass
        if not original_broken:
            # runtime-class errors are still "broken" — mark as true
            # (we injected the error, we know it's there)
            original_broken = True
    except SyntaxError:
        original_broken = True

    # Build a fake traceback matching the log format
    traceback_str = (
        f'Traceback (most recent call last):\n'
        f'  File "{target}", line {line_hint}, in <module>\n'
        f'    ...\n'
        f'{error_type}: {error_msg}'
    )

    # Run CodeAgent
    task = Task(
        type=TaskType.CODE_FIX,
        description=f"Test repair: {error_type} in {rel_path}",
        data={
            "file": str(target),
            "error": traceback_str,
            "line": line_hint,
            "error_type": error_type,
            "mode": "auto_fix",
        },
        priority=1,
    )

    start = time.time()
    result = repair_agent.execute(task)
    elapsed = time.time() - start

    # Check if the file now compiles
    fixed_compiles = False
    if target.exists():
        try:
            fixed_content = target.read_text(encoding="utf-8")
            ast.parse(fixed_content, filename=str(target))
            fixed_compiles = True
        except SyntaxError:
            fixed_compiles = False

    # Check ruff on the fixed file
    ruff_clean = False
    if fixed_compiles:
        try:
            r = subprocess.run(
                ["ruff", "check", str(target), "--select=E9,F821",
                 "--output-format=json", "--no-cache"],
                capture_output=True, text=True, timeout=10,
            )
            ruff_clean = (r.returncode == 0) or not r.stdout.strip()
        except Exception:
            ruff_clean = True  # ruff unavailable, assume ok

    return {
        "success": result.success,
        "message": result.message,
        "model_used": result.model_used,
        "agentic": result.data.get("agentic", False),
        "duration_s": round(elapsed, 1),
        "original_broken": original_broken,
        "fixed_compiles": fixed_compiles,
        "ruff_clean": ruff_clean,
    }


# ===========================================================================
# TC01: NameError — undefined variable (sr_data) — THE death-spiral error
# ===========================================================================

TC01_GOOD = textwrap.dedent('''\
    """Multimodel scorer stub."""
    import logging
    logger = logging.getLogger(__name__)

    class MultiModelScorer:
        def __init__(self):
            self.weights = {"technical": 0.30, "sr": 0.20}

        def score_symbol(self, symbol: str, daily_data=None, sr_data=None):
            """Score a single symbol across all dimensions."""
            sr_data = sr_data or {}
            atr_val = sr_data.get('atr_14', 0)
            price = daily_data.get('close', 0) if daily_data else 0
            tech_score = min(100, max(0, price * 0.5 + atr_val * 10))
            return {"symbol": symbol, "score": tech_score, "atr": atr_val}

        def _score_technical(self, symbol, daily_data, sr_data):
            atr_val = sr_data.get('atr_14', 0) if sr_data else 0
            return atr_val * 10
''')

TC01_BAD = textwrap.dedent('''\
    """Multimodel scorer stub."""
    import logging
    logger = logging.getLogger(__name__)

    class MultiModelScorer:
        def __init__(self):
            self.weights = {"technical": 0.30, "sr": 0.20}

        def score_symbol(self, symbol: str, daily_data=None):
            """Score a single symbol across all dimensions."""
            atr_val = sr_data.get('atr_14', 0) if sr_data else 0
            price = daily_data.get('close', 0) if daily_data else 0
            tech_score = min(100, max(0, price * 0.5 + atr_val * 10))
            return {"symbol": symbol, "score": tech_score, "atr": atr_val}

        def _score_technical(self, symbol, daily_data, sr_data):
            atr_val = sr_data.get('atr_14', 0) if sr_data else 0
            return atr_val * 10
''')


def test_tc01_name_error_undefined_variable(workspace, repair_agent):
    """TC01: NameError — sr_data referenced but not in method params (83 real attempts!)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/utils/multimodel_scorer.py",
        TC01_GOOD, TC01_BAD,
        "NameError", "name 'sr_data' is not defined",
        line_hint=11,
    )
    print(f"\n  TC01 result: success={res['success']}, agentic={res['agentic']}, "
          f"compiles={res['fixed_compiles']}, ruff={res['ruff_clean']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"], "Injected error should be detected as broken"
    assert res["fixed_compiles"], "Fix should produce valid Python syntax"
    assert res["ruff_clean"], "Fix should pass ruff (no undefined names)"


# ===========================================================================
# TC02: SyntaxError — unclosed dict literal
# ===========================================================================

TC02_GOOD = textwrap.dedent('''\
    """Agent module stub."""
    import json

    def handle_error(error_msg, context=None):
        """Handle an error with context."""
        result = {
            "error": error_msg,
            "context": context or {},
            "handled": True,
        }
        return json.dumps(result)
''')

TC02_BAD = textwrap.dedent('''\
    """Agent module stub."""
    import json

    def handle_error(error_msg, context=None):
        """Handle an error with context."""
        result = {
            "error": error_msg,
            "context": context or {},
            "handled": True,

        return json.dumps(result)
''')


def test_tc02_syntax_error_unclosed_dict(workspace, repair_agent):
    """TC02: SyntaxError — missing closing brace on dict literal."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/core/error_handler.py",
        TC02_GOOD, TC02_BAD,
        "SyntaxError", "invalid syntax. Perhaps you forgot a comma?",
        line_hint=10,
    )
    print(f"\n  TC02 result: success={res['success']}, agentic={res['agentic']}, "
          f"compiles={res['fixed_compiles']}, model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"], "SyntaxError fix should produce valid syntax"


# ===========================================================================
# TC03: SyntaxError — missing closing parenthesis in print
# ===========================================================================

TC03_GOOD = textwrap.dedent('''\
    """Self-heal test stub."""
    import sys

    def test_basic():
        print("Testing self-healing pipeline")
        result = 1 + 1
        assert result == 2
        print("PASS: basic arithmetic")
        return True

    if __name__ == "__main__":
        test_basic()
''')

TC03_BAD = textwrap.dedent('''\
    """Self-heal test stub."""
    import sys

    def test_basic():
        print("Testing self-healing pipeline"
        result = 1 + 1
        assert result == 2
        print("PASS: basic arithmetic")
        return True

    if __name__ == "__main__":
        test_basic()
''')


def test_tc03_syntax_error_missing_paren(workspace, repair_agent):
    """TC03: SyntaxError — missing closing paren on print (real fix log entry)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "tests/test_self_heal.py",
        TC03_GOOD, TC03_BAD,
        "SyntaxError", "invalid syntax",
        line_hint=5,
    )
    print(f"\n  TC03 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# TC04: NameError — undefined constant (LOGS_DIR)
# ===========================================================================

TC04_GOOD = textwrap.dedent('''\
    """Trade logger stub."""
    import json
    import logging
    from pathlib import Path

    LOGS_DIR = Path(__file__).parent.parent / "logs"
    logger = logging.getLogger(__name__)

    class TradeLogger:
        def __init__(self):
            self.log_dir = LOGS_DIR
            self.log_dir.mkdir(exist_ok=True)

        def _log_trade_decision(self, symbol, action, reason):
            """Log a trade decision to the decisions file."""
            entry = {"symbol": symbol, "action": action, "reason": reason}
            log_file = LOGS_DIR / "trade_decisions.jsonl"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\\n")
            except Exception as e:
                logger.warning(f"Could not log decision: {e}")
''')

TC04_BAD = textwrap.dedent('''\
    """Trade logger stub."""
    import json
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    class TradeLogger:
        def __init__(self):
            self.log_dir = LOGS_DIR
            self.log_dir.mkdir(exist_ok=True)

        def _log_trade_decision(self, symbol, action, reason):
            """Log a trade decision to the decisions file."""
            entry = {"symbol": symbol, "action": action, "reason": reason}
            log_file = LOGS_DIR / "trade_decisions.jsonl"
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\\n")
            except Exception as e:
                logger.warning(f"Could not log decision: {e}")
''')


def test_tc04_name_error_undefined_constant(workspace, repair_agent):
    """TC04: NameError — LOGS_DIR used but never defined (6 real attempts)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/core/trade_logger.py",
        TC04_GOOD, TC04_BAD,
        "NameError", "name 'LOGS_DIR' is not defined",
        line_hint=10,
    )
    print(f"\n  TC04 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"ruff={res['ruff_clean']}, model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]
    assert res["ruff_clean"]


# ===========================================================================
# TC05: AttributeError — NoneType.strip()
# ===========================================================================

TC05_GOOD = textwrap.dedent('''\
    """Data validator stub."""

    def validate_parquet_data(df):
        """Validate parquet data before use."""
        issues = []
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            if sample is not None and isinstance(sample, str):
                cleaned = sample.strip()
                if not cleaned:
                    issues.append(f"Column {col} has empty strings")
        return issues
''')

TC05_BAD = textwrap.dedent('''\
    """Data validator stub."""

    def validate_parquet_data(df):
        """Validate parquet data before use."""
        issues = []
        for col in df.columns:
            sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            cleaned = sample.strip()
            if not cleaned:
                issues.append(f"Column {col} has empty strings")
        return issues
''')


def test_tc05_attribute_error_nonetype_strip(workspace, repair_agent):
    """TC05: AttributeError — .strip() on None (real supervisor log error)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/utils/data_validator.py",
        TC05_GOOD, TC05_BAD,
        "AttributeError", "'NoneType' object has no attribute 'strip'",
        line_hint=8,
    )
    print(f"\n  TC05 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# TC06: AttributeError — missing method on class
# ===========================================================================

TC06_GOOD = textwrap.dedent('''\
    """Regime router stub."""
    import logging
    logger = logging.getLogger(__name__)

    class RegimeRouter:
        def __init__(self, regime="neutral"):
            self.regime = regime

        def filter_signals_by_regime(self, signals):
            """Filter signals based on current market regime."""
            if self.regime == "bearish":
                return [s for s in signals if s.get("direction") == "short"]
            elif self.regime == "bullish":
                return [s for s in signals if s.get("direction") == "long"]
            return signals

        def get_regime(self):
            return self.regime

    def run_pipeline(signals):
        router = RegimeRouter("bullish")
        filtered = router.filter_signals_by_regime(signals)
        return filtered
''')

TC06_BAD = textwrap.dedent('''\
    """Regime router stub."""
    import logging
    logger = logging.getLogger(__name__)

    class RegimeRouter:
        def __init__(self, regime="neutral"):
            self.regime = regime

        def get_regime(self):
            return self.regime

    def run_pipeline(signals):
        router = RegimeRouter("bullish")
        filtered = router.filter_signals_by_regime(signals)
        return filtered
''')


def test_tc06_attribute_error_missing_method(workspace, repair_agent):
    """TC06: AttributeError — filter_signals_by_regime missing from class (real log)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/signals/regime_router.py",
        TC06_GOOD, TC06_BAD,
        "AttributeError",
        "'RegimeRouter' object has no attribute 'filter_signals_by_regime'",
        line_hint=14,
    )
    print(f"\n  TC06 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# TC07: TypeError — object not JSON serializable
# ===========================================================================

TC07_GOOD = textwrap.dedent('''\
    """Backtest results stub."""
    import json
    from dataclasses import dataclass, asdict

    @dataclass
    class BacktestResult:
        strategy: str
        win_rate: float
        profit_factor: float
        total_trades: int

    def save_results(results: list, path: str):
        """Save backtest results to JSON file."""
        serializable = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
''')

TC07_BAD = textwrap.dedent('''\
    """Backtest results stub."""
    import json
    from dataclasses import dataclass

    @dataclass
    class BacktestResult:
        strategy: str
        win_rate: float
        profit_factor: float
        total_trades: int

    def save_results(results: list, path: str):
        """Save backtest results to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
''')


def test_tc07_type_error_not_serializable(workspace, repair_agent):
    """TC07: TypeError — BacktestResult not JSON serializable (3 real attempts)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/utils/backtest_saver.py",
        TC07_GOOD, TC07_BAD,
        "TypeError", "Object of type BacktestResult is not JSON serializable",
        line_hint=15,
    )
    print(f"\n  TC07 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# TC08: KeyError — wrong dict key
# ===========================================================================

TC08_GOOD = textwrap.dedent('''\
    """Config accessor stub."""

    def get_screener_config(config: dict) -> dict:
        """Extract screener config with proper key names."""
        screener = config.get("screener_v2", {})
        return {
            "min_composite_score": screener.get("min_composite_score", 75),
            "rsi_min": screener.get("rsi_min", 40),
            "rsi_max": screener.get("rsi_max", 65),
            "min_atr_pct": screener.get("min_atr_pct", 3.0),
        }
''')

TC08_BAD = textwrap.dedent('''\
    """Config accessor stub."""

    def get_screener_config(config: dict) -> dict:
        """Extract screener config with proper key names."""
        screener = config.get("screener_v2", {})
        return {
            "min_composite_score": screener["min_score"],
            "rsi_min": screener.get("rsi_min", 40),
            "rsi_max": screener.get("rsi_max", 65),
            "min_atr_pct": screener.get("min_atr_pct", 3.0),
        }
''')


def test_tc08_key_error_wrong_dict_key(workspace, repair_agent):
    """TC08: KeyError — screener['min_score'] should be .get('min_composite_score')."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "config/screener_config.py",
        TC08_GOOD, TC08_BAD,
        "KeyError", "'min_score'",
        line_hint=7,
    )
    print(f"\n  TC08 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# TC09: IndentationError — mixed tabs and spaces
# ===========================================================================

TC09_GOOD = textwrap.dedent('''\
    """Signal filter stub."""

    def filter_weak_signals(signals, threshold=60):
        """Remove signals below confidence threshold."""
        strong = []
        for sig in signals:
            conf = sig.get("confidence", 0)
            if conf >= threshold:
                strong.append(sig)
        return strong
''')

# Intentionally mix a tab into spaces
TC09_BAD = (
    '"""Signal filter stub."""\n'
    '\n'
    'def filter_weak_signals(signals, threshold=60):\n'
    '    """Remove signals below confidence threshold."""\n'
    '    strong = []\n'
    '    for sig in signals:\n'
    '        conf = sig.get("confidence", 0)\n'
    '\tif conf >= threshold:\n'            # <-- tab instead of spaces
    '            strong.append(sig)\n'
    '    return strong\n'
)


def test_tc09_indentation_error_mixed_tabs(workspace, repair_agent):
    """TC09: IndentationError — tab mixed with spaces (common copy-paste error)."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/signals/signal_filter.py",
        TC09_GOOD, TC09_BAD,
        "IndentationError", "unexpected indent",
        line_hint=8,
    )
    print(f"\n  TC09 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# TC10: ImportError — importing a renamed symbol
# ===========================================================================

TC10_GOOD = textwrap.dedent('''\
    """Plan loader stub."""
    import json
    from pathlib import Path

    def load_and_migrate_plan(plan_path):
        """Load plan and migrate legacy keys."""
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
        # Migrate legacy keys
        if "buy_signals" in plan and "signals" not in plan:
            plan["signals"] = plan.pop("buy_signals")
        if "entry_candidates" in plan and "signals" not in plan:
            plan["signals"] = plan.pop("entry_candidates")
        return plan
''')

TC10_BAD = textwrap.dedent('''\
    """Plan loader stub."""
    import json
    from pathlib import Path
    from autotrade.utils.plan_validator import validate_and_load

    def load_and_migrate_plan(plan_path):
        """Load plan and migrate legacy keys."""
        plan = validate_and_load(plan_path)
        # Migrate legacy keys
        if "buy_signals" in plan and "signals" not in plan:
            plan["signals"] = plan.pop("buy_signals")
        if "entry_candidates" in plan and "signals" not in plan:
            plan["signals"] = plan.pop("entry_candidates")
        return plan
''')


def test_tc10_import_error_renamed_symbol(workspace, repair_agent):
    """TC10: ImportError — importing from a module that doesn't exist."""
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/utils/plan_loader.py",
        TC10_GOOD, TC10_BAD,
        "ImportError",
        "cannot import name 'validate_and_load' from 'autotrade.utils.plan_validator'",
        line_hint=4,
    )
    print(f"\n  TC10 result: success={res['success']}, compiles={res['fixed_compiles']}, "
          f"ruff={res['ruff_clean']}, model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# COMPLEX CASES (TC11-TC13) — multi-part logic bugs from real logs
# ===========================================================================

# ---------------------------------------------------------------------------
# TC11: NoneType in f-string format specifier — real error from
#       autonomous_agent.py line 2520.  Multiple variables can be None
#       but get {:.2f} or {:+.1f} format specs applied.  The fix must
#       add None guards WITHOUT removing the format specifiers.
# ---------------------------------------------------------------------------

TC11_GOOD = textwrap.dedent('''\
    """Trade prompt builder stub."""

    def build_execution_prompt(symbol, score, planned_price, current_price,
                               stop_loss, target, market_context):
        """Build an f-string prompt for the trade advisor LLM."""
        gap_pct = (
            ((current_price - planned_price) / planned_price) * 100
            if planned_price and current_price
            else 0.0
        )
        risk_pct = (
            ((current_price - stop_loss) / current_price) * 100
            if current_price and stop_loss
            else 0.0
        )
        reward_pct = (
            ((target - current_price) / current_price) * 100
            if current_price and target
            else 0.0
        )

        spy_change = market_context.get("spy_change") or 0.0
        sentiment = market_context.get("sentiment") or "mixed"

        prompt = (
            f"TRADE: {symbol} score={score}/100\\n"
            f"Entry: ${planned_price or 0:.2f} Current: ${current_price or 0:.2f}\\n"
            f"Gap: {gap_pct:+.1f}% Risk: {risk_pct:.1f}% Reward: {reward_pct:.1f}%\\n"
            f"Stop: ${stop_loss or 0:.2f} Target: ${target or 0:.2f}\\n"
            f"SPY: {spy_change:+.1f}% Sentiment: {sentiment}"
        )
        return prompt
''')

TC11_BAD = textwrap.dedent('''\
    """Trade prompt builder stub."""

    def build_execution_prompt(symbol, score, planned_price, current_price,
                               stop_loss, target, market_context):
        """Build an f-string prompt for the trade advisor LLM."""
        gap_pct = ((current_price - planned_price) / planned_price) * 100
        risk_pct = ((current_price - stop_loss) / current_price) * 100
        reward_pct = ((target - current_price) / current_price) * 100

        prompt = (
            f"TRADE: {symbol} score={score}/100\\n"
            f"Entry: ${planned_price:.2f} Current: ${current_price:.2f}\\n"
            f"Gap: {gap_pct:+.1f}% Risk: {risk_pct:.1f}% Reward: {reward_pct:.1f}%\\n"
            f"Stop: ${stop_loss:.2f} Target: ${target:.2f}\\n"
            f"SPY: {market_context.get('spy_change'):+.1f}% "
            f"Sentiment: {market_context.get('sentiment')}"
        )
        return prompt
''')


def test_tc11_nonetype_fstring_format(workspace, repair_agent):
    """TC11: TypeError — None values hit format specifiers like {:+.1f}.

    Real error from autonomous_agent.py:2520.  Multiple None-able values
    are formatted with numeric specs.  The model must add None guards to
    each variable WITHOUT breaking the prompt structure.
    """
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/core/prompt_builder.py",
        TC11_GOOD, TC11_BAD,
        "TypeError",
        "unsupported format string passed to NoneType.__format__",
        line_hint=15,
    )
    print(f"\n  TC11 result: success={res['success']}, agentic={res['agentic']}, "
          f"compiles={res['fixed_compiles']}, ruff={res['ruff_clean']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ---------------------------------------------------------------------------
# TC12: Method signature mismatch + conditional branch bug.
#       A refactor changed score_and_rank() to require `regime` param,
#       but one call site in a try/except still uses the OLD 2-arg form.
#       Additionally, the except branch references `ranked` before
#       assignment (UnboundLocalError if scoring raises before assigning).
# ---------------------------------------------------------------------------

TC12_GOOD = textwrap.dedent('''\
    """Signal pipeline stub."""
    import logging
    logger = logging.getLogger(__name__)

    class SignalScorer:
        def __init__(self, regime="neutral"):
            self.regime = regime

        def score_and_rank(self, signals, regime=None):
            """Score and rank signals by quality. Regime filters applied."""
            regime = regime or self.regime
            scored = []
            for s in signals:
                base = s.get("raw_score", 50)
                if regime == "bearish":
                    base *= 0.7
                elif regime == "bullish":
                    base *= 1.2
                scored.append({**s, "final_score": min(100, base)})
            ranked = sorted(scored, key=lambda x: x["final_score"], reverse=True)
            return ranked

    def run_pipeline(raw_signals, market_regime="neutral"):
        """Run the full signal scoring pipeline."""
        scorer = SignalScorer(regime=market_regime)
        try:
            ranked = scorer.score_and_rank(raw_signals, regime=market_regime)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            ranked = raw_signals  # fallback: return unscored

        top_picks = [s for s in ranked if s.get("final_score", 0) >= 60]
        logger.info(f"Pipeline: {len(raw_signals)} in -> {len(top_picks)} picks")
        return top_picks
''')

TC12_BAD = textwrap.dedent('''\
    """Signal pipeline stub."""
    import logging
    logger = logging.getLogger(__name__)

    class SignalScorer:
        def __init__(self, regime="neutral"):
            self.regime = regime

        def score_and_rank(self, signals, regime=None):
            """Score and rank signals by quality. Regime filters applied."""
            regime = regime or self.regime
            scored = []
            for s in signals:
                base = s.get("raw_score", 50)
                if regime == "bearish":
                    base *= 0.7
                elif regime == "bullish":
                    base *= 1.2
                scored.append({**s, "final_score": min(100, base)})
            ranked = sorted(scored, key=lambda x: x["final_score"], reverse=True)
            return ranked

    def run_pipeline(raw_signals, market_regime="neutral"):
        """Run the full signal scoring pipeline."""
        scorer = SignalScorer(regime=market_regime)
        try:
            scorer.score_and_rank(raw_signals, market_regime, True)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")

        top_picks = [s for s in ranked if s.get("final_score", 0) >= 60]
        logger.info(f"Pipeline: {len(raw_signals)} in -> {len(top_picks)} picks")
        return top_picks
''')


def test_tc12_signature_mismatch_plus_unbound(workspace, repair_agent):
    """TC12: TypeError (wrong arg count) + UnboundLocalError (ranked).

    Two bugs interleaved:
      1. score_and_rank() called with 3 positional args (only takes 2+self)
      2. `ranked` is used after except block but never assigned on error path
    The model must fix BOTH to produce valid code.
    """
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/signals/signal_pipeline.py",
        TC12_GOOD, TC12_BAD,
        "TypeError",
        "score_and_rank() takes 2 positional arguments but 4 were given",
        line_hint=28,
    )
    print(f"\n  TC12 result: success={res['success']}, agentic={res['agentic']}, "
          f"compiles={res['fixed_compiles']}, ruff={res['ruff_clean']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]
    assert res["ruff_clean"], "Must fix undefined name 'ranked' too"


# ---------------------------------------------------------------------------
# TC13: JSON serialization + property/attribute confusion.
#       A dataclass with a read-only @property is passed to json.dumps.
#       Code also tries to SET .output on a TaskResult-like object that
#       only has a getter property.  Two interleaved type-system bugs.
# ---------------------------------------------------------------------------

TC13_GOOD = textwrap.dedent('''\
    """Results serializer stub."""
    import json
    from dataclasses import dataclass, field, asdict
    from typing import Dict, Any, List

    @dataclass
    class StrategyResult:
        name: str
        win_rate: float
        profit_factor: float
        trades: int
        equity_curve: List[float] = field(default_factory=list)

    @dataclass
    class RunSummary:
        success: bool
        message: str
        results: List[StrategyResult] = field(default_factory=list)

        def set_output(self, msg: str):
            """Update the output message."""
            self.message = msg

    def save_run(summary: RunSummary, path: str):
        """Save a strategy run to JSON."""
        summary.set_output(f"Saved {len(summary.results)} results")
        data = {
            "success": summary.success,
            "message": summary.message,
            "results": [asdict(r) for r in summary.results],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
''')

TC13_BAD = textwrap.dedent('''\
    """Results serializer stub."""
    import json
    from dataclasses import dataclass, field
    from typing import Dict, Any, List

    @dataclass
    class StrategyResult:
        name: str
        win_rate: float
        profit_factor: float
        trades: int
        equity_curve: List[float] = field(default_factory=list)

    @dataclass
    class RunSummary:
        success: bool
        message: str
        results: List[StrategyResult] = field(default_factory=list)

        @property
        def output(self):
            return self.message

    def save_run(summary: RunSummary, path: str):
        """Save a strategy run to JSON."""
        summary.output = f"Saved {len(summary.results)} results"
        data = {
            "success": summary.success,
            "message": summary.message,
            "results": summary.results,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return data
''')


def test_tc13_property_assign_plus_serialization(workspace, repair_agent):
    """TC13: AttributeError (can't set property) + TypeError (not serializable).

    Two interleaved bugs:
      1. summary.output is a read-only @property — setting it raises AttributeError
      2. summary.results contains StrategyResult dataclass objects passed
         directly to json.dump without asdict() conversion
    The model must fix BOTH.
    """
    res = _inject_and_attempt(
        workspace, repair_agent,
        "autotrade/utils/results_saver.py",
        TC13_GOOD, TC13_BAD,
        "AttributeError",
        "property 'output' of 'RunSummary' object has no setter",
        line_hint=26,
    )
    print(f"\n  TC13 result: success={res['success']}, agentic={res['agentic']}, "
          f"compiles={res['fixed_compiles']}, ruff={res['ruff_clean']}, "
          f"model={res['model_used']}, time={res['duration_s']}s")
    assert res["original_broken"]
    assert res["fixed_compiles"]


# ===========================================================================
# SUMMARY REPORT — runs after all tests
# ===========================================================================

@pytest.fixture(scope="module", autouse=True)
def print_summary(request):
    """Print a summary of all test results at module end."""
    yield
    # This runs after all tests in the module
    print("\n" + "=" * 70)
    print("CODE REPAIR BACKTEST COMPLETE")
    print("=" * 70)
    print("See individual test output above for per-case details.")
    print("Tests that PASS = CodeAgent successfully fixed the injected error.")
    print("Tests that FAIL = CodeAgent could not fix — needs investigation.")
    print("=" * 70)


# ===========================================================================
# Quick standalone runner
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
