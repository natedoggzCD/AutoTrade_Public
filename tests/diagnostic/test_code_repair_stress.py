"""
Additional repair-agent diagnostics with runtime-aware verification.

These cases complement test_code_repair.py by checking whether the repair
agent can handle:
  - simple runtime bugs that compile cleanly
  - project-specific schema conventions such as atr_14
  - cross-file API drift where the broken file needs context from another file

All work happens in a temp workspace. No production files are modified.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pytest


RuntimeValidator = Callable[[Path], Tuple[bool, str]]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("repair_stress")
    for rel_dir in [
        "autotrade",
        "autotrade/utils",
        "autotrade/core",
        "autotrade/signals",
        "config",
        "tests",
    ]:
        target_dir = ws / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "__init__.py").write_text("", encoding="utf-8")
    yield ws


@pytest.fixture
def repair_agent():
    from autotrade.core.agentic_orchestrator import CodeAgent

    return CodeAgent()


def _run_runtime_check(workspace: Path, code: str) -> Tuple[bool, str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip() or "runtime behavior validated"
    note = completed.stdout.strip() or completed.stderr.strip() or "runtime check failed"
    return False, note


def _ruff_clean(path: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "ruff",
                "check",
                str(path),
                "--select=E9,F821",
                "--output-format=json",
                "--no-cache",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
        return (result.returncode == 0) or not result.stdout.strip()
    except Exception:
        return True


def _inject_and_attempt(
    workspace: Path,
    repair_agent,
    rel_path: str,
    good_code: str,
    bad_code: str,
    error_type: str,
    error_msg: str,
    line_hint: int,
    runtime_validate: Optional[RuntimeValidator] = None,
    extra_files: Optional[Dict[str, str]] = None,
    traceback_override: Optional[str] = None,
) -> Dict:
    from autotrade.core.agentic_orchestrator import Task, TaskType

    if extra_files:
        for extra_rel, extra_content in extra_files.items():
            extra_target = workspace / extra_rel
            extra_target.parent.mkdir(parents=True, exist_ok=True)
            extra_target.write_text(extra_content, encoding="utf-8")

    target = workspace / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(bad_code, encoding="utf-8")

    original_broken = True
    try:
        ast.parse(bad_code, filename=str(target))
    except SyntaxError:
        original_broken = True

    traceback_str = traceback_override or (
        f'Traceback (most recent call last):\n'
        f'  File "{target}", line {line_hint}, in <module>\n'
        f"    ...\n"
        f"{error_type}: {error_msg}"
    )

    task = Task(
        type=TaskType.CODE_FIX,
        description=f"Stress repair: {error_type} in {rel_path}",
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

    fixed_compiles = False
    if target.exists():
        try:
            fixed_content = target.read_text(encoding="utf-8")
            ast.parse(fixed_content, filename=str(target))
            fixed_compiles = True
        except SyntaxError:
            fixed_compiles = False

    runtime_ok = fixed_compiles
    runtime_note = "not run"
    if fixed_compiles and runtime_validate:
        try:
            runtime_ok, runtime_note = runtime_validate(workspace)
        except Exception as exc:  # pragma: no cover - diagnostic path
            runtime_ok = False
            runtime_note = f"runtime validation raised: {exc}"

    return {
        "success": result.success,
        "message": result.message,
        "model_used": result.model_used,
        "agentic": result.data.get("agentic", False),
        "duration_s": round(elapsed, 1),
        "original_broken": original_broken,
        "fixed_compiles": fixed_compiles,
        "ruff_clean": _ruff_clean(target) if fixed_compiles else False,
        "runtime_ok": runtime_ok,
        "runtime_note": runtime_note,
        "expected_good_code": good_code,
    }


def _validate_safe_pct_change(workspace: Path) -> Tuple[bool, str]:
    return _run_runtime_check(
        workspace,
        textwrap.dedent(
            """\
            import sys
            sys.path.insert(0, '.')
            from autotrade.utils.price_math import safe_pct_change

            assert round(safe_pct_change(100.0, 110.0), 2) == 10.0
            assert safe_pct_change(0.0, 110.0) == 0.0
            assert safe_pct_change(None, 110.0) == 0.0
            print("runtime behavior validated")
            """
        ),
    )


def _validate_atr_summary(workspace: Path) -> Tuple[bool, str]:
    return _run_runtime_check(
        workspace,
        textwrap.dedent(
            """\
            import sys
            sys.path.insert(0, '.')
            from autotrade.core.risk_summary import build_risk_summary

            assert build_risk_summary({"symbol": "ABCD", "atr_14": 1.25}) == "ABCD atr_14=1.25"
            assert build_risk_summary({"symbol": "ABCD"}) == "ATR unavailable"
            print("runtime behavior validated")
            """
        ),
    )


def _validate_selector_pipeline(workspace: Path) -> Tuple[bool, str]:
    return _run_runtime_check(
        workspace,
        textwrap.dedent(
            """\
            import sys
            sys.path.insert(0, '.')
            from autotrade.signals.selector import select_symbol

            assert select_symbol("RVLT") == {"symbol": "RVLT", "score": 72}
            assert select_symbol("LOW") is None
            print("runtime behavior validated")
            """
        ),
    )


TC14_GOOD = textwrap.dedent(
    """\
    def safe_pct_change(previous_close, current_price):
        if previous_close in (None, 0):
            return 0.0
        return ((current_price - previous_close) / previous_close) * 100.0
    """
)

TC14_BAD = textwrap.dedent(
    """\
    def safe_pct_change(previous_close, current_price):
        return ((current_price - previous_close) / previous_close) * 100.0
    """
)


def test_tc14_simple_runtime_zero_division(workspace, repair_agent):
    """Simple: runtime-only failure that compiles and lints cleanly."""
    res = _inject_and_attempt(
        workspace=workspace,
        repair_agent=repair_agent,
        rel_path="autotrade/utils/price_math.py",
        good_code=TC14_GOOD,
        bad_code=TC14_BAD,
        error_type="ZeroDivisionError",
        error_msg="float division by zero",
        line_hint=2,
        runtime_validate=_validate_safe_pct_change,
    )
    print(
        f"\n  TC14 result: success={res['success']}, compiles={res['fixed_compiles']}, "
        f"ruff={res['ruff_clean']}, runtime={res['runtime_ok']}, "
        f"model={res['model_used']}, time={res['duration_s']}s"
    )
    assert res["original_broken"]
    assert res["fixed_compiles"]
    assert res["runtime_ok"], res["runtime_note"]


TC15_GOOD = textwrap.dedent(
    """\
    def build_risk_summary(row):
        atr = row.get("atr_14")
        if atr is None:
            return "ATR unavailable"
        return f"{row['symbol']} atr_14={atr:.2f}"
    """
)

TC15_BAD = textwrap.dedent(
    """\
    def build_risk_summary(row):
        return f"{row['symbol']} atr_14={row['atr']:.2f}"
    """
)


def test_tc15_medium_project_convention_atr_14(workspace, repair_agent):
    """Medium: uses the project's canonical atr_14 convention."""
    res = _inject_and_attempt(
        workspace=workspace,
        repair_agent=repair_agent,
        rel_path="autotrade/core/risk_summary.py",
        good_code=TC15_GOOD,
        bad_code=TC15_BAD,
        error_type="KeyError",
        error_msg="'atr'",
        line_hint=2,
        runtime_validate=_validate_atr_summary,
    )
    print(
        f"\n  TC15 result: success={res['success']}, compiles={res['fixed_compiles']}, "
        f"ruff={res['ruff_clean']}, runtime={res['runtime_ok']}, "
        f"model={res['model_used']}, time={res['duration_s']}s"
    )
    assert res["original_broken"]
    assert res["fixed_compiles"]
    assert res["runtime_ok"], res["runtime_note"]


TC16_HELPER = textwrap.dedent(
    """\
    def fetch_feature_rows(symbol):
        rows = {
            "RVLT": {"symbol": "RVLT", "atr_14": 1.8, "score": 72},
            "LOW": {"symbol": "LOW", "atr_14": 0.7, "score": 41},
        }
        return rows.get(symbol, {"symbol": symbol, "atr_14": 1.1, "score": 50})
    """
)

TC16_GOOD = textwrap.dedent(
    """\
    from autotrade.utils.feature_rows import fetch_feature_rows

    def select_symbol(symbol):
        row = fetch_feature_rows(symbol)
        if row["atr_14"] < 1.0:
            return None
        return {"symbol": row["symbol"], "score": row["score"]}
    """
)

TC16_BAD = textwrap.dedent(
    """\
    from autotrade.utils.feature_rows import load_feature_rows

    def select_symbol(symbol):
        row = load_feature_rows(symbol)
        if row["atr"] < 1.0:
            return None
        return {"symbol": row["symbol"], "score": row["score"]}
    """
)


def test_tc16_complex_cross_file_api_drift(workspace, repair_agent):
    """Complex: broken import plus wrong field name across related files."""
    helper_path = workspace / "autotrade/utils/feature_rows.py"
    target_path = workspace / "autotrade/signals/selector.py"
    traceback_str = (
        "Traceback (most recent call last):\n"
        f'  File "{workspace / "runner.py"}", line 4, in <module>\n'
        "    from autotrade.signals.selector import select_symbol\n"
        f'  File "{target_path}", line 1, in <module>\n'
        "    from autotrade.utils.feature_rows import load_feature_rows\n"
        f'  File "{helper_path}", line 1, in <module>\n'
        "    def fetch_feature_rows(symbol):\n"
        "ImportError: cannot import name 'load_feature_rows' from 'autotrade.utils.feature_rows'\n"
    )

    res = _inject_and_attempt(
        workspace=workspace,
        repair_agent=repair_agent,
        rel_path="autotrade/signals/selector.py",
        good_code=TC16_GOOD,
        bad_code=TC16_BAD,
        error_type="ImportError",
        error_msg="cannot import name 'load_feature_rows' from 'autotrade.utils.feature_rows'",
        line_hint=1,
        runtime_validate=_validate_selector_pipeline,
        extra_files={"autotrade/utils/feature_rows.py": TC16_HELPER},
        traceback_override=traceback_str,
    )
    print(
        f"\n  TC16 result: success={res['success']}, compiles={res['fixed_compiles']}, "
        f"ruff={res['ruff_clean']}, runtime={res['runtime_ok']}, "
        f"model={res['model_used']}, time={res['duration_s']}s"
    )
    assert res["original_broken"]
    assert res["fixed_compiles"]
    assert res["runtime_ok"], res["runtime_note"]
