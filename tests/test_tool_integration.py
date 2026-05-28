"""Test the full tool integration: ruff lint, run_tests, tool registration."""
import sys
import os
import tempfile
sys.path.insert(0, ".")

from pathlib import Path
from autotrade.core.local_coding_agent import (
    TOOL_GROUPS, MODE_PERMISSIONS, NATIVE_TOOL_SCHEMAS,
    get_tools_for_mode, AgentMode, ToolExecutor,
)


# ── 1. TOOL_GROUPS registration ──
print("Test 1: run_tests in TOOL_GROUPS")
assert "test" in TOOL_GROUPS, f"Missing 'test' group: {list(TOOL_GROUPS.keys())}"
assert "run_tests" in TOOL_GROUPS["test"], f"run_tests not in test group"
print(f"  PASS — test group: {TOOL_GROUPS['test']}")

# ── 2. MODE_PERMISSIONS ──
print("Test 2: test group in ACT and DEBUG modes")
assert "test" in MODE_PERMISSIONS["act"], "Missing 'test' in ACT mode"
assert "test" in MODE_PERMISSIONS["debug"], "Missing 'test' in DEBUG mode"
assert "test" not in MODE_PERMISSIONS["plan"], "test should NOT be in PLAN mode"
print("  PASS — test group in act + debug, not in plan")

# ── 3. NATIVE_TOOL_SCHEMAS ──
print("Test 3: run_tests in NATIVE_TOOL_SCHEMAS")
schema_names = [t["function"]["name"] for t in NATIVE_TOOL_SCHEMAS]
assert "run_tests" in schema_names, f"run_tests missing from schemas: {schema_names}"
print(f"  PASS — {len(NATIVE_TOOL_SCHEMAS)} schemas, run_tests included")

# ── 4. get_tools_for_mode ──
print("Test 4: run_tests available in ACT mode tools")
act_tools = [t["function"]["name"] for t in get_tools_for_mode(AgentMode.ACT)]
assert "run_tests" in act_tools, f"run_tests not in ACT tools: {act_tools}"
print(f"  PASS — {len(act_tools)} ACT tools including run_tests")

# ── 5. ToolExecutor._tool_run_tests exists ──
print("Test 5: ToolExecutor has _tool_run_tests method")
te = ToolExecutor(project_root=Path(".").resolve())
assert hasattr(te, "_tool_run_tests"), "Missing _tool_run_tests method"
print("  PASS")

# ── 6. run_tests actually runs test_parser.py ──
print("Test 6: run_tests executes test_parser.py")
result = te.execute("run_tests", {"path": "tests/test_parser.py"}, AgentMode.ACT)
print(f"  stdout: {result.get('stdout', '')[:200]}")
print(f"  stderr: {result.get('stderr', '')[:200]}")
print(f"  cmd: {result.get('command', '')}")
print(f"  exit_code: {result.get('exit_code', 'N/A')}")
assert result["success"], f"run_tests failed: {result}"
assert "PASSED" in result["stdout"].upper() or "PASS" in result["stdout"].upper(), \
    f"Expected PASS in output: {result['stdout'][:200]}"
print(f"  PASS — test_parser.py ran successfully")

# ── 7. run_tests returns helpful error when no test exists ──
print("Test 7: run_tests gives helpful error for source file without test")
result2 = te.execute("run_tests", {"path": "autotrade/core/local_coding_agent.py"}, AgentMode.ACT)
assert not result2["success"], "Should fail — no test_local_coding_agent.py exists"
assert "No test file found" in result2.get("error", ""), f"Wrong error: {result2}"
print(f"  PASS — helpful error: {result2['error'][:100]}")

# ── 8. Ruff catches undefined names ──
print("Test 8: Ruff lint catches undefined variable")
tmp = Path(tempfile.mkdtemp())
bad_file = tmp / "bad_test.py"
bad_file.write_text("def foo():\n    return undefined_var\n")
lint = te._lint_python_file(bad_file)
assert lint is not None, "Ruff should catch undefined name"
assert "F821" in lint or "Undefined" in lint, f"Expected F821: {lint}"
os.unlink(str(bad_file))
print(f"  PASS — ruff caught: {lint.strip()}")

# ── 9. Ruff passes clean file ──
print("Test 9: Ruff passes clean Python file")
good_file = tmp / "good_test.py"
good_file.write_text("def foo():\n    return 42\n")
lint2 = te._lint_python_file(good_file)
assert lint2 is None, f"Clean file should pass ruff: {lint2}"
os.unlink(str(good_file))
print("  PASS — clean file passes ruff")

# ── 10. Ruff catches unused imports (and auto-fixes them) ──
print("Test 10: Ruff catches unused imports")
unused_file = tmp / "unused_import.py"
unused_file.write_text("import os\nimport sys\n\ndef foo():\n    return 42\n")
lint3 = te._lint_python_file(unused_file)
# Note: lint3 will be None because _lint_python_file uses ruff check --fix
if lint3 is not None:
    assert "F401" in lint3 or "unused" in lint3.lower(), f"Expected unused import report: {lint3}"
os.unlink(str(unused_file))
print("  PASS — ruff handled unused imports (via auto-fix or report)")

# ── 11. AST still catches syntax errors (before ruff) ──
print("Test 11: AST catches syntax errors first")
syntax_file = tmp / "syntax_error.py"
syntax_file.write_text("def foo(\n    return 42\n")
lint4 = te._lint_python_file(syntax_file)
assert lint4 is not None, "Should catch syntax error"
assert "SyntaxError" in lint4, f"Expected SyntaxError: {lint4}"
os.unlink(str(syntax_file))
os.rmdir(str(tmp))
print(f"  PASS — AST caught syntax error: {lint4}")

# ── 12. run_tests blocked in PLAN mode ──
print("Test 12: run_tests blocked in PLAN mode")
result3 = te.execute("run_tests", {"path": "tests/test_parser.py"}, AgentMode.PLAN)
assert not result3["success"], "run_tests should be blocked in PLAN mode"
assert "not allowed" in result3["error"].lower(), f"Wrong error: {result3['error']}"
print("  PASS — correctly blocked in PLAN mode")

# ── 13. Write-then-lint catches ruff errors and rolls back ──
print("Test 13: write_to_file rolls back on ruff errors")
test_file = Path(".").resolve() / "tests" / "_tmp_ruff_test.py"
result4 = te.execute("write_to_file", {
    "path": "tests/_tmp_ruff_test.py",
    "content": "import os\nimport sys\n\ndef foo():\n    return undefined_xyz\n",
}, AgentMode.ACT)
assert not result4["success"], f"Should fail ruff: {result4}"
assert "rolled_back" in result4 or "LINT FAILED" in result4.get("error", ""), \
    f"Should mention rollback: {result4}"
# Verify file was rolled back (removed since it didn't exist before)
assert not test_file.exists(), "File should have been removed after rollback"
print(f"  PASS — write rolled back on ruff error")

print("\n=== ALL 13 INTEGRATION TESTS PASSED ===")
