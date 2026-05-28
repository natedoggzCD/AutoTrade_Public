"""Test the replace_in_file diagnostic diff output.

Verifies that when old_text doesn't match, the model gets actionable
feedback: a diff showing exactly what's different + the actual text to copy.
"""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from autotrade.core.local_coding_agent import ToolExecutor, AgentMode


def make_executor(root: Path) -> ToolExecutor:
    te = ToolExecutor(project_root=root)
    # Pre-mark all test files as "read" so the read-before-write guard
    # doesn't interfere — these tests are specifically testing mismatch diagnostics.
    for f in root.glob("*"):
        if f.is_file():
            te._files_read.add(str(f))
    return te


# ── Fixtures ──
TEST_DIR = Path("tests/_tmp_diag")
TEST_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = TEST_DIR / "sample.py"
SAMPLE_FILE.write_text(
    '''class DataLoader:
    def __init__(self, path):
        self.path = path
        self.data = None

    def load(self):
        """Load data from file."""
        with open(self.path) as f:
            self.data = f.read()
        return self.data

    def process(self):
        """Process loaded data."""
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.strip()
''',
    encoding="utf-8",
)


# ---- Test 1: Wrong indentation (the original bug) ----
print("Test 1: Wrong indentation (stripped leading spaces)")
executor = make_executor(TEST_DIR)
result = executor.execute("replace_in_file", {
    "path": "sample.py",
    # Model sends WITHOUT leading indentation (the .strip() bug)
    "old_text": "def load(self):\n    \"\"\"Load data from file.\"\"\"\n    with open(self.path) as f:\n        self.data = f.read()\n    return self.data",
    "new_text": "def load(self):\n    # improved\n    pass",
}, AgentMode.ACT)
assert not result["success"], f"Should fail: {result}"
assert "WHITESPACE MISMATCH" in result["error"], f"Should detect whitespace issue: {result['error'][:200]}"
assert "DIFF" in result["error"], f"Should show diff: {result['error'][:200]}"
assert "ACTUAL TEXT" in result["error"], f"Should show actual text to copy: {result['error'][:200]}"
# The actual text block should have the correct 4-space indentation
assert "    def load(self):" in result["error"], f"Actual text should have correct indent"
print(f"  PASS — detected whitespace mismatch, provided diff + actual text")
print(f"  Error preview: {result['error'][:300]}...")
print()


# ---- Test 2: Content changed (different line) ----
print("Test 2: Near match with one changed line")
result2 = executor.execute("replace_in_file", {
    "path": "sample.py",
    "old_text": "    def process(self):\n        \"\"\"Process loaded data.\"\"\"\n        if self.data is None:\n            raise RuntimeError(\"No data loaded\")\n        return self.data.strip()",
    "new_text": "    def process(self):\n        pass",
}, AgentMode.ACT)
assert not result2["success"], f"Should fail: {result2}"
assert "DIFF" in result2["error"], f"Should show diff"
# It said ValueError in file but model sent RuntimeError — diff should show this
assert "---" in result2["error"] and "+++" in result2["error"], f"Should have unified diff markers"
print(f"  PASS — detected near match with content diff")
print(f"  Error preview: {result2['error'][:300]}...")
print()


# ---- Test 3: Completely wrong text ----
print("Test 3: No match at all")
result3 = executor.execute("replace_in_file", {
    "path": "sample.py",
    "old_text": "def something_totally_different():\n    pass",
    "new_text": "def x():\n    pass",
}, AgentMode.ACT)
assert not result3["success"]
assert "not found" in result3["error"].lower()
print(f"  PASS — correctly reports not found")
print()


# ---- Test 4: Exact match still works ----
print("Test 4: Exact match succeeds (no diagnostic needed)")
result4 = executor.execute("replace_in_file", {
    "path": "sample.py",
    "old_text": "    def process(self):\n        \"\"\"Process loaded data.\"\"\"\n        if self.data is None:\n            raise ValueError(\"No data loaded\")\n        return self.data.strip()",
    "new_text": "    def process(self):\n        \"\"\"Process loaded data (v2).\"\"\"\n        if self.data is None:\n            raise ValueError(\"No data loaded\")\n        return self.data.strip().upper()",
}, AgentMode.ACT)
assert result4["success"], f"Exact match should succeed: {result4}"
print(f"  PASS — exact match works, lint={result4.get('lint')}")
print()


# ---- Test 5: Diff gives the actual text the model should copy ----
print("Test 5: ACTUAL TEXT block is copy-pasteable")
# Re-read file (Test 4 modified it)
SAMPLE_FILE.write_text(
    '''class Example:
    def method_a(self):
        x = 1
        y = 2
        return x + y
''',
    encoding="utf-8",
)
result5 = executor.execute("replace_in_file", {
    "path": "sample.py",
    # Model sends with tabs instead of spaces
    "old_text": "\tdef method_a(self):\n\t\tx = 1\n\t\ty = 2\n\t\treturn x + y",
    "new_text": "\tdef method_a(self):\n\t\treturn 3",
}, AgentMode.ACT)
assert not result5["success"]
err = result5["error"]
# Extract the ACTUAL TEXT section
assert "--- ACTUAL TEXT" in err, f"Should have ACTUAL TEXT section"
actual_section = err.split("--- ACTUAL TEXT")[1]
assert "    def method_a(self):" in actual_section, f"Actual text should have spaces not tabs"
print(f"  PASS — actual text block has correct spacing for copy-paste")
print()


# ---- Cleanup ----
import shutil
shutil.rmtree(TEST_DIR, ignore_errors=True)

print("=== ALL 5 DIAGNOSTIC TESTS PASSED ===")
