"""Quick test for parse_tool_call and parse_json_tool_call."""
import sys
sys.path.insert(0, ".")

from autotrade.core.local_coding_agent import parse_json_tool_call, parse_tool_call, _extract_json_objects

# Test 1: JSON in code fence (qwen2.5-coder style)
t1 = '```json\n{"name": "read_file", "arguments": {"path": "autotrade/core/file.py"}}\n```'
r1 = parse_json_tool_call(t1)
assert r1 is not None, "Test 1 failed: should parse JSON in code fence"
assert r1[0] == "read_file", f"Test 1 tool name wrong: {r1[0]}"
assert r1[1]["path"] == "autotrade/core/file.py", f"Test 1 path wrong: {r1[1]}"
print(f"Test 1 PASS: {r1}")

# Test 2: Bare JSON
t2 = '{"name": "list_files", "arguments": {"path": ".", "recursive": "true"}}'
r2 = parse_json_tool_call(t2)
assert r2 is not None, "Test 2 failed"
assert r2[0] == "list_files"
print(f"Test 2 PASS: {r2}")

# Test 3: parse_tool_call Strategy 4 (JSON via unified parser)
t3 = 'Some analysis text\n```json\n{"name": "execute_command", "arguments": {"command": "python test.py"}}\n```'
r3 = parse_tool_call(t3)
assert r3 is not None, "Test 3 failed: parse_tool_call should find JSON"
assert r3[0] == "execute_command"
print(f"Test 3 PASS: {r3}")

# Test 4: XML still works
t4 = '<tool_call><tool_name>read_file</tool_name><path>file.py</path><start_line>1</start_line><end_line>50</end_line></tool_call>'
r4 = parse_tool_call(t4)
assert r4 is not None, "Test 4 failed: XML should still work"
assert r4[0] == "read_file"
print(f"Test 4 PASS: {r4}")

# Test 5: Invalid tool name rejected
t5 = '{"name": "invalid_tool", "arguments": {"x": "1"}}'
r5 = parse_json_tool_call(t5)
assert r5 is None, "Test 5 failed: should reject unknown tools"
print(f"Test 5 PASS: None (rejected invalid tool)")

# Test 6: Bare <tool_name> strategy still works
t6 = '<tool_name>search_files</tool_name><path>.</path><regex>def main</regex>'
r6 = parse_tool_call(t6)
assert r6 is not None, "Test 6 failed: bare XML should work"
assert r6[0] == "search_files"
print(f"Test 6 PASS: {r6}")

# Test 7: JSON with numeric values (converted to strings)
t7 = '{"name": "read_file", "arguments": {"path": "file.py", "start_line": 1, "end_line": 50}}'
r7 = parse_json_tool_call(t7)
assert r7 is not None, "Test 7 failed"
assert r7[1]["start_line"] == "1", f"Test 7: expected string '1', got {r7[1]['start_line']}"
print(f"Test 7 PASS: {r7}")

# Test 8: Nested braces in JSON string values (the qwen2.5-coder replace_in_file bug)
# This is the exact pattern that was failing: old_text/new_text contain Python code with {} 
t8 = '''```json
{
  "name": "replace_in_file",
  "arguments": {
    "path": "autotrade/utils/financial_db.py",
    "old_text": "\\\"\\\"\\\"\\n        Returns rows sorted by period_end DESC, metric.\\n        \\\"\\\"\\\"\\n        sql = \\"SELECT * FROM financial_statements WHERE ticker = ? AND statement_type = ? AND frequency = ?\\"\\n        params: list = [ticker.upper(), statement_type, frequency]\\n        if metrics:\\n            placeholders = \\",\\".join(\\"?\\" for _ in metrics)\\n            sql += f\\" AND metric IN ({placeholders})\\"\\n            params.extend(metrics)\\n        sql += \\" ORDER BY period_end DESC, metric\\"\\n        return self._query(sql, params)",
    "new_text": "\\\"\\\"\\\"\\n        Returns rows sorted by period_end DESC, metric.\\n        \\\"\\\"\\\"\\n        sql = \\"SELECT * FROM financial_statements WHERE ticker = ? AND statement_type = ? AND frequency = ?\\"\\n        params: list = [ticker.upper(), statement_type, frequency]\\n        if metrics:\\n            placeholders = \\",\\".join(\\"?\\" for _ in metrics)\\n            sql += f\\" AND metric IN ({placeholders})\\"\\n            params.extend(metrics)\\n        sql += \\" ORDER BY period_end DESC, metric\\"\\n        rows = self._query(sql, params)\\n        return rows"
  }
}
```'''
r8 = parse_json_tool_call(t8)
assert r8 is not None, "Test 8 failed: should handle nested braces in string values"
assert r8[0] == "replace_in_file", f"Test 8 tool name wrong: {r8[0]}"
assert r8[1]["path"] == "autotrade/utils/financial_db.py", f"Test 8 path wrong"
assert "old_text" in r8[1] and "new_text" in r8[1], "Test 8 missing old_text or new_text"
print(f"Test 8 PASS: replace_in_file with nested braces (path={r8[1]['path']}, old_text len={len(r8[1]['old_text'])})")

# Test 9: _extract_json_objects handles multiple objects
t9 = 'prefix {"a": 1} middle {"b": {"c": 2}} end'
objects = _extract_json_objects(t9)
assert len(objects) == 2, f"Test 9 failed: expected 2 objects, got {len(objects)}"
print(f"Test 9 PASS: extracted {len(objects)} JSON objects")

# Test 10: _extract_json_objects handles deeply nested braces inside strings
t10 = '{"name": "test", "data": "some {nested} and {more {deep}} braces"}'
objects10 = _extract_json_objects(t10)
assert len(objects10) == 1, f"Test 10 failed: expected 1 object, got {len(objects10)}"
import json
parsed10 = json.loads(objects10[0])
assert parsed10["name"] == "test", f"Test 10 JSON parse wrong"
print(f"Test 10 PASS: braces inside strings handled correctly")

# Test 11: Whitespace preservation in old_text/new_text (THE CRITICAL FIX)
# Models put indented code inside <old_text>...</old_text>. The parser MUST
# preserve leading whitespace (indentation) — .strip() was removing it.
t11 = '<tool_call>\n<tool_name>replace_in_file</tool_name>\n<path>test.py</path>\n<old_text>\n    def foo(self):\n        pass\n</old_text>\n<new_text>\n    def foo(self):\n        return True\n</new_text>\n</tool_call>'
r11 = parse_tool_call(t11)
assert r11 is not None, "Test 11 failed: should parse replace_in_file"
name11, params11 = r11
assert name11 == "replace_in_file", f"Test 11 wrong tool: {name11}"
assert params11["old_text"].startswith("    "), f"Test 11 FAIL: old_text lost leading indentation: {repr(params11['old_text'][:40])}"
assert params11["new_text"].startswith("    "), f"Test 11 FAIL: new_text lost leading indentation: {repr(params11['new_text'][:40])}"
# Verify it would actually match indented file content
file_content = "class Foo:\n    def foo(self):\n        pass\n"
assert params11["old_text"] in file_content, f"Test 11 FAIL: old_text doesn't match file content"
print(f"Test 11 PASS: whitespace preserved in old_text/new_text")

# Test 12: Bare-tag strategy also preserves whitespace
t12 = 'Let me edit this:\n<tool_name>replace_in_file</tool_name>\n<path>test.py</path>\n<old_text>\n        self.value = None\n</old_text>\n<new_text>\n        self.value = 42\n</new_text>'
r12 = parse_tool_call(t12)
assert r12 is not None, "Test 12 failed: should parse bare-tag replace_in_file"
name12, params12 = r12
assert params12["old_text"].startswith("        "), f"Test 12 FAIL: bare-tag old_text lost indentation: {repr(params12['old_text'][:40])}"
print(f"Test 12 PASS: bare-tag strategy preserves whitespace")

# Test 13: Non-text params still get .strip() (path shouldn't have extra spaces) 
t13 = '<tool_call>\n<tool_name>read_file</tool_name>\n<path>  src/main.py  </path>\n<start_line> 1 </start_line>\n</tool_call>'
r13 = parse_tool_call(t13)
assert r13 is not None, "Test 13 failed" 
_, params13 = r13
assert params13["path"] == "src/main.py", f"Test 13 FAIL: path not stripped: {repr(params13['path'])}"
assert params13["start_line"] == "1", f"Test 13 FAIL: start_line not stripped: {repr(params13['start_line'])}"
print(f"Test 13 PASS: non-text params still stripped correctly")

print("\n=== ALL 13 PARSER TESTS PASSED ===")
