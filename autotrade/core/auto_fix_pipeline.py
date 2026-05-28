"""
AutoFixPipeline - Generate, apply, validate, and rollback code fixes.

Default behavior: enabled + auto-apply. Controlled via env:
- AUTO_FIX_ENABLED (default true)
- AUTO_FIX_AUTO_APPLY (default true)
- AUTO_FIX_MAX_ATTEMPTS (default 2)
- AUTO_FIX_VALIDATION_CMDS (optional ';' separated commands)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AutoFixPipeline:
    def __init__(self, project_root: Path, logger=None):
        self.project_root = Path(project_root).resolve()
        self.logger = logger
        self.enabled = self._env_true("AUTO_FIX_ENABLED", default=True)
        self.auto_apply = self._env_true("AUTO_FIX_AUTO_APPLY", default=True)
        self.max_attempts = self._env_int("AUTO_FIX_MAX_ATTEMPTS", default=2)
        self.validation_cmds = self._env_cmds("AUTO_FIX_VALIDATION_CMDS")
        self.use_pyright = self._env_true("AUTO_FIX_USE_PYRIGHT", default=True)
        self.use_pytest = self._env_true("AUTO_FIX_USE_PYTEST", default=True)
        self.test_script = os.environ.get("AUTO_FIX_TEST_SCRIPT", "tests/test_critical_functions.py")
        self._attempts: Dict[str, int] = {}
        self.max_daily_attempts_per_pattern = self._env_int(
            "AUTO_FIX_MAX_DAILY_PER_PATTERN", default=200
        )
        self.max_daily_attempts_total = self._env_int(
            "AUTO_FIX_MAX_DAILY_TOTAL", default=1000
        )
        self._attempt_day = date.today().isoformat()
        self._daily_total_attempts = 0
        self._daily_pattern_attempts: Dict[str, int] = {}
        self._file_inference_cache: Dict[str, Optional[str]] = {}

        self.log_dir = self.project_root / "logs"
        self.backup_root = self.log_dir / "autofix_backups"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)

    def _env_true(self, key: str, default: bool = False) -> bool:
        val = os.environ.get(key)
        if val is None:
            return default
        return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

    def _env_int(self, key: str, default: int = 0) -> int:
        val = os.environ.get(key)
        if not val:
            return default
        try:
            return int(val)
        except Exception:
            return default

    def _env_cmds(self, key: str) -> List[str]:
        val = os.environ.get(key, "").strip()
        if not val:
            return []
        return [c.strip() for c in val.split(";") if c.strip()]

    def _resolve_test_script(self) -> Optional[Path]:
        candidates: List[str] = []
        if self.test_script:
            candidates.append(self.test_script)
        candidates.extend([
            "tests/test_critical_functions.py",
            "test_critical_functions.py",
        ])
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = (self.project_root / path).resolve()
            if path.exists():
                return path
        return None

    def _roll_daily_counters(self) -> None:
        today_key = date.today().isoformat()
        if today_key == self._attempt_day:
            return
        self._attempt_day = today_key
        self._daily_total_attempts = 0
        self._daily_pattern_attempts = {}

    @staticmethod
    def _camel_to_snake(value: str) -> str:
        value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value or "")
        return re.sub(r"_+", "_", value).strip("_").lower()

    def _resolve_existing_file(self, file_hint: Optional[str]) -> Optional[str]:
        if not file_hint:
            return None
        p = Path(str(file_hint).strip())
        if not p.is_absolute():
            p = self.project_root / p
        try:
            p = p.resolve()
        except Exception:
            return None
        if not p.exists():
            return None
        return str(p)

    def _is_project_file(self, file_path: Optional[str]) -> bool:
        if not file_path:
            return False
        try:
            Path(file_path).resolve().relative_to(self.project_root)
            return True
        except Exception:
            return False

    def _infer_file_hint_from_error(self, combined_error_text: str) -> Optional[str]:
        text = str(combined_error_text or "")
        norm = text.strip().lower()
        if not norm:
            return None
        cached = self._file_inference_cache.get(norm)
        if cached is not None:
            return cached

        logger_name = None
        message = text

        # RuntimeError from log router:
        # [LOG:ERROR] Logger.Name: message
        m = re.search(r"\[LOG:[A-Z]+\]\s*([^:]+):\s*(.+)", text)
        if m:
            logger_name = m.group(1).strip()
            message = m.group(2).strip()

        # Strong mappings first.
        lower_message = message.lower()
        if "failed to analyze" in lower_message and "momentum" in lower_message:
            candidate = self.project_root / "autotrade" / "signals" / "momentum_engine.py"
            if candidate.exists():
                resolved = str(candidate.resolve())
                self._file_inference_cache[norm] = resolved
                return resolved
        if "opencode.json" in lower_message and "alpaca" in lower_message:
            candidate = self.project_root / "autotrade" / "utils" / "mcp_client.py"
            if candidate.exists():
                resolved = str(candidate.resolve())
                self._file_inference_cache[norm] = resolved
                return resolved

        def _check_candidate(path: Path) -> Optional[str]:
            if path.exists():
                try:
                    rp = path.resolve()
                    rp.relative_to(self.project_root)
                    return str(rp)
                except Exception:
                    return None
            return None

        if logger_name:
            if logger_name.startswith("autotrade."):
                resolved = _check_candidate(
                    self.project_root / (logger_name.replace(".", "/") + ".py")
                )
                if resolved:
                    self._file_inference_cache[norm] = resolved
                    return resolved

            if logger_name.startswith("AutoTrade."):
                suffix = logger_name.split(".", 1)[1]
                if suffix.startswith("AutonomousAgent"):
                    resolved = _check_candidate(
                        self.project_root / "autotrade" / "core" / "autonomous_agent.py"
                    )
                    if resolved:
                        self._file_inference_cache[norm] = resolved
                        return resolved
                leaf = self._camel_to_snake(suffix.split(".")[-1])
                for root in ("autotrade/core", "autotrade/utils", "autotrade/signals", "langgraph_workflow"):
                    resolved = _check_candidate(self.project_root / root / f"{leaf}.py")
                    if resolved:
                        self._file_inference_cache[norm] = resolved
                        return resolved

            if "." not in logger_name:
                for root in ("autotrade/core", "autotrade/utils", "autotrade/signals", "langgraph_workflow"):
                    resolved = _check_candidate(self.project_root / root / f"{logger_name}.py")
                    if resolved:
                        self._file_inference_cache[norm] = resolved
                        return resolved

        self._file_inference_cache[norm] = None
        return None

    def run(
        self,
        task_router,
        error: Exception,
        traceback_str: str,
        context: str = "",
        file_hint: Optional[str] = None,
        line_hint: Optional[int] = None,
    ) -> Dict:
        if not self.enabled:
            return {"success": False, "skipped": True, "reason": "AUTO_FIX_ENABLED is false"}
        if not task_router:
            return {"success": False, "skipped": True, "reason": "TaskRouter unavailable"}

        self._roll_daily_counters()

        combined_error_text = (
            f"{type(error).__name__}: {str(error)}\n{traceback_str or ''}\n{context or ''}"
        )

        # Try to extract file/line if not provided
        if not file_hint:
            file_hint, line_hint = self._extract_file_line_from_traceback(traceback_str)
        resolved_hint = self._resolve_existing_file(file_hint)
        inferred_hint = None if self._is_project_file(resolved_hint) else self._infer_file_hint_from_error(combined_error_text)
        file_hint = inferred_hint or resolved_hint
        if not file_hint:
            return {
                "success": False,
                "skipped": True,
                "reason": "source_file_unresolved",
            }

        attempt_key = f"{type(error).__name__}:{str(error)[:120]}"
        attempts = self._attempts.get(attempt_key, 0)
        if attempts >= self.max_attempts:
            return {"success": False, "skipped": True, "reason": "max_attempts reached"}
        if self._daily_total_attempts >= self.max_daily_attempts_total:
            return {"success": False, "skipped": True, "reason": "daily_total_cap_reached"}
        if (
            self._daily_pattern_attempts.get(attempt_key, 0)
            >= self.max_daily_attempts_per_pattern
        ):
            return {
                "success": False,
                "skipped": True,
                "reason": "daily_pattern_cap_reached",
            }
        self._attempts[attempt_key] = attempts + 1
        self._daily_total_attempts += 1
        self._daily_pattern_attempts[attempt_key] = (
            self._daily_pattern_attempts.get(attempt_key, 0) + 1
        )

        run_meta = {
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "context": context,
            "attempt_key": attempt_key,
            "attempt_num": self._attempts.get(attempt_key, 0),
            "file_hint": file_hint,
            "line_hint": line_hint,
        }

        try:
            from autotrade.core.agentic_orchestrator import Task, TaskType
        except Exception as e:
            return {"success": False, "skipped": True, "reason": f"Task import failed: {e}"}

        task = Task(
            type=TaskType.CODE_FIX,
            description=f"Auto-fix runtime error: {type(error).__name__}",
            data={
                "file": file_hint,
                "line": line_hint,
                "error": str(error),
                "traceback": traceback_str,
                "context": context,
                "mode": "auto_fix",
            },
            priority=1,
        )

        result = task_router.route(task)
        if not result.success:
            self._log_result({"success": False, "reason": result.message}, meta=run_meta)
            return {"success": False, "reason": result.message}

        result_data = getattr(result, "data", {}) or {}
        already_applied = int(result_data.get("changes_applied", 0) or 0)
        if already_applied > 0:
            parsed_result = self._parse_fix_spec(result.message or "") or {}
            modified_files = [
                str(Path(path).resolve())
                for path in (result_data.get("modified_files") or [])
                if str(path or "").strip()
            ]
            if not modified_files:
                for change in result_data.get("fix_details") or []:
                    file_path = str(change.get("file") or "").strip()
                    if not file_path:
                        continue
                    modified_files.append(str(Path(file_path).resolve()))
            deduped_files = list(dict.fromkeys(modified_files))
            apply_result = {
                "success": True,
                "already_applied": True,
                "persisted_by": "codeagent",
                "modified_files": deduped_files,
                "summary": str(
                    parsed_result.get("summary")
                    or result.message
                    or "persisted_by_codeagent"
                ),
            }
            self._log_result(apply_result, meta=run_meta)
            return apply_result

        fix_spec = self._parse_fix_spec(result.message)
        if not fix_spec:
            self._log_result(
                {"success": False, "reason": "No fix spec parsed", "response": result.message},
                meta=run_meta,
            )
            return {"success": False, "reason": "No fix spec parsed"}

        # If agent omitted file paths, fall back to extracted file_hint
        if file_hint and isinstance(fix_spec, dict):
            changes = fix_spec.get("changes") or []
            for change in changes:
                if not change.get("file"):
                    change["file"] = file_hint

        if not self.auto_apply:
            self._log_result({"success": False, "dry_run": True, "fix_spec": fix_spec}, meta=run_meta)
            return {"success": False, "dry_run": True, "fix_spec": fix_spec}

        apply_result = self._apply_fix_spec(fix_spec)
        self._log_result(apply_result, meta=run_meta)
        return apply_result

    def _extract_file_line_from_traceback(self, tb: str) -> Tuple[Optional[str], Optional[int]]:
        matches = re.findall(r'File ["\']([^"\']+\.py)["\'], line (\d+)', tb or "")
        if not matches:
            return None, None
        file_path, line_str = matches[-1]
        try:
            line_num = int(line_str)
        except Exception:
            line_num = None
        return file_path, line_num

    def _parse_fix_spec(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        # Try direct JSON
        try:
            return json.loads(text)
        except Exception:
            pass
        # Extract first JSON object from text
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        return json.loads(blob)
                    except Exception:
                        return None
        return None

    def _apply_fix_spec(self, spec: Dict) -> Dict:
        changes = spec.get("changes") or []
        if not changes:
            return {"success": False, "reason": "No changes in fix spec"}

        # Create temporary working directory for this attempt
        temp_work_dir = self.log_dir / "autofix_work" / datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_work_dir.mkdir(parents=True, exist_ok=True)

        backup_dir = self.backup_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)

        backups: Dict[Path, Path] = {}
        modified_files: List[Path] = []
        temp_copies: Dict[Path, Path] = {}  # original_path -> temp_copy_path

        try:
            # 1. Create temp copies and apply changes to them
            for change in changes:
                file_path = change.get("file")
                search = change.get("search", "")
                replace = change.get("replace", "")
                count = change.get("count")

                if file_path is None or str(file_path).strip().lower() in ("", "none", "null"):
                    return {
                        "success": False,
                        "reason": "Missing file path in fix spec change",
                        "change": change,
                    }
                if not search:
                    return {"success": False, "reason": "Invalid change entry", "change": change}

                full_path = Path(file_path)
                if not full_path.is_absolute():
                    full_path = self.project_root / full_path
                full_path = full_path.resolve()

                if not full_path.exists():
                    return {"success": False, "reason": f"File not found: {full_path}"}

                # Create temp copy if not already done
                if full_path not in temp_copies:
                    rel_path = full_path.relative_to(self.project_root)
                    temp_copy = temp_work_dir / rel_path
                    temp_copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, temp_copy)
                    temp_copies[full_path] = temp_copy

                temp_path = temp_copies[full_path]
                content = temp_path.read_text(encoding="utf-8")
                
                occurrences = content.count(search)
                if count is None:
                    count = 1
                if occurrences != count:
                    return {
                        "success": False,
                        "reason": "Search text occurrences mismatch in temp copy",
                        "file": str(full_path),
                        "expected": count,
                        "found": occurrences,
                    }

                new_content = content.replace(search, replace, count)
                temp_path.write_text(new_content, encoding="utf-8")

            # 2. Pre-validation cleanup on temp copies
            for full_path, temp_path in temp_copies.items():
                if temp_path.suffix == ".py":
                    # Try ruff check --fix
                    subprocess.run(
                        ["ruff", "check", "--fix", str(temp_path.resolve())],
                        capture_output=True, text=True, timeout=15, cwd=self.project_root
                    )
                    # Try ruff format
                    subprocess.run(
                        ["ruff", "format", str(temp_path.resolve())],
                        capture_output=True, text=True, timeout=15, cwd=self.project_root
                    )

            # 3. Validation on temp copies
            # NOTE: _run_validation needs to be aware of temp copies if it runs tests that import these files.
            # This is complex because imports will still point to original files.
            # For now, we py_compile temp copies, but for full validation, we MUST swap them.
            
            # Temporary Swap for Validation
            for full_path, temp_path in temp_copies.items():
                # Backup original
                backup_path = backup_dir / full_path.name
                shutil.copy2(full_path, backup_path)
                backups[full_path] = backup_path
                
                # Overwrite original with fixed temp
                shutil.copy2(temp_path, full_path)
                modified_files.append(full_path)

            valid, outputs = self._run_validation(modified_files, spec.get("validation"))
            
            if not valid:
                self._rollback(backups)
                return {"success": False, "reason": "Validation failed", "validation": outputs}

            # Success! Cleanup temp work dir
            shutil.rmtree(temp_work_dir, ignore_errors=True)
            return {
                "success": True,
                "modified_files": [str(p) for p in modified_files],
                "validation": outputs,
                "summary": str(spec.get("summary") or "auto_fix_pipeline_applied_patch"),
            }

        except Exception as e:
            self._rollback(backups)
            return {"success": False, "reason": f"Apply failed: {e}"}
        finally:
            shutil.rmtree(temp_work_dir, ignore_errors=True)

    def _run_validation(self, modified_files: List[Path], validation_cmds: Optional[List[str]]):
        cmds = validation_cmds or self.validation_cmds
        outputs = []

        if not cmds:
            cmds = self._default_validation_cmds(modified_files)
            if not cmds:
                # Emergency fallback: py_compile all modified python files
                for path in modified_files:
                    if path.suffix == ".py":
                        cmds.append(f"\"{sys.executable}\" -m py_compile \"{path}\"")

        for cmd in cmds:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=self.project_root,
                )
                outputs.append({
                    "command": cmd,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:],
                })
                if result.returncode != 0:
                    # If pyright is missing, fall back to py_compile instead of failing
                    if cmd.strip().startswith("pyright") and (
                        "not recognized" in result.stderr.lower() or "not found" in result.stderr.lower()
                    ):
                        for path in modified_files:
                            if path.suffix == ".py":
                                fallback = f"\"{sys.executable}\" -m py_compile \"{path}\""
                                fallback_res = subprocess.run(
                                    fallback,
                                    shell=True,
                                    capture_output=True,
                                    text=True,
                                    timeout=120,
                                    cwd=self.project_root,
                                )
                                outputs.append({
                                    "command": fallback,
                                    "returncode": fallback_res.returncode,
                                    "stdout": fallback_res.stdout[-2000:],
                                    "stderr": fallback_res.stderr[-2000:],
                                })
                                if fallback_res.returncode != 0:
                                    return False, outputs
                        continue
                    return False, outputs
            except Exception as e:
                outputs.append({"command": cmd, "error": str(e)})
                return False, outputs

        return True, outputs

    def _default_validation_cmds(self, modified_files: List[Path]) -> List[str]:
        cmds: List[str] = []

        # 1) Pyright (Pylance engine) on modified files if available
        if self.use_pyright and self._tool_exists("pyright"):
            file_args = " ".join(f"\"{p}\"" for p in modified_files if p.suffix == ".py")
            if file_args:
                cmds.append(f"pyright {file_args}")
            else:
                cmds.append("pyright")

        # 2) Execute critical test script if present
        script_path = self._resolve_test_script()
        if script_path:
            cmds.append(f"\"{sys.executable}\" \"{script_path}\"")

        # 3) Pytest (if available)
        if self.use_pytest and self._tool_exists("pytest"):
            cmds.append("pytest -q")

        return cmds

    def _tool_exists(self, tool: str) -> bool:
        return shutil.which(tool) is not None

    def _rollback(self, backups: Dict[Path, Path]) -> None:
        for original, backup in backups.items():
            try:
                if backup.exists():
                    original.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

    def _log_result(self, result: Dict, meta: Optional[Dict] = None) -> None:
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "meta": meta or {},
            }
            path = self.log_dir / f"autofix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
