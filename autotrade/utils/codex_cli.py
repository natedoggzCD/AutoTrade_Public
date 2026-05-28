"""
Codex CLI wrapper.

Provides a minimal interface to invoke the Codex CLI and return output.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Callable, List, Optional, Tuple


def codex_available(command: str = "codex") -> bool:
    """Check if Codex CLI is available on PATH."""
    return shutil.which(command) is not None


def run_codex(
    prompt: str,
    command: str = "codex",
    timeout: int = 120,
    use_stdin: bool = False,
    extra_args: Optional[List[str]] = None,
    log_path: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Run Codex CLI with the given prompt.

    Returns:
        (success, stdout, stderr)
    """
    args = [command, "exec"]
    if extra_args:
        args.extend(extra_args)

    try:
        if use_stdin:
            result = subprocess.run(
                args + ["-"],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        else:
            result = subprocess.run(
                args + [prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        success = result.returncode == 0
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if log_path:
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(stdout)
                    if stderr:
                        f.write("\n\n[stderr]\n")
                        f.write(stderr)
            except Exception:
                pass
        return success, stdout, stderr
    except Exception as exc:
        return False, "", f"Codex CLI error: {exc}"


def run_codex_stream(
    prompt: str,
    command: str = "codex",
    timeout: int = 120,
    use_stdin: bool = False,
    extra_args: Optional[List[str]] = None,
    log_path: Optional[str] = None,
    json_events: bool = False,
    on_stdout: Optional[Callable[[str], None]] = None,
    on_stderr: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str, str]:
    """
    Run Codex CLI and stream stdout/stderr line-by-line.
    Returns (success, stdout, stderr) after process completes.
    """
    args = [command, "exec"]
    if extra_args:
        args.extend(extra_args)
    if json_events:
        args.append("--json")

    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    try:
        if use_stdin:
            proc = subprocess.Popen(
                args + ["-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
        else:
            proc = subprocess.Popen(
                args + [prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )

        assert proc.stdout is not None
        assert proc.stderr is not None

        # Read streams line-by-line
        for line in proc.stdout:
            line = line.rstrip("\n")
            stdout_chunks.append(line)
            if on_stdout:
                on_stdout(line)
        for line in proc.stderr:
            line = line.rstrip("\n")
            stderr_chunks.append(line)
            if on_stderr:
                on_stderr(line)

        proc.wait(timeout=timeout)

        stdout = "\n".join(stdout_chunks).strip()
        stderr = "\n".join(stderr_chunks).strip()

        if log_path:
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    if stdout:
                        f.write(stdout + "\n")
                    if stderr:
                        f.write("\n[stderr]\n")
                        f.write(stderr + "\n")
            except Exception:
                pass

        return proc.returncode == 0, stdout, stderr
    except Exception as exc:
        return False, "", f"Codex CLI error: {exc}"
