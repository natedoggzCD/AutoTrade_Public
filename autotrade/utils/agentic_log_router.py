"""
Route log errors to the agentic exception pipeline.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import deque
from typing import Deque, Tuple

from autotrade.utils.agentic_exceptions import route_exception


_KEYWORDS = ("error", "failed", "failure", "critical", "fatal", "panic", "exception")


class AgenticLogRouter(logging.Handler):
    def __init__(self, level: int = logging.ERROR, dedupe_window_sec: int = 60, max_seen: int = 200):
        super().__init__(level=level)
        self._guard = threading.local()
        self._dedupe_window = dedupe_window_sec
        self._seen: Deque[Tuple[float, str]] = deque(maxlen=max_seen)

    def emit(self, record: logging.LogRecord) -> None:
        if os.environ.get("AUTOTRADE_AGENTIC_LOG_ROUTER", "0").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
        }:
            return
        if getattr(self._guard, "active", False):
            return

        msg = record.getMessage()
        if not msg:
            return

        msg_l = msg.lower()
        if record.levelno < logging.ERROR and not any(k in msg_l for k in _KEYWORDS):
            return

        normalized = self._normalize_message(msg_l)
        now = time.time()
        if self._is_duplicate(normalized, now):
            return

        self._guard.active = True
        try:
            err = RuntimeError(f"[LOG:{record.levelname}] {record.name}: {msg}")
            file_hint = None
            line_hint = None
            try:
                if record.pathname and str(record.pathname).endswith(".py"):
                    file_hint = str(record.pathname)
                if record.lineno:
                    line_hint = int(record.lineno)
            except Exception:
                file_hint = None
                line_hint = None
            route_exception(
                err,
                context="log_router",
                file_hint=file_hint,
                line_hint=line_hint,
            )
        finally:
            self._guard.active = False

    def _is_duplicate(self, msg: str, now: float) -> bool:
        # purge old
        while self._seen and now - self._seen[0][0] > self._dedupe_window:
            self._seen.popleft()
        for ts, seen_msg in self._seen:
            if seen_msg == msg:
                return True
        self._seen.append((now, msg))
        return False

    @staticmethod
    def _normalize_message(msg: str) -> str:
        normalized = str(msg or "").strip().lower()
        if not normalized:
            return normalized

        # Collapse volatile fragments so recurring errors dedupe correctly.
        normalized = re.sub(
            r"\b\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:\d{2})?\b",
            "<ts>",
            normalized,
        )
        normalized = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<time>", normalized)
        normalized = re.sub(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            "<uuid>",
            normalized,
        )
        normalized = re.sub(r"\border[_ ]id[=: ]+[\w\-]+\b", "order_id=<id>", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()


def install_log_router() -> None:
    if os.environ.get("AUTOTRADE_AGENTIC_LOG_ROUTER", "0").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, AgenticLogRouter):
            return
    root.addHandler(AgenticLogRouter())
