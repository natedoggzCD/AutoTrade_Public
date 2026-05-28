import json
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict

# Try to import SafeStreamHandler for Windows console compatibility
try:
    from autotrade.utils.safe_logging import SafeStreamHandler

    _SAFE_HANDLER_AVAILABLE = True
except ImportError:
    _SAFE_HANDLER_AVAILABLE = False

_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_BACKUP_COUNT = 5

_STANDARD_ATTRS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        # Include extra fields passed via logger.*(..., extra={...})
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value

        return json.dumps(payload, default=str)


class WindowsSafeRotatingFileHandler(RotatingFileHandler):
    """
    RotatingFileHandler variant that tolerates Windows file-lock collisions.

    On Windows, any process tailing or otherwise holding the active log file can
    block ``os.rename`` during rollover. The standard handler reports a logging
    error on every subsequent emit once the file remains over maxBytes. This
    handler fails open by appending to the active file and retrying rollover
    after a cooldown.
    """

    def __init__(
        self,
        *args: Any,
        rollover_cooldown_seconds: float = 300.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rollover_cooldown_seconds = max(0.0, rollover_cooldown_seconds)
        self._rollover_blocked_until = 0.0
        self.rollover_failures = 0

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self._rollover_blocked_until > time.monotonic():
            return False
        return super().shouldRollover(record)

    def doRollover(self) -> None:
        try:
            super().doRollover()
            self._rollover_blocked_until = 0.0
            self.rollover_failures = 0
        except PermissionError:
            self.rollover_failures += 1
            self._rollover_blocked_until = (
                time.monotonic() + self.rollover_cooldown_seconds
            )
            if self.stream is None or self.stream.closed:
                self.stream = self._open()


def configure_logging(
    logs_dir: Path,
    level: str | int = "INFO",
    filename: str = "app.jsonl",
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    console: bool = True,
) -> logging.Logger:
    """
    Configure root logging to JSON with rotation.

    Returns the root logger.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if getattr(root_logger, "_json_configured", False):
        return root_logger

    root_logger.setLevel(level)

    formatter = JsonFormatter()

    file_handler = WindowsSafeRotatingFileHandler(
        logs_dir / filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console:
        # Use SafeStreamHandler if available (Windows console compatible)
        if _SAFE_HANDLER_AVAILABLE:
            console_handler = SafeStreamHandler()
        else:
            console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    root_logger._json_configured = True

    # H8 (2026-05-22): sitecustomize.py is NOT reliably auto-imported by
    # Python in this env (verified: yfinance logger has no filters even
    # after a normal launch). Call install() here so the filter is
    # guaranteed to be attached to the file_handler we just added and
    # to the yfinance logger, regardless of sitecustomize timing.
    try:
        from autotrade.utils.yfinance_404_filter import install as _install_yf404
        _install_yf404()
    except Exception:
        pass

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Log a structured event with extra fields."""
    logger.log(level, message, extra=fields)
