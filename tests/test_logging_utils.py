import logging

from autotrade.utils.logging_utils import WindowsSafeRotatingFileHandler


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_windows_safe_rotating_file_handler_backs_off_locked_rollover(tmp_path):
    log_path = tmp_path / "app.jsonl"
    handler = WindowsSafeRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        rollover_cooldown_seconds=60,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    rotate_calls = 0

    def locked_rotate(source, dest):
        nonlocal rotate_calls
        rotate_calls += 1
        raise PermissionError(32, "locked", source)

    handler.rotate = locked_rotate

    try:
        handler.emit(_record("first"))
        assert rotate_calls == 1
        assert handler.rollover_failures == 1
        assert "first" in log_path.read_text(encoding="utf-8")

        handler.emit(_record("second"))
        assert rotate_calls == 1
        assert "second" in log_path.read_text(encoding="utf-8")
    finally:
        handler.close()
