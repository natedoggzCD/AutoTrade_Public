import logging
from datetime import datetime
from io import StringIO

from autotrade.utils import safe_logging


class _FrozenDateTime:
    current = datetime(2026, 3, 12, 23, 55)

    @classmethod
    def now(cls):
        return cls.current


def test_date_scoped_file_handler_rolls_over_by_date(tmp_path, monkeypatch):
    monkeypatch.setattr(safe_logging, "datetime", _FrozenDateTime)

    handler = safe_logging.DateScopedFileHandler(tmp_path / "runtime_20260312.log")
    handler.setFormatter(logging.Formatter("%(message)s"))

    first = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="before_midnight",
        args=(),
        exc_info=None,
    )
    handler.emit(first)

    _FrozenDateTime.current = datetime(2026, 3, 13, 0, 5)
    second = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="after_midnight",
        args=(),
        exc_info=None,
    )
    handler.emit(second)
    handler.close()

    first_path = tmp_path / "runtime_20260312.log"
    second_path = tmp_path / "runtime_20260313.log"

    assert first_path.read_text(encoding="utf-8").strip() == "before_midnight"
    assert second_path.read_text(encoding="utf-8").strip() == "after_midnight"


def test_sanitize_for_console_fails_closed_when_console_codec_is_broken(monkeypatch):
    class _BrokenText(str):
        def encode(self, encoding="utf-8", errors="strict"):
            raise TypeError("encoder must return a tuple (object, integer)")

    monkeypatch.setattr(
        safe_logging,
        "_resolve_console_encoding",
        lambda: "cp1252",
    )

    result = safe_logging.sanitize_for_console(_BrokenText("Hello 🌍"))

    assert isinstance(result, str)
    assert "Hello" in result


def test_safe_stream_handler_falls_back_to_ascii_when_sanitize_breaks(monkeypatch):
    stream = StringIO()
    handler = safe_logging.SafeStreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    monkeypatch.setattr(
        safe_logging,
        "sanitize_for_console",
        lambda text: (_ for _ in ()).throw(TypeError("bad codec path")),
    )

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Hello 🌍",
        args=(),
        exc_info=None,
    )
    handler.emit(record)

    assert "Hello" in stream.getvalue()


def test_safe_exception_text_survives_broken_str():
    class _BrokenExc(Exception):
        def __str__(self):
            raise RecursionError("maximum recursion depth exceeded while getting str")

    rendered = safe_logging.safe_exception_text(_BrokenExc("boom"))

    assert rendered.startswith("_BrokenExc:")
    assert "boom" in rendered or "unrenderable" in rendered
