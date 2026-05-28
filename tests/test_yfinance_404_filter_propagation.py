"""H8 (2026-05-21): verify the yfinance 404 filter catches records from any
logger, not just the `yfinance` logger.

Before H8, records like `HTTP Error 404: Quote not found for symbol: ARMN`
leaked through non-yfinance loggers and produced ~9 ERROR-level repeats per
delisted symbol per session. Fix: attach the pattern-gated filter to root
handlers and monkey-patch addHandler so future handlers also get it.
"""

from __future__ import annotations

import logging

import pytest

from autotrade.utils.yfinance_404_filter import (
    YFinance404Filter,
    _seen_404s,
    install,
)


@pytest.fixture(autouse=True)
def _reset_filter_state():
    _seen_404s.clear()
    yield
    _seen_404s.clear()


def _capture_handler():
    """Return a fresh stream handler that records what gets emitted."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _Capture(level=logging.DEBUG)
    return h, records


def test_filter_drops_repeats_through_non_yfinance_logger():
    install()
    h, records = _capture_handler()
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    try:
        other = logging.getLogger("autotrade.signals.something")
        msg = "HTTP Error 404: Quote not found for symbol: ARMN"
        other.error(msg)
        other.error(msg)
        other.error(msg)
    finally:
        root.removeHandler(h)

    armn_records = [
        r for r in records if "ARMN" in r.getMessage()
    ]
    assert len(armn_records) == 1, [r.getMessage() for r in armn_records]
    assert armn_records[0].levelno == logging.WARNING


def test_filter_attached_to_handlers_added_after_install():
    """addHandler monkey-patch ensures handlers added post-install still
    receive the filter."""
    install()
    h, _ = _capture_handler()
    root = logging.getLogger()
    root.addHandler(h)
    try:
        assert any(isinstance(f, YFinance404Filter) for f in h.filters), (
            "addHandler patch did not attach filter to new root handler"
        )
    finally:
        root.removeHandler(h)


def test_configure_logging_installs_filter(tmp_path):
    """H8 (2026-05-22): in production sitecustomize.py is not reliably
    auto-imported, so the filter never attached. configure_logging() must
    install it directly so the json file handler we just registered gets
    the filter and yfinance ERRORs are deduped on the real path.
    """
    import logging
    import autotrade.utils.logging_utils as logging_utils

    root = logging.getLogger()
    # Reset the configure_logging idempotency guard and clear handlers so
    # this test exercises a fresh setup.
    if hasattr(root, "_json_configured"):
        delattr(root, "_json_configured")
    prior_handlers = list(root.handlers)
    for h in prior_handlers:
        root.removeHandler(h)
    try:
        logging_utils.configure_logging(
            tmp_path, level="DEBUG", filename="app.jsonl", console=False
        )
        yfinance_logger = logging.getLogger("yfinance")
        assert any(
            isinstance(f, YFinance404Filter) for f in yfinance_logger.filters
        ), "configure_logging did not attach filter to yfinance logger"
        assert root.handlers, "configure_logging should add a file handler"
        file_h = root.handlers[0]
        assert any(
            isinstance(f, YFinance404Filter) for f in file_h.filters
        ), "configure_logging did not attach filter to its file handler"

        # End-to-end: 3 raw 404s collapse to 1 rewritten WARNING in the file.
        msg = (
            'HTTP Error 404: {"quoteSummary":{"result":null,"error":'
            '{"code":"Not Found","description":"Quote not found for symbol: ZZZZ"}}}'
        )
        for _ in range(3):
            yfinance_logger.error(msg)
        for h in root.handlers:
            h.flush()
        contents = (tmp_path / "app.jsonl").read_text(encoding="utf-8")
        assert contents.count("ZZZZ") == 1, contents
        assert "symbol 'ZZZZ' returned 404" in contents
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        if hasattr(root, "_json_configured"):
            delattr(root, "_json_configured")
        for h in prior_handlers:
            root.addHandler(h)


def test_filter_passes_unrelated_records_untouched():
    install()
    h, records = _capture_handler()
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
    try:
        logging.getLogger("autotrade.misc").info("normal log message")
        logging.getLogger("autotrade.misc").error("unrelated error 500")
    finally:
        root.removeHandler(h)
    msgs = [r.getMessage() for r in records]
    assert "normal log message" in msgs
    assert "unrelated error 500" in msgs
