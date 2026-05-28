"""Suppress repeat yfinance 404 spam for known-bad symbols.

Per-process in-memory set: first 404 for a symbol logs at WARNING (visible);
subsequent 404s for the same symbol are dropped silently.
Reset by restarting the process.
"""

import logging
import re

_FOUR_OH_FOUR_RE = re.compile(
    r'HTTP Error 404:.*Quote not found for symbol:\s*([A-Z0-9\.\-=]+)',
    re.IGNORECASE,
)

_seen_404s: set[str] = set()


class YFinance404Filter(logging.Filter):
    """Drop yfinance 404 records for symbols we've already flagged once."""

    def filter(self, record: logging.LogRecord) -> bool:  # True keeps, False drops
        try:
            msg = record.getMessage()
        except Exception:
            return True

        m = _FOUR_OH_FOUR_RE.search(msg)
        if not m:
            return True

        symbol = m.group(1).upper()
        if symbol in _seen_404s:
            return False  # drop repeat

        _seen_404s.add(symbol)
        # Downgrade the first occurrence to WARNING with a clearer message.
        record.levelno = logging.WARNING
        record.levelname = "WARNING"
        record.msg = (
            f"[yfinance] symbol '{symbol}' returned 404; suppressing further "
            f"404s for this symbol this session"
        )
        record.args = ()
        return True


def _attach_to(target) -> None:
    """Attach the filter to a logger or a handler, idempotent."""
    for f in target.filters:
        if isinstance(f, YFinance404Filter):
            return
    target.addFilter(YFinance404Filter())


_addHandler_patched = False


def install() -> None:
    """Attach the filter to the yfinance logger AND every handler on the
    root logger, present and future.

    H8 (2026-05-21): records matching "Quote not found for symbol: X"
    were leaking through non-yfinance loggers (saw 9+ ARMN, 1+ AMZS at
    ERROR level after the yfinance-only attachment was in place).

    Python logging only re-applies *handler* filters to propagated
    records — logger-level filters on parents do NOT fire for records
    originating from a child logger. To catch records from any logger,
    we attach to every handler on the root logger. Because install() is
    called from sitecustomize before app code configures logging, we
    also monkey-patch Logger.addHandler so handlers added later still
    receive the filter. The filter is pattern-gated, so non-matching
    records short-circuit on the regex. Idempotent.
    """
    global _addHandler_patched

    _attach_to(logging.getLogger("yfinance"))
    root = logging.getLogger()
    for handler in root.handlers:
        _attach_to(handler)

    if not _addHandler_patched:
        _orig_addHandler = logging.Logger.addHandler

        def _patched_addHandler(self, handler):  # type: ignore[no-redef]
            _orig_addHandler(self, handler)
            if self is logging.getLogger() or self.name == "root":
                _attach_to(handler)

        logging.Logger.addHandler = _patched_addHandler  # type: ignore[assignment]
        _addHandler_patched = True
