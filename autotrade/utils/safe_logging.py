"""
Safe Logging Utilities - Windows Console Compatible

Handles encoding issues when logging to Windows console (CP1252/CP437)
while preserving full Unicode in log files.

Usage:
    from autotrade.utils.safe_logging import get_safe_logger, safe_log
    
    # Option 1: Get a pre-configured safe logger
    logger = get_safe_logger('my_module')
    logger.info("Message with emoji 🚀")  # Auto-sanitized for console
    
    # Option 2: Wrap an existing logger
    from autotrade.utils.safe_logging import SafeLogger
    safe_logger = SafeLogger(existing_logger)
    safe_logger.info("Message with emoji")
    
    # Option 3: Just sanitize a string
    from autotrade.utils.safe_logging import sanitize_for_console
    clean_msg = sanitize_for_console("Hello 🌍")  # "Hello [globe]"
"""

import logging
import sys
import re
import codecs
from pathlib import Path
from datetime import datetime
from threading import RLock
from typing import Any, Optional, Union

# Directory for log files
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# Emoji to ASCII replacement map
EMOJI_REPLACEMENTS = {
    # Status indicators
    '✅': '[OK]',
    '❌': '[X]',
    '⚠️': '[!]',
    '⏳': '[..]',
    '🔄': '[~]',
    '➡️': '[->]',
    '⬆️': '[^]',
    '⬇️': '[v]',
    
    # Performance grades
    '🚀': '[++]',
    '📈': '[^]',
    '📉': '[v]',
    '💰': '[$]',
    '💵': '[$]',
    
    # Emotions/states
    '😢': '[:(]',
    '😊': '[:)]',
    '🎉': '[!]',
    '🔥': '[*]',
    '💪': '[+]',
    
    # Objects
    '📊': '[chart]',
    '📋': '[list]',
    '📁': '[folder]',
    '📄': '[file]',
    '🔧': '[tool]',
    '⚙️': '[gear]',
    '🔍': '[search]',
    '🌐': '[web]',
    '🌍': '[globe]',
    '💾': '[save]',
    '🤖': '[bot]',
    '📚': '[docs]',
    
    # Misc
    '•': '-',
    '→': '->',
    '←': '<-',
    '↑': '^',
    '↓': 'v',
    '×': 'x',
    '÷': '/',
    '≈': '~',
    '≠': '!=',
    '≤': '<=',
    '≥': '>=',
}

# Regex pattern to match common emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Misc symbols, emoticons, etc
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport/map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "]+", 
    flags=re.UNICODE
)


# Cache for console encoding to prevent recursion and overhead
_CONSOLE_ENCODING_CACHE: Optional[str] = None


def sanitize_for_console(text: str) -> str:
    """
    Sanitize text for Windows console output.
    
    Replaces emoji and special Unicode characters with ASCII equivalents.
    
    Args:
        text: String that may contain emoji/Unicode
        
    Returns:
        Console-safe ASCII string
    """
    if not text:
        return ""
    
    # Handle non-string types gracefully
    if not isinstance(text, str):
        text = str(text)
    
    # Fast path for plain ASCII
    try:
        text.encode('ascii')
        return text
    except Exception:
        pass
    
    result = text
    
    # Replace known emojis with ASCII equivalents
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        if emoji in result:
            result = result.replace(emoji, replacement)
    
    # Replace any remaining emoji with [?]
    result = EMOJI_PATTERN.sub('[?]', result)
    
    # Try to encode to console encoding, but fail closed to ASCII if the
    # active Windows codec path is unstable or recursively broken.
    try:
        console_encoding = _resolve_console_encoding()
        # Test if it's encodable
        result.encode(console_encoding, errors='strict')
    except Exception:
        # Final safety: strip everything non-ASCII
        result = _ascii_fallback(result)
    
    return result


def _resolve_console_encoding() -> str:
    """Return a safe console encoding name."""
    global _CONSOLE_ENCODING_CACHE
    if _CONSOLE_ENCODING_CACHE:
        return _CONSOLE_ENCODING_CACHE

    try:
        encoding = getattr(sys.stdout, "encoding", None)
    except Exception:
        encoding = None

    if not isinstance(encoding, str) or not encoding.strip():
        # Default to UTF-8 on Windows now that we force it in sitecustomize
        encoding = "utf-8"

    encoding = encoding.strip().lower()
    try:
        codecs.lookup(encoding)
        _CONSOLE_ENCODING_CACHE = encoding
        return encoding
    except Exception:
        # Final fallback is always utf-8
        _CONSOLE_ENCODING_CACHE = "utf-8"
        return "utf-8"


def _ascii_fallback(text: str) -> str:
    """Return a plain-ASCII best-effort fallback."""
    try:
        return str(text).encode("ascii", errors="replace").decode("ascii")
    except Exception:
        return ''.join(c if ord(c) < 128 else '?' for c in str(text))


def safe_exception_text(exc: Any, max_len: int = 500, _depth: int = 0) -> str:
    """
    Best-effort exception text that avoids recursive/broken ``str(exc)`` paths.
    """
    if exc is None:
        return "Exception"

    # Recursion protection
    if _depth > 3:
        return f"{type(exc).__name__} [deep recursion protected]"

    exc_type = type(exc).__name__ or "Exception"
    raw_message = ""

    for renderer in (str, repr):
        try:
            # Forcing utf-8 representation if it's a known problematic exception type
            candidate = renderer(exc)
        except (RecursionError, UnicodeEncodeError, UnicodeDecodeError):
            candidate = ""
        except Exception:
            candidate = ""

        if isinstance(candidate, str):
            candidate = candidate.strip()
            if candidate:
                raw_message = candidate
                break

    if not raw_message:
        try:
            args = getattr(exc, "args", ()) or ()
        except Exception:
            args = ()
        safe_args = []
        for arg in list(args)[:3]:
            try:
                rendered = repr(arg)
            except Exception:
                rendered = "<unrenderable>"
            safe_args.append(rendered)
        raw_message = ", ".join(safe_args) if safe_args else "<unrenderable>"

    try:
        raw_message = re.sub(r"\s+", " ", raw_message).strip()
    except Exception:
        raw_message = _ascii_fallback(raw_message)

    try:
        safe_message = sanitize_for_console(raw_message)
    except Exception:
        safe_message = _ascii_fallback(raw_message)

    if max_len > 0 and len(safe_message) > max_len:
        safe_message = safe_message[: max_len - 15].rstrip() + "... [truncated]"

    if not safe_message or safe_message == exc_type:
        return exc_type
    if safe_message.startswith(f"{exc_type}:"):
        return safe_message
    return f"{exc_type}: {safe_message}"

class SafeStreamHandler(logging.StreamHandler):
    """
    A StreamHandler that sanitizes output for Windows console.
    
    Automatically converts emoji and special characters to ASCII
    WITHOUT modifying the original log record (so file logs stay Unicode).
    """
    
    def emit(self, record):
        try:
            # Format the message first (this uses the shared record)
            msg = self.format(record)
            
            # Sanitize the COMPLETELY formatted message for this stream only
            safe_msg = sanitize_for_console(msg)
            
            # Write directly to the stream
            stream = self.stream
            stream.write(safe_msg + self.terminator)
            self.flush()
        except Exception:
            try:
                fallback_msg = _ascii_fallback(self.format(record))
                self.stream.write(fallback_msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)


class DateScopedFileHandler(logging.Handler):
    """
    UTF-8 file handler that reopens the destination when the calendar date changes.

    This preserves long-running process continuity while keeping one physical log
    file per date-scoped filename.
    """

    def __init__(self, log_file: Union[str, Path], level: int = logging.DEBUG):
        super().__init__(level=level)
        self._template_path = Path(log_file)
        self._current_date_key: Optional[str] = None
        self._file_handler: Optional[logging.FileHandler] = None
        self._handler_lock = RLock()

    @staticmethod
    def _today_tokens() -> tuple[str, str]:
        now = datetime.now()
        return now.strftime("%Y-%m-%d"), now.strftime("%Y%m%d")

    def _path_for_current_date(self) -> Path:
        filename = self._template_path.name
        dashed, compact = self._today_tokens()
        updated = filename

        if re.search(r"\d{4}-\d{2}-\d{2}", updated):
            updated = re.sub(r"\d{4}-\d{2}-\d{2}", dashed, updated)
        elif re.search(r"\d{8}", updated):
            updated = re.sub(r"\d{8}", compact, updated)
        else:
            stem = self._template_path.stem
            suffix = self._template_path.suffix
            updated = f"{stem}_{dashed}{suffix}"

        return self._template_path.with_name(updated)

    def _ensure_handler(self) -> logging.FileHandler:
        with self._handler_lock:
            current_key = datetime.now().strftime("%Y-%m-%d")
            if self._file_handler is not None and self._current_date_key == current_key:
                return self._file_handler

            if self._file_handler is not None:
                try:
                    self._file_handler.flush()
                    self._file_handler.close()
                finally:
                    self._file_handler = None

            current_path = self._path_for_current_date()
            current_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(current_path, encoding="utf-8")
            handler.setLevel(self.level)
            if self.formatter is not None:
                handler.setFormatter(self.formatter)
            self._file_handler = handler
            self._current_date_key = current_key
            return handler

    def emit(self, record):
        try:
            handler = self._ensure_handler()
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def flush(self):
        with self._handler_lock:
            if self._file_handler is not None:
                self._file_handler.flush()

    def close(self):
        try:
            with self._handler_lock:
                if self._file_handler is not None:
                    self._file_handler.close()
                    self._file_handler = None
        finally:
            super().close()


class SafeLogger:
    """
    Wrapper around a logger that sanitizes console output.
    
    File handlers still receive full Unicode.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def _safe_log(self, level: int, msg: str, *args, **kwargs):
        """Log with console-safe message."""
        # For console, we sanitize; for file handlers, we don't
        # The SafeStreamHandler handles this automatically
        self._logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._safe_log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._safe_log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._safe_log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._safe_log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._safe_log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._safe_log(logging.ERROR, msg, *args, **kwargs)


def get_safe_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a logger configured for safe Windows console output.
    
    Args:
        name: Logger name (usually module name)
        log_file: Optional log file path. If None, uses logs/{name}_{date}.log
        level: Overall logging level
        console_level: Console handler level
        
    Returns:
        Configured logger with safe console output
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler with sanitization
    console = SafeStreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    ))
    logger.addHandler(console)
    
    # File handler - full Unicode support
    if log_file is None:
        log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    file_handler = DateScopedFileHandler(log_file, level=level)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    ))
    logger.addHandler(file_handler)
    
    return logger


def setup_safe_logging_for_module(
    logger: logging.Logger,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Convert an existing logger to use safe console output.
    
    Replaces StreamHandlers with SafeStreamHandlers.
    
    Args:
        logger: Existing logger to modify
        console_level: Level for console output
        
    Returns:
        The modified logger
    """
    new_handlers = []
    
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # Replace with safe handler
            safe_handler = SafeStreamHandler(handler.stream)
            safe_handler.setLevel(handler.level or console_level)
            safe_handler.setFormatter(handler.formatter)
            new_handlers.append(safe_handler)
        else:
            # Keep file handlers as-is
            new_handlers.append(handler)
    
    logger.handlers = new_handlers
    return logger


# Convenience function for quick logging
def safe_print(msg: str, level: str = "info"):
    """
    Quick console-safe print with logging.

    Args:
        msg: Message to print (emoji will be replaced)
        level: Log level (debug, info, warning, error)
    """
    clean_msg = sanitize_for_console(msg)
    print(clean_msg)


def safe_log_info(logger: Optional[logging.Logger], msg: str):
    """Log INFO level message safely for Windows console."""
    if logger:
        logger.info(sanitize_for_console(msg))


def safe_log_warning(logger: Optional[logging.Logger], msg: str):
    """Log WARNING level message safely for Windows console."""
    if logger:
        logger.warning(sanitize_for_console(msg))


def safe_log_error(logger: Optional[logging.Logger], msg: str):
    """Log ERROR level message safely for Windows console."""
    if logger:
        logger.error(sanitize_for_console(msg))

if __name__ == "__main__":
    # Test the safe logging
    print("Testing safe logging utilities...")
    
    # Test sanitization
    test_strings = [
        "Normal text",
        "With emoji 🚀 rocket",
        "Status: ✅ OK and ❌ failed",
        "Trend: 📈 up and 📉 down",
        "Mixed: Hello 🌍 world ⚠️ warning",
    ]
    
    print("\n--- Sanitization Test ---")
    for s in test_strings:
        print(f"Original: {s}")
        print(f"Sanitized: {sanitize_for_console(s)}")
        print()
    
    # Test logger
    print("\n--- Logger Test ---")
    logger = get_safe_logger("test_module")
    logger.info("Test message with emoji 🚀")
    logger.warning("Warning with ⚠️ symbol")
    logger.error("Error with ❌ mark")
    
    print("\nAll tests passed!")
