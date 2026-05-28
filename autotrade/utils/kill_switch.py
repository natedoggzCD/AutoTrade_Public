"""File-based runtime kill switches for entry paths.

Flag contents are ignored; presence alone activates the switch. Existing
positions are not auto-closed by these helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from autotrade.utils.safe_logging import get_safe_logger

logger = get_safe_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLAGS_DIR = PROJECT_ROOT / "flags"

FLAG_FILES: Dict[str, str] = {
    "disable_shorts": "disable_shorts.flag",
    "disable_longs": "disable_longs.flag",
    "disable_inverse_etf": "disable_inverse_etf.flag",
    "halt_all_entries": "halt_all_entries.flag",
}


def _normalize_flag_name(flag_name: str) -> str:
    text = str(flag_name or "").strip().lower()
    if text.endswith(".flag"):
        text = text[:-5]
    return text


def flag_path(flag_name: str, *, flags_dir: Optional[Path] = None) -> Path:
    """Return the path for a known flag name."""
    key = _normalize_flag_name(flag_name)
    filename = FLAG_FILES.get(key, f"{key}.flag")
    return (flags_dir or FLAGS_DIR) / filename


def is_disabled(flag_name: str, *, flags_dir: Optional[Path] = None) -> bool:
    """Return true when a flag file is present."""
    return flag_path(flag_name, flags_dir=flags_dir).exists()


def active_flags(*, flags_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Return active known flags and their paths."""
    base = flags_dir or FLAGS_DIR
    return {
        name: base / filename
        for name, filename in FLAG_FILES.items()
        if (base / filename).exists()
    }


def entry_scope_disabled(scope: str, *, flags_dir: Optional[Path] = None) -> bool:
    """Return true if new entries for a scope are blocked."""
    scope_key = str(scope or "").lower().strip()
    if is_disabled("halt_all_entries", flags_dir=flags_dir):
        return True
    if scope_key in {"short", "shorts", "sell_short"}:
        return is_disabled("disable_shorts", flags_dir=flags_dir)
    if scope_key in {"long", "longs", "buy"}:
        return is_disabled("disable_longs", flags_dir=flags_dir)
    if scope_key in {"inverse_etf", "inverse", "hedge"}:
        return is_disabled("disable_inverse_etf", flags_dir=flags_dir)
    return False


def log_active_entry_flags(
    *,
    log=logger,
    flags: Optional[Iterable[str]] = None,
    flags_dir: Optional[Path] = None,
) -> None:
    """Emit a warning for every active flag so tail output stays visible."""
    names = list(flags or active_flags(flags_dir=flags_dir).keys())
    for name in names:
        path = flag_path(name, flags_dir=flags_dir)
        mtime = ""
        try:
            mtime = f" mtime={path.stat().st_mtime:.0f}"
        except OSError:
            pass
        log.warning("[KILL SWITCH] %s active at %s%s", name, path, mtime)
