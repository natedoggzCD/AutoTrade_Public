"""
Strategy pool loaders for Strategy Lab artifacts.

Primary source:
- data/strategy_lab/validated_strategies.json

Optional per-symbol source:
- data/strategy_lab/validated_strategies_by_symbol.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.config_loader import get_config
from autotrade.signals.contracts import StrategyProfile
from autotrade.signals.alpha_catalog_runtime import (
    generate_alpha_catalog_signal,
    is_alpha_catalog_row,
)
from autotrade.utils.strategy_params import sanitize_strategy_params

logger = logging.getLogger(__name__)

VALIDATED_STRATEGIES_PATH = Path("data/strategy_lab/validated_strategies.json")
VALIDATED_STRATEGIES_BY_SYMBOL_PATH = Path(
    "data/strategy_lab/validated_strategies_by_symbol.json"
)
_LEGACY_RUNTIME_FIELD_HINTS = {
    "expression",
    "expr",
    "code",
    "rule",
    "rules",
    "screener_code",
    "entry_expression",
    "exit_expression",
}
_DATAFRAME_ALIAS_PATTERN = re.compile(r"(?<![A-Za-z0-9_.])DataFrame(?![A-Za-z0-9_.])")


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _extract_strategy_definition(strategy_row: Dict[str, Any]) -> Dict[str, Any]:
    defn = strategy_row.get("strategy_definition", {})
    if isinstance(defn, dict) and defn:
        return defn
    return {}


def _extract_strategy_id(strategy_row: Dict[str, Any], defn: Dict[str, Any]) -> str:
    for candidate in (
        strategy_row.get("strategy_id"),
        defn.get("strategy_id"),
        defn.get("name"),
        strategy_row.get("strategy_name"),
        strategy_row.get("name"),
    ):
        if candidate:
            return str(candidate)
    return ""


def _extract_setup_type(strategy_row: Dict[str, Any], defn: Dict[str, Any]) -> str:
    entry = defn.get("entry", {}) if isinstance(defn.get("entry"), dict) else {}
    for candidate in (
        strategy_row.get("setup_type"),
        entry.get("setup_type"),
        defn.get("setup_type"),
    ):
        if candidate:
            return str(candidate)
    return ""


def _extract_strategy_params(
    strategy_row: Dict[str, Any], defn: Dict[str, Any]
) -> Dict[str, Any]:
    existing = strategy_row.get("strategy_params", {})
    params: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    exit_rules = defn.get("exit", {}) if isinstance(defn.get("exit"), dict) else {}
    patch = strategy_row.get("config_patch", {})
    backtest_patch = patch.get("backtest", {}) if isinstance(patch, dict) else {}
    if not isinstance(backtest_patch, dict):
        backtest_patch = {}

    params.setdefault(
        "stop_atr_mult",
        _as_float(
            exit_rules.get("stop_atr_mult", backtest_patch.get("stop_atr")), 2.0
        ),
    )
    params.setdefault(
        "target_atr_mult",
        _as_float(
            exit_rules.get("target_atr_mult", backtest_patch.get("target_atr")), 3.0
        ),
    )
    params.setdefault(
        "trailing_stop", _as_bool(exit_rules.get("trailing_stop"), True)
    )
    params.setdefault(
        "trailing_atr_mult",
        _as_float(exit_rules.get("trailing_atr_mult"), 1.5),
    )
    params.setdefault(
        "max_hold_days",
        _as_int(exit_rules.get("max_hold_days", backtest_patch.get("max_hold_days")), 5),
    )
    params.setdefault(
        "time_stop_if_flat_days",
        _as_int(exit_rules.get("time_stop_if_flat_days"), 3),
    )

    return sanitize_strategy_params(params, fallback_stop=2.0, fallback_target=3.0)


def _extract_backtest_provenance(
    strategy_row: Dict[str, Any], defn: Dict[str, Any]
) -> Tuple[float, float, bool]:
    metrics = strategy_row.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    bt = defn.get("backtest_results", {}) if isinstance(defn.get("backtest_results"), dict) else {}

    win_rate = _as_float(
        strategy_row.get("backtest_win_rate", bt.get("win_rate", metrics.get("win_rate"))),
        0.0,
    )
    profit_factor = _as_float(
        strategy_row.get(
            "backtest_profit_factor",
            bt.get("profit_factor", metrics.get("profit_factor")),
        ),
        0.0,
    )
    walk_forward_validated = _as_bool(
        strategy_row.get(
            "walk_forward_validated",
            bt.get("walk_forward_validated", metrics.get("walk_forward_validated")),
        ),
        False,
    )
    return win_rate, profit_factor, walk_forward_validated


def _normalize_legacy_runtime_fields(
    value: Any,
    *,
    parent_key: str = "",
    touched_fields: Optional[set[str]] = None,
) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            key_str = str(key)
            out[key_str] = _normalize_legacy_runtime_fields(
                child,
                parent_key=key_str,
                touched_fields=touched_fields,
            )
        return out
    if isinstance(value, list):
        return [
            _normalize_legacy_runtime_fields(
                child,
                parent_key=parent_key,
                touched_fields=touched_fields,
            )
            for child in value
        ]
    if isinstance(value, str):
        key = parent_key.strip().lower()
        looks_like_runtime_field = (
            key in _LEGACY_RUNTIME_FIELD_HINTS
            or key.endswith("_expression")
            or key.endswith("_expr")
            or key.endswith("_code")
        )
        if looks_like_runtime_field:
            normalized = _DATAFRAME_ALIAS_PATTERN.sub("pd.DataFrame", value)
            if normalized != value and touched_fields is not None:
                touched_fields.add(parent_key)
            return normalized
    return value


def _with_strategy_provenance(strategy_row: Dict[str, Any]) -> Dict[str, Any]:
    """Attach canonical strategy provenance fields used by signal contracts."""
    if not isinstance(strategy_row, dict):
        return {}

    row = dict(strategy_row)
    touched_fields: set[str] = set()
    row = _normalize_legacy_runtime_fields(row, touched_fields=touched_fields)
    defn = _extract_strategy_definition(row)
    strategy_id = _extract_strategy_id(row, defn)
    setup_type = _extract_setup_type(row, defn)
    strategy_params = _extract_strategy_params(row, defn)
    backtest_win_rate, backtest_profit_factor, walk_forward_validated = (
        _extract_backtest_provenance(row, defn)
    )

    row["strategy_id"] = strategy_id
    if setup_type:
        row["setup_type"] = setup_type
    if strategy_id and not row.get("strategy_name"):
        row["strategy_name"] = strategy_id
    row["strategy_params"] = strategy_params
    row["backtest_win_rate"] = backtest_win_rate
    row["backtest_profit_factor"] = backtest_profit_factor
    row["walk_forward_validated"] = walk_forward_validated
    if touched_fields:
        logger.warning(
            "Normalized legacy bare DataFrame aliases in strategy '%s' fields: %s",
            strategy_id or row.get("strategy_name") or "unknown",
            ", ".join(sorted(touched_fields)),
        )

    return row


def _strategy_profit_factor(strategy_row: Dict[str, Any]) -> float:
    """Best-effort PF resolver across legacy/new export shapes."""
    try:
        metrics = strategy_row.get("metrics", {}) or {}
        pf = metrics.get("profit_factor")
        if pf is None:
            defn = strategy_row.get("strategy_definition", {}) or {}
            bt = defn.get("backtest_results", {}) or {}
            pf = bt.get("profit_factor")
        return float(pf or 0.0)
    except Exception:
        return 0.0


def _strategy_sort_key(strategy_row: Dict[str, Any]) -> Tuple[float, float, float]:
    pf = _strategy_profit_factor(strategy_row)
    final_score = float(strategy_row.get("final_score", 0.0) or 0.0)
    wr = float((strategy_row.get("metrics", {}) or {}).get("win_rate", 0.0) or 0.0)
    return (pf, final_score, wr)


def _symbol_sort_key(strategy_row: Dict[str, Any]) -> Tuple[float, float, float, str, str]:
    symbol_score = float(strategy_row.get("symbol_score", 0.0) or 0.0)
    symbol_metrics = strategy_row.get("symbol_metrics", {}) or {}
    trades = float(symbol_metrics.get("trades", 0.0) or 0.0)
    wr = float(symbol_metrics.get("win_rate", 0.0) or 0.0)
    strategy_name = str(strategy_row.get("strategy_name") or "")
    setup_type = str(strategy_row.get("setup_type") or "")
    return (symbol_score, trades, wr, strategy_name, setup_type)


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _resolve_fallback_to_global(override: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)
    try:
        cfg = get_config().strategy_lab
        return bool(getattr(cfg, "per_symbol_fallback_to_global", True))
    except Exception:
        return True


# Lab strategies with absurd ATR multiples (>5x) reflect the pre-alignment
# strategy lab that ignored live exit gates. They would pass entries that live
# execution will immediately stop out, so block them at load time until the
# lab is realigned with live exit rules.
ABSURD_ATR_MULT_THRESHOLD = 5.0


def _has_absurd_atr_mult(row: Dict[str, Any]) -> bool:
    defn = row.get("strategy_definition") or {}
    exit_rules = defn.get("exit") or {}
    patch = row.get("config_patch") or {}
    backtest_patch = patch.get("backtest") or {}
    candidates = [
        exit_rules.get("stop_atr_mult"),
        exit_rules.get("target_atr_mult"),
        backtest_patch.get("stop_atr"),
        backtest_patch.get("target_atr"),
    ]
    for v in candidates:
        try:
            if v is not None and float(v) > ABSURD_ATR_MULT_THRESHOLD:
                return True
        except (TypeError, ValueError):
            continue
    return False


def load_validated_strategies(
    path: Path = VALIDATED_STRATEGIES_PATH,
) -> List[Dict[str, Any]]:
    """Load WF-validated strategies exported by Strategy Lab factory.

    Returns [] if file is missing or corrupt (graceful fallback).
    Applies config-driven pool controls:
    - minimum PF floor (`strategy_lab.validated_strategy_min_profit_factor`)
    - top-N cap (`strategy_lab.validated_strategy_pool_top_n`)
    """
    if not path.exists():
        logger.info("No validated_strategies.json found; using default screener only")
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            logger.warning("validated_strategies.json is not a list; ignoring")
            return []
    except Exception as e:
        logger.warning("Failed to load validated_strategies.json: %s", e)
        return []

    try:
        cfg = get_config().strategy_lab
        min_pf = float(getattr(cfg, "validated_strategy_min_profit_factor", 1.05))
        top_n = int(getattr(cfg, "validated_strategy_pool_top_n", 5))
    except Exception:
        min_pf = 1.05
        top_n = 5

    absurd = [row for row in raw if _has_absurd_atr_mult(row)]
    if absurd:
        # Demoted from WARNING (H3 2026-05-26): fires per-symbol per-cycle
        # (515x in 30min observed); pure lab-artifact noise once the live
        # clamp does its job.
        logger.debug(
            "Rejected %d/%d validated strategies with stop/target ATR mult > %.1fx "
            "(lab pre-alignment artifacts; live exit gates would stop them out)",
            len(absurd), len(raw), ABSURD_ATR_MULT_THRESHOLD,
        )
    raw_filtered = [row for row in raw if not _has_absurd_atr_mult(row)]

    filtered = [row for row in raw_filtered if _strategy_profit_factor(row) >= min_pf]
    if not filtered:
        # Defensive fallback: if none meet floor, still keep best strategies
        # from the absurd-ATR-filtered pool.
        filtered = list(raw_filtered)

    filtered.sort(key=_strategy_sort_key, reverse=True)
    selected = filtered[: max(1, top_n)]
    selected = [_with_strategy_provenance(row) for row in selected]
    logger.info(
        "Loaded %d validated strategies (selected %d; min_pf=%.2f; top_n=%d)",
        len(raw),
        len(selected),
        min_pf,
        top_n,
    )
    return selected


def load_validated_strategies_by_symbol(
    path: Path = VALIDATED_STRATEGIES_BY_SYMBOL_PATH,
    *,
    fallback_to_global: Optional[bool] = None,
    global_path: Path = VALIDATED_STRATEGIES_PATH,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load Strategy Lab per-symbol strategy rankings.

    Returns:
    - `{SYMBOL: [strategy_rows...]}` when artifact is valid.
    - `{"*": global_rows}` when artifact is missing/corrupt and fallback is enabled.
    - `{}` when unavailable and fallback is disabled.
    """
    use_global_fallback = _resolve_fallback_to_global(fallback_to_global)

    if not path.exists():
        if use_global_fallback:
            global_rows = load_validated_strategies(path=global_path)
            return {"*": global_rows} if global_rows else {}
        logger.info("No validated_strategies_by_symbol.json found")
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to parse validated_strategies_by_symbol.json: %s", e)
        if use_global_fallback:
            global_rows = load_validated_strategies(path=global_path)
            return {"*": global_rows} if global_rows else {}
        return {}

    raw_symbols = raw.get("symbols") if isinstance(raw, dict) else None
    if not isinstance(raw_symbols, dict):
        logger.warning("validated_strategies_by_symbol.json missing symbols map")
        if use_global_fallback:
            global_rows = load_validated_strategies(path=global_path)
            return {"*": global_rows} if global_rows else {}
        return {}

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for symbol, rows in raw_symbols.items():
        symbol_key = _normalize_symbol(symbol)
        if not symbol_key or not isinstance(rows, list):
            continue

        valid_rows: List[Dict[str, Any]] = []
        rejected_absurd = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            strategy_name = str(row.get("strategy_name") or "").strip()
            setup_type = str(row.get("setup_type") or "").strip()
            if not strategy_name or not setup_type:
                continue
            if _has_absurd_atr_mult(row):
                rejected_absurd += 1
                continue
            valid_rows.append(_with_strategy_provenance(dict(row)))
        if rejected_absurd:
            logger.warning(
                "%s: rejected %d strategies with absurd ATR mult (>%.1fx)",
                symbol_key, rejected_absurd, ABSURD_ATR_MULT_THRESHOLD,
            )

        valid_rows.sort(key=_symbol_sort_key, reverse=True)
        if valid_rows:
            normalized[symbol_key] = valid_rows

    if normalized:
        logger.info("Loaded per-symbol strategy map for %d symbols", len(normalized))
        return normalized

    if use_global_fallback:
        global_rows = load_validated_strategies(path=global_path)
        return {"*": global_rows} if global_rows else {}
    return {}


def get_top_strategies_for_symbol(
    symbol: str,
    *,
    symbol_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    symbol_path: Path = VALIDATED_STRATEGIES_BY_SYMBOL_PATH,
    global_path: Path = VALIDATED_STRATEGIES_PATH,
    fallback_to_global: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Return deterministic top strategy rows for a symbol with fallback safety."""
    symbol_key = _normalize_symbol(symbol)
    if not symbol_key:
        return []

    if symbol_map is None:
        symbol_map = load_validated_strategies_by_symbol(
            path=symbol_path,
            fallback_to_global=fallback_to_global,
            global_path=global_path,
        )

    rows = symbol_map.get(symbol_key)
    if rows:
        ordered = list(rows)
        ordered.sort(key=_symbol_sort_key, reverse=True)
        return ordered

    wildcard_rows = symbol_map.get("*")
    if wildcard_rows:
        ordered = list(wildcard_rows)
        ordered.sort(key=_strategy_sort_key, reverse=True)
        return ordered

    if _resolve_fallback_to_global(fallback_to_global):
        return load_validated_strategies(path=global_path)
    return []


def generate_signal_for_strategy_row(
    strategy_row: Dict[str, Any],
    *,
    symbol: str,
    daily_bars: Any,
    alpha_context: Any = None,
    current_regime: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Dispatch a strategy row to its signal generator.

    Alpha-catalog rows (those carrying ``alpha_metadata.alpha_id`` and no
    populated ``strategy_definition``) are routed to the alpha-catalog runtime,
    which executes the registered alpha generator and returns a legacy-shape
    signal dict (or None when the alpha does not fire today).

    Legacy rows return None — they are still consumed via the existing
    screener/regime-router path. This adapter is purely additive: callers can
    invoke it for every row in a per-symbol pool and treat ``None`` as "the
    legacy path owns this row".
    """
    if not isinstance(strategy_row, dict):
        return None
    if not is_alpha_catalog_row(strategy_row):
        return None
    return generate_alpha_catalog_signal(
        strategy_row,
        symbol=symbol,
        daily_bars=daily_bars,
        alpha_ctx=alpha_context,
        current_regime=current_regime,
    )


def select_robust_profiles(min_trades: int = 10, min_win_rate: float = 0.45) -> List[StrategyProfile]:
    """Select the best performing strategies from the validated pool."""
    strategies = load_validated_strategies()
    robust = []
    
    for s in strategies:
        metrics = s.get("metrics", {}) or {}
        trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0)
        
        if trades >= min_trades and win_rate >= min_win_rate:
            defn = s.get("strategy_definition", {})
            profile = StrategyProfile(
                strategy_id=s.get("strategy_name", "unknown"),
                setup_type=s.get("setup_type", "unknown"),
                entry_rules=defn.get("entry", {}),
                exit_rules=defn.get("exit", {}),
                risk_params=defn.get("portfolio", {}),
                validation_stats=metrics
            )
            robust.append(profile)
            
    return robust


def generate_profile_signals(
    profiles: List[StrategyProfile],
    start_date: str,
    end_date: str,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """Generate candidate signals from strategy profiles using hourly bars."""
    try:
        from autotrade.backtesting.hourly_runner import HourlyBacktestRunner
    except Exception as e:
        logger.warning("Hourly backtest runner unavailable for profile signals: %s", e)
        return []

    runner = HourlyBacktestRunner()
    all_signals = []
    
    for profile in profiles:
        logger.info(f"Running profile discovery: {profile.strategy_id}")
        
        # Use runner to find what would have triggered recently
        result = runner.backtest_hourly_setup(
            signal_criteria=profile.entry_rules,
            start_date=start_date,
            end_date=end_date,
            hold_hours=profile.exit_rules.get("max_hold_days", 3) * 7, # Approx hourly hold
            top_n=top_n
        )
        
        # In a real signal factory, we would return the actual tickers that triggered.
        # For now, we'll return the profile metadata and its recent performance.
        if result.total_trades > 0:
            all_signals.append({
                "strategy_id": profile.strategy_id,
                "setup_type": profile.setup_type,
                "performance": result.to_dict(),
                "entry_rules": profile.entry_rules,
                "exit_rules": profile.exit_rules
            })
            
    return all_signals
