from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal


SourceFlag = Literal[
    "overnight_pick",
    "claw_yesterday",
    "strength_reentry",
    "pm_plan",
    "thesis_intact",
]
CatalystFlag = Literal[
    "earnings_within_5d",
    "recent_offering",
    "dilution_risk",
    "news_today",
]
ThesisState = Literal["intact", "weakening", "broken", "none"]


@dataclass(frozen=True)
class SourceFreshness:
    name: str
    path: str | None
    fresh: bool
    message: str
    age_hours: float | None = None
    rejected_count: int = 0
    reject_path: str | None = None


@dataclass(frozen=True)
class TickerFacts:
    symbol: str
    price: float
    gap_pct: float | None
    atr_14: float
    rel_strength_5d: float
    source_flags: frozenset[str] = field(default_factory=frozenset)
    catalyst_flags: frozenset[str] = field(default_factory=frozenset)
    thesis_state: ThesisState = "none"
    key_level: float | None = None
    score: float = 0.0
    recent_loss_count: int = 0
    recommendation: str = ""
    setup_type: str = ""
    stop_loss: float | None = None
    target: float | None = None
    risk_reward: float | None = None
    confidence: float | None = None
    volume_ratio: float | None = None
    rsi_14: float | None = None
    overnight_expected_open_to_high_pct: float | None = None
    overnight_expected_open_to_close_pct: float | None = None


@dataclass(frozen=True)
class TapeContext:
    regime_label: str = "unknown"
    breadth_pct: float | None = None
    source_label: str = "tape unavailable"
    is_live: bool = False
    claw_theme: str = "no DecisionClaw theme available"
    fade_theme: str = "no fade theme available"


@dataclass(frozen=True)
class FactAssemblyResult:
    facts: list[TickerFacts]
    freshness: list[SourceFreshness]
    tape: TapeContext = field(default_factory=TapeContext)


def assemble_facts(
    root: str | Path = ".",
    now: datetime | None = None,
    freshness_hours: int = 24,
) -> FactAssemblyResult:
    repo = Path(root)
    current_time = _coerce_now(now)
    freshness: list[SourceFreshness] = []

    signals_path = _latest_file(repo / "logs", "signals_*.json")
    signals_data, signals_fresh = _read_json_source(
        "overnight_signals", signals_path, current_time, freshness_hours
    )
    freshness.append(signals_fresh)
    if signals_path is None:
        return FactAssemblyResult(facts=[], freshness=freshness)

    claw_path = _latest_file(repo / "plans", "decision_claw_final_watchlist_*.json")
    claw_data, claw_fresh = _read_json_source(
        "decision_claw_yesterday", claw_path, current_time, freshness_hours * 2
    )
    freshness.append(claw_fresh)

    pm_path = _latest_file(repo / "plans", "pm_plan_*.json")
    pm_data, pm_fresh = _read_json_source(
        "pm_plan", pm_path, current_time, freshness_hours * 2
    )
    freshness.append(pm_fresh)

    reentry_path = _latest_file(repo / "reports", "historical_bullish_revisit_*.json")
    reentry_data, reentry_fresh = _read_json_source(
        "strength_reentry", reentry_path, current_time, freshness_hours * 7
    )
    freshness.append(reentry_fresh)

    thesis_path = _locate_thesis_path(repo)
    thesis_rows, thesis_fresh = _read_thesis_source(thesis_path, current_time)
    freshness.append(thesis_fresh)

    financial_db = repo / "data" / "financial.db"
    catalyst_by_symbol, financial_fresh = _read_financial_flags(
        financial_db, current_time.date()
    )
    freshness.append(financial_fresh)

    trade_journal = repo / "logs" / "trade_journal.json"
    loss_counts, journal_fresh = _read_loss_counts(trade_journal, current_time.date())
    freshness.append(journal_fresh)

    intraday_path = repo / "data" / "intraday_analysis.json"
    intraday_data, intraday_fresh = _read_json_source(
        "intraday_tape",
        intraday_path if intraday_path.exists() else None,
        current_time,
        8,
    )
    freshness.append(intraday_fresh)

    claw_symbols = _extract_symbols(claw_data)
    pm_symbols = _extract_symbols(pm_data)
    reentry_symbols = _extract_symbols(reentry_data)
    thesis_by_symbol = {row["symbol"]: row for row in thesis_rows if row.get("symbol")}

    facts: dict[str, TickerFacts] = {}
    rejected: list[dict[str, Any]] = []
    for row in _extract_signal_rows(signals_data):
        symbol = _clean_symbol(row.get("symbol"))
        if not symbol:
            continue
        flags = {"overnight_pick"}
        if symbol in claw_symbols:
            flags.add("claw_yesterday")
        if symbol in pm_symbols:
            flags.add("pm_plan")
        if symbol in reentry_symbols:
            flags.add("strength_reentry")

        thesis_state = _thesis_state(thesis_by_symbol.get(symbol))
        if thesis_state == "intact":
            flags.add("thesis_intact")

        catalysts = set(_catalyst_flags_from_row(row))
        catalysts.update(catalyst_by_symbol.get(symbol, set()))
        price = _first_float(
            row,
            "price",
            "latest_close",
            "close",
            "entry_price",
            "last_price",
        )
        atr_14 = _first_float(row, "atr_14", "atr", "atr_percent")
        key_level = _first_float(row, "key_level", "r1_price", "target", "prior_high")
        parsed = TickerFacts(
            symbol=symbol,
            price=price or 0.0,
            gap_pct=_first_float(row, "gap_pct", "premarket_gap_pct"),
            atr_14=atr_14 or 0.0,
            rel_strength_5d=_first_float(row, "rel_strength_5d", "weekly_return")
            or 0.0,
            source_flags=frozenset(flags),
            catalyst_flags=frozenset(catalysts),
            thesis_state=thesis_state,
            key_level=key_level,
            score=_first_float(
                row,
                "final_score",
                "ranking_score",
                "conviction_priority_score",
                "confidence",
            )
            or 0.0,
            recent_loss_count=loss_counts.get(symbol, 0),
            recommendation=str(row.get("recommendation") or ""),
            setup_type=str(row.get("setup_type") or row.get("strategy_name") or ""),
            stop_loss=_first_float(row, "stop_loss", "stop", "s1_price"),
            target=_first_float(row, "target", "target_price", "r1_price"),
            risk_reward=_first_float(row, "risk_reward"),
            confidence=_first_float(row, "confidence"),
            volume_ratio=_first_float(row, "volume_ratio", "vol_trend_ratio"),
            rsi_14=_first_float(row, "rsi_14", "rsi"),
            overnight_expected_open_to_high_pct=_first_float(
                row, "overnight_expected_open_to_high_pct"
            ),
            overnight_expected_open_to_close_pct=_first_float(
                row, "overnight_expected_open_to_close_pct"
            ),
        )
        malformed, reason = is_malformed(parsed)
        if malformed:
            rejected.append(
                {
                    "symbol": symbol,
                    "reason": reason,
                    "raw_row_snapshot": row,
                }
            )
            continue
        facts[symbol] = parsed

    if rejected:
        reject_path = _write_rejection_log(
            repo, signals_data, current_time.date(), rejected
        )
        freshness.append(
            SourceFreshness(
                "morning_brief_input_gate",
                str(reject_path),
                True,
                f"filtered {len(rejected)} malformed signals",
                rejected_count=len(rejected),
                reject_path=str(reject_path),
            )
        )

    ordered = sorted(facts.values(), key=lambda item: (-item.score, item.symbol))
    return FactAssemblyResult(
        facts=ordered,
        freshness=freshness,
        tape=_build_tape_context(
            signals_data, claw_data, pm_data, intraday_data, current_time.date()
        ),
    )


def is_malformed(facts: TickerFacts) -> tuple[bool, str]:
    e, s, t = facts.price, facts.stop_loss, facts.target
    if not e or e <= 0:
        return True, "no_entry"
    if not s or s <= 0:
        return True, "zero_or_missing_stop"
    if not t or t <= 0:
        return True, "zero_or_missing_target"
    if s >= e:
        return True, "stop_at_or_above_entry"
    if t <= e:
        return True, "target_at_or_below_entry"
    risk = e - s
    reward = t - e
    rr = reward / risk
    if rr < 1.2:
        return True, "rr_below_1.2"
    if t / e > 1.40:
        return True, "target_above_40pct_fantasy"
    if (e - s) / e > 0.25:
        return True, "stop_more_than_25pct"
    return False, ""


def _write_rejection_log(
    repo: Path,
    signals_data: Any,
    fallback_date: date,
    rejected: list[dict[str, Any]],
) -> Path:
    target_date = (
        _parse_date(signals_data.get("date"))
        if isinstance(signals_data, dict)
        else None
    )
    target_date = target_date or fallback_date
    path = repo / "logs" / f"morning_brief_rejects_{target_date.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rejected, indent=2, default=str), encoding="utf-8")
    return path


def _coerce_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now().astimezone()
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now


def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.exists():
        return None
    files = [path for path in directory.glob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: (_date_key(path), path.stat().st_mtime))


def _date_key(path: Path) -> str:
    match = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", path.name)
    if match:
        return "".join(match.groups())
    return "00000000"


def _read_json_source(
    name: str,
    path: Path | None,
    now: datetime,
    freshness_hours: int,
) -> tuple[Any, SourceFreshness]:
    if path is None:
        return None, SourceFreshness(name, None, False, "missing source")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, SourceFreshness(name, str(path), False, f"unreadable: {exc}")
    age = _age_hours(path, now)
    return data, SourceFreshness(
        name=name,
        path=str(path),
        fresh=age <= freshness_hours,
        age_hours=round(age, 2),
        message="fresh" if age <= freshness_hours else f"stale > {freshness_hours}h",
    )


def _read_thesis_source(
    path: Path | None,
    now: datetime,
) -> tuple[list[dict[str, Any]], SourceFreshness]:
    if path is None:
        return [], SourceFreshness("position_thesis", None, False, "missing source")
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = _dicts_from_any(data)
    except (OSError, json.JSONDecodeError) as exc:
        return [], SourceFreshness(
            "position_thesis", str(path), False, f"unreadable: {exc}"
        )
    age = _age_hours(path, now)
    return rows, SourceFreshness(
        "position_thesis",
        str(path),
        age <= 24 * 7,
        "fresh" if age <= 24 * 7 else "stale > 7d",
        round(age, 2),
    )


def _locate_thesis_path(repo: Path) -> Path | None:
    candidates = sorted((repo / "data").glob("position_thesis_cache*"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    archive = repo / "data" / "thesis_archive.jsonl"
    return archive if archive.exists() else None


def _read_financial_flags(
    db_path: Path, target_date: date
) -> tuple[dict[str, set[str]], SourceFreshness]:
    if not db_path.exists():
        return {}, SourceFreshness("financial_db", None, False, "missing source")
    flags: dict[str, set[str]] = {}
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in conn.execute("select name from sqlite_master where type='table'")
        }
        if "earnings_calendar" in tables:
            for row in conn.execute("select * from earnings_calendar"):
                symbol = _clean_symbol(
                    row["symbol"] if "symbol" in row.keys() else None
                )
                event_date = _first_date(row)
                if symbol and event_date and abs((event_date - target_date).days) <= 5:
                    flags.setdefault(symbol, set()).add("earnings_within_5d")
        if "financial_events" in tables:
            for row in conn.execute("select * from financial_events"):
                symbol = _clean_symbol(
                    row["symbol"] if "symbol" in row.keys() else None
                )
                text = " ".join(str(row[key]).lower() for key in row.keys() if row[key])
                if not symbol:
                    continue
                if "offering" in text:
                    flags.setdefault(symbol, set()).add("recent_offering")
                if "dilution" in text or "shelf" in text or "warrant" in text:
                    flags.setdefault(symbol, set()).add("dilution_risk")
        conn.close()
    except sqlite3.Error as exc:
        return {}, SourceFreshness(
            "financial_db", str(db_path), False, f"unreadable: {exc}"
        )
    return flags, SourceFreshness("financial_db", str(db_path), True, "fresh")


def _read_loss_counts(
    path: Path, target_date: date
) -> tuple[dict[str, int], SourceFreshness]:
    if not path.exists():
        return {}, SourceFreshness("trade_journal", None, False, "missing source")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, SourceFreshness(
            "trade_journal", str(path), False, f"unreadable: {exc}"
        )
    counts: dict[str, int] = {}
    cutoff = target_date - timedelta(days=7)
    for row in _dicts_from_any(data):
        symbol = _clean_symbol(row.get("symbol"))
        trade_date = _parse_date(
            row.get("date") or row.get("timestamp") or row.get("exit_time")
        )
        pnl = _first_float(row, "pnl", "realized_pnl", "profit_loss", "net_pnl")
        if (
            symbol
            and trade_date
            and cutoff <= trade_date <= target_date
            and pnl is not None
            and pnl < 0
        ):
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts, SourceFreshness("trade_journal", str(path), True, "fresh")


def _extract_signal_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in (
            "signals",
            "buy_signals",
            "actionable_top50",
            "candidates",
            "watchlist",
            "final_watchlist",
        ):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    return (
        [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []
    )


def _extract_symbols(data: Any) -> set[str]:
    symbols: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            if "symbol" in key.lower() and isinstance(value, list):
                symbols.update(
                    _clean_symbol(item) for item in value if _clean_symbol(item)
                )
            elif isinstance(value, list):
                for row in value:
                    if isinstance(row, dict):
                        symbol = _clean_symbol(row.get("symbol") or row.get("ticker"))
                        if symbol:
                            symbols.add(symbol)
                    elif isinstance(row, str):
                        symbol = _clean_symbol(row)
                        if symbol:
                            symbols.add(symbol)
    elif isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                symbol = _clean_symbol(row.get("symbol") or row.get("ticker"))
            else:
                symbol = _clean_symbol(row)
            if symbol:
                symbols.add(symbol)
    return symbols


def _dicts_from_any(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, dict):
                rows.append(value)
            elif isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        return rows
    return []


def _clean_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    symbol = value.strip().upper()
    return (
        symbol
        if symbol and symbol.replace(".", "").replace("-", "").isalnum()
        else None
    )


def _first_float(row: Any, *keys: str) -> float | None:
    for key in keys:
        try:
            value = row.get(key)
        except AttributeError:
            continue
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _catalyst_flags_from_row(row: dict[str, Any]) -> set[str]:
    flags: set[str] = set()
    text = json.dumps(row, default=str).lower()
    if row.get("fresh_news") or row.get("has_catalyst") or "news_today" in text:
        flags.add("news_today")
    if "offering" in text:
        flags.add("recent_offering")
    if "dilution" in text or "shelf" in text or "warrant" in text:
        flags.add("dilution_risk")
    if "earnings" in text:
        flags.add("earnings_within_5d")
    return flags


def _thesis_state(row: dict[str, Any] | None) -> ThesisState:
    if not row:
        return "none"
    text = json.dumps(row, default=str).lower()
    if "broken" in text:
        return "broken"
    if "weakening" in text or "weak" in text:
        return "weakening"
    if "intact" in text or row.get("thesis_intact") is True:
        return "intact"
    return "none"


def _first_date(row: sqlite3.Row) -> date | None:
    for key in row.keys():
        if "date" in key.lower() or "time" in key.lower():
            parsed = _parse_date(row[key])
            if parsed:
                return parsed
    return None


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value)[:10]
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _age_hours(path: Path, now: datetime) -> float:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=now.tzinfo)
    return max(0.0, (now - modified).total_seconds() / 3600)


def _build_tape_context(
    signals_data: Any,
    claw_data: Any,
    pm_data: Any,
    intraday_data: Any,
    target_date: date,
) -> TapeContext:
    intraday_context = _same_day_intraday_context(intraday_data, target_date)
    symbols = sorted(_extract_symbols(claw_data))[:5]
    theme = (
        f"DecisionClaw confirms {', '.join(symbols)}"
        if symbols
        else "DecisionClaw has no symbols"
    )
    if intraday_context:
        breadth = _first_float(intraday_context, "breadth_pct", "market_breadth_pct")
        label = (
            intraday_context.get("regime_label")
            or intraday_context.get("regime")
            or "live context"
        )
        return TapeContext(
            _normalize_regime_label(label),
            breadth,
            "live intraday tape",
            True,
            theme,
            str(
                intraday_data.get("tape_inference")
                or "compare live breadth with watchlist strength"
            ),
        )

    regime = "unknown"
    breadth = None
    source = "plan posture, not live tape"
    for data in (pm_data, signals_data, claw_data):
        if not isinstance(data, dict):
            continue
        candidate_regime = (
            data.get("regime")
            or data.get("resolved_regime")
            or data.get("regime_analysis", {}).get("regime")
        )
        if candidate_regime:
            regime = _normalize_regime_label(candidate_regime)
            source = _regime_source_label(candidate_regime)
            breadth = _regime_breadth(candidate_regime) or breadth
        breadth = (
            _first_float(data, "breadth_pct", "market_breadth_pct", "advancing_pct")
            if breadth is None
            else breadth
        )
    return TapeContext(
        str(regime),
        breadth,
        source,
        False,
        theme,
        "live tape missing; do not treat plan regime as current market state",
    )


def _normalize_regime_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("regime", "label", "resolved_regime", "bucket"):
            nested = value.get(key)
            if nested:
                return str(nested)
        return "unknown"
    return str(value)


def _same_day_intraday_context(data: Any, target_date: date) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    timestamp = _parse_date(data.get("timestamp"))
    if timestamp != target_date:
        return None
    context = data.get("market_context")
    return context if isinstance(context, dict) else data


def _regime_breadth(value: Any) -> float | None:
    if isinstance(value, dict):
        return _first_float(
            value,
            "breadth_pct",
            "breadth_pct_positive",
            "market_breadth_pct",
            "advancing_pct",
        )
    return None


def _regime_source_label(value: Any) -> str:
    if isinstance(value, dict):
        method = value.get("method")
        if method:
            return f"plan posture via {method}, not live tape"
    return "plan posture, not live tape"
