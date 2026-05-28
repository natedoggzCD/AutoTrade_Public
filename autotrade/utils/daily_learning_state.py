"""
Structured daily learning state built from reports and feedback artifacts.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from autotrade.utils.daily_lessons_parser import DailyLessonsParser

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"
DEFAULT_STATE_PATH = DATA_DIR / "daily_learning_state.json"
MARKET_CONTEXT_SYMBOLS = {"SPY", "QQQ"}
LEARNING_SYMBOL_EXCLUSIONS = MARKET_CONTEXT_SYMBOLS.union(
    {
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "AMZN",
        "GOOGL",
        "META",
        "DIA",
        "IWM",
        "VXX",
        "UVXY",
        "NEE",
        "SO",
    }
)

REPORT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("daily_lessons", "daily_lessons_*.md"),
    ("historical_bullish_revisit", "historical_bullish_revisit_*.json"),
    ("top_pick_deep_research", "top_pick_deep_research_*.json"),
    ("vl_quality", "vl_quality_*.json"),
    ("vl_signal_benchmark", "vl_signal_benchmark_*.json"),
    ("sequential_shadow_accuracy", "sequential_shadow_accuracy_*.json"),
    ("workflow_report_markdown", "workflow_report_*.md"),
)


def load_learning_state(
    state_path: Path = DEFAULT_STATE_PATH,
) -> Dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load daily learning state from %s: %s", path, exc)
        return {}


def build_learning_state(
    review: Optional[Dict[str, Any]] = None,
    feedback: Optional[Dict[str, Any]] = None,
    lessons_payload: Optional[Dict[str, Any]] = None,
    reports_dir: Path = REPORTS_DIR,
    state_path: Path = DEFAULT_STATE_PATH,
    as_of_date: Optional[str] = None,
) -> Dict[str, Any]:
    reports_dir = Path(reports_dir)
    parser = DailyLessonsParser(reports_dir=reports_dir)
    lessons = parser.parse_recent(n=5) or {}
    feedback = feedback or {}
    lessons_payload = lessons_payload or {}

    symbol_rules = _build_symbol_rules(lessons)
    setup_rules = _build_setup_rules(feedback)
    threshold_rules = _build_threshold_rules(feedback)
    bias_rules = _build_bias_rules(lessons)
    workflow_rules = _build_workflow_rules(
        lessons=lessons,
        feedback=feedback,
        reports_dir=reports_dir,
    )
    source_summary = _build_source_summary(reports_dir)
    artifact_status = _build_learning_artifact_status(
        lessons=lessons,
        feedback=feedback,
        reports_dir=reports_dir,
        expected_date=as_of_date
        or feedback.get("date")
        or lessons.get("days", [{}])[-1].get("date", ""),
    )

    blocked_symbol_ids = {
        rule["symbol"]: rule["rule_id"] for rule in symbol_rules["block_rules"]
    }
    preferred_symbol_ids = {
        rule["symbol"]: rule["rule_id"] for rule in symbol_rules["prefer_rules"]
    }
    blocked_setup_ids = {
        rule["setup_type"]: rule["rule_id"]
        for rule in setup_rules["blocked_setup_rules"]
    }
    blocked_sector_ids = {
        rule["sector"]: rule["rule_id"] for rule in setup_rules["blocked_sector_rules"]
    }

    state: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "as_of_date": as_of_date
        or feedback.get("date")
        or lessons.get("days", [{}])[-1].get("date", ""),
        "learning_artifact_status": artifact_status,
        "source_summary": source_summary,
        "symbol_rules": symbol_rules,
        "setup_rules": setup_rules,
        "workflow_rules": workflow_rules,
        "threshold_rules": threshold_rules,
        "bias_rules": bias_rules,
        "active_rules": {
            "blocked_symbols": sorted(blocked_symbol_ids),
            "preferred_symbols": sorted(preferred_symbol_ids),
            "mixed_symbols": sorted(symbol_rules["mixed_symbols"]),
            "blocked_setup_types": sorted(blocked_setup_ids),
            "preferred_setup_types": sorted(setup_rules["preferred_setup_types"]),
            "blocked_sectors": sorted(blocked_sector_ids),
            "min_candidate_score": threshold_rules["min_candidate_score"],
            "boosted_symbols": [
                rule["symbol"]
                for rule in sorted(
                    bias_rules.get("symbol_boost_rules", []),
                    key=lambda rule: str(rule.get("symbol", "") or ""),
                )
            ],
            "preferred_setup_keywords": [
                rule["keyword"]
                for rule in bias_rules.get("setup_keyword_boost_rules", [])
            ],
        },
        "rule_index": {
            "blocked_symbols": blocked_symbol_ids,
            "preferred_symbols": preferred_symbol_ids,
            "blocked_setup_types": blocked_setup_ids,
            "blocked_sectors": blocked_sector_ids,
        },
        "learning_digest": _build_learning_digest(
            lessons=lessons,
            feedback=feedback,
            review=review or {},
            symbol_rules=symbol_rules,
            setup_rules=setup_rules,
            threshold_rules=threshold_rules,
            bias_rules=bias_rules,
            workflow_rules=workflow_rules,
        ),
    }

    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def apply_learning_gates(
    candidates: List[Any],
    state: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any]]:
    if not candidates or not state:
        return candidates, {"blocked": [], "passed": [], "state_loaded": bool(state)}

    if not _learning_state_usable_for_bias(state):
        status = dict(state.get("learning_artifact_status") or {})
        for candidate in candidates:
            metadata = getattr(candidate, "metadata", {}) or {}
            metadata["learning_gate_status"] = "neutral"
            metadata["learning_gate_reasons"] = list(status.get("warnings") or [])
            metadata["learning_rule_ids"] = []
            metadata["learning_confidence"] = 0.0
            metadata["daily_learning"] = {
                "as_of_date": state.get("as_of_date"),
                "status": status.get("status", "unknown"),
                "usable_for_bias": False,
                "neutral_reason": status.get(
                    "neutral_reason", "learning_state_not_usable_for_bias"
                ),
            }
            candidate.metadata = metadata
        return candidates, {
            "blocked": [],
            "passed": [
                {"symbol": getattr(candidate, "symbol", ""), "reasons": [], "rule_ids": []}
                for candidate in candidates
            ],
            "state_loaded": True,
            "as_of_date": state.get("as_of_date"),
            "learning_neutral": True,
            "artifact_status": status,
        }

    allowed: List[Any] = []
    blocked: List[Dict[str, Any]] = []
    passed: List[Dict[str, Any]] = []

    for candidate in candidates:
        gate = evaluate_candidate_learning_gate(candidate, state)
        metadata = getattr(candidate, "metadata", {}) or {}
        metadata["learning_gate_status"] = "blocked" if gate["blocked"] else "pass"
        metadata["learning_gate_reasons"] = list(gate["reasons"])
        metadata["learning_rule_ids"] = list(gate["rule_ids"])
        metadata["learning_confidence"] = gate["confidence"]
        metadata["daily_learning"] = {
            "as_of_date": state.get("as_of_date"),
            "min_candidate_score": state.get("threshold_rules", {}).get(
                "min_candidate_score", 0.0
            ),
        }
        boost = _learning_score_boost(candidate, state)
        if boost["score_boost"] > 0.0 and not gate["blocked"]:
            original_score = float(getattr(candidate, "final_score", 0.0) or 0.0)
            candidate.final_score = max(0.0, min(100.0, original_score + boost["score_boost"]))
            metadata["daily_learning"]["score_boost"] = boost["score_boost"]
            metadata["daily_learning"]["score_before_boost"] = round(original_score, 4)
            metadata["daily_learning"]["score_after_boost"] = round(
                float(getattr(candidate, "final_score", 0.0) or 0.0), 4
            )
            metadata["daily_learning"]["boost_reasons"] = boost["reasons"]
            metadata["learning_rule_ids"].extend(boost["rule_ids"])
        candidate.metadata = metadata

        summary = {
            "symbol": getattr(candidate, "symbol", ""),
            "reasons": gate["reasons"],
            "rule_ids": gate["rule_ids"],
        }
        if gate["blocked"]:
            blocked.append(summary)
        else:
            allowed.append(candidate)
            passed.append(summary)

    allowed.sort(key=lambda c: float(getattr(c, "final_score", 0.0)), reverse=True)
    return allowed, {
        "blocked": blocked,
        "passed": passed,
        "state_loaded": True,
        "as_of_date": state.get("as_of_date"),
    }


def evaluate_candidate_learning_gate(
    candidate: Any,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    symbol = str(getattr(candidate, "symbol", "") or "").upper()
    metadata = getattr(candidate, "metadata", {}) or {}
    setup_type = str(
        metadata.get("setup_type") or getattr(candidate, "scan_type", "") or ""
    ).strip()
    sector = str(
        metadata.get("sector") or getattr(candidate, "sector", "") or ""
    ).strip()
    final_score = float(getattr(candidate, "final_score", 0.0) or 0.0)

    reasons: List[str] = []
    rule_ids: List[str] = []
    confidence = 0.0

    rule_index = state.get("rule_index", {})
    threshold_rules = state.get("threshold_rules", {})
    min_candidate_score = float(threshold_rules.get("min_candidate_score", 0.0) or 0.0)

    if symbol and symbol not in MARKET_CONTEXT_SYMBOLS:
        symbol_rule = rule_index.get("blocked_symbols", {}).get(symbol)
        if symbol_rule:
            reasons.append(f"blocked repeated loser symbol: {symbol}")
            rule_ids.append(symbol_rule)
            confidence = max(confidence, 0.8)

    if setup_type:
        setup_rule = rule_index.get("blocked_setup_types", {}).get(setup_type)
        if setup_rule:
            reasons.append(f"blocked weak setup type: {setup_type}")
            rule_ids.append(setup_rule)
            confidence = max(confidence, 0.7)

    if sector:
        sector_rule = rule_index.get("blocked_sectors", {}).get(sector)
        if sector_rule:
            reasons.append(f"blocked underperforming sector: {sector}")
            rule_ids.append(sector_rule)
            confidence = max(confidence, 0.65)

    if min_candidate_score > 0 and final_score < min_candidate_score:
        reasons.append(
            f"score {final_score:.2f} below learning floor {min_candidate_score:.2f}"
        )
        rule_ids.append("threshold:min_candidate_score")
        confidence = max(confidence, 0.6)

    return {
        "blocked": bool(reasons),
        "reasons": reasons,
        "rule_ids": rule_ids,
        "confidence": round(confidence, 2),
    }


def _learning_state_usable_for_bias(state: Dict[str, Any]) -> bool:
    status = state.get("learning_artifact_status")
    if not isinstance(status, dict):
        return True
    return bool(status.get("usable_for_bias", False))


def _learning_score_boost(candidate: Any, state: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(getattr(candidate, "symbol", "") or "").upper().strip()
    if not symbol or symbol in LEARNING_SYMBOL_EXCLUSIONS:
        return {"score_boost": 0.0, "rule_ids": [], "reasons": []}

    boost = 0.0
    rule_ids: List[str] = []
    reasons: List[str] = []
    active_rules = state.get("active_rules", {}) or {}
    bias_rules = state.get("bias_rules", {}) or {}

    boost_by_symbol = {
        str(rule.get("symbol") or "").upper().strip(): rule
        for rule in bias_rules.get("symbol_boost_rules", []) or []
        if isinstance(rule, dict)
    }
    if symbol in boost_by_symbol:
        rule = boost_by_symbol[symbol]
        score_boost = float(rule.get("score_boost", 0.0) or 0.0)
        if score_boost > 0.0:
            boost += score_boost
            rule_ids.append(str(rule.get("rule_id") or f"bias:symbol:{symbol}"))
            reasons.append(str(rule.get("reason") or "daily lesson symbol boost"))
    elif symbol in {str(s).upper().strip() for s in active_rules.get("preferred_symbols", []) if s}:
        boost += 2.0
        rule_ids.append(f"symbol:prefer:{symbol}")
        reasons.append("Repeated fresh lessons preferred this symbol")

    metadata = getattr(candidate, "metadata", {}) or {}
    setup_text = " ".join(
        str(item or "")
        for item in (
            metadata.get("setup_type"),
            metadata.get("strategy_name"),
            metadata.get("reason"),
            getattr(candidate, "scan_type", ""),
            getattr(candidate, "catalyst_note", ""),
        )
    ).lower()
    final_score = float(getattr(candidate, "final_score", 0.0) or 0.0)
    for rule in bias_rules.get("setup_keyword_boost_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        keyword = str(rule.get("keyword") or "").lower().strip()
        if not keyword or keyword not in setup_text:
            continue
        min_score = float(rule.get("min_score", 0.0) or 0.0)
        if min_score > 0.0 and final_score < min_score:
            continue
        score_boost = float(rule.get("score_boost", 0.0) or 0.0)
        if score_boost <= 0.0:
            continue
        boost += score_boost
        rule_ids.append(str(rule.get("rule_id") or f"bias:setup_keyword:{keyword}"))
        reasons.append(str(rule.get("reason") or f"daily lesson setup boost: {keyword}"))

    return {
        "score_boost": round(min(boost, 8.0), 4),
        "rule_ids": rule_ids,
        "reasons": reasons,
    }


def _build_symbol_rules(lessons: Dict[str, Any]) -> Dict[str, Any]:
    prefer_counter: Counter[str] = Counter()
    avoid_counter: Counter[str] = Counter()
    prefer_dates: defaultdict[str, List[str]] = defaultdict(list)
    avoid_dates: defaultdict[str, List[str]] = defaultdict(list)

    for day in lessons.get("days", []):
        date = str(day.get("date", "") or "")
        for symbol in {str(s).upper() for s in day.get("prefer_symbols", []) if s}:
            if symbol in LEARNING_SYMBOL_EXCLUSIONS:
                continue
            prefer_counter[symbol] += 1
            if date:
                prefer_dates[symbol].append(date)
        for symbol in {str(s).upper() for s in day.get("avoid_symbols", []) if s}:
            if symbol in LEARNING_SYMBOL_EXCLUSIONS:
                continue
            avoid_counter[symbol] += 1
            if date:
                avoid_dates[symbol].append(date)

    block_rules: List[Dict[str, Any]] = []
    prefer_rules: List[Dict[str, Any]] = []
    mixed_symbols = sorted(
        set(prefer_counter).intersection(avoid_counter) - LEARNING_SYMBOL_EXCLUSIONS
    )

    for symbol, avoid_count in sorted(avoid_counter.items()):
        if avoid_count >= 2 and prefer_counter.get(symbol, 0) == 0:
            block_rules.append(
                {
                    "rule_id": f"symbol:block:{symbol}",
                    "symbol": symbol,
                    "avoid_count": avoid_count,
                    "dates": avoid_dates.get(symbol, []),
                    "reason": "Repeated avoid-only mentions in recent daily lessons",
                }
            )

    for symbol, prefer_count in sorted(prefer_counter.items()):
        if prefer_count >= 2 and avoid_counter.get(symbol, 0) == 0:
            prefer_rules.append(
                {
                    "rule_id": f"symbol:prefer:{symbol}",
                    "symbol": symbol,
                    "prefer_count": prefer_count,
                    "dates": prefer_dates.get(symbol, []),
                    "reason": "Repeated prefer-only mentions in recent daily lessons",
                }
            )

    return {
        "block_rules": block_rules,
        "prefer_rules": prefer_rules,
        "mixed_symbols": mixed_symbols,
    }


def _build_setup_rules(feedback: Dict[str, Any]) -> Dict[str, Any]:
    blocked_setup_rules: List[Dict[str, Any]] = []
    preferred_setup_rules: List[Dict[str, Any]] = []
    blocked_sector_rules: List[Dict[str, Any]] = []

    for family in feedback.get("weak_signal_families", []):
        setup_type = str(family.get("signal_family", "") or "").strip()
        if (
            setup_type
            and float(family.get("score_adjustment", 0.0) or 0.0) <= -1.5
            and float(family.get("confidence", 0.0) or 0.0) >= 0.4
            and int(family.get("sample_size", 0) or 0) >= 3
        ):
            blocked_setup_rules.append(
                {
                    "rule_id": f"setup:block:{setup_type}",
                    "setup_type": setup_type,
                    "reason": family.get("reason", ""),
                    "score_adjustment": float(
                        family.get("score_adjustment", 0.0) or 0.0
                    ),
                    "confidence": float(family.get("confidence", 0.0) or 0.0),
                    "sample_size": int(family.get("sample_size", 0) or 0),
                }
            )

    for family in feedback.get("best_signal_families", []):
        setup_type = str(family.get("signal_family", "") or "").strip()
        if (
            setup_type
            and float(family.get("score_adjustment", 0.0) or 0.0) >= 1.5
            and float(family.get("confidence", 0.0) or 0.0) >= 0.4
            and int(family.get("sample_size", 0) or 0) >= 3
        ):
            preferred_setup_rules.append(
                {
                    "rule_id": f"setup:prefer:{setup_type}",
                    "setup_type": setup_type,
                    "reason": family.get("reason", ""),
                    "score_adjustment": float(
                        family.get("score_adjustment", 0.0) or 0.0
                    ),
                    "confidence": float(family.get("confidence", 0.0) or 0.0),
                    "sample_size": int(family.get("sample_size", 0) or 0),
                }
            )

    for sector_detail in feedback.get("underperforming_sector_details", []):
        sector = str(sector_detail.get("sector", "") or "").strip()
        sample_size = int(sector_detail.get("sample_size", 0) or 0)
        avg_change = float(sector_detail.get("avg_change_pct", 0.0) or 0.0)
        if sector and sample_size >= 2 and avg_change <= -1.5:
            blocked_sector_rules.append(
                {
                    "rule_id": f"sector:block:{sector}",
                    "sector": sector,
                    "avg_change_pct": avg_change,
                    "sample_size": sample_size,
                    "reason": "Recent watchlist names in this sector underperformed",
                }
            )

    return {
        "blocked_setup_rules": blocked_setup_rules,
        "preferred_setup_rules": preferred_setup_rules,
        "blocked_sector_rules": blocked_sector_rules,
        "preferred_setup_types": [rule["setup_type"] for rule in preferred_setup_rules],
    }


def _build_threshold_rules(feedback: Dict[str, Any]) -> Dict[str, Any]:
    buckets = feedback.get("score_bucket_performance", {}) or {}
    min_candidate_score = 0.0

    for floor, label in ((80.0, "65-79"), (65.0, "50-64"), (50.0, "35-49")):
        bucket = buckets.get(label, {})
        count = int(bucket.get("count", 0) or 0)
        positive_rate = float(bucket.get("positive_rate", 0.0) or 0.0)
        avg_change = float(bucket.get("avg_change_pct", 0.0) or 0.0)
        if count >= 2 and positive_rate <= 0.25 and avg_change < 0:
            min_candidate_score = max(min_candidate_score, floor)

    return {"min_candidate_score": min_candidate_score}


def _build_bias_rules(lessons: Dict[str, Any]) -> Dict[str, Any]:
    latest_day = {}
    for day in lessons.get("days", []):
        if isinstance(day, dict) and day.get("date"):
            latest_day = day

    if not latest_day:
        return {"symbol_boost_rules": [], "setup_keyword_boost_rules": []}

    actionable_lessons = [
        str(item or "").strip() for item in latest_day.get("actionable_lessons", [])
    ]
    lesson_text = " ".join(actionable_lessons).lower()

    symbol_boost_rules: List[Dict[str, Any]] = []
    setup_keyword_boost_rules: List[Dict[str, Any]] = []

    breakout_threshold_too_strict = (
        "breakout" in lesson_text
        and ("too strict" in lesson_text or "lower" in lesson_text)
        and ("threshold" in lesson_text or "entry criteria" in lesson_text)
    )
    if breakout_threshold_too_strict:
        setup_keyword_boost_rules.append(
            {
                "rule_id": "bias:setup_keyword:breakout",
                "keyword": "breakout",
                "score_boost": 5.0,
                "min_score": 70.0,
                "source_date": str(latest_day.get("date", "") or ""),
                "reason": "Latest daily lessons said breakout conviction/entry thresholds were too strict",
            }
        )

    for item in latest_day.get("missed_opportunities", []) or []:
        symbol = str(item.get("symbol", "") or "").upper().strip()
        if not symbol or symbol in LEARNING_SYMBOL_EXCLUSIONS:
            continue
        move_pct = item.get("move_pct")
        try:
            move_pct = float(move_pct) if move_pct is not None else None
        except (TypeError, ValueError):
            move_pct = None
        if move_pct is not None and move_pct < 3.0:
            continue
        symbol_boost_rules.append(
            {
                "rule_id": f"bias:symbol:{symbol}",
                "symbol": symbol,
                "score_boost": 4.0 if (move_pct or 0.0) >= 5.0 else 3.0,
                "source_date": str(latest_day.get("date", "") or ""),
                "reason": "Latest daily lessons flagged this as a missed opportunity",
            }
        )

    return {
        "symbol_boost_rules": symbol_boost_rules,
        "setup_keyword_boost_rules": setup_keyword_boost_rules,
    }


def _build_workflow_rules(
    lessons: Dict[str, Any],
    feedback: Dict[str, Any],
    reports_dir: Path,
) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []

    if any(
        "workflow journal" in lesson.lower()
        for day in lessons.get("days", [])
        for lesson in day.get("actionable_lessons", [])
    ):
        rules.append(
            {
                "rule_id": "workflow:journal_reasoning",
                "status": "warning",
                "reason": "Recent lessons explicitly ask for better workflow journal reasoning",
            }
        )

    vl_quality = _load_latest_json(reports_dir, "vl_quality_*.json")
    if vl_quality and not bool(vl_quality.get("vl_reliable", True)):
        rules.append(
            {
                "rule_id": "workflow:vl_unreliable",
                "status": "warning",
                "reason": str(vl_quality.get("quarantine_reason", "vl_unreliable")),
            }
        )

    seq_shadow = _load_latest_json(reports_dir, "sequential_shadow_accuracy_*.json")
    if seq_shadow:
        evaluated_events = int(
            seq_shadow.get("summary", {}).get("evaluated_events", 0) or 0
        )
        if evaluated_events == 0:
            rules.append(
                {
                    "rule_id": "workflow:shadow_feedback_sparse",
                    "status": "info",
                    "reason": "Sequential shadow accuracy has no evaluated events yet",
                }
            )

    if feedback.get("lessons_updated") is False:
        rules.append(
            {
                "rule_id": "workflow:lessons_not_updated",
                "status": "warning",
                "reason": "Trade learner did not emit fresh lessons for the current review",
            }
        )

    return rules


def _build_source_summary(reports_dir: Path) -> Dict[str, Any]:
    families: Dict[str, Dict[str, Any]] = {}
    for family, pattern in REPORT_PATTERNS:
        files = sorted(reports_dir.glob(pattern))
        families[family] = {
            "count": len(files),
            "latest": files[-1].name if files else None,
        }
    return {"reports_dir": str(reports_dir), "families": families}


def _build_learning_artifact_status(
    *,
    lessons: Dict[str, Any],
    feedback: Dict[str, Any],
    reports_dir: Path,
    expected_date: Optional[str],
) -> Dict[str, Any]:
    expected = str(expected_date or "").strip()
    days = [day for day in lessons.get("days", []) if isinstance(day, dict)]
    latest_day = days[-1] if days else {}
    latest_date = str(latest_day.get("date") or "").strip()
    latest_file = str(latest_day.get("file") or "").strip()

    daily_path = Path(reports_dir) / f"daily_lessons_{expected}.md" if expected else None
    legacy_path = (
        Path(reports_dir) / f"post_market_lesson_{expected}.json" if expected else None
    )
    daily_exists = bool(daily_path and daily_path.exists())
    legacy_exists = bool(legacy_path and legacy_path.exists())
    lessons_updated = feedback.get("lessons_updated")

    warnings: List[str] = []
    if not expected:
        warnings.append("learning_status:missing_expected_date")
    if expected and latest_date and latest_date != expected:
        warnings.append("learning_status:stale_daily_lessons")
    if expected and not daily_exists:
        warnings.append("learning_status:daily_lessons_missing")
    if expected and not legacy_exists:
        warnings.append("learning_status:post_market_lesson_missing")
    if lessons_updated is False:
        warnings.append("learning_status:lessons_not_updated")
    if not latest_date:
        warnings.append("learning_status:no_parsed_daily_lessons")

    fresh = bool(
        expected
        and latest_date == expected
        and daily_exists
        and legacy_exists
        and lessons_updated is not False
    )
    actionable_count = sum(
        len(day.get("actionable_lessons", []) or [])
        + len(day.get("missed_opportunities", []) or [])
        + len(day.get("prefer_symbols", []) or [])
        + len(day.get("avoid_symbols", []) or [])
        for day in days
    )
    usable = bool(fresh and actionable_count > 0)
    if fresh and actionable_count == 0:
        warnings.append("learning_status:no_actionable_lesson_content")

    neutral_reason = ""
    if not fresh:
        neutral_reason = "stale_or_missing_learning_artifacts"
    elif not usable:
        neutral_reason = "no_actionable_lesson_content"

    return {
        "status": "fresh" if fresh else "degraded",
        "usable_for_bias": usable,
        "neutral_reason": neutral_reason,
        "expected_date": expected,
        "latest_daily_lessons_date": latest_date,
        "latest_daily_lessons_file": latest_file,
        "daily_lessons_exists": daily_exists,
        "post_market_lesson_exists": legacy_exists,
        "lessons_updated": lessons_updated,
        "actionable_item_count": actionable_count,
        "warnings": warnings,
    }


def _build_learning_digest(
    lessons: Dict[str, Any],
    feedback: Dict[str, Any],
    review: Dict[str, Any],
    symbol_rules: Dict[str, Any],
    setup_rules: Dict[str, Any],
    threshold_rules: Dict[str, Any],
    bias_rules: Dict[str, Any],
    workflow_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "daily_lessons_files": len(lessons.get("days", [])),
        "blocked_symbols": [rule["symbol"] for rule in symbol_rules["block_rules"]],
        "preferred_symbols": [rule["symbol"] for rule in symbol_rules["prefer_rules"]],
        "blocked_setup_types": [
            rule["setup_type"] for rule in setup_rules["blocked_setup_rules"]
        ],
        "blocked_sectors": [
            rule["sector"] for rule in setup_rules["blocked_sector_rules"]
        ],
        "min_candidate_score": threshold_rules.get("min_candidate_score", 0.0),
        "boosted_symbols": [
            rule["symbol"]
            for rule in sorted(
                bias_rules.get("symbol_boost_rules", []),
                key=lambda rule: str(rule.get("symbol", "") or ""),
            )
        ],
        "preferred_setup_keywords": [
            rule["keyword"]
            for rule in bias_rules.get("setup_keyword_boost_rules", [])
        ],
        "workflow_flags": [rule["rule_id"] for rule in workflow_rules],
        "watchlist_count": len(review.get("watchlist_performance", []) or []),
        "missed_opportunities": sum(
            1
            for row in review.get("watchlist_performance", []) or []
            if row.get("missed_opportunity")
        ),
        "best_signal_family_count": len(feedback.get("best_signal_families", [])),
        "weak_signal_family_count": len(feedback.get("weak_signal_families", [])),
    }


def _load_latest_json(reports_dir: Path, pattern: str) -> Dict[str, Any]:
    files = sorted(Path(reports_dir).glob(pattern))
    if not files:
        return {}

    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Failed to load %s: %s", files[-1], exc)
        return {}


class DailyLearningState:
    """
    Class-based interface for daily learning state, expected by DecisionClaw.
    Wraps the functional implementation in this module.
    """

    def __init__(self, project_root: Union[str, Path, None] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.state_path = self.project_root / "data" / "daily_learning_state.json"

    def build_advisory(self) -> Dict[str, Any]:
        """
        Build the advisory rules from the current learning state.
        """
        state = load_learning_state(state_path=self.state_path)
        if not state:
            return {}
        active_rules = dict(state.get("active_rules", {}) or {})
        digest = dict(state.get("learning_digest", {}) or {})
        return {
            "as_of_date": state.get("as_of_date"),
            "artifact_status": dict(state.get("learning_artifact_status", {}) or {}),
            "workflow_flags": list(digest.get("workflow_flags", []) or []),
            **active_rules,
        }
