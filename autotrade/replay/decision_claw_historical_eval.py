from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from autotrade.core.decision_claw import DecisionClaw, DecisionClawResult
from autotrade.replay.overnight_signal_historical import load_candidate_pool_for_date
from autotrade.replay.runtime_session_replay import (
    DEFAULT_BENCHMARK_DATE_SETS,
    RuntimeSessionReplay,
)
from config.config_loader import get_config

logger = logging.getLogger("decision_claw_historical_eval")

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_HISTORICAL_EVAL_DATE_SETS: Dict[str, Sequence[str]] = {
    "recent_failure_focus": (
        "2026-03-17",
        "2026-03-18",
        "2026-03-20",
        "2026-03-23",
    ),
    "strict_core": DEFAULT_BENCHMARK_DATE_SETS["strict_core"],
    "this_week_focus": DEFAULT_BENCHMARK_DATE_SETS["this_week_focus"],
}

_NOTE_KEYWORDS = (
    "critical",
    "warning",
    "missed",
    "wave capacity",
    "not in waves",
    "no entries",
    "zero entries",
    "stale",
    "expired",
    "reconciliation",
    "failed",
    "blocking",
    "whipsaw",
    "trim",
    "exit",
    "watchlist",
)
_MARKDOWN_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s*")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _extract_failure_notes(markdown_text: str, *, limit: int = 12) -> List[str]:
    notes: List[str] = []
    for raw_line in str(markdown_text or "").splitlines():
        line = _MARKDOWN_BULLET_RE.sub("", raw_line).strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        lowered = line.lower()
        if any(keyword in lowered for keyword in _NOTE_KEYWORDS):
            notes.append(line)
    return _dedupe_preserve(notes)[:limit]


def _load_review_notes(project_dir: Path, session_date: str) -> List[str]:
    candidates = [
        project_dir / "reports" / "Claude" / f"daily_review_{session_date}.md",
        project_dir / "reports" / "Claude" / f"daily_trades_{session_date}.md",
        project_dir / "reports" / "Codex" / f"daily_review_{session_date}.md",
        project_dir / "reports" / "Codex" / f"daily_trades_{session_date}.md",
    ]
    notes: List[str] = []
    for path in candidates:
        notes.extend(_extract_failure_notes(_read_text(path)))
    return _dedupe_preserve(notes)[:12]


def _load_recorded_entry_symbols(project_dir: Path, session_date: str) -> List[str]:
    path = project_dir / "logs" / f"trade_decisions_{session_date.replace('-', '')}.json"
    if not path.exists():
        return []
    try:
        payload = _load_json(path) or []
    except Exception:
        return []
    symbols: List[str] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if not bool(row.get("actually_executed")):
            continue
        symbol = str(row.get("symbol") or "").upper().strip()
        if symbol:
            symbols.append(symbol)
    return _dedupe_preserve(symbols)


def _top_missed_candidates(eod_payload: Dict[str, Any], *, limit: int = 8) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in (eod_payload.get("missed_watchlist_opportunities") or [])
        if isinstance(row, dict)
        and str(row.get("symbol") or "").strip()
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.get("validated_score", row.get("raw_score", 0.0)) or 0.0),
            float(row.get("ranking_position", 9999) or 9999),
        ),
    )
    result: List[Dict[str, Any]] = []
    for row in ranked[:limit]:
        result.append(
            {
                "symbol": str(row.get("symbol") or "").upper().strip(),
                "score": float(
                    row.get("validated_score", row.get("raw_score", 0.0)) or 0.0
                ),
                "entry_source": str(row.get("entry_source") or ""),
                "reason": str(
                    row.get("blocking_reason")
                    or row.get("blocking_rule")
                    or row.get("status")
                    or ""
                ),
                "ranking_position": int(row.get("ranking_position", 9999) or 9999),
            }
        )
    return result


def _plan_candidate_map(project_dir: Path, session_date: str) -> Dict[str, Dict[str, Any]]:
    try:
        rows = load_candidate_pool_for_date(datetime.fromisoformat(session_date).date())
    except Exception:
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
        if not symbol:
            continue
        result[symbol] = dict(row)
    return result


def _top_weak_positions(eod_payload: Dict[str, Any], *, limit: int = 5) -> List[Dict[str, Any]]:
    rows = [
        dict(row)
        for row in (eod_payload.get("trades") or [])
        if isinstance(row, dict)
        and str(row.get("symbol") or "").strip()
    ]
    ranked = sorted(
        rows,
        key=lambda row: float(row.get("unrealized_plpc", 0.0) or 0.0),
    )
    result: List[Dict[str, Any]] = []
    for row in ranked[:limit]:
        result.append(
            {
                "symbol": str(row.get("symbol") or "").upper().strip(),
                "pnl_pct": round(float(row.get("unrealized_plpc", 0.0) or 0.0) * 100.0, 2),
                "qty": int(float(row.get("qty", 0) or 0)),
                "market_value": round(
                    float(row.get("current_price", 0.0) or 0.0)
                    * float(row.get("qty", 0.0) or 0.0),
                    2,
                ),
                "reason": str(
                    row.get("signal_entry_source")
                    or row.get("plan_score_source")
                    or "weak_hold"
                ),
            }
        )
    return result


@dataclass
class HistoricalEvalScenario:
    session_date: str
    phase_agent: str
    payload: Dict[str, Any]
    legacy_recommendation: Dict[str, Any]
    failure_notes: List[str]
    missed_symbols: List[str]
    weak_symbols: List[str]
    replay_report: Dict[str, Any]


def score_decision_claw_result(
    result: DecisionClawResult,
    *,
    scenario: HistoricalEvalScenario,
) -> Dict[str, Any]:
    promote_actions = {
        "submit_entry",
        "promote_symbol",
        "force_signal_recheck",
        "refresh_watchlist",
    }
    manage_actions = {"trim_position", "exit_position"}
    promoted_symbols = {
        action.symbol
        for action in result.actions
        if action.action_type in promote_actions and action.symbol
    }
    managed_symbols = {
        action.symbol
        for action in result.actions
        if action.action_type in manage_actions and action.symbol
    }
    missed_set = {str(sym).upper() for sym in scenario.missed_symbols if str(sym).strip()}
    weak_set = {str(sym).upper() for sym in scenario.weak_symbols if str(sym).strip()}
    missed_hits = sorted(promoted_symbols & missed_set)
    weak_hits = sorted(managed_symbols & weak_set)
    action_count = len(result.actions)
    was_active = action_count > 0 and str(result.decision or "").lower() != "hold"
    missed_hit_rate = (
        round(len(missed_hits) / len(missed_set), 4) if missed_set else 0.0
    )
    weak_hit_rate = round(len(weak_hits) / len(weak_set), 4) if weak_set else 0.0
    activity_score = 1.0 if was_active else 0.0
    total_score = round(
        (missed_hit_rate * 0.55) + (weak_hit_rate * 0.30) + (activity_score * 0.15),
        4,
    )
    return {
        "missed_hits": missed_hits,
        "weak_hits": weak_hits,
        "missed_hit_rate": missed_hit_rate,
        "weak_hit_rate": weak_hit_rate,
        "activity_score": activity_score,
        "total_score": total_score,
        "action_count": action_count,
        "decision": result.decision,
        "confidence": round(float(result.confidence or 0.0), 4),
    }


class DecisionClawHistoricalEvaluator:
    def __init__(
        self,
        *,
        project_dir: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None,
    ) -> None:
        self.project_dir = Path(project_dir) if project_dir else PROJECT_DIR
        default_output = (
            self.project_dir
            / "logs"
            / f"decision_claw_historical_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.output_path = Path(output_path) if output_path else default_output
        self.config = get_config().decision_claw
        self.portfolio_cfg = get_config().portfolio
        self.controller = DecisionClaw(self.config, project_root=self.project_dir)

    def build_scenario(
        self,
        *,
        session_date: str,
        phase_agent: str = "market_state",
        include_replay: bool = True,
    ) -> HistoricalEvalScenario:
        eod_path = self.project_dir / "data" / f"eod_review_{session_date}.json"
        eod_payload = _load_json(eod_path) if eod_path.exists() else {}
        replay_report: Dict[str, Any] = {}
        if include_replay:
            replay_report = RuntimeSessionReplay(
                session_date=session_date,
                project_dir=self.project_dir,
            ).run(persist=False)
        failure_notes = _load_review_notes(self.project_dir, session_date)
        plan_candidates = _plan_candidate_map(self.project_dir, session_date)
        missed_candidates = _top_missed_candidates(eod_payload, limit=8)
        for row in missed_candidates:
            symbol = str(row.get("symbol") or "").upper().strip()
            source_row = plan_candidates.get(symbol) or {}
            for key in (
                "ranking_score",
                "conviction_priority_score",
                "confidence",
                "risk_reward",
                "volume_ratio",
                "support_dist_atr",
                "resistance_dist_atr",
                "setup_type",
                "strategy_name",
                "overnight_actionability_score",
                "overnight_hit_2pct_prob",
                "overnight_positive_close_prob",
                "overnight_bad_close_prob",
            ):
                if key in source_row and row.get(key) is None:
                    row[key] = source_row.get(key)
        weak_positions = _top_weak_positions(eod_payload, limit=5)
        blocked_summary = dict(eod_payload.get("watchlist_causality_summary") or {})
        replay_handoff = dict(replay_report.get("handoff_diagnostics") or {})
        actual_context = dict(replay_report.get("actual_day_context") or {})
        legacy_choice = _load_recorded_entry_symbols(self.project_dir, session_date)[:5]
        if not legacy_choice:
            legacy_choice = [
                str(row.get("symbol") or "").upper()
                for row in missed_candidates[:3]
                if str(row.get("symbol") or "").strip()
            ]

        trades = list(eod_payload.get("trades") or [])
        open_slots = max(
            int(getattr(self.portfolio_cfg, "max_positions", 20) or 20) - len(trades),
            0,
        )
        payload = {
            "phase": "market_hours",
            "wave": None,
            "open_slots": open_slots,
            "free_capital": 0.0,
            "positions_count": len(trades),
            "candidate_count": len(missed_candidates),
            "candidates": missed_candidates,
            "weak_positions": weak_positions,
            "failure_notes": failure_notes,
            "missed_symbols": [row["symbol"] for row in missed_candidates],
            "handoff_recoverable_symbols": list(
                replay_handoff.get("recoverable_candidates_examples", []) or []
            ),
            "replay_divergences": int(replay_report.get("divergences_count", 0) or 0),
            "blocked_candidates": int(blocked_summary.get("blocked", 0) or 0),
            "actual_total_trades": int(actual_context.get("total_trades", 0) or 0),
            "actual_net_pnl": float(actual_context.get("net_pnl", 0.0) or 0.0),
            "legacy_wave": "historical_eval",
        }
        legacy_recommendation = {
            "legacy_choice": legacy_choice,
            "legacy_reason": "recorded_day_entries",
            "failure_notes": failure_notes[:5],
        }
        return HistoricalEvalScenario(
            session_date=session_date,
            phase_agent=phase_agent,
            payload=payload,
            legacy_recommendation=legacy_recommendation,
            failure_notes=failure_notes,
            missed_symbols=[row["symbol"] for row in missed_candidates],
            weak_symbols=[row["symbol"] for row in weak_positions],
            replay_report=replay_report,
        )

    def evaluate_dates(
        self,
        *,
        dates: Sequence[str],
        phase_agent: str = "market_state",
        providers: Optional[List[Dict[str, str]]] = None,
        persist: bool = True,
        include_replay: bool = True,
    ) -> Dict[str, Any]:
        report_rows: List[Dict[str, Any]] = []
        provider_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"score": 0.0, "cost_usd": 0.0, "latency_sec": 0.0, "count": 0.0}
        )
        provider_model_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"score": 0.0, "cost_usd": 0.0, "latency_sec": 0.0, "count": 0.0}
        )
        for session_date in dates:
            try:
                scenario = self.build_scenario(
                    session_date=session_date,
                    phase_agent=phase_agent,
                    include_replay=include_replay,
                )
            except TypeError:
                scenario = self.build_scenario(
                    session_date=session_date,
                    phase_agent=phase_agent,
                )
            raw_rows = self.controller.benchmark_providers(
                phase_agent=phase_agent,
                trigger=f"historical_eval_{session_date}",
                payload=scenario.payload,
                legacy_recommendation=scenario.legacy_recommendation,
                providers=providers,
            )
            provider_rows: List[Dict[str, Any]] = []
            for raw_row in raw_rows:
                result = self.controller.interpret_provider_response(
                    phase_agent=phase_agent,
                    trigger=f"historical_eval_{session_date}",
                    payload=scenario.payload,
                    legacy_recommendation=scenario.legacy_recommendation,
                    response=raw_row,
                )
                score = score_decision_claw_result(result, scenario=scenario)
                provider_name = str(raw_row.get("provider") or result.provider or "unknown")
                model_name = str(raw_row.get("model") or result.model or "unknown")
                provider_model_key = f"{provider_name}:{model_name}"
                provider_totals[provider_name]["score"] += float(score["total_score"])
                provider_totals[provider_name]["cost_usd"] += float(result.cost_usd or 0.0)
                provider_totals[provider_name]["latency_sec"] += float(
                    raw_row.get("latency_sec", 0.0) or 0.0
                )
                provider_totals[provider_name]["count"] += 1.0
                provider_model_totals[provider_model_key]["score"] += float(
                    score["total_score"]
                )
                provider_model_totals[provider_model_key]["cost_usd"] += float(
                    result.cost_usd or 0.0
                )
                provider_model_totals[provider_model_key]["latency_sec"] += float(
                    raw_row.get("latency_sec", 0.0) or 0.0
                )
                provider_model_totals[provider_model_key]["count"] += 1.0
                provider_rows.append(
                    {
                        "provider": provider_name,
                        "model": model_name,
                        "provider_model_key": provider_model_key,
                        "success": bool(raw_row.get("success")),
                        "latency_sec": round(float(raw_row.get("latency_sec", 0.0) or 0.0), 4),
                        "tokens_used": int(raw_row.get("tokens_used") or result.tokens_used or 0),
                        "cost_usd": round(float(result.cost_usd or 0.0), 8),
                        "error": str(raw_row.get("error") or ""),
                        "decision": result.decision,
                        "confidence": round(float(result.confidence or 0.0), 4),
                        "actions": [action.__dict__ for action in result.actions],
                        "score": score,
                    }
                )
            report_rows.append(
                {
                    "date": session_date,
                    "phase_agent": phase_agent,
                    "failure_notes": scenario.failure_notes,
                    "missed_symbols": scenario.missed_symbols,
                    "weak_symbols": scenario.weak_symbols,
                    "legacy_choice": list(scenario.legacy_recommendation.get("legacy_choice", [])),
                    "replay_summary": {
                        "divergences_count": int(
                            scenario.replay_report.get("divergences_count", 0) or 0
                        ),
                        "handoff_status": str(
                            scenario.replay_report.get("handoff_diagnostics", {}).get(
                                "status", ""
                            )
                        ),
                        "recoverable_candidates_examples": list(
                            scenario.replay_report.get("handoff_diagnostics", {}).get(
                                "recoverable_candidates_examples", []
                            )
                            or []
                        ),
                        "actual_net_pnl": float(
                            scenario.replay_report.get("actual_day_context", {}).get(
                                "net_pnl", 0.0
                            )
                            or 0.0
                        ),
                    },
                    "consensus": _build_consensus_summary(provider_rows),
                    "provider_results": provider_rows,
                }
            )

        provider_summary: Dict[str, Dict[str, Any]] = {}
        for provider_name, totals in provider_totals.items():
            count = max(int(totals["count"]), 1)
            provider_summary[provider_name] = {
                "avg_total_score": round(totals["score"] / count, 4),
                "total_cost_usd": round(totals["cost_usd"], 8),
                "avg_latency_sec": round(totals["latency_sec"] / count, 4),
                "dates_evaluated": int(totals["count"]),
            }
        provider_model_summary: Dict[str, Dict[str, Any]] = {}
        for provider_model_key, totals in provider_model_totals.items():
            count = max(int(totals["count"]), 1)
            provider_model_summary[provider_model_key] = {
                "avg_total_score": round(totals["score"] / count, 4),
                "total_cost_usd": round(totals["cost_usd"], 8),
                "avg_latency_sec": round(totals["latency_sec"] / count, 4),
                "dates_evaluated": int(totals["count"]),
            }

        output = {
            "summary": {
                "dates_total": len(report_rows),
                "phase_agent": phase_agent,
                "providers": sorted(provider_summary.keys()),
                "provider_summary": provider_summary,
                "provider_model_summary": provider_model_summary,
            },
            "dates": report_rows,
        }
        if persist:
            _write_json(self.output_path, output)
            latest_path = self.output_path.with_name("decision_claw_historical_eval_latest.json")
            _write_json(latest_path, output)
        return output


def _parse_provider_specs(raw_values: Sequence[str]) -> List[Dict[str, str]]:
    providers: List[Dict[str, str]] = []
    for raw_value in raw_values:
        text = str(raw_value or "").strip()
        if not text:
            continue
        if ":" in text:
            provider, model = text.split(":", 1)
            providers.append({"provider": provider.strip(), "model": model.strip()})
        else:
            cfg = getattr(get_config().decision_claw, "market_state", None)
            default_model = str(getattr(cfg, "model", "gpt-4.1-mini"))
            if text.strip().lower() == "local":
                default_model = "qwen2.5-coder:7b"
            elif text.strip().lower() == "openrouter":
                default_model = str(
                    getattr(cfg, "fallback_model", "anthropic/claude-3.5-sonnet")
                )
            providers.append({"provider": text.strip(), "model": default_model})
    return providers


def _build_consensus_summary(provider_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    promoted_counter: Counter[str] = Counter()
    managed_counter: Counter[str] = Counter()
    active_models = 0
    for row in provider_rows:
        if not isinstance(row, dict) or not bool(row.get("success")):
            continue
        active_models += 1
        seen_promoted: set[str] = set()
        seen_managed: set[str] = set()
        for action in row.get("actions", []) or []:
            if not isinstance(action, dict):
                continue
            symbol = str(action.get("symbol") or "").upper().strip()
            action_type = str(action.get("action_type") or "").strip()
            if not symbol:
                continue
            if action_type in {
                "submit_entry",
                "promote_symbol",
                "force_signal_recheck",
                "refresh_watchlist",
            }:
                seen_promoted.add(symbol)
            elif action_type in {"trim_position", "exit_position"}:
                seen_managed.add(symbol)
        for symbol in seen_promoted:
            promoted_counter[symbol] += 1
        for symbol in seen_managed:
            managed_counter[symbol] += 1
    threshold = max(1, (active_models // 2) + 1)
    return {
        "active_models": active_models,
        "majority_threshold": threshold,
        "promote_majority_symbols": sorted(
            [symbol for symbol, count in promoted_counter.items() if count >= threshold]
        ),
        "manage_majority_symbols": sorted(
            [symbol for symbol, count in managed_counter.items() if count >= threshold]
        ),
    }


def _resolve_dates(date_values: Sequence[str], benchmark_set: str) -> List[str]:
    if date_values:
        return [str(value).strip() for value in date_values if str(value).strip()]
    if benchmark_set in DEFAULT_HISTORICAL_EVAL_DATE_SETS:
        return list(DEFAULT_HISTORICAL_EVAL_DATE_SETS[benchmark_set])
    if benchmark_set in DEFAULT_BENCHMARK_DATE_SETS:
        return list(DEFAULT_BENCHMARK_DATE_SETS[benchmark_set])
    raise ValueError(f"Unknown benchmark set: {benchmark_set}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark DecisionClaw against historical trading days and review notes.",
    )
    parser.add_argument("--dates", nargs="*", default=[])
    parser.add_argument(
        "--benchmark-set",
        default="recent_failure_focus",
        help="Named date set to evaluate when --dates is omitted.",
    )
    parser.add_argument("--phase-agent", default="market_state")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider spec like local:qwen2.5-coder:7b or openai:gpt-4.1-nano",
    )
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Skip RuntimeSessionReplay when building scenarios; benchmark from EOD + plan artifacts only.",
    )
    args = parser.parse_args(argv)

    dates = _resolve_dates(args.dates, args.benchmark_set)
    providers = _parse_provider_specs(args.provider) if args.provider else None
    evaluator = DecisionClawHistoricalEvaluator(
        output_path=Path(args.output) if str(args.output).strip() else None,
    )
    report = evaluator.evaluate_dates(
        dates=dates,
        phase_agent=str(args.phase_agent or "market_state"),
        providers=providers,
        persist=True,
        include_replay=not bool(args.skip_replay),
    )
    print("=" * 100)
    print("DECISION CLAW HISTORICAL EVAL")
    print("=" * 100)
    print(
        f"Dates={report['summary']['dates_total']} | Phase={report['summary']['phase_agent']}"
    )
    for provider, summary in (report["summary"].get("provider_summary") or {}).items():
        print(
            f"{provider:<12} avg_score={summary['avg_total_score']:.4f} "
            f"avg_latency={summary['avg_latency_sec']:.2f}s total_cost=${summary['total_cost_usd']:.6f} "
            f"dates={summary['dates_evaluated']}"
        )
    for row in report.get("dates", []):
        print("-" * 100)
        print(
            f"{row['date']} | missed={len(row['missed_symbols'])} weak={len(row['weak_symbols'])} "
            f"replay_divergences={row['replay_summary']['divergences_count']}"
        )
        for provider_row in row.get("provider_results", []):
            score = provider_row.get("score", {})
            print(
                f"  {provider_row['provider']:<10} decision={provider_row['decision']:<18} "
                f"score={float(score.get('total_score', 0.0)):.4f} "
                f"missed_hits={len(score.get('missed_hits', []))} weak_hits={len(score.get('weak_hits', []))} "
                f"latency={float(provider_row.get('latency_sec', 0.0)):.2f}s "
                f"cost=${float(provider_row.get('cost_usd', 0.0)):.6f}"
            )
    print(f"Saved: {evaluator.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
