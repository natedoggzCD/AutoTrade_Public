"""
Offline PM batch runner for sequential shadow evaluation.

Usage:
  python -m autotrade.analysis.sequential_shadow_runner --date 2026-02-26
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.config_loader import get_config
from autotrade.analysis.sequential_shadow_schema import (
    append_jsonl,
    event_ids,
    prediction_path_for_day,
    read_jsonl,
    ready_path_for_day,
    report_path_for_day,
)
from autotrade.analysis.sequential_outcome_scorer import (
    build_comparative_report,
    score_event_outcome,
)
from autotrade.reasoning.sequential_engine import (
    SequentialEngineConfig,
    run_sequential_shadow_inference,
)


PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("sequential_shadow_runner")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / f"sequential_shadow_runner_{datetime.now().strftime('%Y-%m-%d')}.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _infer_with_retries(
    row: Dict[str, Any],
    engine_cfg: SequentialEngineConfig,
    retry_attempts: int,
) -> Any:
    attempts = max(0, int(retry_attempts))
    last_error: Exception | None = None
    for attempt in range(attempts + 1):
        try:
            return run_sequential_shadow_inference(row, engine_cfg)
        except Exception as e:
            last_error = e
            if attempt < attempts:
                logger.warning(
                    "Retrying prediction for event_id=%s attempt=%d/%d error=%s",
                    row.get("event_id"),
                    attempt + 1,
                    attempts + 1,
                    e,
                )
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("inference_retry_failed")


def _summarize_outcomes(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {
            "evaluated_events": 0,
            "sequential_more_accurate_rate": 0.0,
            "mean_score_delta": 0.0,
            "by_event_type": {},
        }
    wins = sum(1 for r in rows if bool(r.get("sequential_more_accurate")))
    mean_delta = sum(float(r.get("score_delta", 0.0) or 0.0) for r in rows) / float(total)
    by_event: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        et = str(row.get("event_type", "unknown"))
        slot = by_event.setdefault(et, {"count": 0, "wins": 0})
        slot["count"] += 1
        if bool(row.get("sequential_more_accurate")):
            slot["wins"] += 1
    for et, slot in by_event.items():
        count = max(1, int(slot["count"]))
        slot["rate"] = float(slot["wins"]) / float(count)
    return {
        "evaluated_events": total,
        "sequential_more_accurate_rate": float(wins) / float(total),
        "mean_score_delta": float(mean_delta),
        "by_event_type": by_event,
    }


def run_for_day(day_str: str, max_workers: int = 2, max_events: int = 0, timeout_seconds: int = 10) -> Dict[str, Any]:
    cfg = get_config()
    shadow_cfg = getattr(cfg, "sequential_shadow_eval", None)
    ready_path = ready_path_for_day(day_str)
    pred_path = prediction_path_for_day(day_str)
    rpt_path = report_path_for_day(day_str)

    ready_rows = read_jsonl(ready_path)
    pred_rows_existing = read_jsonl(pred_path)
    seen_event_ids = event_ids(pred_rows_existing)

    todo = [r for r in ready_rows if str(r.get("event_id", "")) not in seen_event_ids]
    if max_events > 0:
        todo = todo[: max_events]
    batch_size = int(getattr(shadow_cfg, "batch_size", 50) or 50)
    batch_size = max(1, batch_size)
    retry_attempts = int(getattr(shadow_cfg, "retry_attempts", 0) or 0)
    retry_attempts = max(0, retry_attempts)
    horizon_minutes = int(getattr(shadow_cfg, "horizon_minutes", 120) or 120)
    horizon_minutes = max(1, horizon_minutes)
    batches: List[List[Dict[str, Any]]] = [
        todo[i : i + batch_size] for i in range(0, len(todo), batch_size)
    ]

    logger.info(
        "Sequential shadow batch start | day=%s ready=%d existing_predictions=%d todo=%d workers=%d batch_size=%d retries=%d",
        day_str,
        len(ready_rows),
        len(pred_rows_existing),
        len(todo),
        max_workers,
        batch_size,
        retry_attempts,
    )

    engine_cfg = SequentialEngineConfig(
        model=str(
            getattr(shadow_cfg, "model", "")
            or getattr(cfg.llm, "model_medium", "deepseek-r1:8b")
        ),
        timeout_seconds=int(timeout_seconds),
        num_ctx=int(getattr(cfg.llm, "num_ctx", 4096) or 4096),
        temperature=0.1,
        max_thoughts=6,
    )

    generated_predictions: List[Dict[str, Any]] = []
    outcomes: List[Dict[str, Any]] = []
    errors = 0

    for batch_idx, batch_rows in enumerate(batches, start=1):
        logger.info(
            "Sequential shadow processing batch %d/%d size=%d",
            batch_idx,
            len(batches),
            len(batch_rows),
        )
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {
                pool.submit(_infer_with_retries, row, engine_cfg, retry_attempts): row
                for row in batch_rows
            }
            for fut in as_completed(futures):
                row = futures[fut]
                try:
                    pred = fut.result()
                    pred_dict = pred.to_dict()
                    append_jsonl(pred_path, pred_dict)
                    generated_predictions.append(pred_dict)

                    outcome = score_event_outcome(
                        event=row,
                        prediction=pred_dict,
                        horizon_minutes=horizon_minutes,
                    )
                    outcomes.append(outcome.to_dict())
                except Exception as e:
                    errors += 1
                    logger.warning("Prediction failed for event_id=%s: %s", row.get("event_id"), e)

    combined_outcomes = outcomes
    summary = _summarize_outcomes(combined_outcomes)
    comparative_report = build_comparative_report(combined_outcomes)
    report = {
        "date": day_str,
        "generated_at": datetime.now().isoformat(),
        "ready_events": len(ready_rows),
        "new_predictions": len(generated_predictions),
        "errors": int(errors),
        "batches_processed": len(batches),
        "retry_attempts": retry_attempts,
        "summary": summary,
        "comparative_report": comparative_report,
    }
    rpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rpt_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Sequential shadow batch complete | day=%s new_predictions=%d errors=%d report=%s",
        day_str,
        len(generated_predictions),
        errors,
        rpt_path.name,
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run offline sequential shadow evaluation batch.")
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Target date (YYYY-MM-DD)")
    p.add_argument("--workers", type=int, default=2, help="Max parallel inference workers")
    p.add_argument("--max-events", type=int, default=0, help="Cap events processed this run (0 = all)")
    p.add_argument("--timeout-seconds", type=int, default=10, help="Per-event inference timeout")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    run_for_day(
        day_str=str(args.date),
        max_workers=int(args.workers),
        max_events=int(args.max_events),
        timeout_seconds=int(args.timeout_seconds),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
