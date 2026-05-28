import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from autotrade.utils.safe_logging import get_safe_logger
from config.config_loader import get_config

logger = get_safe_logger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
LOG_DIR = ROOT_DIR / 'logs'
PLANS_DIR = ROOT_DIR / 'plans'
REPORTS_DIR = ROOT_DIR / 'reports'
PROMPTS_DIR = ROOT_DIR / 'prompts' / 'llm'

class DailyLessonsAnalyzer:
    """Analyzes daily trading activity and generates a qualitative lessons report."""
    
    def __init__(self):
        self.config = get_config()
        REPORTS_DIR.mkdir(exist_ok=True)
        
    def run(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Run daily lessons analysis and return structured metadata."""
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        result: Dict[str, Any] = {
            "success": False,
            "date": date_str,
            "report_path": None,
            "model": None,
            "mode": "llm",
            "key_decisions_count": 0,
            "top_signals_count": 0,
        }

        logger.info(f"Starting Daily Lessons Analysis for {date_str}...")

        # 1. Load Data
        daily_review = self._load_daily_review(date_str)
        workflow_journal = self._load_workflow_journal(date_str)
        morning_plan = self._load_morning_plan(date_str)

        # Extract key decisions from workflow journal
        key_decisions = []
        for entry in workflow_journal:
            if entry.get('action') in ['TRIM', 'EXIT', 'BUY', 'SELL']:
                key_decisions.append({
                    'symbol': entry.get('symbol'),
                    'action': entry.get('action'),
                    'reasoning': entry.get('reasoning', entry.get('reason', 'No reason provided'))
                })
        result["key_decisions_count"] = len(key_decisions)
        journal_str = json.dumps(key_decisions, indent=2)[:5000]
        
        # Extract top signals from morning plan
        top_signals = []
        if morning_plan:
            signals = morning_plan.get('entry_candidates', morning_plan.get('signals', []))
            for s in signals[:20]:  # Top 20
                top_signals.append({
                    'symbol': s.get('symbol'),
                    'score': s.get('score', s.get('conviction_score')),
                    'setup': s.get('setup', s.get('strategy'))
                })
        result["top_signals_count"] = len(top_signals)

        if not daily_review:
            logger.error(f"No daily review found for {date_str}. Run daily_review.py first.")
            return self._finalize_fallback_result(
                result=result,
                date_str=date_str,
                daily_review={},
                key_decisions=key_decisions,
                top_signals=top_signals,
                llm_error="daily_review_missing",
            )

        # 2. Prepare Prompt
        prompt_path = PROMPTS_DIR / 'daily_lessons.txt'
        if not prompt_path.exists():
            logger.error(f"Prompt template not found at {prompt_path}")
            return self._finalize_fallback_result(
                result=result,
                date_str=date_str,
                daily_review=daily_review,
                key_decisions=key_decisions,
                top_signals=top_signals,
                llm_error="prompt_template_missing",
            )

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        # Format data for prompt (truncate if too large)
        review_str = json.dumps(daily_review, indent=2)[:5000]
        plan_str = json.dumps(top_signals, indent=2)[:3000]
        
        prompt = prompt_template.format(
            date=date_str,
            daily_review=review_str,
            workflow_journal=journal_str,
            morning_plan=plan_str
        )
        
        # 3. Generate Report using LLM
        logger.info("Generating lessons report with LLM...")
        model = self.config.llm.model_decision  # Use the decision model (e.g., qwen2.5-coder:14b)
        result["model"] = model
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 8192
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            report_content = response.json().get("response", "")
            
            # 4. Save Report
            report_path = self._write_report(date_str, report_content)
            legacy_path = self._write_legacy_lesson_export(
                date_str=date_str,
                report_path=report_path,
                content=report_content,
                mode="llm",
                key_decisions_count=result["key_decisions_count"],
                top_signals_count=result["top_signals_count"],
                model=model,
            )
                
            logger.info(f"Successfully generated daily lessons report: {report_path}")
            result["success"] = True
            result["report_path"] = str(report_path)
            result["legacy_report_path"] = str(legacy_path)
            result["report_excerpt"] = report_content[:1200]
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate lessons report: {e}")
            return self._finalize_fallback_result(
                result=result,
                date_str=date_str,
                daily_review=daily_review,
                key_decisions=key_decisions,
                top_signals=top_signals,
                llm_error=str(e),
            )

    def _finalize_fallback_result(
        self,
        *,
        result: Dict[str, Any],
        date_str: str,
        daily_review: Optional[Dict[str, Any]],
        key_decisions: List[Dict[str, Any]],
        top_signals: List[Dict[str, Any]],
        llm_error: str,
    ) -> Dict[str, Any]:
        fallback_content = self._build_fallback_report(
            date_str=date_str,
            daily_review=daily_review or {},
            key_decisions=key_decisions,
            top_signals=top_signals,
            llm_error=llm_error,
        )
        report_path = self._write_report(date_str, fallback_content)
        legacy_path = self._write_legacy_lesson_export(
            date_str=date_str,
            report_path=report_path,
            content=fallback_content,
            mode="fallback",
            key_decisions_count=result["key_decisions_count"],
            top_signals_count=result["top_signals_count"],
            model=result.get("model"),
            llm_error=llm_error,
        )
        result["success"] = True
        result["mode"] = "fallback"
        result["report_path"] = str(report_path)
        result["legacy_report_path"] = str(legacy_path)
        result["report_excerpt"] = fallback_content[:1200]
        result["error"] = llm_error
        return result

    def _write_report(self, date_str: str, content: str) -> Path:
        report_path = REPORTS_DIR / f'daily_lessons_{date_str}.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return report_path

    def _write_legacy_lesson_export(
        self,
        *,
        date_str: str,
        report_path: Path,
        content: str,
        mode: str,
        key_decisions_count: int,
        top_signals_count: int,
        model: Optional[str],
        llm_error: Optional[str] = None,
    ) -> Path:
        """
        Keep the legacy post-market lesson JSON family populated as a
        compatibility export for tooling that has not migrated yet.
        """
        legacy_path = REPORTS_DIR / f"post_market_lesson_{date_str}.json"
        payload: Dict[str, Any] = {
            "date": date_str,
            "artifact_type": "daily_lessons_compatibility_export",
            "source_report_path": str(report_path),
            "generated_at": datetime.now().isoformat(),
            "mode": mode,
            "model": model,
            "metrics": {
                "key_decisions_count": int(key_decisions_count or 0),
                "top_signals_count": int(top_signals_count or 0),
            },
            "adherence": {
                "status": "migrated_to_daily_lessons_markdown",
            },
            "lesson": content,
        }
        if llm_error:
            payload["llm_error"] = llm_error

        with open(legacy_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        return legacy_path

    def _build_fallback_report(
        self,
        *,
        date_str: str,
        daily_review: Dict[str, Any],
        key_decisions: List[Dict[str, Any]],
        top_signals: List[Dict[str, Any]],
        llm_error: str,
    ) -> str:
        summary = daily_review.get('summary') or daily_review.get('executive_summary') or ""
        grade = daily_review.get('grade') or daily_review.get('overall_grade') or "N/A"
        pnl = (
            daily_review.get('net_pnl')
            or daily_review.get('actual_net_pnl')
            or daily_review.get('pnl')
            or "unknown"
        )
        critical_issues = daily_review.get('critical_issues') or []
        lessons: List[str] = [
            f"# Daily Lessons - {date_str}",
            "",
            "> Fallback report generated because the LLM lessons pass was unavailable.",
            f"> LLM error: {llm_error}",
            "",
            "## Session Summary",
            f"- Grade: {grade}",
            f"- Net PnL: {pnl}",
        ]
        if summary:
            lessons.append(f"- Review summary: {str(summary).strip()[:500]}")
        lessons.extend(["", "## Critical Issues"])
        if critical_issues:
            for issue in critical_issues[:8]:
                lessons.append(f"- {str(issue).strip()[:300]}")
        else:
            lessons.append("- No structured critical issues were present in the daily review payload.")

        lessons.extend(["", "## Key Decisions"])
        if key_decisions:
            for item in key_decisions[:10]:
                lessons.append(
                    f"- {item.get('symbol')}: {item.get('action')} | {str(item.get('reasoning') or 'No reason provided').strip()[:240]}"
                )
        else:
            lessons.append("- No journal decisions matched BUY/SELL/TRIM/EXIT actions.")

        lessons.extend(["", "## Top Signals Reviewed"])
        if top_signals:
            for signal in top_signals[:10]:
                lessons.append(
                    f"- {signal.get('symbol')}: score={signal.get('score')} | setup={signal.get('setup') or 'unknown'}"
                )
        else:
            lessons.append("- Morning game plan signals were unavailable.")

        lessons.extend(
            [
                "",
                "## Immediate Takeaways",
                "- Preserve this report as a valid learning artifact for next-session review.",
                "- Re-run the richer LLM lessons analysis later if the local model service is restored.",
                "",
            ]
        )
        return "\n".join(lessons)
            
    def _load_daily_review(self, date_str: str) -> Optional[Dict]:
        path = LOG_DIR / f'daily_review_{date_str}.json'
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
        
    def _load_workflow_journal(self, date_str: str) -> List[Dict]:
        path = LOG_DIR / f'workflow_journal_{date_str}.jsonl'
        journal = []
        if path.exists():
            with open(path, 'r') as f:
                for line in f:
                    try:
                        journal.append(json.loads(line))
                    except Exception:
                        pass
        return journal
        
    def _load_morning_plan(self, date_str: str) -> Optional[Dict]:
        date_compact = date_str.replace('-', '')
        path = PLANS_DIR / f'morning_game_plan_{date_compact}.json'
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Daily Lessons Report")
    parser.add_argument("--date", type=str, help="Date in YYYY-MM-DD format (default: today)")
    args = parser.parse_args()
    
    analyzer = DailyLessonsAnalyzer()
    analyzer.run(args.date)
