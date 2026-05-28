"""
Agentic Advisor - Bridge between Multi-Agent LangGraph Workflow and Day Manager
================================================================================
Drop-in replacement for PositionAdvisor using multi-agent LangGraph workflow.

NO HUMAN-IN-THE-LOOP - advisory decisions only by default.

Interface (required by Day Manager):
    - build_context(position, entry_time) -> Dict
    - get_advice(context) -> Dict with action, reasoning, etc.

Usage:
    from autotrade.core.agentic_advisor import AgenticAdvisor

    advisor = AgenticAdvisor(trading_client=client, dry_run=False)
    context = advisor.build_context(position, entry_time)
    advice = advisor.get_advice(context)
"""

import json
import logging
import os
import threading
import time
from copy import deepcopy
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Structured JSON logger for automated analysis
json_logger = logging.getLogger("app")


class _AdviceCache:
    def __init__(self, ttl_seconds: int = 300):
        self._ttl = max(0, int(ttl_seconds or 0))
        self._lock = threading.RLock()
        self._cache: Dict[str, Any] = {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _key(self, ctx: Dict[str, Any]) -> str:
        symbol = str(ctx.get("symbol", "")).upper()
        pnl_bucket = round(self._safe_float(ctx.get("pnl_pct", 0.0)) * 4) / 4
        score_bucket = round(self._safe_float(ctx.get("score", 0.0)) / 5) * 5
        qty = int(self._safe_float(ctx.get("qty", 0.0)))
        return f"{symbol}:{pnl_bucket}:{score_bucket}:{qty}"

    def get(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._ttl <= 0:
            return None
        with self._lock:
            entry = self._cache.get(self._key(ctx))
            if not entry:
                return None
            ts, advice = entry
            if time.monotonic() - ts > self._ttl:
                return None
            return deepcopy(advice)

    def put(self, ctx: Dict[str, Any], advice: Dict[str, Any]) -> None:
        if self._ttl <= 0:
            return
        with self._lock:
            self._cache[self._key(ctx)] = (time.monotonic(), deepcopy(advice))


class AgenticAdvisor:
    """
    Agentic position advisor using multi-agent LangGraph workflow.

    Key Features:
    - NO human-in-the-loop (advisory-only unless explicitly opted into execution)
    - Uses proper model allocation (8B for loops, 14B for risk, 27B for decision)
    - Strict state schema with audit logging
    - Hard risk rules always enforced (offering=exit, loss>2xATR=exit)

    Workflow Nodes:
    1. fetch_compute (NO LLM) - Get data, compute indicators
    2. risk_gate (NO LLM) - Fast path checks
    3. news_sentiment (8B optional) - Analyze headlines
    4. technical_read (8B) - Interpret setup
    5. candidate_gen (8B) - Generate action candidates
    6. risk_manager (14B + hard rules) - Validate constraints
    7. decision (27B) - Final synthesis (ONCE only)
    8. execute (NO LLM) - Advisory-only by default; DayManager brokers orders
    9. journal (NO LLM) - Log everything
    """

    def __init__(
        self,
        trading_client=None,
        dry_run: bool = True,
        fallback_to_rules: bool = True,
        allow_workflow_execution: bool = False,
    ):
        """
        Initialize the Agentic Advisor.

        Args:
            trading_client: Alpaca TradingClient; only used for direct workflow
                execution when allow_workflow_execution is explicitly enabled.
            dry_run: If True, don't execute trades (just log)
            fallback_to_rules: If True, fall back to rule-based if workflow fails
            allow_workflow_execution: Explicit opt-in for LangGraph order placement.
        """
        self.trading_client = trading_client
        self.dry_run = dry_run
        self.fallback_to_rules = fallback_to_rules
        self.allow_workflow_execution = bool(allow_workflow_execution and not dry_run)
        self._graph = None
        self._initialized = False
        cache_enabled = str(os.environ.get("AUTOTRADE_LLM_CACHE", "1")).lower()
        cache_ttl = int(os.environ.get("AUTOTRADE_LLM_CACHE_TTL_SECONDS", "300"))
        self._advice_cache = (
            _AdviceCache(ttl_seconds=cache_ttl)
            if cache_enabled not in {"0", "false", "off", "no"}
            else None
        )

        logger.info(f"AgenticAdvisor created (dry_run={dry_run})")
        self._log_json(
            {"event": "advisor_created", "advisor_type": "agentic", "dry_run": dry_run}
        )

    def _log_json(self, data: Dict[str, Any]):
        """Log structured JSON for automation."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "module": "agentic_advisor",
                **data,
            }
            json_logger.info(json.dumps(log_entry))
        except Exception:
            pass  # Don't fail on logging errors

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _ensure_initialized(self):
        """Lazy initialize the LangGraph workflow."""
        if self._initialized:
            return

        try:
            from langgraph_workflow.graph import TradingGraph

            self._graph = TradingGraph(
                trading_client=self.trading_client,
                dry_run=(self.dry_run or not self.allow_workflow_execution),
                enable_checkpointing=True,
                allow_live_execute=self.allow_workflow_execution,
            )
            self._initialized = True
            logger.info("AgenticAdvisor: LangGraph workflow initialized")
            self._log_json({"event": "workflow_initialized", "success": True})
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph workflow: {e}")
            self._log_json(
                {"event": "workflow_initialized", "success": False, "error": str(e)}
            )
            self._initialized = True  # Don't retry on every call

    def build_context(self, position, entry_time: datetime = None) -> Dict[str, Any]:
        """
        Build context for a position (compatibility interface).

        Args:
            position: Alpaca position object
            entry_time: When the position was entered

        Returns:
            Context dict for get_advice
        """
        # Calculate hold duration
        hold_minutes = 0
        if entry_time:
            hold_minutes = (datetime.now() - entry_time).total_seconds() / 60

        # Get cost basis
        cost_basis = None
        if hasattr(position, "cost_basis"):
            cost_basis = float(position.cost_basis)
        elif hasattr(position, "avg_entry_price") and hasattr(position, "qty"):
            cost_basis = float(position.avg_entry_price) * int(position.qty)

        return {
            "symbol": position.symbol,
            "entry_price": float(position.avg_entry_price),
            "current_price": float(position.current_price),
            "pnl_pct": float(position.unrealized_plpc) * 100,
            "qty": int(position.qty),
            "market_value": float(position.market_value),
            "cost_basis": cost_basis,
            "entry_time": entry_time,
            "hold_duration_minutes": hold_minutes,
            "_position_obj": position,  # Keep reference for advanced use
        }

    def get_advice(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get advice for a position using the multi-agent LangGraph workflow.

        Note: This method returns AUTO-EXECUTABLE advice on paper account.
        No human approval required.

        Args:
            context: Context dict from build_context

        Returns:
            Dict with:
            - action: 'hold', 'trim', 'exit', 'add'
            - confidence: 0-1
            - reasoning: str
            - stop_price: float (optional)
            - executed: bool (True if order was placed)
            - risk_level: str
            - flags: list of warning flags
        """
        symbol = context["symbol"]
        bypass = self._bypass_llm_advice_if_clear_signal(context)
        if bypass is not None:
            logger.info("[AGENTIC] %s: fast-path LLM bypass", symbol)
            if self._advice_cache:
                self._advice_cache.put(context, bypass)
            return bypass
        cached = self._advice_cache.get(context) if self._advice_cache else None
        if cached is not None:
            logger.info("[AGENTIC] %s: cache hit", symbol)
            cached["cache_hit"] = True
            return cached

        self._ensure_initialized()

        if self._graph is None:
            logger.warning(
                f"[AGENTIC] Workflow not available for {symbol}, using fallback"
            )
            self._log_json(
                {
                    "event": "advisor_fallback",
                    "symbol": symbol,
                    "reason": "workflow_unavailable",
                }
            )
            return self._fallback_advice(context)

        try:
            logger.info(f"[AGENTIC] Running LangGraph workflow for {symbol}...")

            # Build initial state for the workflow
            from langgraph_workflow.state import create_initial_state

            # Create initial state using the factory function
            entry_time = context.get("entry_time")
            _metadata = {}
            if context.get("thesis"):
                _metadata["thesis"] = context["thesis"]
            initial_state = create_initial_state(
                symbol=context["symbol"],
                entry_price=context["entry_price"],
                current_price=context["current_price"],
                qty=context["qty"],
                market_value=context["market_value"],
                unrealized_plpc=context["pnl_pct"] / 100,
                entry_time=entry_time,
                hold_duration_minutes=context.get("hold_duration_minutes", 0),
                metadata=_metadata if _metadata else None,
            )

            # Run the workflow
            result = self._graph.run(initial_state)

            # Extract final action from result
            final_action = result.get("final_action", {})
            risk_gate_result = result.get("risk_gate_result", {}) or {}
            risk_check = result.get("risk_check", {})
            execution = result.get("execution_result", {})
            news_context = result.get("news_context", {})
            technical_read = result.get("technical_read", {})
            completed_nodes = result.get("completed_nodes", []) or []
            risk_gate_skip_agents = bool(risk_gate_result.get("skip_agents", False))
            risk_gate_rules = list(risk_gate_result.get("triggered_rules", []) or [])
            advisor_source = (
                "risk_gate_fast_path"
                if risk_gate_skip_agents
                else "llm_decision"
                if "decision" in completed_nodes
                else "agentic_workflow"
            )

            # Build advice dict
            action = final_action.get("action", "hold")
            if isinstance(action, str):
                action = action.lower()
            else:
                action = str(action).lower() if action else "hold"
            reasoning = str(final_action.get("reasoning", "") or "")
            if self._should_force_hold_on_cooldown_fallback(
                action=action,
                reasoning=reasoning,
                context=context,
            ):
                action = "hold"
                reasoning = f"fallback:cooldown_forced_hold (was: {reasoning.lower()})"

            advice = {
                "action": action,
                "confidence": final_action.get("confidence", 0.5),
                "reasoning": reasoning or "Multi-agent workflow recommendation",
                "stop_price": final_action.get("stop_price"),
                # Execution status
                "executed": execution.get("executed", False),
                "order_id": execution.get("order_id"),
                "execution_error": execution.get("error"),
                # Risk assessment
                "risk_level": risk_check.get("risk_level", "unknown"),
                "flags": risk_check.get("violations", []),
                "loss_to_atr": risk_check.get("loss_to_atr_ratio", 0),
                "hard_exit": risk_check.get("hard_exit_required", False),
                # News context
                "critical_news": None,
                "has_offering": news_context.get("has_offering", False),
                # Technical context
                "trend": technical_read.get("trend", "neutral"),
                "setup_quality": technical_read.get("setup_quality", 50),
                # Metadata
                "advisor_used": True,
                "advisor_type": "agentic",
                "advisor_source": advisor_source,
                "risk_gate_skip_agents": risk_gate_skip_agents,
                "risk_gate_rules": risk_gate_rules,
                "final_action_risk_warnings": final_action.get("risk_warnings", []),
                "workflow_id": result.get("workflow_id"),
                "nodes_completed": completed_nodes,
                "total_duration_ms": sum(
                    a.get("duration_ms", 0) for a in result.get("audit_log", [])
                ),
                "errors": result.get("errors", []),
            }

            # Pass thesis updates back to Day Manager for multi-cycle memory
            advice["thesis_update"] = {
                "bear_case": final_action.get("updated_bear_case", ""),
                "key_level_update": final_action.get("key_level_update"),
            }

            # Check for critical news
            if news_context.get("has_offering"):
                advice["critical_news"] = "OFFERING DETECTED"
            elif news_context.get("has_dilution"):
                advice["critical_news"] = "DILUTION DETECTED"
            elif news_context.get("critical_news"):
                advice["critical_news"] = news_context["critical_news"]

            # Log the advice
            action_str = advice["action"].upper()
            if advice["executed"]:
                action_str += f" [EXECUTED: {advice['order_id']}]"

            logger.info(
                f"[AGENTIC] {symbol}: {action_str} "
                f"(confidence={advice['confidence']:.0%}) "
                f"in {advice['total_duration_ms']:.0f}ms"
            )

            self._log_json(
                {
                    "event": "advisor_advice",
                    "symbol": symbol,
                    "advisor_type": "agentic",
                    "advisor_source": advisor_source,
                    "advisor_used": True,
                    "action": advice["action"],
                    "confidence": advice["confidence"],
                    "reasoning": advice["reasoning"][:100],
                    "executed": advice["executed"],
                    "risk_level": advice["risk_level"],
                    "risk_gate_skip_agents": risk_gate_skip_agents,
                    "risk_gate_rules": risk_gate_rules,
                    "duration_ms": advice["total_duration_ms"],
                }
            )
            if self._advice_cache:
                self._advice_cache.put(context, advice)
            return advice

        except Exception as e:
            logger.error(f"[AGENTIC] Workflow error for {symbol}: {e}")
            self._log_json(
                {
                    "event": "advisor_error",
                    "symbol": symbol,
                    "advisor_type": "agentic",
                    "error": str(e),
                }
            )

            if self.fallback_to_rules:
                return self._fallback_advice(context)
            else:
                return {
                    "action": "hold",
                    "confidence": 0.3,
                    "reasoning": f"Workflow error: {str(e)}",
                    "stop_price": None,
                    "executed": False,
                    "advisor_used": True,
                    "advisor_type": "agentic",
                    "risk_level": "unknown",
                    "flags": ["workflow_error"],
                }

    @staticmethod
    def _should_force_hold_on_cooldown_fallback(
        *,
        action: str,
        reasoning: str,
        context: Dict[str, Any],
    ) -> bool:
        if str(os.environ.get("AUTOTRADE_FORCE_HOLD_ON_COOLDOWN", "1")).lower() in {
            "0",
            "false",
            "off",
            "no",
        }:
            return False
        if action not in {"trim", "exit", "exit_immediately"}:
            return False
        if not str(reasoning or "").lower().startswith("fallback:cooldown_or_budget"):
            return False
        try:
            pnl_pct = float(context.get("pnl_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            pnl_pct = 0.0
        return pnl_pct >= -7.0

    @classmethod
    def _bypass_llm_advice_if_clear_signal(
        cls, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if str(os.environ.get("AUTOTRADE_LLM_BYPASS", "1")).lower() in {
            "0",
            "false",
            "off",
            "no",
        }:
            return None
        score = cls._safe_float(
            context.get("final_score")
            or context.get("realtime_score")
            or context.get("score")
            or context.get("confidence"),
            0.0,
        )
        floor = cls._safe_float(
            os.environ.get("AUTOTRADE_LLM_BYPASS_SCORE_FLOOR", "85"),
            85.0,
        )
        if score < floor:
            return None
        if bool(context.get("validation_gated")):
            return None
        entry_source = (
            str(
                context.get("entry_source")
                or context.get("origin_entry_source")
                or context.get("source")
                or ""
            )
            .strip()
            .lower()
        )
        if entry_source not in {
            "overnight_plan",
            "overnight_plan_full_watchlist",
            "premarket_adjusted",
        }:
            return None
        if bool(context.get("critical_news")):
            return None
        if bool(context.get("failsafe_halt_new_entries")):
            return None
        entry_authority = context.get("entry_authority")
        if isinstance(entry_authority, dict) and not bool(
            entry_authority.get("eligible", True)
        ):
            return None
        if context.get("strict_plan_authority") is False:
            return None
        spread_pct = cls._safe_float(
            context.get("l2_spread_pct") or context.get("spread_pct"),
            0.0,
        )
        if spread_pct > 0.1:
            return None
        return {
            "action": "buy",
            "confidence": 0.85,
            "reasoning": "fast_path:high_score_clear_signal",
            "executed": False,
            "advisor_used": False,
            "advisor_type": "fast_path_bypass",
            "risk_level": "low",
            "flags": ["llm_bypassed"],
            "cache_hit": False,
        }

    def run_for_position(self, position, entry_time: datetime = None) -> Dict[str, Any]:
        """
        Convenience method to run workflow directly on a position.

        Args:
            position: Alpaca position object
            entry_time: When the position was entered

        Returns:
            Advice dict with execution status
        """
        context = self.build_context(position, entry_time)
        return self.get_advice(context)

    def _fallback_advice(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to simple rule-based advice.

        Used when the multi-agent workflow is unavailable.
        Does NOT execute trades - just returns recommendation.
        """
        symbol = context.get("symbol", "UNKNOWN")
        pnl_pct = context.get("pnl_pct", 0)

        logger.warning(f"[AGENTIC] Using fallback rules for {symbol}")
        self._log_json(
            {
                "event": "advisor_fallback",
                "symbol": symbol,
                "advisor_type": "agentic",
                "reason": "workflow_failed",
            }
        )

        # Simple P&L-based rules
        if pnl_pct < -7:
            advice = {
                "action": "exit",
                "confidence": 0.7,
                "reasoning": f"Fallback: Large loss ({pnl_pct:.1f}%) exceeds -7% threshold",
                "stop_price": None,
                "executed": False,
                "advisor_used": True,
                "advisor_type": "agentic_fallback",
                "risk_level": "high",
                "flags": [f"large_loss_{pnl_pct:.0f}pct", "fallback_mode"],
            }
        elif pnl_pct < -4:
            advice = {
                "action": "watch",
                "confidence": 0.6,
                "reasoning": f"Fallback: Moderate loss ({pnl_pct:.1f}%) needs monitoring",
                "stop_price": context["entry_price"] * 0.92,
                "executed": False,
                "advisor_used": True,
                "advisor_type": "agentic_fallback",
                "risk_level": "medium",
                "flags": [f"moderate_loss_{pnl_pct:.0f}pct", "fallback_mode"],
            }
        elif pnl_pct > 10:
            advice = {
                "action": "trim",
                "confidence": 0.6,
                "reasoning": f"Fallback: Good profit ({pnl_pct:.1f}%), consider taking some",
                "stop_price": context["current_price"] * 0.95,
                "executed": False,
                "advisor_used": True,
                "advisor_type": "agentic_fallback",
                "risk_level": "low",
                "flags": ["fallback_mode"],
            }
        else:
            advice = {
                "action": "hold",
                "confidence": 0.5,
                "reasoning": "Fallback: Position within normal range",
                "stop_price": context["entry_price"] * 0.95,
                "executed": False,
                "advisor_used": True,
                "advisor_type": "agentic_fallback",
                "risk_level": "low",
                "flags": ["fallback_mode"],
            }

        self._log_json(
            {
                "event": "advisor_advice",
                "symbol": symbol,
                "advisor_type": "agentic_fallback",
                "advisor_used": True,
                "action": advice["action"],
                "confidence": advice["confidence"],
                "reasoning": advice["reasoning"][:100],
            }
        )

        return advice

    def quick_check_offering(self, symbol: str) -> tuple:
        """
        Quick check for offering news.

        Args:
            symbol: Stock ticker

        Returns:
            Tuple of (has_offering: bool, headline: str)
        """
        try:
            from autotrade.analysis.news_sentiment import NewsSentimentAnalyzer

            analyzer = NewsSentimentAnalyzer(max_news_age_days=7)
            result = analyzer.analyze_ticker(symbol)

            # Check for offering keywords
            headlines = [h.get("headline", "") for h in result.get("headlines", [])]
            offering_kw = [
                "offering",
                "secondary",
                "dilution",
                "atm",
                "shelf",
                "warrant",
            ]

            for h in headlines:
                for kw in offering_kw:
                    if kw in h.lower():
                        return True, h

            return False, ""

        except Exception as e:
            logger.warning(f"Quick offering check failed for {symbol}: {e}")
            return False, ""


# Factory function for easy integration
def create_agentic_advisor(trading_client=None, dry_run: bool = True) -> AgenticAdvisor:
    """
    Create an AgenticAdvisor instance.

    Args:
        trading_client: Alpaca TradingClient for order execution
        dry_run: If True, don't execute trades

    Returns:
        AgenticAdvisor instance
    """
    return AgenticAdvisor(trading_client=trading_client, dry_run=dry_run)


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("AgenticAdvisor module loaded successfully")

    # Quick self-test
    advisor = AgenticAdvisor(dry_run=True)
    print("AgenticAdvisor created (dry_run=True)")
    print("Module ready for Day Manager import")
