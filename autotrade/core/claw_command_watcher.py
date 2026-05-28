import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from autotrade.core.decision_claw import DecisionClawAction, DecisionClawResult
from tools.claw_remote.signal_brief import load_signal_brief

logger = logging.getLogger(__name__)

class CommandWatcher(threading.Thread):
    def __init__(self, service_health: Any, journal: Any, config: Any, recovery_agent: Any = None, project_root: Optional[Path] = None):
        super().__init__(name="CommandWatcher", daemon=True)
        self.service_health = service_health
        self.journal = journal
        self.config = config
        self.recovery_agent = recovery_agent
        self.project_root = project_root or Path.cwd()
        self.command_dir = self.project_root / config.claw_remote.command_dir
        self.pending_dir = self.command_dir / "pending"
        self.processing_dir = self.command_dir / "processing"
        self.completed_dir = self.command_dir / "completed"
        self._stop_event = threading.Event()
        
        # Ensure directories exist
        for d in [self.pending_dir, self.processing_dir, self.completed_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def stop(self):
        self._stop_event.set()

    def run(self):
        logger.info(f"[CLAW-REMOTE] Orchestrator started, polling {self.pending_dir}")
        while not self._stop_event.is_set():
            try:
                self._poll_commands()
            except Exception as e:
                logger.error(f"[CLAW-REMOTE] Error polling commands: {e}", exc_info=True)
            time.sleep(self.config.claw_remote.poll_interval_sec)

    def _poll_commands(self):
        commands = sorted(self.pending_dir.glob("cmd_*.json"))
        for cmd_path in commands:
            self._process_command(cmd_path)

    def _process_command(self, cmd_path: Path):
        processing_path = self.processing_dir / cmd_path.name
        try:
            cmd_path.rename(processing_path)
            with open(processing_path, "r") as f:
                cmd = json.load(f)
            
            # The intent now contains target_agent from the Dispatcher
            target_agent = cmd.get("intent", {}).get("target_agent", "CHAT")
            logger.info(f"[CLAW-REMOTE] Dispatching {cmd.get('id')} to agent: {target_agent}")
            
            result = self._execute_command(cmd)
            
            completed_path = self.completed_dir / cmd_path.name
            with open(completed_path, "w") as f:
                json.dump({
                    "id": cmd.get("id"),
                    "status": "completed",
                    "result": result,
                    "completed_at": datetime.now().isoformat()
                }, f, indent=2)
            
            processing_path.unlink()
        except Exception as e:
            logger.error(f"[CLAW-REMOTE] Error processing {cmd_path.name}: {e}", exc_info=True)
            if processing_path.exists():
                error_path = self.completed_dir / f"error_{cmd_path.name}"
                with open(error_path, "w") as f:
                    json.dump({"error": str(e), "failed_at": datetime.now().isoformat()}, f)
                processing_path.unlink()

    def _execute_command(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        intent_data = cmd.get("intent", {})
        target = intent_data.get("target_agent", "CHAT")
        params = intent_data.get("params", {})
        
        # Route to specialized agents
        if target == "TRADING":
            return self._dispatch_trading(params)
        elif target == "CODING":
            return self._dispatch_coding(params)
        elif target == "DIAGNOSTIC":
            return self._dispatch_diagnostic(params)
        elif target == "SIGNALS":
            return self._dispatch_signals(params)
        else:
            # CHAT is handled by the bridge generating a status response
            return self._handle_status()

    def _dispatch_trading(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Injects a manual trade override for DecisionClaw to pick up."""
        try:
            # DecisionClaw now polls data/manual_overrides.json
            override_path = self.project_root / "data" / "manual_overrides.json"
            action = params.get("action")
            symbol = params.get("symbol", "").upper()
            
            from autotrade.core.decision_claw import ACTION_ALIASES
            action_type = ACTION_ALIASES.get(action, action)
            
            # Load existing if any
            existing_actions = []
            if override_path.exists():
                try:
                    with open(override_path, "r") as f:
                        existing_data = json.load(f)
                        existing_actions = existing_data.get("actions", [])
                except: pass
            
            new_action = {
                "action_type": action_type,
                "symbol": symbol,
                "reason": f"Manual override via Hotline: {params.get('reason', 'user request')}",
                "metadata": params.get("metadata", {}),
                "timestamp": datetime.now().isoformat()
            }
            existing_actions.append(new_action)
            
            with open(override_path, "w") as f:
                json.dump({"actions": existing_actions}, f, indent=2)
            
            self.journal.record("hotline_trading_override", action=action_type, symbol=symbol)
            return {"status": "ok", "message": f"INJECTED: {action_type} for {symbol}. Will be executed in the next trade cycle."}
        except Exception as e:
            return {"status": "error", "message": f"Trading dispatch failed: {e}"}

    def _dispatch_coding(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Invokes the high-fidelity CodeAgent for logical repairs or adjustments."""
        try:
            from autotrade.core.agentic_orchestrator import CodeAgent, Task, TaskType
            agent = CodeAgent()
            
            request = params.get("request")
            file_hint = params.get("file")
            
            logger.info(f"[CLAW-REMOTE] Invoking CodeAgent for request: {request} (file: {file_hint})")
            
            task = Task(
                id=f"hotline_{int(time.time())}",
                type=TaskType.CODE_FIX,
                data={
                    "request": request,
                    "file_hint": file_hint,
                    "source": "hotline"
                }
            )
            
            # This will run the full agentic loop (read, search, edit, validate)
            result = agent.execute(task)
            
            return {
                "status": "ok" if result.status == "resolved" else "error",
                "summary": result.summary,
                "actions": result.actions_taken
            }
        except Exception as e:
            logger.error(f"Coding dispatch failed: {e}", exc_info=True)
            return {"status": "error", "message": f"Coding dispatch failed: {e}"}

    def _dispatch_diagnostic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Uses the RecoveryAgent's tools to perform deep system analysis."""
        try:
            query = params.get("query", "system health check")
            logger.info(f"[CLAW-REMOTE] Performing diagnostic: {query}")
            
            # We can use the RecoveryAgent's existing capability to analyze state
            if not self.recovery_agent:
                return self._handle_status()
                
            # If the user asked a specific "why" question, we can try to get an answer
            # For now, we'll return the elite status which includes logs and state
            return self._handle_status()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _dispatch_signals(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a top-N signal brief for the phone UI."""
        try:
            limit = int(params.get("limit", 10) or 10)
            as_of = params.get("as_of") or params.get("date")
            brief = load_signal_brief(self.project_root, limit=limit, as_of=as_of)
            logger.info(
                "[CLAW-REMOTE] Built signal brief: %s returned %s of %s",
                brief.get("summary", {}).get("date", "unknown"),
                brief.get("summary", {}).get("returned", 0),
                brief.get("summary", {}).get("total_signals", 0),
            )
            return brief
        except Exception as e:
            logger.error(f"[CLAW-REMOTE] Signal brief dispatch failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "result_type": "signal_brief"}

    def _handle_status(self) -> Dict[str, Any]:
        """Gather deep system state for the Gemini bridge to interpret."""
        try:
            state_path = self.project_root / "data" / "workflow_claw_state.json"
            positions_path = self.project_root / "data" / "positions_cache.json"
            theses_path = self.project_root / "data" / "position_theses.json"
            failsafe_path = self.project_root / "plans" / "strategy_failsafe_state.json"
            config_path = self.project_root / "config" / "trading_config.yaml"
            
            status = {
                "timestamp": datetime.now().isoformat(),
                "phase": "UNKNOWN",
                "health": "OK",
                "strategic_context": {}
            }
            
            if hasattr(self.service_health, 'get_overall_status'):
                status["health"] = self.service_health.get_overall_status()
            
            if state_path.exists():
                with open(state_path, "r") as f:
                    status["workflow_state"] = json.load(f)
                    status["phase"] = status["workflow_state"].get("current_phase", "UNKNOWN")
            
            if positions_path.exists():
                with open(positions_path, "r") as f:
                    status["positions_raw"] = json.load(f)
            
            if theses_path.exists():
                with open(theses_path, "r") as f:
                    status["strategic_context"]["theses"] = json.load(f)
            
            if failsafe_path.exists():
                with open(failsafe_path, "r") as f:
                    status["strategic_context"]["failsafe"] = json.load(f)
            
            if config_path.exists():
                import yaml
                with open(config_path, "r") as f:
                    status["strategic_context"]["config_snapshot"] = yaml.safe_load(f)
            
            # RECENT LOGS
            status["recent_logs"] = []
            log_paths = [
                self.project_root / "logs" / "app.jsonl",
                self.project_root / "logs" / "supervisor.log"
            ]
            for lp in log_paths:
                if lp.exists():
                    try:
                        with open(lp, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            status["recent_logs"].append({"file": lp.name, "content": lines[-50:]})
                    except: pass
            
            return {"status": "ok", "data": status}
        except Exception as e:
            return {"status": "error", "message": str(e)}
