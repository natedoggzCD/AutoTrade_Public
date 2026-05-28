import logging
from types import SimpleNamespace
from pathlib import Path

from autotrade.core.auto_fix_pipeline import AutoFixPipeline
from autotrade.utils.agentic_log_router import AgenticLogRouter


def test_log_router_passes_file_and_line_hints(monkeypatch):
    captured = {}
    monkeypatch.setenv("AUTOTRADE_AGENTIC_LOG_ROUTER", "1")

    def _fake_route_exception(exc, context="manual_route", file_hint=None, line_hint=None):
        captured["exc"] = exc
        captured["context"] = context
        captured["file_hint"] = file_hint
        captured["line_hint"] = line_hint
        return {"success": False, "reason": "test"}

    monkeypatch.setattr(
        "autotrade.utils.agentic_log_router.route_exception",
        _fake_route_exception,
    )

    handler = AgenticLogRouter(level=logging.ERROR, dedupe_window_sec=1, max_seen=20)
    record = logging.LogRecord(
        name="AutoTrade.MomentumEngine",
        level=logging.ERROR,
        pathname=str(Path("autotrade/signals/momentum_engine.py").resolve()),
        lineno=45,
        msg="Failed to analyze AAPL for momentum: synthetic failure",
        args=(),
        exc_info=None,
    )

    handler.emit(record)

    assert captured["context"] == "log_router"
    normalized = str(captured["file_hint"]).replace("/", "\\")
    assert normalized.endswith("autotrade\\signals\\momentum_engine.py")
    assert captured["line_hint"] == 45


def test_autofix_pipeline_inferrs_momentum_source_file():
    project_root = Path(__file__).resolve().parents[1]
    pipeline = AutoFixPipeline(project_root)
    inferred = pipeline._infer_file_hint_from_error(
        "[LOG:ERROR] AutoTrade.MomentumEngine: Failed to analyze AAPL for momentum: synthetic"
    )
    assert inferred is not None
    assert inferred.replace("/", "\\").endswith("autotrade\\signals\\momentum_engine.py")


def test_autofix_pipeline_inferrs_mcp_client_source_file():
    project_root = Path(__file__).resolve().parents[1]
    pipeline = AutoFixPipeline(project_root)
    inferred = pipeline._infer_file_hint_from_error(
        "[LOG:ERROR] AutoTrade.MCPClient: Server 'alpaca' not defined in opencode.json"
    )
    assert inferred is not None
    assert inferred.replace("/", "\\").endswith("autotrade\\utils\\mcp_client.py")


def test_autofix_pipeline_accepts_already_persisted_codeagent_patch(tmp_path):
    project_root = tmp_path
    target_file = project_root / "autotrade" / "core" / "day_manager.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("patched\n", encoding="utf-8")

    pipeline = AutoFixPipeline(project_root)
    pipeline._apply_fix_spec = lambda spec: (_ for _ in ()).throw(
        AssertionError("_apply_fix_spec should not run for already-persisted patches")
    )

    task_router = SimpleNamespace(
        route=lambda task: SimpleNamespace(
            success=True,
            message='{"summary":"fixed on disk","changes":[]}',
            data={
                "changes_applied": 1,
                "modified_files": [str(target_file)],
                "persisted_to_disk": True,
            },
        )
    )

    err = TypeError("slice indices must be integers or None or have an __index__ method")
    result = pipeline.run(
        task_router,
        err,
        "Traceback...\n",
        context="Day manager cycle",
        file_hint=str(target_file),
        line_hint=18480,
    )

    assert result["success"] is True
    assert result["already_applied"] is True
    assert result["persisted_by"] == "codeagent"
    assert result["modified_files"] == [str(target_file.resolve())]
