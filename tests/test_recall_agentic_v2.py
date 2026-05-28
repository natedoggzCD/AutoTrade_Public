import asyncio
import json
import sqlite3
import time

from tools.mcp import recall_mcp_server as rcs
from tools.mcp import research_mcp_server as rms


def _agentic_db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        """
        CREATE TABLE retrieval_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.0,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            repos_considered INTEGER NOT NULL DEFAULT 0,
            snippets_returned INTEGER NOT NULL DEFAULT 0,
            clarification_count INTEGER NOT NULL DEFAULT 0,
            ingest_proposed INTEGER NOT NULL DEFAULT 0,
            optimizer_used INTEGER NOT NULL DEFAULT 0,
            optimizer_path TEXT DEFAULT '',
            optimizer_model TEXT DEFAULT '',
            optimizer_latency_ms INTEGER NOT NULL DEFAULT 0,
            optimizer_failed_open INTEGER NOT NULL DEFAULT 0,
            optimizer_variant_count INTEGER NOT NULL DEFAULT 0,
            chunk_optimizer_used INTEGER NOT NULL DEFAULT 0,
            chunk_optimizer_model TEXT DEFAULT '',
            chunk_optimizer_latency_ms INTEGER NOT NULL DEFAULT 0,
            chunk_pool_count INTEGER NOT NULL DEFAULT 0,
            chunk_optimizer_input_chunks INTEGER NOT NULL DEFAULT 0,
            chunk_optimizer_selected_chunks INTEGER NOT NULL DEFAULT 0,
            chunk_optimizer_failed_open INTEGER NOT NULL DEFAULT 0,
            progress_heartbeats INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return db


def _row(repo: str, path: str, score: float = 0.45, chunk_id: int | None = None):
    return {
        "chunk_id": chunk_id if chunk_id is not None else hash((repo, path, score)) & 0x7FFFFFFF,
        "source_id": 1,
        "chunk_index": 0,
        "text": "[CODE:python]\ndef alpha():\n    return 1\n",
        "chunk_type": "code",
        "repo": repo,
        "path": path,
        "line_start": 10,
        "line_end": 40,
        "sha256": "abc123",
        "hybrid_score": score,
        "score": score,
        "rerank_score": score,
    }


def _many_rows(n: int, base_score: float = 0.6):
    rows = []
    for i in range(n):
        rows.append(_row("repo/a", f"src/mod_{i}.py", score=base_score - (i * 0.0001), chunk_id=10_000 + i))
    return rows


def _run(coro):
    return asyncio.run(coro)


class _Ctx:
    def __init__(self):
        self.progress = []

    async def report_progress(self, step: int, total: int, message: str):
        self.progress.append((step, total, message))


def test_context_packer_budget_and_ordering():
    rows = [
        _row("r", "tests/test_strategy.py", 0.5),
        _row("r", "src/engine/core.py", 0.4),
        _row("r", "lib/helpers.py", 0.3),
    ]
    pack = rcs._build_context_pack(rows, max_latency_s=15)
    assert 1 <= len(pack["snippets"]) <= 24
    assert all(len(s["code"]) <= 2200 for s in pack["snippets"])
    assert pack["snippets"][0]["path"] == "src/engine/core.py"


def test_reranker_prefers_tighter_function_chunk_over_file_header_chunk():
    header = _row("repo/a", "autotrade/core/workflow_manager.py", 0.88, chunk_id=1)
    header["line_start"] = 1
    header["line_end"] = 240
    header["text"] = (
        "[CODE:python]\n"
        "import json\n"
        "import time\n"
        "from pathlib import Path\n"
    )

    function_body = _row(
        "repo/a", "autotrade/core/workflow_manager.py", 0.81, chunk_id=2
    )
    function_body["line_start"] = 1188
    function_body["line_end"] = 1248
    function_body["text"] = (
        "[CODE:python]\n"
        "def _run_post_market_idle(self) -> Dict[str, Any]:\n"
        "    if pending:\n"
        "        return {'pm_workflow_pending': True}\n"
    )

    ranked = rcs._rerank_code_candidates(
        "pm workflow pending trigger run_post_market_idle workflow_manager",
        [header, function_body],
        {"repo/a|autotrade/core/workflow_manager.py": 0.18},
    )

    assert ranked[0]["chunk_id"] == 2
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_reranker_boosts_exact_file_hint_matches_from_query():
    workflow_manager = _row(
        "repo/a", "autotrade/core/workflow_manager.py", 0.76, chunk_id=3
    )
    autonomous_agent = _row(
        "repo/a", "autotrade/execution/post_market_workflow.py", 0.88, chunk_id=4
    )

    ranked = rcs._rerank_code_candidates(
        "PM_WORKFLOW skip pending workflow_manager _run_post_market_idle",
        [autonomous_agent, workflow_manager],
        {},
    )

    assert ranked[0]["chunk_id"] == 3
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_reranker_demotes_doclike_artifacts_for_code_queries():
    doc_row = _row("repo/a", "CHANGELOG.md", 0.90, chunk_id=5)
    doc_row["chunk_type"] = "prose"
    doc_row["line_start"] = None
    doc_row["line_end"] = None
    doc_row["sha256"] = ""
    doc_row["text"] = "release notes for pm workflow"

    code_row = _row("repo/a", "autotrade/core/workflow_manager.py", 0.72, chunk_id=6)
    code_row["line_start"] = 1188
    code_row["line_end"] = 1248
    code_row["text"] = (
        "[CODE:python]\n"
        "def _run_post_market_idle(self):\n"
        "    return {'pm_workflow_pending': True}\n"
    )

    ranked = rcs._rerank_code_candidates(
        "workflow_manager pm workflow catchup _run_post_market_idle",
        [doc_row, code_row],
        {},
    )

    assert ranked[0]["chunk_id"] == 6


def test_optimizer_parser_rejects_summarization_style_output():
    bad = json.dumps(
        {
            "canonical_query": "In summary, this code defines support and resistance and should be improved.",
            "variants": ["Overall the script can be enhanced by adding indicators."],
            "symbol_terms": [],
            "file_hints": [],
            "likely_intent": "explanation",
        }
    )
    parsed, err = rcs._validate_optimizer_output(bad, "support and resistance")
    assert parsed is None
    assert err.startswith("optimizer_summary_like")


def test_chunk_optimizer_parser_rejects_non_json_and_unknown_id():
    parsed, err = rcs._validate_chunk_optimizer_output("Here are best chunks", {1, 2})
    assert parsed is None
    assert err == "chunk_optimizer_not_json_only"

    parsed2, err2 = rcs._validate_chunk_optimizer_output('{"selected_chunk_ids":[99]}', {1, 2})
    assert parsed2 is None
    assert err2 == "chunk_optimizer_unknown_chunk_id"


def test_optimizer_parser_accepts_wrapped_json_object():
    wrapped = """Here is the optimized JSON:
{"canonical_query":"workflow_manager pm workflow catchup missing plan","variants":["workflow_manager catchup missing pm_plan"],"symbol_terms":["pm_workflow_ran"],"file_hints":["autotrade/core/workflow_manager.py"],"likely_intent":"bug_fix"}
"""
    parsed, err = rcs._validate_optimizer_output(
        wrapped, "pm workflow catchup missing plan"
    )
    assert err == ""
    assert parsed is not None
    assert parsed["canonical_query"] == "workflow_manager pm workflow catchup missing plan"


def test_deterministic_optimizer_fallback_extracts_file_and_symbol_hints():
    fallback = rcs._deterministic_optimizer_fallback(
        "Need the exact implementation path for PM workflow catch-up in workflow_manager.py, including the state flag or artifact checks that trigger it.",
        repo_hint="local::autotrade",
        topic="local_project:autotrade",
        lang="python",
        intent="bug_fix",
    )

    assert "workflow_manager.py" in fallback["file_hints"]
    assert "workflow_manager" in fallback["symbol_terms"]
    assert fallback["likely_intent"] == "bug_fix"


def test_chunk_optimizer_parser_accepts_wrapped_json_object():
    wrapped = """Selected best chunks:
{"selected_chunk_ids":[1,2],"line_focus":[{"chunk_id":1,"line_start":10,"line_end":20}]}
"""
    parsed, err = rcs._validate_chunk_optimizer_output(wrapped, {1, 2})
    assert err == ""
    assert parsed is not None
    assert parsed["selected_chunk_ids"] == [1, 2]
    assert parsed["line_focus"][1] == (10, 20)


def test_fast_rewrite_only_by_default_no_heavy_rewrite(monkeypatch):
    db = _agentic_db()
    calls = []

    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ALLOW_HEAVY_REWRITE", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": [], "flags": {}})
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": [_row("repo/a", "src/main.py", 0.9)], "candidates_scanned": 10, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.92)

    async def _fake_opt(**kwargs):
        calls.append(kwargs["model"])
        return {
            "ok": True,
            "error": "",
            "raw": "",
            "parsed": {
                "canonical_query": "fast query",
                "variants": [],
                "symbol_terms": [],
                "file_hints": [],
                "likely_intent": "feature_impl",
            },
            "model": kwargs["model"],
        }

    monkeypatch.setattr(rcs, "_run_optimizer_pass", _fake_opt)

    out = json.loads(_run(rcs.recall_code_assist_v2(request="determing support and resist levels in python")))
    assert out["status"] == "ready"
    assert calls == [rcs.RECALL_QUERY_OPT_MODEL_FAST]
    assert out["diagnostics"]["query_optimization"]["path"] == "fast_only"


def test_optimizer_summary_like_output_uses_deterministic_fallback(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ALLOW_HEAVY_REWRITE", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "bug_fix", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.92)

    captured = {}

    def _fake_retrieve(*_a, **kwargs):
        captured["queries"] = kwargs["retrieval_queries"]
        captured["file_hints"] = kwargs["file_hints"]
        return {"rows": [_row("repo/a", "autotrade/core/workflow_manager.py", 0.9)], "candidates_scanned": 10, "symbol_map": {}}

    monkeypatch.setattr(rcs, "_agentic_retrieve_multi_query", _fake_retrieve)

    async def _bad_opt(**kwargs):
        return {
            "ok": False,
            "error": "optimizer_summary_like_canonical_query",
            "raw": "",
            "parsed": None,
            "model": kwargs["model"],
        }

    monkeypatch.setattr(rcs, "_run_optimizer_pass", _bad_opt)

    out = json.loads(_run(rcs.recall_code_assist_v2(request="Need the exact implementation path for PM workflow catch-up in workflow_manager.py, including the state flag or artifact checks that trigger it.")))
    assert out["status"] == "ready"
    assert out["diagnostics"]["query_optimization"]["path"] == "deterministic_fallback"
    assert out["diagnostics"]["query_optimization"]["failed_open"] is False
    assert "workflow_manager.py" in captured["file_hints"]
    assert any("workflow_manager.py" in q for q in captured["queries"])


def test_agentic_retrieve_candidates_hard_narrows_to_exact_file_hints(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(
        rcs,
        "code_symbol_search",
        lambda *_a, **_k: [{"repo": "repo/a", "path": "autotrade/core/workflow_manager.py", "score": 1.0}],
    )
    monkeypatch.setattr(
        rcs,
        "rank_repo_files_for_query",
        lambda *_a, **_k: [
            {"file_id": 1, "repo": "repo/a", "path": "autotrade/core/workflow_manager.py", "path_hint_score": 0.56, "file_score": 0.9},
            {"file_id": 2, "repo": "repo/a", "path": "tools/strategy_lab.py", "path_hint_score": 0.0, "file_score": 0.95},
        ],
    )
    monkeypatch.setattr(
        rcs,
        "repo_scoped_search",
        lambda *_a, **_k: [
            dict(_row("repo/a", "tools/strategy_lab.py", 0.95, chunk_id=101), file_id=2),
            dict(_row("repo/a", "autotrade/core/workflow_manager.py", 0.80, chunk_id=102), file_id=1),
        ],
    )
    monkeypatch.setattr(rcs, "expand_adjacent_chunks", lambda *_a, **_k: [])

    out = rcs._agentic_retrieve_candidates(
        db,
        request="workflow_manager pm workflow catchup",
        file_hints=["workflow_manager.py"],
        topic="",
        lang="",
        repo_candidates=[{"repo": "repo/a", "score": 1.0}],
        max_latency_s=15,
    )

    assert out["rows"]
    assert all(r["path"] == "autotrade/core/workflow_manager.py" for r in out["rows"])


def test_chunk_count_trigger_runs_heavy_chunk_optimizer(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_TRIGGER_CHUNK_COUNT", 90)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": _many_rows(91), "candidates_scanned": 220, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.9)

    async def _chunk_opt(_request, rows, timeout_s):
        assert len(rows) == 91
        return {
            "ok": True,
            "error": "",
            "model": rcs.RECALL_CHUNK_OPT_MODEL_HEAVY,
            "raw": "{}",
            "rows": rows[:24],
            "input_chunks": 91,
            "selected_chunks": 24,
        }

    monkeypatch.setattr(rcs, "_run_chunk_optimizer_pass", _chunk_opt)

    out = json.loads(_run(rcs.recall_code_assist_v2(request="determing support and resist levels in python")))
    diag = out["diagnostics"]["chunk_optimization"]
    assert out["status"] == "ready"
    assert diag["triggered"] is True
    assert diag["trigger_reason"] == "chunk_pool_count"
    assert diag["chunk_pool_count"] == 91
    assert diag["selected_chunks"] == 24


def test_chunk_count_threshold_90_does_not_trigger(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_TRIGGER_CHUNK_COUNT", 90)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": _many_rows(90), "candidates_scanned": 200, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.9)

    async def _never(*_a, **_k):
        raise AssertionError("chunk optimizer should not run at chunk_pool_count == threshold")

    monkeypatch.setattr(rcs, "_run_chunk_optimizer_pass", _never)
    out = json.loads(_run(rcs.recall_code_assist_v2(request="determing support and resist levels in python")))
    diag = out["diagnostics"]["chunk_optimization"]
    assert diag["triggered"] is False
    assert diag["chunk_pool_count"] == 90


def test_chunk_optimizer_fail_open_keeps_actionable_output(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_TRIGGER_CHUNK_COUNT", 90)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": _many_rows(95), "candidates_scanned": 205, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.9)

    async def _bad(*_a, **_k):
        return {
            "ok": False,
            "error": "chunk_optimizer_invalid_json",
            "model": rcs.RECALL_CHUNK_OPT_MODEL_HEAVY,
            "raw": "bad",
            "rows": _many_rows(95),
            "input_chunks": 95,
            "selected_chunks": 0,
        }

    monkeypatch.setattr(rcs, "_run_chunk_optimizer_pass", _bad)
    out = json.loads(_run(rcs.recall_code_assist_v2(request="determing support and resist levels in python")))
    assert out["status"] == "ready"
    assert out["context_pack"]["snippets"]
    assert out["diagnostics"]["chunk_optimization"]["failed_open"] is True


def test_keepalive_heartbeats_emitted_during_slow_chunk_optimizer(monkeypatch):
    db = _agentic_db()
    ctx = _Ctx()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_TRIGGER_CHUNK_COUNT", 90)
    monkeypatch.setattr(rcs, "RECALL_PROGRESS_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_PROGRESS_HEARTBEAT_S", 0.05)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": _many_rows(92), "candidates_scanned": 210, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.9)

    async def _slow(_request, rows, timeout_s):
        await asyncio.sleep(0.18)
        return {
            "ok": True,
            "error": "",
            "model": rcs.RECALL_CHUNK_OPT_MODEL_HEAVY,
            "raw": "{}",
            "rows": rows[:24],
            "input_chunks": len(rows),
            "selected_chunks": 24,
        }

    monkeypatch.setattr(rcs, "_run_chunk_optimizer_pass", _slow)
    out = json.loads(_run(rcs.recall_code_assist_v2(request="determing support and resist levels in python", ctx=ctx)))
    assert out["status"] == "ready"
    assert out["diagnostics"]["progress"]["heartbeats"] >= 1
    assert len(ctx.progress) >= 1


def test_chunk_telemetry_fields_persist(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_TRIGGER_CHUNK_COUNT", 90)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": _many_rows(93), "candidates_scanned": 220, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.9)

    async def _ok(_request, rows, timeout_s):
        return {
            "ok": True,
            "error": "",
            "model": rcs.RECALL_CHUNK_OPT_MODEL_HEAVY,
            "raw": "{}",
            "rows": rows[:24],
            "input_chunks": len(rows),
            "selected_chunks": 24,
        }

    monkeypatch.setattr(rcs, "_run_chunk_optimizer_pass", _ok)

    out = json.loads(_run(rcs.recall_code_assist_v2(request="determing support and resist levels in python")))
    assert out["status"] == "ready"

    row = db.execute(
        """
        SELECT chunk_optimizer_used, chunk_optimizer_model, chunk_pool_count,
               chunk_optimizer_input_chunks, chunk_optimizer_selected_chunks,
               chunk_optimizer_failed_open
        FROM retrieval_events ORDER BY id DESC LIMIT 1
        """
    ).fetchone()
    assert int(row["chunk_optimizer_used"]) == 1
    assert row["chunk_optimizer_model"] == rcs.RECALL_CHUNK_OPT_MODEL_HEAVY
    assert int(row["chunk_pool_count"]) == 93
    assert int(row["chunk_optimizer_input_chunks"]) == 93
    assert int(row["chunk_optimizer_selected_chunks"]) == 24
    assert int(row["chunk_optimizer_failed_open"]) == 0


def test_recall_code_assist_v2_latency_profile_p95(monkeypatch):
    db = _agentic_db()
    monkeypatch.setattr(rcs, "RECALL_AGENTIC_V2_ENABLED", True)
    monkeypatch.setattr(rcs, "RECALL_QUERY_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "RECALL_CHUNK_OPT_ENABLED", False)
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "_classify_agentic_intent", lambda *_a, **_k: {"intent": "feature_impl", "llm_ambiguity": 0.0})
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 0.9}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": [_row("repo/a", "src/main.py", 0.9)], "candidates_scanned": 10, "symbol_map": {}},
    )
    monkeypatch.setattr(rcs, "_is_scope_ambiguous", lambda *_a, **_k: {"ambiguous": False, "reasons": [], "lang_signals": ["python"], "flags": {}})
    monkeypatch.setattr(rcs, "_compute_confidence", lambda *_a, **_k: 0.9)

    latencies = []
    for _ in range(20):
        t0 = time.time()
        out = json.loads(_run(rcs.recall_code_assist_v2(request="implement x", repo_hint="repo/a")))
        assert out["status"] == "ready"
        latencies.append((time.time() - t0) * 1000.0)
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    assert p95 <= 15000


def test_summary_model_router_code_heavy_and_general():
    code_rows = [_row("repo/a", "src/x.py", 0.8)]
    prose_rows = [{"chunk_type": "prose", "text": "plain prose"} for _ in range(4)]

    code_pick = rms._select_summary_model("optimize python function in repo", code_rows)
    prose_pick = rms._select_summary_model("summarize macro trends", prose_rows)

    assert code_pick["model_route"] == "code_heavy_primary"
    assert code_pick["model_selected"] == rms.RESEARCH_SUMMARY_MODEL_CODE_HEAVY
    assert prose_pick["model_route"] == "general_default"
    assert prose_pick["model_selected"] == rms.RESEARCH_SUMMARY_MODEL_DEFAULT


def test_summary_model_router_openai_provider(monkeypatch):
    code_rows = [_row("repo/a", "src/x.py", 0.8)]
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_PROVIDER", "openai")
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_OPENAI_MODEL_CODE_HEAVY", "gpt-5-mini")
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_OPENAI_MODEL_DEFAULT", "gpt-5-mini")
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_OPENAI_MODEL_FALLBACK", "gpt-5-mini")

    code_pick = rms._select_summary_model("optimize python function in repo", code_rows)

    assert code_pick["provider"] == "openai"
    assert code_pick["model_selected"] == "gpt-5-mini"


def test_summary_routing_fallback_engages(monkeypatch):
    calls = []

    def _fake_llm(prompt, model, max_tokens, timeout_s=None, temperature=0.3):
        calls.append((model, int(timeout_s or 0)))
        if model == rms.RESEARCH_SUMMARY_MODEL_CODE_HEAVY:
            return "[LLM error: timeout]"
        return "fallback-ok"

    monkeypatch.setattr(rms, "_llm_generate", _fake_llm)
    out = rms._synthesize_with_model_routing(
        prompt="p",
        query="python function in repo",
        retrieved_rows=[_row("repo/a", "src/x.py", 0.8)],
        max_tokens=512,
        context_chars_sent=1234,
    )
    assert out["text"] == "fallback-ok"
    assert out["fallback_used"] is True
    assert out["model_route"] == "fallback"
    assert calls[0][0] == rms.RESEARCH_SUMMARY_MODEL_CODE_HEAVY
    assert calls[1][0] == rms.RESEARCH_SUMMARY_MODEL_FALLBACK


def test_extract_openai_output_text_handles_responses_shape():
    text = rms._extract_openai_output_text(
        {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "alpha"},
                        {"type": "output_text", "text": "beta"},
                    ]
                }
            ]
        }
    )
    assert text == "alpha\nbeta"


def test_synthesis_routing_uses_openai_provider(monkeypatch):
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_PROVIDER", "openai")
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_OPENAI_MODEL_CODE_HEAVY", "gpt-5-mini")
    monkeypatch.setattr(rms, "RESEARCH_SUMMARY_OPENAI_MODEL_FALLBACK", "gpt-5-mini")

    calls = []

    def _fake_openai(prompt, *, model, max_tokens, timeout_s):
        calls.append((model, int(timeout_s)))
        return "openai-ok"

    monkeypatch.setattr(rms, "_llm_generate_openai", _fake_openai)
    out = rms._synthesize_with_model_routing(
        prompt="p",
        query="python function in repo",
        retrieved_rows=[_row("repo/a", "src/x.py", 0.8)],
        max_tokens=512,
        context_chars_sent=1234,
    )
    assert out["text"] == "openai-ok"
    assert out["provider"] == "openai"
    assert out["model_selected"] == "gpt-5-mini"
    assert calls[0][0] == "gpt-5-mini"


def test_synthesis_packer_enforces_32k_limits():
    blocks = [("x" * 4000) for _ in range(30)]
    packed = rms._pack_synthesis_context_blocks(
        blocks,
        target_chars=36000,
        max_chunk_chars=1800,
        max_chunks=14,
    )
    assert packed["blocks_used"] <= 14
    assert packed["context_chars_sent"] <= 36000
    assert all(len(b) <= 1800 for b in packed["context"].split("\n\n---\n\n") if b)


def test_recall_synthesize_includes_model_route_metadata(monkeypatch):
    rows = [
        {
            "title": "Doc A",
            "domain": "example.com",
            "url": "https://example.com/a",
            "text": "sample code content",
            "chunk_type": "code",
            "repo": "repo/a",
            "path": "src/a.py",
            "line_start": 1,
            "line_end": 10,
            "score": 0.7,
        }
    ]
    monkeypatch.setattr(rcs, "_get_db", lambda: sqlite3.connect(":memory:"))
    monkeypatch.setattr(rcs, "hybrid_search", lambda *_a, **_k: rows)
    monkeypatch.setattr(
        rcs,
        "_synthesize_with_model_routing",
        lambda **_k: {
            "text": "synthetic answer",
            "provider": "ollama",
            "model_selected": "qwen2.5-coder:14b",
            "model_route": "code_heavy_primary",
            "fallback_used": False,
            "timeout_s_used": 240,
            "context_chars_sent": 1234,
            "code_ratio": 1.0,
        },
    )
    out = rcs.recall_synthesize("optimizing RAG retrieval python", top_k=8)
    assert "**LLM:** ollama:qwen2.5-coder:14b (route=code_heavy_primary, fallback=no, timeout=240s, context_chars=1234)" in out


def test_recall_synthesize_code_intent_uses_agentic_retrieval(monkeypatch):
    rows = [
        {
            "title": "Workflow Manager",
            "domain": "local",
            "url": "file://workflow_manager",
            "text": "[CODE:python]\ndef _run_overnight(self):\n    if not self.state.daily_flags.get('pm_workflow_ran', False):\n        self._run_pm_workflow()\n",
            "chunk_type": "code",
            "repo": "repo/a",
            "path": "autotrade/core/workflow_manager.py",
            "line_start": 1130,
            "line_end": 1200,
            "score": 0.8,
            "hybrid_score": 0.8,
        }
    ]
    monkeypatch.setattr(rcs, "_get_db", lambda: sqlite3.connect(":memory:"))
    monkeypatch.setattr(rcs, "_discover_candidate_repos", lambda *_a, **_k: [{"repo": "repo/a", "score": 1.0}])
    monkeypatch.setattr(
        rcs,
        "_agentic_retrieve_multi_query",
        lambda *_a, **_k: {"rows": rows, "candidates_scanned": 1, "symbol_map": {}},
    )

    def _fail_hybrid(*_a, **_k):
        raise AssertionError("code-intent synthesis should use agentic retrieval first")

    monkeypatch.setattr(rcs, "hybrid_search", _fail_hybrid)
    monkeypatch.setattr(
        rcs,
        "_synthesize_with_model_routing",
        lambda **_k: {
            "text": "synthetic answer",
            "provider": "ollama",
            "model_selected": "qwen2.5-coder:14b",
            "model_route": "code_heavy_primary",
            "fallback_used": False,
            "timeout_s_used": 240,
            "context_chars_sent": 1234,
            "code_ratio": 1.0,
        },
    )

    out = rcs.recall_synthesize(
        "How does PM workflow catch-up work in autotrade/core/workflow_manager.py?",
        top_k=8,
    )
    assert "autotrade/core/workflow_manager.py" in out


def test_infer_local_repo_hint_from_query():
    assert (
        rcs._infer_local_repo_hint_from_query(
            "How does PM workflow catch-up work in autotrade/core/workflow_manager.py?"
        )
        == "local::autotrade"
    )


def test_infer_default_local_repo_hint_prefers_dominant_local_repo():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        """
        CREATE TABLE source_files (
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            repo_full TEXT,
            repo_ref TEXT,
            path TEXT,
            lang TEXT,
            content_raw TEXT,
            content_rendered TEXT,
            sha256 TEXT,
            line_count INTEGER,
            byte_count INTEGER,
            file_kind TEXT,
            status TEXT,
            reason TEXT
        )
        """
    )
    db.execute(
        """
        INSERT INTO source_files (source_id, repo_full, path, status) VALUES
        (1, 'local::autotrade', 'autotrade/core/a.py', 'selected'),
        (1, 'local::autotrade', 'autotrade/core/b.py', 'selected'),
        (1, 'local::autotrade', 'autotrade/core/c.py', 'selected'),
        (1, 'local::autotrade', 'autotrade/core/d.py', 'selected'),
        (1, 'local::autotrade', 'autotrade/core/e.py', 'selected'),
        (1, 'local::tiny', 'src/a.py', 'selected')
        """
    )
    assert rcs._infer_default_local_repo_hint(db) == "local::autotrade"


def test_build_ingest_proposal_prefers_local_repo_recovery_calls():
    proposal = rcs._build_ingest_proposal(
        request="selected_symbols empty final_watchlist _persist_result",
        repos=[{"repo": "local::autotrade", "score": 0.3}],
        status_reason="insufficient evidence",
    )
    calls = proposal.get("suggested_calls", [])
    assert any("recall_repo_search" in c for c in calls)
    assert any("rescan_local_project" in c for c in calls)


def test_discover_candidate_repos_falls_back_to_local_repo_when_hybrid_empty(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        """
        CREATE TABLE source_files (
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            repo_full TEXT,
            repo_ref TEXT,
            path TEXT,
            lang TEXT,
            content_raw TEXT,
            content_rendered TEXT,
            sha256 TEXT,
            line_count INTEGER,
            byte_count INTEGER,
            file_kind TEXT,
            status TEXT,
            reason TEXT
        )
        """
    )
    db.execute(
        """
        INSERT INTO source_files
        (source_id, repo_full, path, status)
        VALUES
        (1, 'local::autotrade', 'autotrade/core/autonomous_agent.py', 'selected'),
        (1, 'local::autotrade', 'autotrade/core/decision_claw.py', 'selected')
        """
    )
    monkeypatch.setattr(rcs, "hybrid_search", lambda **_k: [])
    repos = rcs._discover_candidate_repos(
        db=db, query="runtime handoff symbol propagation mismatch"
    )
    assert repos
    assert repos[0]["repo"] == "local::autotrade"


def test_recall_repo_file_matches_normalized_path_without_selected_status(monkeypatch):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY,
            url TEXT,
            topic TEXT
        );
        CREATE TABLE source_files (
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            repo_full TEXT,
            repo_ref TEXT,
            path TEXT,
            lang TEXT,
            content_raw TEXT,
            content_rendered TEXT,
            sha256 TEXT,
            line_count INTEGER,
            byte_count INTEGER,
            file_kind TEXT,
            status TEXT,
            reason TEXT
        );
        """
    )
    body = "def a():\n    return 1\n"
    db.execute("INSERT INTO sources (id, url, topic) VALUES (1, 'file://x', 'local')")
    db.execute(
        """
        INSERT INTO source_files
        (source_id, repo_full, path, lang, content_raw, sha256, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (1, "local::autotrade", "autotrade/core/decision_claw.py", "python", body, "abc", "skipped"),
    )
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    out = rcs.recall_repo_file(
        repo="local::autotrade",
        path=".\\autotrade\\core\\decision_claw.py",
        full_file=True,
    )
    assert "Repo File Range: local::autotrade/autotrade/core/decision_claw.py" in out
    assert "def a()" in out


def test_recall_repo_file_uses_local_fs_fallback_for_local_repo(monkeypatch, tmp_path):
    repo_root = tmp_path / "autotrade"
    code_path = repo_root / "autotrade" / "core" / "decision_claw.py"
    code_path.parent.mkdir(parents=True)
    code_path.write_text("def local_only():\n    return True\n", encoding="utf-8")

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY,
            url TEXT,
            topic TEXT
        );
        CREATE TABLE source_files (
            id INTEGER PRIMARY KEY,
            source_id INTEGER,
            repo_full TEXT,
            repo_ref TEXT,
            path TEXT,
            lang TEXT,
            content_raw TEXT,
            content_rendered TEXT,
            sha256 TEXT,
            line_count INTEGER,
            byte_count INTEGER,
            file_kind TEXT,
            status TEXT,
            reason TEXT
        );
        """
    )
    monkeypatch.setattr(rcs, "_get_db", lambda: db)
    monkeypatch.setattr(rcs, "PROJECT_ROOT", repo_root)
    out = rcs.recall_repo_file(
        repo="local::autotrade", path="autotrade/core/decision_claw.py", full_file=True
    )
    assert "local_fs_fallback" in out
    assert "def local_only()" in out


def test_recall_synthesize_fails_closed_without_grounded_code(monkeypatch):
    rows = [
        {
            "title": "Change Log",
            "domain": "local",
            "url": "file://changelog",
            "text": "pm workflow was adjusted",
            "chunk_type": "prose",
            "repo": "repo/a",
            "path": "CHANGELOG.md",
            "line_start": None,
            "line_end": None,
            "score": 0.8,
        }
    ]
    monkeypatch.setattr(rcs, "_get_db", lambda: sqlite3.connect(":memory:"))
    monkeypatch.setattr(rcs, "hybrid_search", lambda *_a, **_k: rows)

    def _should_not_run(**_k):
        raise AssertionError("synthesis should fail closed before LLM call")

    monkeypatch.setattr(rcs, "_synthesize_with_model_routing", _should_not_run)

    out = rcs.recall_synthesize("how does pm workflow catchup work in workflow_manager.py", top_k=8)
    assert "No grounded code evidence found" in out
    assert "refusing to guess" in out


def test_recall_synthesize_filters_doclike_rows_from_code_context(monkeypatch):
    rows = [
        {
            "title": "Change Log",
            "domain": "local",
            "url": "file://changelog",
            "text": "release notes",
            "chunk_type": "prose",
            "repo": "repo/a",
            "path": "CHANGELOG.md",
            "line_start": None,
            "line_end": None,
            "score": 0.9,
        },
        {
            "title": "Workflow Manager",
            "domain": "local",
            "url": "file://workflow_manager",
            "text": "[CODE:python]\ndef _run_post_market_idle(self):\n    return {'pm_workflow_pending': True}\n",
            "chunk_type": "code",
            "repo": "repo/a",
            "path": "autotrade/core/workflow_manager.py",
            "line_start": 1188,
            "line_end": 1248,
            "score": 0.7,
        },
    ]
    captured = {}

    monkeypatch.setattr(rcs, "_get_db", lambda: sqlite3.connect(":memory:"))
    monkeypatch.setattr(rcs, "hybrid_search", lambda *_a, **_k: rows)

    def _fake_synth(**kwargs):
        captured["rows"] = kwargs["retrieved_rows"]
        return {
            "text": "synthetic answer",
            "provider": "ollama",
            "model_selected": "qwen2.5-coder:14b",
            "model_route": "code_heavy_primary",
            "fallback_used": False,
            "timeout_s_used": 240,
            "context_chars_sent": 1200,
            "code_ratio": 1.0,
        }

    monkeypatch.setattr(rcs, "_synthesize_with_model_routing", _fake_synth)
    out = rcs.recall_synthesize("how does pm workflow catchup work in workflow_manager.py", top_k=8)

    assert len(captured["rows"]) == 1
    assert captured["rows"][0]["path"] == "autotrade/core/workflow_manager.py"
    assert "**Sources used:** 1 chunks from 1 unique sources" in out
