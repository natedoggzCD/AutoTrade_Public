import sqlite3

from tools.mcp import research_mcp_server as rms


def test_query_symbol_tokens_prioritize_symbol_like_identifiers():
    tokens = rms._query_symbol_tokens(
        "PM_WORKFLOW skip pending pm_plan generation workflow_manager "
        "post_market idle _run_pm_workflow _run_post_market_idle"
    )

    assert "workflow_manager" in tokens[:6]
    assert "_run_pm_workflow" in tokens[:6]
    assert "_run_post_market_idle" in tokens[:6]
    assert "skip" not in tokens[:6]


def test_rank_repo_files_prefers_explicit_file_hint_over_metadata(monkeypatch):
    def _fake_repo_scoped_search(*args, **kwargs):
        return [
            {
                "file_id": 1,
                "repo": "local::autotrade",
                "path": "conductor/tracks/pm_workflow_v2_20260309/metadata.json",
                "hybrid_score": 0.020,
            },
            {
                "file_id": 2,
                "repo": "local::autotrade",
                "path": "autotrade/core/workflow_manager.py",
                "hybrid_score": 0.015,
            },
            {
                "file_id": 2,
                "repo": "local::autotrade",
                "path": "autotrade/core/workflow_manager.py",
                "hybrid_score": 0.014,
            },
            {
                "file_id": 3,
                "repo": "local::autotrade",
                "path": "autotrade/execution/post_market_workflow.py",
                "hybrid_score": 0.018,
            },
        ]

    def _fake_code_symbol_search(*args, **kwargs):
        return [
            {
                "file_id": 2,
                "repo": "local::autotrade",
                "path": "autotrade/core/workflow_manager.py",
                "symbol_name": "_run_post_market_idle",
                "score": 1.0,
            }
        ]

    monkeypatch.setattr(rms, "repo_scoped_search", _fake_repo_scoped_search)
    monkeypatch.setattr(rms, "code_symbol_search", _fake_code_symbol_search)

    ranked = rms.rank_repo_files_for_query(
        "PM_WORKFLOW skip pending workflow_manager _run_post_market_idle",
        db=None,
        repo="local::autotrade",
        top_k=5,
    )

    assert ranked[0]["path"] == "autotrade/core/workflow_manager.py"
    assert ranked[-1]["path"] == "conductor/tracks/pm_workflow_v2_20260309/metadata.json"


def test_code_query_row_weight_penalizes_doclike_sources():
    code_row = {
        "chunk_type": "code",
        "path": "autotrade/core/workflow_manager.py",
        "line_start": 100,
        "line_end": 140,
        "sha256": "abc",
    }
    doc_row = {
        "chunk_type": "prose",
        "path": "CHANGELOG.md",
        "line_start": None,
        "line_end": None,
        "sha256": "",
    }

    code_weight = rms._code_query_row_weight(
        code_row, "workflow_manager pm workflow catchup"
    )
    doc_weight = rms._code_query_row_weight(doc_row, "workflow_manager pm workflow catchup")

    assert code_weight > 1.0
    assert doc_weight < 0.3


def test_hybrid_search_prefer_code_demotes_doclike_chunks(monkeypatch):
    code_row = {
        "chunk_id": 1,
        "chunk_type": "code",
        "path": "autotrade/core/workflow_manager.py",
        "line_start": 120,
        "line_end": 180,
        "sha256": "abc",
        "score": 0.45,
    }
    doc_row = {
        "chunk_id": 2,
        "chunk_type": "prose",
        "path": "CHANGELOG.md",
        "line_start": None,
        "line_end": None,
        "sha256": "",
        "score": 0.45,
    }

    monkeypatch.setattr(rms, "vector_search", lambda *_a, **_k: [doc_row, code_row])
    monkeypatch.setattr(rms, "fts_search", lambda *_a, **_k: [doc_row, code_row])

    out = rms.hybrid_search(
        "workflow_manager pm workflow catchup", sqlite3.connect(":memory:"), top_k=2, prefer_code=True
    )

    assert out[0]["path"] == "autotrade/core/workflow_manager.py"
    assert out[1]["path"] == "CHANGELOG.md"
