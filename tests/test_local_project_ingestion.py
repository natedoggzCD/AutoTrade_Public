import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from tools.mcp import research_mcp_server as rms
from tools.mcp import research_web_server as rws


def _patch_local_runtime(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "research_evidence.db"
    monkeypatch.setattr(rms, "DB_PATH", db_path)
    monkeypatch.setattr(rms, "RESEARCH_LOCAL_ALLOWLIST_ROOTS", str(tmp_path))
    monkeypatch.setattr(rms, "RESEARCH_LOCAL_CATEGORY_EMBED_FALLBACK", False)
    monkeypatch.setattr(rms, "_SYMBOL_EMBED_LIMIT", 0)
    monkeypatch.setattr(
        rms,
        "embed_texts",
        lambda texts, model=rms.EMBED_MODEL: [[0.01, 0.02, 0.03] for _ in texts],
    )
    monkeypatch.setattr(
        rms,
        "embed_single",
        lambda text, model=rms.EMBED_MODEL: [0.01, 0.02, 0.03],
    )
    return db_path


def _wait_job_terminal(client: TestClient, job_id: str, timeout_s: float = 8.0):
    end = time.time() + timeout_s
    last = {}
    while time.time() < end:
        res = client.get(f"/api/local/projects/jobs/{job_id}")
        assert res.status_code == 200, res.text
        last = res.json()
        if last.get("status") in {"completed", "failed", "cancelled"}:
            return last
        time.sleep(0.1)
    return last


def test_local_path_allowlist(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    allowed_root = tmp_path / "allowed"
    denied_root = tmp_path / "denied"
    allowed_root.mkdir(parents=True, exist_ok=True)
    denied_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rms, "RESEARCH_LOCAL_ALLOWLIST_ROOTS", str(allowed_root))

    ok_path, ok_err = rms._validate_local_project_root(str(allowed_root))
    bad_path, bad_err = rms._validate_local_project_root(str(denied_root))

    assert ok_path is not None
    assert ok_err == ""
    assert bad_path is None
    assert bad_err == "root_path_not_allowlisted"


def test_extract_local_project_file_first_without_readme(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "proj_no_readme"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "strategy.py").write_text(
        "def levels(prices):\n    return {'support': min(prices), 'resistance': max(prices)}\n",
        encoding="utf-8",
    )
    (proj / "indicators.ts").write_text(
        "export function trend(v:number[]){ return v.length > 5 ? 'up' : 'flat'; }\n",
        encoding="utf-8",
    )

    out = rms._extract_local_project(root_path=str(proj), alias="myproj")
    selected = [f for f in out["source_files"] if f.get("status") == "selected"]

    assert out["method"] == "local_project_scan"
    assert out["local_repo"] == "local::myproj"
    assert "## FILE: strategy.py" in out["text"]
    assert "## FILE: indicators.ts" in out["text"]
    assert "# Local Project Snapshot" not in out["text"]
    assert selected


def test_extract_local_project_exact_path_include_skips_os_walk(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "proj_targeted"
    (proj / "pkg").mkdir(parents=True, exist_ok=True)
    (proj / "alpha.py").write_text(
        "def alpha(values):\n    return sum(values)\n",
        encoding="utf-8",
    )
    (proj / "pkg" / "beta.py").write_text(
        "def beta(values):\n    return max(values)\n",
        encoding="utf-8",
    )

    def _os_walk_should_not_run(*args, **kwargs):
        raise AssertionError("os.walk should not run for exact targeted include paths")

    monkeypatch.setattr(rms.os, "walk", _os_walk_should_not_run)

    out = rms._extract_local_project(
        root_path=str(proj),
        alias="targeted-proj",
        include_globs=["alpha.py", "pkg/beta.py"],
    )

    selected = [f for f in out["source_files"] if f.get("status") == "selected"]

    assert out["file_count"] == 2
    assert {f["path"] for f in selected} == {"alpha.py", "pkg/beta.py"}
    assert "## FILE: alpha.py" in out["text"]
    assert "## FILE: pkg/beta.py" in out["text"]


def test_local_ingest_persists_namespace_kind_and_categories(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "proj_ingest"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "src").mkdir(exist_ok=True)
    (proj / "src" / "core.py").write_text(
        (
            "class Engine:\n"
            "    \"\"\"Trading engine for local ingestion tests.\"\"\"\n"
            "    def run(self, candles, risk_limit=0.02):\n"
            "        total = 0\n"
            "        for candle in candles:\n"
            "            total += candle.get('close', 0)\n"
            "        avg = total / max(1, len(candles))\n"
            "        if avg > 100 and risk_limit < 0.05:\n"
            "            return {'signal': 'long', 'avg': avg}\n"
            "        return {'signal': 'flat', 'avg': avg}\n"
        ),
        encoding="utf-8",
    )

    res = rms._ingest_local_project(root_path=str(proj), alias="engine_proj")
    assert res["ok"] is True
    assert res["repo_full"] == "local::engine_proj"
    assert res["chunks"] > 0

    db = rms._get_db()
    src = db.execute("SELECT source_kind, topic FROM sources WHERE id = ?", (res["source_id"],)).fetchone()
    assert src is not None
    assert src["source_kind"] == "local"
    assert src["topic"] == "local_project:engine_proj"

    target = db.execute(
        "SELECT target_kind, target_payload_json FROM ingest_targets WHERE id = ?",
        (res["project_id"],),
    ).fetchone()
    assert target is not None
    assert target["target_kind"] == "local_project"
    payload = json.loads(target["target_payload_json"] or "{}")
    assert payload.get("alias") == "engine_proj"
    assert payload.get("root_path") == str(proj.resolve())

    file_rows = db.execute(
        "SELECT repo_full, file_category, status FROM source_files WHERE source_id = ?",
        (res["source_id"],),
    ).fetchall()
    assert file_rows
    assert any(r["repo_full"] == "local::engine_proj" for r in file_rows)
    assert any((r["file_category"] or "") for r in file_rows if r["status"] == "selected")

    chunk_rows = db.execute(
        "SELECT chunk_category FROM chunks WHERE source_id = ?",
        (res["source_id"],),
    ).fetchall()
    assert chunk_rows
    assert any((r["chunk_category"] or "") for r in chunk_rows)


def test_rescan_override_does_not_persist_scoped_include_globs(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "proj_rescan_scope"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "alpha.py").write_text(
        (
            "def alpha(values):\n"
            "    total = 0\n"
            "    for value in values:\n"
            "        total += value\n"
            "    return {'signal': 'alpha', 'avg': total / max(1, len(values))}\n"
        ),
        encoding="utf-8",
    )
    (proj / "beta.py").write_text(
        (
            "def beta(values):\n"
            "    peak = max(values) if values else 0\n"
            "    floor = min(values) if values else 0\n"
            "    return {'signal': 'beta', 'spread': peak - floor}\n"
        ),
        encoding="utf-8",
    )

    res = rms._ingest_local_project(root_path=str(proj), alias="scope-proj")
    assert res["ok"] is True

    rescanned = rms._rescan_local_project(
        alias="scope-proj",
        include_globs_override=["alpha.py"],
    )
    assert rescanned["ok"] is True

    db = rms._get_db()
    target = db.execute(
        "SELECT target_payload_json FROM ingest_targets WHERE id = ?",
        (rescanned["project_id"],),
    ).fetchone()
    assert target is not None
    payload = json.loads(target["target_payload_json"] or "{}")
    assert payload.get("include_globs") == []


def test_category_embedding_fallback_path(monkeypatch):
    monkeypatch.setattr(rms, "RESEARCH_LOCAL_CATEGORY_EMBED_FALLBACK", True)
    monkeypatch.setattr(rms, "_categorize_file", lambda **_k: ("unknown", 0.0))

    calls = {"n": 0}

    def _fake_embed(text: str, model: str = ""):
        calls["n"] += 1
        if "configuration settings yaml json toml ini env variables" in text:
            return [1.0, 0.0]
        if "application source implementation logic class function module" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]

    monkeypatch.setattr(rms, "embed_single", _fake_embed)
    rms._CATEGORY_PROTO_EMB_CACHE.clear()
    cat = rms._categorize_with_embed_fallback(
        path="mystery.abc",
        lang="",
        text="mystery config payload with enough length to trigger embedding fallback classification",
    )

    assert calls["n"] >= 2
    assert cat in {"config", "source", "docs", "build", "data", "script", "notebook", "assets", "test", "unknown"}


def test_api_local_project_add_and_files(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "api_local_proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "algo.py").write_text(
        (
            "def signal(series):\n"
            "    \"\"\"Generate a deterministic signal from simple moving averages.\"\"\"\n"
            "    fast = sum(series[-5:]) / max(1, len(series[-5:]))\n"
            "    slow = sum(series[-20:]) / max(1, len(series[-20:]))\n"
            "    if fast > slow:\n"
            "        return {'bias': 'bullish', 'fast': fast, 'slow': slow}\n"
            "    return {'bias': 'neutral', 'fast': fast, 'slow': slow}\n"
        ),
        encoding="utf-8",
    )

    client = TestClient(rws.app)
    add = client.post(
        "/api/local/projects",
        json={
            "root_path": str(proj),
            "alias": "api-proj",
            "topic": "",
            "include_globs": [],
            "exclude_globs": [],
        },
    )
    assert add.status_code == 200, add.text
    payload = add.json()
    assert payload["ok"] is True
    assert payload["repo_full"] == "local::api-proj"
    project_id = int(payload["project_id"])

    listed = client.get("/api/local/projects?limit=200")
    assert listed.status_code == 200
    projects = listed.json().get("projects", [])
    assert any(int(p["project_id"]) == project_id for p in projects)

    files = client.get(f"/api/local/projects/{project_id}/files?limit=2000")
    assert files.status_code == 200
    assert files.json().get("count", 0) >= 1


def test_api_local_directory_browser(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    browse_root = tmp_path / "browse_root"
    browse_root.mkdir(parents=True, exist_ok=True)
    (browse_root / "alpha").mkdir()
    (browse_root / "beta").mkdir()
    monkeypatch.setattr(rms, "RESEARCH_LOCAL_ALLOWLIST_ROOTS", str(browse_root))

    client = TestClient(rws.app)

    roots = client.get("/api/local/roots")
    assert roots.status_code == 200
    root_rows = roots.json().get("roots", [])
    assert root_rows
    assert str(browse_root.resolve()) in root_rows

    browse = client.get("/api/local/browse", params={"path": str(browse_root), "limit": 100})
    assert browse.status_code == 200
    dirs = browse.json().get("dirs", [])
    names = {d.get("name") for d in dirs}
    assert {"alpha", "beta"}.issubset(names)

    outside = client.get("/api/local/browse", params={"path": str(tmp_path.parent)})
    assert outside.status_code in {400, 403}


def test_api_local_ingest_queue_enqueue_and_complete(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "queue_proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "mod.py").write_text(
        (
            "def calc(xs):\n"
            "    \"\"\"Calculate average with enough lexical signal for indexing tests.\"\"\"\n"
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += x\n"
            "    avg = total / max(1, len(xs))\n"
            "    if avg > 5:\n"
            "        return {'state': 'high', 'avg': avg}\n"
            "    return {'state': 'normal', 'avg': avg}\n"
        ),
        encoding="utf-8",
    )

    with TestClient(rws.app) as client:
        enq = client.post(
            "/api/local/projects/jobs",
            json={"root_path": str(proj), "alias": "queue-proj", "topic": "", "include_globs": [], "exclude_globs": []},
        )
        assert enq.status_code == 200, enq.text
        payload = enq.json()
        assert payload["ok"] is True
        assert payload["status"] == "queued"
        job_id = payload["job_id"]

        done = _wait_job_terminal(client, job_id, timeout_s=12.0)
        assert done.get("status") == "completed", done
        assert int(done.get("heartbeat_count", 0)) >= 0
        assert (done.get("result") or {}).get("repo_full") == "local::queue-proj"

        listing = client.get("/api/local/projects/jobs?limit=50&active_only=0")
        assert listing.status_code == 200
        jobs = listing.json().get("jobs", [])
        assert any(j.get("job_id") == job_id for j in jobs)


def test_api_local_ingest_queue_cancel_queued(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "queue_cancel"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "a.py").write_text("def a(x):\n    return x\n", encoding="utf-8")

    def _slow_ingest(**kwargs):
        time.sleep(1.2)
        alias = kwargs.get("alias") or "proj"
        return {
            "ok": True,
            "project_id": 101,
            "source_id": 201,
            "alias": alias,
            "repo_full": f"local::{alias}",
            "selected_files": 1,
            "file_rows": 1,
            "chunks": 1,
        }

    monkeypatch.setattr(rws, "_ingest_local_project", _slow_ingest)

    with TestClient(rws.app) as client:
        client.post("/api/local/projects/jobs", json={"root_path": str(proj), "alias": "first"}).json()
        second = client.post("/api/local/projects/jobs", json={"root_path": str(proj), "alias": "second"}).json()
        job2 = second["job_id"]

        # Give worker a moment to pick job1 and leave job2 queued.
        time.sleep(0.2)
        cancel = client.post(f"/api/local/projects/jobs/{job2}/cancel")
        if cancel.status_code == 200:
            body = cancel.json()
            assert body["ok"] is True
            assert body["job"]["status"] == "cancelled"

            done2 = _wait_job_terminal(client, job2, timeout_s=4.0)
            assert done2.get("status") == "cancelled"
            assert done2.get("error") == "cancelled_by_user"
        else:
            # If the worker grabbed job2 quickly, cancellation returns 409.
            assert cancel.status_code == 409, cancel.text
            body = cancel.json()
            assert body.get("error") == "running_job_not_cancellable"


def test_api_local_ingest_running_not_cancellable(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "queue_running"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "b.py").write_text("def b(x):\n    return x*2\n", encoding="utf-8")

    def _slow_ingest(**kwargs):
        time.sleep(1.0)
        alias = kwargs.get("alias") or "proj"
        return {
            "ok": True,
            "project_id": 301,
            "source_id": 401,
            "alias": alias,
            "repo_full": f"local::{alias}",
            "selected_files": 1,
            "file_rows": 1,
            "chunks": 1,
        }

    monkeypatch.setattr(rws, "_ingest_local_project", _slow_ingest)

    with TestClient(rws.app) as client:
        enq = client.post("/api/local/projects/jobs", json={"root_path": str(proj), "alias": "runner"})
        assert enq.status_code == 200, enq.text
        job_id = enq.json()["job_id"]

        # Wait until running.
        end = time.time() + 3.0
        status = ""
        while time.time() < end:
            snap = client.get(f"/api/local/projects/jobs/{job_id}").json()
            status = snap.get("status", "")
            if status == "running":
                break
            time.sleep(0.05)
        assert status == "running"

        cancel = client.post(f"/api/local/projects/jobs/{job_id}/cancel")
        assert cancel.status_code == 409
        assert "running_job_not_cancellable" in cancel.text


def test_api_local_ingest_progress_updates(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "queue_progress"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "c.py").write_text("def c(x):\n    return x+1\n", encoding="utf-8")

    def _progress_ingest(**kwargs):
        cb = kwargs.get("progress_cb")
        for i in range(1, 6):
            if cb:
                cb(
                    {
                        "stage": "scan_index",
                        "message": "Scanning local files...",
                        "total_files": 5,
                        "processed_files": i,
                        "selected_files": min(i, 3),
                        "current_path": f"src/file_{i}.py",
                    }
                )
            time.sleep(0.05)
        alias = kwargs.get("alias") or "proj"
        return {
            "ok": True,
            "project_id": 501,
            "source_id": 601,
            "alias": alias,
            "repo_full": f"local::{alias}",
            "selected_files": 3,
            "file_rows": 5,
            "chunks": 9,
        }

    monkeypatch.setattr(rws, "_ingest_local_project", _progress_ingest)

    with TestClient(rws.app) as client:
        enq = client.post("/api/local/projects/jobs", json={"root_path": str(proj), "alias": "progress"})
        assert enq.status_code == 200, enq.text
        job_id = enq.json()["job_id"]

        seen_processed = 0
        end = time.time() + 4.0
        final = {}
        while time.time() < end:
            snap = client.get(f"/api/local/projects/jobs/{job_id}")
            assert snap.status_code == 200
            final = snap.json()
            prog = final.get("progress") or {}
            seen_processed = max(seen_processed, int(prog.get("processed_files", 0) or 0))
            if final.get("status") == "completed":
                break
            time.sleep(0.05)

        assert final.get("status") == "completed", final
        assert seen_processed >= 5


def test_api_local_rescan_job_queues_into_ingestion_queue(monkeypatch, tmp_path):
    _patch_local_runtime(monkeypatch, tmp_path)
    proj = tmp_path / "rescan_proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "main.py").write_text(
        (
            "def main(values):\n"
            "    total = 0\n"
            "    for v in values:\n"
            "        total += v\n"
            "    return total / max(1, len(values))\n"
        ),
        encoding="utf-8",
    )

    with TestClient(rws.app) as client:
        add = client.post(
            "/api/local/projects",
            json={"root_path": str(proj), "alias": "rescan-proj", "topic": "", "include_globs": [], "exclude_globs": []},
        )
        assert add.status_code == 200, add.text
        project_id = int(add.json()["project_id"])

        rq = client.post(f"/api/local/projects/{project_id}/rescan-job")
        assert rq.status_code == 200, rq.text
        data = rq.json()
        assert data["ok"] is True
        assert data["status"] == "queued"
        job_id = data["job_id"]

        done = _wait_job_terminal(client, job_id, timeout_s=12.0)
        assert done.get("status") in {"completed", "failed"}
