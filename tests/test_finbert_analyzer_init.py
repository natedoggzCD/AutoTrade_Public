import autotrade.analysis.finbert_analyzer as finbert_module
from pathlib import Path


def test_get_shared_finbert_analyzer_caches_instance(monkeypatch):
    instances = []

    class _StubAnalyzer:
        def __init__(self, model_name="ProsusAI/finbert", device=None, verbose=False):
            instances.append((model_name, device, verbose))

    monkeypatch.setattr(finbert_module, "FinBERTAnalyzer", _StubAnalyzer)
    monkeypatch.setattr(finbert_module, "_SHARED_ANALYZER", None, raising=False)
    monkeypatch.setattr(finbert_module, "_SHARED_ANALYZER_ERROR", None, raising=False)

    first = finbert_module.get_shared_finbert_analyzer(verbose=False)
    second = finbert_module.get_shared_finbert_analyzer(verbose=False)

    assert first is second
    assert len(instances) == 1
    assert finbert_module.get_finbert_initialization_status() == {"ready": True, "error": None}


def test_get_shared_finbert_analyzer_tracks_error(monkeypatch):
    class _BrokenAnalyzer:
        def __init__(self, model_name="ProsusAI/finbert", device=None, verbose=False):
            raise RuntimeError("cache missing")

    monkeypatch.setattr(finbert_module, "FinBERTAnalyzer", _BrokenAnalyzer)
    monkeypatch.setattr(finbert_module, "_SHARED_ANALYZER", None, raising=False)
    monkeypatch.setattr(finbert_module, "_SHARED_ANALYZER_ERROR", None, raising=False)

    try:
        finbert_module.get_shared_finbert_analyzer(verbose=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")

    status = finbert_module.get_finbert_initialization_status()
    assert status["ready"] is False
    assert status["error"].startswith("RuntimeError:")


def test_finbert_init_wraps_broken_exception_text(monkeypatch):
    class _BrokenExc(Exception):
        def __str__(self):
            raise RecursionError("maximum recursion depth exceeded while getting str")

    monkeypatch.setitem(__import__("sys").modules, "transformers", None)

    original_import = __import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers":
            raise _BrokenExc("cache missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _fake_import)

    try:
        finbert_module.FinBERTAnalyzer(verbose=False)
    except RuntimeError as exc:
        rendered = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "FinBERT runtime dependencies are unavailable for local inference" in rendered
    assert "_BrokenExc:" in rendered


def test_resolve_local_model_dir_uses_cached_snapshot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    snapshot = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--ProsusAI--finbert"
        / "snapshots"
        / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(finbert_module.Path, "home", staticmethod(lambda: home))

    resolved = finbert_module.FinBERTAnalyzer._resolve_local_model_dir("ProsusAI/finbert")

    assert resolved == snapshot


def test_resolve_local_model_dir_rejects_missing_local_snapshot(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(finbert_module.Path, "home", staticmethod(lambda: home))

    try:
        finbert_module.FinBERTAnalyzer._resolve_local_model_dir("ProsusAI/finbert")
    except RuntimeError as exc:
        rendered = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "local snapshot not found" in rendered
    assert "offline FinBERT only" in rendered
