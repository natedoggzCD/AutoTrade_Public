from datetime import datetime
from types import SimpleNamespace

from autotrade.utils.research_retrigger import attempt_retrigger_if_stale


def test_attempt_retrigger_if_stale_runs_once_per_day():
    state = {}
    calls = {"count": 0}

    def retrigger():
        calls["count"] += 1
        return True

    freshness = {"age_stale": True}
    now_et = datetime(2026, 3, 3, 6, 30)
    cfg = SimpleNamespace(premarket_retrigger_if_stale=True)

    ok_first = attempt_retrigger_if_stale(
        freshness, now_et=now_et, state=state, retrigger_fn=retrigger, cfg=cfg
    )
    ok_second = attempt_retrigger_if_stale(
        freshness, now_et=now_et, state=state, retrigger_fn=retrigger, cfg=cfg
    )

    assert ok_first is True
    assert ok_second is False
    assert calls["count"] == 1


def test_attempt_retrigger_if_stale_skips_when_disabled():
    state = {}
    calls = {"count": 0}

    def retrigger():
        calls["count"] += 1
        return True

    freshness = {"age_stale": True}
    now_et = datetime(2026, 3, 3, 6, 30)
    cfg = SimpleNamespace(premarket_retrigger_if_stale=False)

    ok = attempt_retrigger_if_stale(
        freshness, now_et=now_et, state=state, retrigger_fn=retrigger, cfg=cfg
    )

    assert ok is False
    assert calls["count"] == 0


def test_attempt_retrigger_if_stale_skips_when_fresh():
    state = {}
    calls = {"count": 0}

    def retrigger():
        calls["count"] += 1
        return True

    freshness = {"age_stale": False}
    now_et = datetime(2026, 3, 3, 6, 30)
    cfg = SimpleNamespace(premarket_retrigger_if_stale=True)

    ok = attempt_retrigger_if_stale(
        freshness, now_et=now_et, state=state, retrigger_fn=retrigger, cfg=cfg
    )

    assert ok is False
    assert calls["count"] == 0


def test_attempt_retrigger_if_stale_handles_failure():
    state = {}
    calls = {"count": 0}

    class Logger:
        def __init__(self):
            self.infos = 0
            self.warnings = 0

        def info(self, *args, **kwargs):
            self.infos += 1

        def warning(self, *args, **kwargs):
            self.warnings += 1

    def retrigger():
        calls["count"] += 1
        raise RuntimeError("boom")

    freshness = {"age_stale": True}
    now_et = datetime(2026, 3, 3, 6, 30)
    cfg = SimpleNamespace(premarket_retrigger_if_stale=True)
    logger = Logger()

    ok = attempt_retrigger_if_stale(
        freshness,
        now_et=now_et,
        state=state,
        retrigger_fn=retrigger,
        cfg=cfg,
        logger=logger,
    )

    assert ok is False
    assert calls["count"] == 1
    assert logger.warnings == 1
