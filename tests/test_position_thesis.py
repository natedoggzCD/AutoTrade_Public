"""Tests for PositionThesis and PositionThesisCache."""

import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from autotrade.core.position_thesis import PositionThesis, PositionThesisCache


# ── PositionThesis dataclass ──────────────────────────────────────


class TestPositionThesis:
    def test_default_creation(self):
        t = PositionThesis(symbol="ACME")
        assert t.symbol == "ACME"
        assert t.consecutive_holds == 0
        assert t.last_action == "hold"
        assert t.created_at  # auto-set
        assert t.updated_at

    def test_record_hold_increments(self):
        t = PositionThesis(symbol="ACME")
        t.record_action("hold")
        assert t.consecutive_holds == 1
        t.record_action("HOLD")
        assert t.consecutive_holds == 2
        assert t.history == []  # holds don't push to history

    def test_record_non_hold_resets_and_logs(self):
        t = PositionThesis(symbol="ACME")
        t.record_action("hold")
        t.record_action("hold")
        assert t.consecutive_holds == 2
        t.record_action("trim", "P&L at +5%, taking partial")
        assert t.consecutive_holds == 0
        assert t.last_action == "trim"
        assert len(t.history) == 1
        assert t.history[0]["action"] == "trim"

    def test_history_capped(self):
        t = PositionThesis(symbol="ACME")
        for i in range(15):
            t.record_action("trim", f"reason {i}")
        assert len(t.history) == 10
        assert t.history[0]["reasoning"] == "reason 5"

    def test_update_from_advisor(self):
        t = PositionThesis(symbol="ACME", bear_case="old bear")
        t.update_from_advisor({"bear_case": "new bear", "key_level_update": {"stop": 12.5}})
        assert t.bear_case == "new bear"
        assert t.key_levels["stop"] == 12.5

    def test_update_from_advisor_ignores_empty(self):
        t = PositionThesis(symbol="ACME", bear_case="keep this")
        t.update_from_advisor({})
        assert t.bear_case == "keep this"
        t.update_from_advisor(None)
        assert t.bear_case == "keep this"

    def test_to_prompt_context(self):
        t = PositionThesis(
            symbol="ACME",
            entry_reason="Momentum breakout above 20 SMA",
            strategy_type="breakout",
            bull_case="Strong volume",
            bear_case="Fails at resistance 15.00",
            key_levels={"stop": 12.0, "target": 18.0},
            last_action="hold",
            last_action_reasoning="Thesis intact",
            consecutive_holds=3,
        )
        ctx = t.to_prompt_context()
        assert "PRIOR THESIS:" in ctx
        assert "Momentum breakout" in ctx
        assert "breakout" in ctx
        assert "stop=12.00" in ctx
        assert "Consecutive holds: 3" in ctx
        assert "Has anything changed" in ctx

    def test_round_trip_dict(self):
        t = PositionThesis(
            symbol="ACME",
            entry_reason="test",
            key_levels={"stop": 10.0},
        )
        t.record_action("trim", "partial exit")
        d = t.to_dict()
        t2 = PositionThesis.from_dict(d)
        assert t2.symbol == "ACME"
        assert t2.key_levels == {"stop": 10.0}
        assert len(t2.history) == 1

    def test_from_dict_ignores_unknown_keys(self):
        d = {"symbol": "XYZ", "unknown_field": "whatever"}
        t = PositionThesis.from_dict(d)
        assert t.symbol == "XYZ"


# ── PositionThesisCache ──────────────────────────────────────────


class TestPositionThesisCache:
    def _make_cache(self, tmp_dir):
        return PositionThesisCache(
            persist_path=os.path.join(tmp_dir, "theses.json"),
            archive_path=os.path.join(tmp_dir, "archive.jsonl"),
        )

    def test_seed_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            thesis = cache.seed("ACME", entry_reason="Breakout", strategy_type="breakout")
            assert thesis.symbol == "ACME"
            assert cache.get("acme") is thesis  # case-insensitive
            assert cache.get("ACME") is thesis

    def test_seed_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME", entry_reason="First")
            cache.seed("ACME", entry_reason="Second")  # should NOT overwrite
            assert cache.get("ACME").entry_reason == "First"

    def test_seed_fills_empty_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME", entry_reason="First")
            cache.seed("ACME", bull_case="Strong volume")
            assert cache.get("ACME").entry_reason == "First"
            assert cache.get("ACME").bull_case == "Strong volume"

    def test_record_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME")
            cache.record_action("ACME", "hold")
            assert cache.get("ACME").consecutive_holds == 1

    def test_record_action_missing_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.record_action("NOPE", "hold")  # should not raise

    def test_remove_and_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME", entry_reason="test")
            removed = cache.remove("ACME")
            assert removed is not None
            assert removed.symbol == "ACME"
            assert cache.get("ACME") is None
            # Check archive written
            archive_path = Path(tmp) / "archive.jsonl"
            assert archive_path.exists()
            line = json.loads(archive_path.read_text(encoding="utf-8").strip())
            assert line["symbol"] == "ACME"
            assert "archived_at" in line

    def test_remove_without_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME")
            cache.remove("ACME", archive=False)
            assert not (Path(tmp) / "archive.jsonl").exists()

    def test_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME")
            cache.seed("BCDE")
            cache.seed("CDEF")
            archived = cache.reconcile(["ACME", "CDEF"])
            assert "BCDE" in archived
            assert cache.get("BCDE") is None
            assert cache.get("ACME") is not None

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist = os.path.join(tmp, "theses.json")
            archive = os.path.join(tmp, "archive.jsonl")

            # Save
            c1 = PositionThesisCache(persist_path=persist, archive_path=archive)
            c1.seed("ACME", entry_reason="Breakout", key_levels={"stop": 10.0})
            c1.record_action("ACME", "hold")
            c1.record_action("ACME", "hold")
            c1.save()

            # Load in new instance
            c2 = PositionThesisCache(persist_path=persist, archive_path=archive)
            thesis = c2.get("ACME")
            assert thesis is not None
            assert thesis.entry_reason == "Breakout"
            assert thesis.consecutive_holds == 2
            assert thesis.key_levels == {"stop": 10.0}

    def test_load_skips_stale_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            persist = os.path.join(tmp, "theses.json")
            # Write a file with yesterday's date
            payload = {
                "session_date": "2020-01-01",
                "theses": {"ACME": PositionThesis(symbol="ACME").to_dict()},
            }
            Path(persist).write_text(json.dumps(payload), encoding="utf-8")

            cache = PositionThesisCache(persist_path=persist, archive_path=os.path.join(tmp, "a.jsonl"))
            assert cache.get("ACME") is None  # stale session, not loaded

    def test_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME")
            cache.seed("BCDE")
            assert sorted(cache.symbols()) == ["ACME", "BCDE"]

    def test_get_prompt_context_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            assert cache.get_prompt_context("NOPE") == ""

    def test_get_prompt_context_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = self._make_cache(tmp)
            cache.seed("ACME", entry_reason="Momentum", strategy_type="momentum_ride")
            ctx = cache.get_prompt_context("ACME")
            assert "PRIOR THESIS:" in ctx
            assert "Momentum" in ctx
