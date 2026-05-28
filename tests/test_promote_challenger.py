"""Round-trip + reversibility tests for tools/promote_challenger.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import promote_challenger as pc  # noqa: E402


def _make_row(
    alpha_id: str, post_pf: float, post_se: float, beats_bnh: bool, pf: float
) -> dict:
    return {
        "strategy_name": f"alpha_catalog__{alpha_id}",
        "setup_type": alpha_id,
        "config_patch": {},
        "backtest_profit_factor": pf,
        "backtest_win_rate": 0.55,
        "metrics": {
            "profit_factor": pf,
            "win_rate": 0.55,
            "sharpe_ratio": 1.0,
            "total_trades": 100,
            "posterior_pf": post_pf,
            "posterior_se": post_se,
            "beats_bnh": beats_bnh,
        },
        "alpha_metadata": {"alpha_id": alpha_id, "family": "momentum"},
    }


def _champion_payload() -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00",
        "promotion_mode": "legacy",
        "symbols": {
            "AAPL": [
                _make_row(
                    "legacy_ichimoku",
                    post_pf=1.0,
                    post_se=0.0,
                    beats_bnh=False,
                    pf=1.05,
                )
            ],
            "NVDA": [
                _make_row(
                    "legacy_ichimoku",
                    post_pf=1.0,
                    post_se=0.0,
                    beats_bnh=False,
                    pf=1.20,
                )
            ],
        },
    }


def _challenger_payload() -> dict:
    return {
        "generated_at": "2026-05-24T12:00:00",
        "promotion_mode": "alpha_catalog_purged_wf_shrinkage",
        "source_run_id": "lab_orchestrator_test",
        "symbols": {
            # AAPL: passes gate (post_pf-se > champion 1.05, beats_bnh)
            "AAPL": [
                _make_row(
                    "ts_momentum_12_1",
                    post_pf=1.40,
                    post_se=0.05,
                    beats_bnh=True,
                    pf=1.45,
                )
            ],
            # NVDA: fails (does not beat bnh)
            "NVDA": [
                _make_row(
                    "ts_momentum_12_1",
                    post_pf=1.50,
                    post_se=0.10,
                    beats_bnh=False,
                    pf=1.55,
                )
            ],
            # AMD: champion has no entry; challenger introduces
            "AMD": [
                _make_row(
                    "donchian_20_10",
                    post_pf=1.60,
                    post_se=0.20,
                    beats_bnh=True,
                    pf=1.62,
                )
            ],
        },
    }


@pytest.fixture()
def temp_pool(tmp_path: Path, monkeypatch):
    champ = tmp_path / "validated_strategies_by_symbol.json"
    chal = tmp_path / "validated_strategies_by_symbol_challenger.json"
    archive_dir = tmp_path / "archive"
    champ.write_text(json.dumps(_champion_payload(), indent=2), encoding="utf-8")
    chal.write_text(json.dumps(_challenger_payload(), indent=2), encoding="utf-8")
    monkeypatch.setattr(pc, "ARCHIVE_DIR", archive_dir)
    return champ, chal, archive_dir


def test_dry_run_does_not_mutate(temp_pool, capsys):
    champ, chal, archive_dir = temp_pool
    original = champ.read_bytes()
    rc = pc.cmd_dry_run(champ, chal)
    assert rc == 0
    out = capsys.readouterr().out
    assert "AAPL" in out and "AMD" in out
    assert "PASS" in out
    assert champ.read_bytes() == original
    assert not archive_dir.exists() or not list(archive_dir.glob("*"))


def test_compare_pool_gates(temp_pool):
    champ, chal, _ = temp_pool
    comparisons = pc.compare_pool(
        json.loads(champ.read_text()), json.loads(chal.read_text())
    )
    by_sym = {c.symbol: c for c in comparisons}
    assert by_sym["AAPL"].gate_pass is True
    assert by_sym["NVDA"].gate_pass is False
    assert "challenger_does_not_beat_bnh" in by_sym["NVDA"].gate_reasons
    assert by_sym["AMD"].gate_pass is True


def test_compare_pool_uses_wildcard_champion_and_min_trades():
    champion = {
        "symbols": {
            "*": [
                _make_row(
                    "legacy_wildcard",
                    post_pf=1.0,
                    post_se=0.0,
                    beats_bnh=True,
                    pf=1.25,
                )
            ]
        }
    }
    challenger = {
        "symbols": {
            "OPEN": [
                _make_row(
                    "sparse_winner",
                    post_pf=999.0,
                    post_se=0.0,
                    beats_bnh=True,
                    pf=999.0,
                )
            ],
            "AAL": [
                _make_row(
                    "too_weak",
                    post_pf=1.20,
                    post_se=0.0,
                    beats_bnh=True,
                    pf=1.20,
                )
            ],
        }
    }
    challenger["symbols"]["OPEN"][0]["metrics"]["total_trades"] = 2

    comparisons = pc.compare_pool(champion, challenger)
    by_sym = {c.symbol: c for c in comparisons}

    assert by_sym["OPEN"].champion_alpha == "legacy_wildcard"
    assert by_sym["OPEN"].challenger_posterior_pf == pc.MAX_POSTERIOR_PF
    assert by_sym["OPEN"].gate_pass is False
    assert "n_trades 2 < 20" in by_sym["OPEN"].gate_reasons
    assert by_sym["AAL"].gate_pass is False
    assert any("champion_pf 1.250" in reason for reason in by_sym["AAL"].gate_reasons)


def test_promote_then_rollback_byte_identical(temp_pool):
    champ, chal, archive_dir = temp_pool
    original = champ.read_bytes()
    rc = pc.cmd_promote(champ, chal, yes=True)
    assert rc == 0
    after = json.loads(champ.read_text(encoding="utf-8"))
    aapl_row = after["symbols"]["AAPL"][0]
    assert aapl_row["alpha_metadata"]["alpha_id"] == "ts_momentum_12_1"
    # NVDA stays on champion (gate failed)
    assert (
        after["symbols"]["NVDA"][0]["alpha_metadata"]["alpha_id"] == "legacy_ichimoku"
    )
    # AMD added from challenger
    assert "AMD" in after["symbols"]
    # archive exists
    archives = list(archive_dir.glob("validated_strategies_by_symbol_*.json"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == original

    rc = pc.cmd_rollback(champ)
    assert rc == 0
    assert champ.read_bytes() == original


def test_promote_refuses_without_passing_gates(tmp_path, monkeypatch, capsys):
    champ = tmp_path / "champ.json"
    chal = tmp_path / "chal.json"
    archive_dir = tmp_path / "archive"
    monkeypatch.setattr(pc, "ARCHIVE_DIR", archive_dir)

    # Champion strong, challenger weak everywhere.
    champ.write_text(
        json.dumps(
            {
                "symbols": {
                    "AAPL": [
                        _make_row(
                            "legacy", post_pf=2.0, post_se=0.0, beats_bnh=True, pf=2.0
                        )
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    chal.write_text(
        json.dumps(
            {
                "symbols": {
                    "AAPL": [
                        _make_row(
                            "weak", post_pf=1.1, post_se=0.2, beats_bnh=True, pf=1.1
                        )
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    original = champ.read_bytes()

    rc = pc.cmd_promote(champ, chal, yes=True)
    assert rc == 1
    assert champ.read_bytes() == original
    assert not archive_dir.exists() or not list(archive_dir.glob("*"))


def test_promote_with_replay_fails_closed_when_measurement_unavailable(
    temp_pool, monkeypatch
):
    champ, chal, archive_dir = temp_pool
    original = champ.read_bytes()
    monkeypatch.setattr(
        pc,
        "_replay_delta_via_swap",
        lambda champion_path, challenger_path: {
            "baseline": None,
            "treatment": None,
            "delta": None,
        },
    )

    rc = pc.cmd_promote(champ, chal, yes=True, with_replay=True)

    assert rc == 1
    assert champ.read_bytes() == original
    assert not archive_dir.exists() or not list(archive_dir.glob("*"))
