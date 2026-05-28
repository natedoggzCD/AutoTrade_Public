from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, Field


DEFAULT_BUNDLE_NAME = "overnight_research_bundle_latest.json"


class CatalystArtifact(BaseModel):
    score: float = 0.0
    tags: list[str] = Field(default_factory=list)
    note: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class SupportResistanceArtifact(BaseModel):
    s1_price: float = 0.0
    r1_price: float = 0.0
    sr_quality_score: float = 0.0

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class ResearchArtifactBundle(BaseModel):
    trade_date: str
    generated_at_et: str
    full_watchlist: list[dict[str, Any]] = Field(default_factory=list)
    top_picks: list[dict[str, Any]] = Field(default_factory=list)
    catalysts: dict[str, CatalystArtifact] = Field(default_factory=dict)
    support_resistance: dict[str, SupportResistanceArtifact] = Field(default_factory=dict)
    quantitative_regime: dict[str, Any] = Field(default_factory=dict)
    youtube_intelligence: dict[str, Any] = Field(default_factory=dict)
    resolved_regime: dict[str, Any] = Field(default_factory=dict)
    workflow_completion: dict[str, Any] = Field(default_factory=dict)


def _symbol_from_row(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or row.get("ticker") or "").strip().upper()


def _merge_watchlist_rows(
    game_plan_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    state_by_symbol = {
        _symbol_from_row(row): dict(row)
        for row in state_rows
        if isinstance(row, Mapping) and _symbol_from_row(row)
    }
    seen: set[str] = set()

    for row in game_plan_rows:
        if not isinstance(row, Mapping):
            continue
        symbol = _symbol_from_row(row)
        if not symbol or symbol in seen:
            continue
        combined = dict(state_by_symbol.get(symbol, {}))
        combined.update(dict(row))
        combined["symbol"] = symbol
        merged.append(combined)
        seen.add(symbol)

    for symbol, row in state_by_symbol.items():
        if symbol in seen:
            continue
        row["symbol"] = symbol
        merged.append(row)
        seen.add(symbol)

    return merged


def build_research_artifact_bundle(
    state: Mapping[str, Any],
    game_plan: Mapping[str, Any],
    now_et: Optional[datetime] = None,
) -> ResearchArtifactBundle:
    timestamp = now_et or datetime.now()
    trade_date = str(game_plan.get("date") or timestamp.strftime("%Y-%m-%d"))

    game_plan_watchlist = list(game_plan.get("full_watchlist") or [])
    state_watchlist = list(state.get("watchlist") or [])
    full_watchlist = _merge_watchlist_rows(game_plan_watchlist, state_watchlist)
    if not full_watchlist:
        full_watchlist = state_watchlist
    top_picks = list(
        game_plan.get("signals")
        or game_plan.get("buy_signals")
        or game_plan.get("top_picks")
        or []
    )

    catalysts: Dict[str, CatalystArtifact] = {}
    support_resistance: Dict[str, SupportResistanceArtifact] = {}

    for row in full_watchlist:
        if not isinstance(row, Mapping):
            continue
        symbol = _symbol_from_row(row)
        if not symbol:
            continue
        catalysts[symbol] = CatalystArtifact(
            score=float(row.get("catalyst_score") or 0.0),
            tags=[str(tag) for tag in (row.get("catalyst_tags") or []) if str(tag).strip()],
            note=str(row.get("catalyst_note") or ""),
        )
        support_resistance[symbol] = SupportResistanceArtifact(
            s1_price=float(row.get("s1_price") or 0.0),
            r1_price=float(row.get("r1_price") or 0.0),
            sr_quality_score=float(row.get("sr_quality_score") or 0.0),
        )

    return ResearchArtifactBundle(
        trade_date=trade_date,
        generated_at_et=timestamp.isoformat(),
        full_watchlist=full_watchlist,
        top_picks=top_picks,
        catalysts=catalysts,
        support_resistance=support_resistance,
        quantitative_regime=dict(state.get("quantitative_regime") or {}),
        youtube_intelligence=dict(game_plan.get("youtube_intelligence") or {}),
        resolved_regime=dict(
            game_plan.get("resolved_regime") or state.get("resolved_regime") or {}
        ),
        workflow_completion=dict(state.get("workflow_completion") or {}),
    )


def save_research_artifact_bundle(
    bundle: ResearchArtifactBundle,
    output_dir: Path,
) -> Path:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    latest = base / DEFAULT_BUNDLE_NAME
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(bundle.model_dump(mode="json"), f, indent=2, default=str)
    return latest


def load_latest_research_artifact_bundle(
    output_dir: Path,
) -> Optional[ResearchArtifactBundle]:
    latest = Path(output_dir) / DEFAULT_BUNDLE_NAME
    if not latest.exists():
        return None
    try:
        with open(latest, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return ResearchArtifactBundle.model_validate(payload)
    except Exception:
        return None
