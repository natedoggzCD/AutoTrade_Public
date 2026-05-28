from __future__ import annotations

import base64
import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autotrade.backtesting.data import resolve_backtest_paths
from autotrade.replay.overnight_signal_historical import apply_outcome_metrics
from autotrade.replay.overnight_signal_historical import compute_outcomes_from_local_bars
from autotrade.replay.overnight_signal_historical import load_signal_rows
from autotrade.signals.llm_signal_benchmark import attach_outcome_labels


PLAN_SCORE_COLUMNS: List[str] = [
    "final_score",
    "ranking_score",
    "confidence",
    "score",
    "technical_score",
    "sentiment_score",
    "risk_reward",
    "volume_ratio",
    "vol_trend_ratio",
    "weekly_return",
    "atr_percent",
    "entry_price",
    "entry_score",
    "ml_score",
    "sr_score",
    "backtest_win_rate",
    "backtest_profit_factor",
    "historical_win_rate",
    "avg_5d_return",
    "bullish_score",
    "conviction_priority_score",
    "dynamic_conviction_score",
    "allocation_weight",
    "catalyst_count",
    "catalyst_priority_adjustment",
    "catalyst_monitor_score_delta",
    "discovery_family_count",
    "fallback_repeat_count",
    "fallback_repeat_penalty",
    "empirical_band_adjustment",
    "_yt_sector_adj",
    "premarket_gap_pct",
    "premarket_gap",
    "overnight_actionability_score",
    "overnight_expected_profit_proxy",
    "overnight_expected_open_to_close_pct",
    "overnight_expected_open_to_high_pct",
    "overnight_expected_close_loss_pct",
    "overnight_hit_2pct_prob",
    "overnight_positive_close_prob",
    "overnight_bad_close_prob",
    "overnight_fade_risk",
    "overnight_trap_risk",
    "overnight_stickiness_score",
    "overnight_strategy_edge",
    "overnight_setup_edge",
    "overnight_symbol_edge",
    "overnight_catalyst_gate",
    "support_dist_atr",
    "resistance_dist_atr",
    "stop_loss",
    "target_atr_mult",
    "stop_atr_mult",
    "position_size",
    "priority",
    "conviction_priority",
]

PLAN_RANK_COLUMNS: List[str] = [
    "final_score",
    "ranking_score",
    "technical_score",
    "sentiment_score",
    "risk_reward",
    "volume_ratio",
    "weekly_return",
    "atr_percent",
    "overnight_expected_profit_proxy",
]

PRIOR_SESSION_COLUMNS: List[str] = [
    "prior_close_price",
    "prior_bullish_score",
    "prior_rr_ratio",
    "prior_s1_price",
    "prior_s1_strength",
    "prior_s2_price",
    "prior_s2_strength",
    "prior_r1_price",
    "prior_r1_strength",
    "prior_r2_price",
    "prior_r2_strength",
    "prior_atr_14",
    "prior_atr_percent",
    "prior_distance_to_s1_pct",
    "prior_distance_to_r1_pct",
    "prior_spread_pct",
    "prior_momentum_score",
    "prior_poc_distance_pct",
    "prior_in_value_area",
    "prior_stop_loss",
    "prior_target1",
    "prior_target2",
    "prior_actual_open",
    "prior_actual_high",
    "prior_actual_low",
    "prior_actual_close",
    "prior_s1_tested",
    "prior_s1_held",
    "prior_r1_tested",
    "prior_r1_rejected",
    "prior_r2_tested",
    "prior_r2_rejected",
]

CODE_COLUMNS: List[str] = [
    "recommendation_code",
    "action_code",
    "overnight_regime_bucket_code",
    "overnight_execution_intent_code",
    "prior_regime_code",
    "prior_action_plan_code",
]

FEATURE_COLUMNS: List[str] = [
    "rank",
    "day_candidate_count",
    "rank_pct",
    "final_score_rank_pct",
    "ranking_score_rank_pct",
    "technical_score_rank_pct",
    "sentiment_score_rank_pct",
    "risk_reward_rank_pct",
    "volume_ratio_rank_pct",
    "weekly_return_rank_pct",
    "atr_percent_rank_pct",
    "overnight_expected_profit_proxy_rank_pct",
    *PLAN_SCORE_COLUMNS,
    *PRIOR_SESSION_COLUMNS,
    *CODE_COLUMNS,
]

LABEL_COLUMNS: List[str] = [
    "open_to_high_pct",
    "open_to_close_pct",
    "profit_proxy",
    "hit_2pct",
    "positive_close",
    "bad_close",
    "trap_risk",
    "fade_risk",
    "winner",
    "loser",
    "ambiguous",
]

BLEND_WEIGHTS: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

RECOMMENDATION_CODE = {
    "WATCH": 0.0,
    "WEAK BUY": 1.0,
    "BUY": 2.0,
    "STRONG BUY": 3.0,
}

ACTION_CODE = {
    "": 0.0,
    "hold": 0.0,
    "watch": 1.0,
    "watch_pullback": 1.0,
    "buy_dip": 2.0,
    "buy_open": 3.0,
    "buy": 3.0,
}

REGIME_CODE = {
    "": 0.0,
    "neutral": 1.0,
    "mean_reversion": 0.0,
    "momentum": 2.0,
    "momo": 2.0,
    "weak": 0.0,
    "strong": 2.0,
}

EXECUTION_INTENT_CODE = {
    "": 0.0,
    "hold_candidate": 0.0,
    "quick_turnover": 1.0,
    "scalp": 1.0,
    "trend_follow": 2.0,
    "patient": 0.5,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return False


def _safe_date(value: Any) -> Optional[date]:
    try:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        return pd.to_datetime(value).date()
    except Exception:
        return None


def discover_plan_date_range(plans_dir: Path) -> tuple[Optional[date], Optional[date]]:
    dates: List[date] = []
    for path in sorted(plans_dir.glob("morning_game_plan_*.json")):
        match = re.search(r"(\d{8})", path.stem)
        if not match:
            continue
        try:
            dates.append(datetime.strptime(match.group(1), "%Y%m%d").date())
        except Exception:
            continue
    if not dates:
        return None, None
    return min(dates), max(dates)


def _coerce_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _series_or_default(frame: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _code_value(raw: Any, mapping: Dict[str, float], default: float = 0.0) -> float:
    key = str(raw or "").strip().lower()
    return float(mapping.get(key, default))


def _recommendation_to_code(raw: Any) -> float:
    return _code_value(raw, RECOMMENDATION_CODE, 0.0)


def _action_to_code(raw: Any) -> float:
    return _code_value(raw, ACTION_CODE, 0.0)


def _regime_to_code(raw: Any) -> float:
    return _code_value(raw, REGIME_CODE, 1.0)


def _execution_intent_to_code(raw: Any) -> float:
    return _code_value(raw, EXECUTION_INTENT_CODE, 0.0)


def attach_prior_session_features(
    rows: pd.DataFrame,
    *,
    daily_features_path: Optional[Path] = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    daily_path = daily_features_path or resolve_backtest_paths().daily_features_path
    start_date = pd.to_datetime(rows["sig_date"], errors="coerce").dt.date.min()
    end_date = pd.to_datetime(rows["sig_date"], errors="coerce").dt.date.max()
    if start_date is None or end_date is None:
        return rows.copy()

    tickers = sorted({str(t).upper().strip() for t in rows["ticker"].dropna().tolist() if str(t).strip()})
    if not tickers:
        return rows.copy()

    ticker_df = pd.DataFrame({"ticker": tickers})
    conn = duckdb.connect(database=":memory:")
    try:
        conn.register("tickers_df", ticker_df)
        query = f"""
        SELECT
            UPPER(CAST(p.ticker AS VARCHAR)) AS ticker,
            CAST(p.Date AS DATE) AS feature_date,
            CAST(p.Open AS DOUBLE) AS open,
            CAST(p.High AS DOUBLE) AS high,
            CAST(p.Low AS DOUBLE) AS low,
            CAST(p.Close AS DOUBLE) AS close,
            CAST(p.Volume AS DOUBLE) AS volume
        FROM read_parquet('{daily_path.as_posix()}') p
        INNER JOIN tickers_df t
          ON UPPER(CAST(p.ticker AS VARCHAR)) = t.ticker
        WHERE CAST(p.Date AS DATE) BETWEEN DATE '{(start_date - timedelta(days=max(lookback_days, 30))).isoformat()}'
                                      AND DATE '{end_date.isoformat()}'
        ORDER BY ticker, feature_date
        """
        daily = conn.execute(query).df()
    finally:
        conn.close()
    if daily.empty:
        return rows.copy()

    daily = daily.copy()
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    if "feature_date" not in daily.columns:
        if "date" in daily.columns:
            daily["feature_date"] = pd.to_datetime(daily["date"], errors="coerce").dt.date
        elif "Date" in daily.columns:
            daily["feature_date"] = pd.to_datetime(daily["Date"], errors="coerce").dt.date
        else:
            raise KeyError("feature_date")
    else:
        daily["feature_date"] = pd.to_datetime(daily["feature_date"], errors="coerce").dt.date
    daily = daily.sort_values(["ticker", "feature_date"]).reset_index(drop=True)

    grp = daily.groupby("ticker", group_keys=False)
    prev_close = grp["close"].shift(1)
    tr = pd.concat(
        [
            (daily["high"] - daily["low"]).abs(),
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr_14"] = tr.groupby(daily["ticker"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    )
    fallback_atr = (daily["close"].abs() * 0.03).clip(lower=0.05)
    daily["atr_14"] = pd.Series(
        np.where(daily["atr_14"].isna(), fallback_atr, daily["atr_14"]),
        index=daily.index,
    )
    daily["sma_20"] = grp["close"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    daily["vol_sma_20"] = grp["volume"].transform(
        lambda s: s.rolling(20, min_periods=5).mean()
    )
    daily["weekly_return"] = grp["close"].transform(
        lambda s: (s / s.shift(5) - 1.0) * 100.0
    )
    daily["volume_ratio_20"] = (
        daily["volume"] / daily["vol_sma_20"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    daily["s1_price"] = grp["low"].transform(lambda s: s.rolling(10, min_periods=3).min().shift(1))
    daily["s2_price"] = grp["low"].transform(lambda s: s.rolling(20, min_periods=5).min().shift(1))
    daily["r1_price"] = grp["high"].transform(lambda s: s.rolling(10, min_periods=3).max().shift(1))
    daily["r2_price"] = grp["high"].transform(lambda s: s.rolling(20, min_periods=5).max().shift(1))
    daily["s1_price"] = pd.Series(
        np.where(daily["s1_price"].isna(), daily["close"] - daily["atr_14"] * 1.5, daily["s1_price"]),
        index=daily.index,
    )
    daily["s2_price"] = pd.Series(
        np.where(daily["s2_price"].isna(), daily["close"] - daily["atr_14"] * 2.5, daily["s2_price"]),
        index=daily.index,
    )
    daily["r1_price"] = pd.Series(
        np.where(daily["r1_price"].isna(), daily["close"] + daily["atr_14"] * 2.0, daily["r1_price"]),
        index=daily.index,
    )
    daily["r2_price"] = pd.Series(
        np.where(daily["r2_price"].isna(), daily["close"] + daily["atr_14"] * 3.0, daily["r2_price"]),
        index=daily.index,
    )
    close_safe = daily["close"].replace(0, np.nan)
    s1_dist_pct = ((daily["close"] - daily["s1_price"]) / close_safe * 100.0).abs().fillna(0.0)
    r1_dist_pct = ((daily["r1_price"] - daily["close"]) / close_safe * 100.0).abs().fillna(0.0)
    daily["s1_strength"] = (100.0 - s1_dist_pct * 15.0).clip(20.0, 95.0)
    daily["s2_strength"] = (100.0 - (s1_dist_pct + 1.5) * 15.0).clip(20.0, 90.0)
    daily["r1_strength"] = (100.0 - r1_dist_pct * 15.0).clip(20.0, 95.0)
    daily["r2_strength"] = (100.0 - (r1_dist_pct + 1.5) * 15.0).clip(20.0, 90.0)
    trend_pct = ((daily["close"] / daily["sma_20"].replace(0, np.nan)) - 1.0) * 100.0
    bullish_raw = (
        5.0
        + 0.40 * daily["weekly_return"].fillna(0.0)
        + 0.15 * trend_pct.fillna(0.0)
        + 1.50 * (daily["volume_ratio_20"].fillna(1.0) - 1.0)
    )
    daily["bullish_score"] = bullish_raw.clip(0.0, 10.0)
    risk = daily["close"] - daily["s1_price"]
    reward = daily["r1_price"] - daily["close"]
    daily["rr_ratio"] = np.where(risk > 0, reward / risk, np.nan)
    daily["rr_ratio"] = pd.to_numeric(daily["rr_ratio"], errors="coerce").fillna(0.0)
    daily["spread_pct"] = daily["atr_14"].div(close_safe).mul(100.0).mul(0.05).clip(lower=0.02, upper=1.25)
    daily["momentum_score"] = daily["weekly_return"].fillna(0.0).clip(-15.0, 15.0)
    daily["poc_distance_pct"] = trend_pct.fillna(0.0).clip(-25.0, 25.0)
    daily["in_value_area"] = daily["poc_distance_pct"].abs() <= 2.5
    daily["regime"] = np.where(
        daily["weekly_return"].fillna(0.0) >= 3.0,
        "MOMENTUM",
        np.where(
            daily["weekly_return"].fillna(0.0) <= -3.0,
            "MEAN_REVERSION",
            "NEUTRAL",
        ),
    )
    daily["stop_loss"] = daily["close"] - daily["atr_14"] * 2.0
    daily["target1"] = daily["close"] + daily["atr_14"] * 2.5
    daily["target2"] = daily["close"] + daily["atr_14"] * 3.5
    daily["actual_open"] = daily["open"]
    daily["actual_high"] = daily["high"]
    daily["actual_low"] = daily["low"]
    daily["actual_close"] = daily["close"]
    daily["s1_tested"] = daily["actual_low"] <= daily["s1_price"]
    daily["s1_held"] = daily["actual_close"] >= daily["s1_price"]
    daily["r1_tested"] = daily["actual_high"] >= daily["r1_price"]
    daily["r1_rejected"] = daily["actual_close"] < daily["r1_price"]
    daily["r2_tested"] = daily["actual_high"] >= daily["r2_price"]
    daily["r2_rejected"] = daily["actual_close"] < daily["r2_price"]
    daily["action_plan"] = np.where(
        daily["bullish_score"] >= 6.0,
        "bullish_momentum",
        np.where(daily["bullish_score"] >= 4.0, "watch_pullback", "hold"),
    )

    daily["sig_date"] = daily.groupby("ticker")["feature_date"].shift(-1)
    daily = daily[daily["sig_date"].notna()].copy()
    daily["sig_date"] = pd.to_datetime(daily["sig_date"], errors="coerce").dt.date

    rename_map = {
        "close": "prior_close_price",
        "regime": "prior_regime",
        "bullish_score": "prior_bullish_score",
        "rr_ratio": "prior_rr_ratio",
        "s1_price": "prior_s1_price",
        "s1_strength": "prior_s1_strength",
        "s2_price": "prior_s2_price",
        "s2_strength": "prior_s2_strength",
        "r1_price": "prior_r1_price",
        "r1_strength": "prior_r1_strength",
        "r2_price": "prior_r2_price",
        "r2_strength": "prior_r2_strength",
        "atr_14": "prior_atr_14",
        "atr_percent": "prior_atr_percent",
        "distance_to_s1_pct": "prior_distance_to_s1_pct",
        "distance_to_r1_pct": "prior_distance_to_r1_pct",
        "spread_pct": "prior_spread_pct",
        "momentum_score": "prior_momentum_score",
        "poc_distance_pct": "prior_poc_distance_pct",
        "in_value_area": "prior_in_value_area",
        "stop_loss": "prior_stop_loss",
        "target1": "prior_target1",
        "target2": "prior_target2",
        "actual_open": "prior_actual_open",
        "actual_high": "prior_actual_high",
        "actual_low": "prior_actual_low",
        "actual_close": "prior_actual_close",
        "s1_tested": "prior_s1_tested",
        "s1_held": "prior_s1_held",
        "r1_tested": "prior_r1_tested",
        "r1_rejected": "prior_r1_rejected",
        "r2_tested": "prior_r2_tested",
        "r2_rejected": "prior_r2_rejected",
        "action_plan": "prior_action_plan",
    }

    keep_columns = ["ticker", "sig_date", *rename_map.keys()]
    keep_columns = [column for column in keep_columns if column in daily.columns]
    prior = daily[keep_columns].rename(columns=rename_map)
    merged = rows.merge(prior, on=["ticker", "sig_date"], how="left")
    return merged


def build_day_relative_features(
    rows: pd.DataFrame,
    *,
    score_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    frame = rows.copy()
    frame["sig_date"] = pd.to_datetime(frame["sig_date"], errors="coerce").dt.date
    frame["day_candidate_count"] = frame.groupby("sig_date")["ticker"].transform("size")
    frame["rank_pct"] = (
        pd.to_numeric(frame["rank"], errors="coerce").fillna(0.0)
        / frame["day_candidate_count"].replace(0, np.nan)
    ).fillna(1.0)

    score_columns = list(score_columns or PLAN_RANK_COLUMNS)
    for column in score_columns:
        if column not in frame.columns:
            frame[column] = np.nan
        numeric = pd.to_numeric(frame[column], errors="coerce")
        grouped = numeric.groupby(frame["sig_date"])
        frame[f"{column}_rank_pct"] = grouped.rank(method="average", pct=True).fillna(0.5)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0.0, np.nan)
        frame[f"{column}_zscore"] = ((numeric - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return frame


def prepare_broad_training_frame(
    rows: pd.DataFrame,
    *,
    daily_features_path: Optional[Path] = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    prepared = rows.copy()
    prepared["sig_date"] = pd.to_datetime(prepared["sig_date"], errors="coerce").dt.date
    prepared = attach_prior_session_features(
        prepared,
        daily_features_path=daily_features_path,
        lookback_days=lookback_days,
    )
    prepared = build_day_relative_features(prepared)

    prepared["recommendation_code"] = _series_or_default(
        prepared, "recommendation", ""
    ).apply(_recommendation_to_code)
    prepared["action_code"] = _series_or_default(prepared, "action", "").apply(
        _action_to_code
    )
    prepared["overnight_regime_bucket_code"] = _series_or_default(
        prepared, "overnight_regime_bucket", ""
    ).apply(_regime_to_code)
    prepared["overnight_execution_intent_code"] = _series_or_default(
        prepared, "overnight_execution_intent", ""
    ).apply(_execution_intent_to_code)
    prepared["prior_regime_code"] = _series_or_default(
        prepared, "prior_regime", ""
    ).apply(_regime_to_code)
    prepared["prior_action_plan_code"] = _series_or_default(
        prepared, "prior_action_plan", ""
    ).apply(_action_to_code)

    for column in FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = np.nan

    prepared = _coerce_numeric_frame(
        prepared,
        [
            *PLAN_SCORE_COLUMNS,
            *PRIOR_SESSION_COLUMNS,
            *CODE_COLUMNS,
            "rank",
            "day_candidate_count",
            "rank_pct",
            *[f"{column}_rank_pct" for column in PLAN_RANK_COLUMNS if column in prepared.columns],
            *[f"{column}_zscore" for column in PLAN_RANK_COLUMNS if column in prepared.columns],
        ],
    )

    bool_columns = [
        "prior_in_value_area",
        "prior_s1_tested",
        "prior_s1_held",
        "prior_r1_tested",
        "prior_r1_rejected",
        "prior_r2_tested",
        "prior_r2_rejected",
        "in_value_area",
        "s1_tested",
        "s1_held",
        "r1_tested",
        "r1_rejected",
        "r2_tested",
        "r2_rejected",
    ]
    for column in bool_columns:
        if column in prepared.columns:
            prepared[column] = prepared[column].apply(lambda value: 1.0 if _safe_bool(value) else 0.0)

    return prepared


def prepare_broad_training_rows(
    start_date: date,
    end_date: date,
    *,
    prefer_full_watchlist: bool = True,
    daily_features_path: Optional[Path] = None,
    fetch_missing: bool = False,
) -> pd.DataFrame:
    rows = load_signal_rows(
        start_date=start_date,
        end_date=end_date,
        prefer_full_watchlist=prefer_full_watchlist,
    )
    outcomes = compute_outcomes_from_local_bars(rows, fetch_missing=fetch_missing)
    enriched = apply_outcome_metrics(outcomes)
    labeled = attach_outcome_labels(enriched)
    prepared = prepare_broad_training_frame(
        labeled,
        daily_features_path=daily_features_path,
    )
    return prepared


def make_walk_forward_folds(
    dates: Sequence[Any],
    *,
    min_train_days: int = 20,
    validation_days: int = 5,
    step_days: int = 5,
    final_test_days: int = 5,
    embargo_days: int = 1,
) -> Dict[str, Any]:
    unique_dates = sorted(
        {
            _safe_date(value)
            for value in dates
            if _safe_date(value) is not None
        }
    )
    if len(unique_dates) < max(min_train_days + validation_days, final_test_days + 1):
        return {
            "folds": [],
            "train_dates": [str(value) for value in unique_dates],
            "holdout_dates": [],
        }

    train_dates = unique_dates[:-max(final_test_days, 1)]
    holdout_dates = unique_dates[-max(final_test_days, 1):]
    folds: List[Dict[str, Any]] = []
    cursor = max(min_train_days, 1)
    while cursor + validation_days <= len(train_dates):
        train_end = max(0, cursor - max(embargo_days, 0))
        fold_train = train_dates[:train_end]
        fold_valid = train_dates[cursor : cursor + validation_days]
        if fold_train and fold_valid:
            folds.append(
                {
                    "train_dates": [str(value) for value in fold_train],
                    "validation_dates": [str(value) for value in fold_valid],
                }
            )
        cursor += max(step_days, 1)

    return {
        "folds": folds,
        "train_dates": [str(value) for value in train_dates],
        "holdout_dates": [str(value) for value in holdout_dates],
    }


def _build_family_pipeline(family: str, problem: str) -> Pipeline:
    family = str(family).strip().lower()
    if problem == "classifier":
        if family == "hgb":
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_depth=4,
                            learning_rate=0.05,
                            max_iter=180,
                            min_samples_leaf=20,
                            random_state=42,
                        ),
                    ),
                ]
            )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    if family == "hgb":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=180,
                        min_samples_leaf=20,
                        random_state=42,
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _serialize_model(model: Any) -> str:
    return base64.b64encode(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")


def _deserialize_model(serialized_model: str) -> Any:
    return pickle.loads(base64.b64decode(serialized_model.encode("ascii")))


def _feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    matrix = frame.copy()
    for column in feature_columns:
        if column not in matrix.columns:
            matrix[column] = np.nan
    matrix = matrix.loc[:, list(feature_columns)].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def _score_regression_output(values: np.ndarray, *, target_mean: float, target_std: float) -> np.ndarray:
    scale = max(abs(float(target_std)), 1e-6)
    normalized = (values - float(target_mean)) / scale
    return 1.0 / (1.0 + np.exp(-normalized))


def fit_family_models(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    family: str = "logit",
) -> Optional[Dict[str, Any]]:
    if frame.empty:
        return None

    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    work = frame.copy()
    work = work[work["open_to_close_pct"].notna()].copy()
    work = work[work["profit_proxy"].notna()].copy()
    if len(work) < 50:
        return None

    matrix = _feature_matrix(work, feature_columns)
    valid_columns = [column for column in matrix.columns if matrix[column].notna().any()]
    if not valid_columns:
        return None
    matrix = matrix[valid_columns].copy()
    target = work["winner"].fillna(False).astype(int)
    if target.nunique() < 2:
        return None

    classifier = _build_family_pipeline(family, "classifier")
    classifier.fit(matrix, target)

    reg_target = pd.to_numeric(work["profit_proxy"], errors="coerce")
    valid_reg = reg_target.notna()
    if int(valid_reg.sum()) < 50:
        regressor = None
    else:
        regressor = _build_family_pipeline(family, "regression")
        regressor.fit(
            matrix.loc[valid_reg].reset_index(drop=True),
            reg_target.loc[valid_reg].reset_index(drop=True),
        )

    return {
        "family": str(family).strip().lower(),
        "feature_columns": list(valid_columns),
        "classifier": classifier,
        "regressor": regressor,
        "reg_target_mean": float(reg_target.mean()),
        "reg_target_std": float(reg_target.std(ddof=0) or 1.0),
        "train_rows": int(len(work)),
        "train_winner_rate": float(target.mean()),
        "train_profit_proxy_mean": float(reg_target.mean()),
    }


def _predict_family_components(
    frame: pd.DataFrame,
    artifact: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    feature_columns = artifact.get("feature_columns") or list(FEATURE_COLUMNS)
    matrix = _feature_matrix(frame, feature_columns)
    classifier = artifact.get("classifier")
    regressor = artifact.get("regressor")
    if classifier is None:
        raise ValueError("missing_classifier")
    class_prob = classifier.predict_proba(matrix)[:, 1]
    if regressor is None:
        reg_component = np.full(len(matrix), 0.5, dtype=float)
    else:
        reg_raw = np.asarray(regressor.predict(matrix), dtype=float)
        reg_component = _score_regression_output(
            reg_raw,
            target_mean=_safe_float(artifact.get("reg_target_mean"), 0.0),
            target_std=_safe_float(artifact.get("reg_target_std"), 1.0),
        )
    return {"winner_prob": class_prob, "reg_component": reg_component}


def score_family_frame(
    frame: pd.DataFrame,
    artifact: Dict[str, Any],
    *,
    blend_weight: Optional[float] = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if not artifact:
        scored = frame.copy()
        scored["model_score"] = np.nan
        return scored

    components = _predict_family_components(frame, artifact)
    weight = _safe_float(blend_weight, _safe_float(artifact.get("blend_weight"), 0.75))
    weight = max(0.0, min(1.0, weight))
    score = (weight * components["winner_prob"]) + ((1.0 - weight) * components["reg_component"])

    scored = frame.copy()
    scored["winner_prob"] = components["winner_prob"]
    scored["reg_component"] = components["reg_component"]
    scored["model_score"] = score
    return scored


def rank_candidates_by_day(
    frame: pd.DataFrame,
    *,
    score_column: str,
    top_n: int = 10,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.sort_values(
        by=["sig_date", score_column, "rank"],
        ascending=[True, False, True],
    )
    return ranked.groupby("sig_date", group_keys=False).head(max(1, int(top_n))).reset_index(drop=True)


def summarize_selected_rows(selected: pd.DataFrame) -> Dict[str, Any]:
    if selected.empty:
        return {
            "days": 0,
            "rows": 0,
            "avg_open_to_high_pct": 0.0,
            "avg_open_to_close_pct": 0.0,
            "avg_profit_proxy": 0.0,
            "positive_close_rate": 0.0,
            "hit_2pct_rate": 0.0,
            "bad_close_rate": 0.0,
        }

    daily = (
        selected.groupby("sig_date")
        .agg(
            avg_open_to_high_pct=("open_to_high_pct", "mean"),
            avg_open_to_close_pct=("open_to_close_pct", "mean"),
            avg_profit_proxy=("profit_proxy", "mean"),
            positive_close_rate=("positive_close", "mean"),
            hit_2pct_rate=("hit_2pct", "mean"),
            bad_close_rate=("bad_close", "mean"),
        )
        .reset_index(drop=True)
    )

    return {
        "days": int(len(daily)),
        "rows": int(len(selected)),
        "avg_open_to_high_pct": round(float(daily["avg_open_to_high_pct"].mean()), 4),
        "avg_open_to_close_pct": round(float(daily["avg_open_to_close_pct"].mean()), 4),
        "avg_profit_proxy": round(float(daily["avg_profit_proxy"].mean()), 4),
        "positive_close_rate": round(float(daily["positive_close_rate"].mean()), 4),
        "hit_2pct_rate": round(float(daily["hit_2pct_rate"].mean()), 4),
        "bad_close_rate": round(float(daily["bad_close_rate"].mean()), 4),
        "row_positive_close_rate": round(float((selected["open_to_close_pct"] > 0).mean()), 4),
        "row_bad_close_rate": round(float((selected["open_to_close_pct"] <= -2.0).mean()), 4),
    }


def compare_selection_metrics(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "avg_open_to_high_pct": round(
            _safe_float(model_metrics.get("avg_open_to_high_pct"), 0.0)
            - _safe_float(baseline_metrics.get("avg_open_to_high_pct"), 0.0),
            4,
        ),
        "avg_open_to_close_pct": round(
            _safe_float(model_metrics.get("avg_open_to_close_pct"), 0.0)
            - _safe_float(baseline_metrics.get("avg_open_to_close_pct"), 0.0),
            4,
        ),
        "avg_profit_proxy": round(
            _safe_float(model_metrics.get("avg_profit_proxy"), 0.0)
            - _safe_float(baseline_metrics.get("avg_profit_proxy"), 0.0),
            4,
        ),
        "positive_close_rate": round(
            _safe_float(model_metrics.get("positive_close_rate"), 0.0)
            - _safe_float(baseline_metrics.get("positive_close_rate"), 0.0),
            4,
        ),
        "hit_2pct_rate": round(
            _safe_float(model_metrics.get("hit_2pct_rate"), 0.0)
            - _safe_float(baseline_metrics.get("hit_2pct_rate"), 0.0),
            4,
        ),
        "bad_close_rate": round(
            _safe_float(model_metrics.get("bad_close_rate"), 0.0)
            - _safe_float(baseline_metrics.get("bad_close_rate"), 0.0),
            4,
        ),
    }


def select_blend_weight(
    frame: pd.DataFrame,
    artifact: Dict[str, Any],
    *,
    top_n: int = 10,
    baseline_score_column: str = "final_score",
) -> Dict[str, Any]:
    if frame.empty:
        return {
            "blend_weight": 0.75,
            "validation_metrics": summarize_selected_rows(frame),
            "baseline_metrics": summarize_selected_rows(frame),
            "lift": compare_selection_metrics(
                summarize_selected_rows(frame), summarize_selected_rows(frame)
            ),
        }

    baseline_frame = rank_candidates_by_day(
        frame,
        score_column=baseline_score_column,
        top_n=top_n,
    )
    baseline_metrics = summarize_selected_rows(baseline_frame)

    best: Dict[str, Any] | None = None
    for weight in BLEND_WEIGHTS:
        scored = score_family_frame(frame, artifact, blend_weight=weight)
        selected = rank_candidates_by_day(scored, score_column="model_score", top_n=top_n)
        metrics = summarize_selected_rows(selected)
        lift = compare_selection_metrics(metrics, baseline_metrics)
        composite = (
            _safe_float(metrics.get("avg_open_to_close_pct"), 0.0)
            + 0.25 * _safe_float(metrics.get("positive_close_rate"), 0.0)
            + 0.15 * _safe_float(metrics.get("avg_profit_proxy"), 0.0)
        )
        if best is None or composite > _safe_float(best.get("composite"), float("-inf")):
            best = {
                "blend_weight": float(weight),
                "validation_metrics": metrics,
                "baseline_metrics": baseline_metrics,
                "lift": lift,
                "composite": float(composite),
            }

    assert best is not None
    return best


def _baseline_score_frame(frame: pd.DataFrame, score_column: str = "final_score") -> pd.Series:
    if score_column in frame.columns:
        base = pd.to_numeric(frame[score_column], errors="coerce")
    else:
        base = pd.Series(np.nan, index=frame.index, dtype="float64")
    if base.notna().any():
        return base
    for fallback in ("ranking_score", "confidence", "score"):
        if fallback in frame.columns:
            fallback_series = pd.to_numeric(frame[fallback], errors="coerce")
            if fallback_series.notna().any():
                return fallback_series
    return pd.Series(0.0, index=frame.index, dtype="float64")


def build_holdout_summary(
    frame: pd.DataFrame,
    artifact: Dict[str, Any],
    *,
    top_n: int = 10,
    baseline_score_column: str = "final_score",
) -> Dict[str, Any]:
    if frame.empty:
        return {
            "model": summarize_selected_rows(frame),
            "baseline": summarize_selected_rows(frame),
            "lift": compare_selection_metrics(
                summarize_selected_rows(frame), summarize_selected_rows(frame)
            ),
        }

    scored = score_family_frame(frame, artifact, blend_weight=artifact.get("blend_weight"))
    model_selected = rank_candidates_by_day(scored, score_column="model_score", top_n=top_n)
    baseline = frame.copy()
    baseline["baseline_score"] = _baseline_score_frame(baseline, baseline_score_column)
    baseline_selected = rank_candidates_by_day(
        baseline,
        score_column="baseline_score",
        top_n=top_n,
    )
    model_metrics = summarize_selected_rows(model_selected)
    baseline_metrics = summarize_selected_rows(baseline_selected)
    return {
        "model": model_metrics,
        "baseline": baseline_metrics,
        "lift": compare_selection_metrics(model_metrics, baseline_metrics),
    }


def train_broad_signal_model(
    rows: pd.DataFrame,
    *,
    top_n: int = 10,
    final_test_days: int = 5,
    baseline_score_column: str = "final_score",
    families: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if rows.empty:
        raise ValueError("Cannot train broad signal model on an empty frame.")

    work = rows.copy()
    work["sig_date"] = pd.to_datetime(work["sig_date"], errors="coerce").dt.date
    work = work[work["open_to_close_pct"].notna()].copy()
    work = work[work["profit_proxy"].notna()].copy()
    work = work.sort_values(["sig_date", "rank", "ticker"]).reset_index(drop=True)

    dates = work["sig_date"].dropna().tolist()
    split = make_walk_forward_folds(
        dates,
        min_train_days=20,
        validation_days=5,
        step_days=5,
        final_test_days=final_test_days,
        embargo_days=1,
    )
    train_dates = {pd.to_datetime(value).date() for value in split["train_dates"]}
    holdout_dates = {pd.to_datetime(value).date() for value in split["holdout_dates"]}

    families = [str(family).strip().lower() for family in (families or ("logit", "hgb"))]
    fold_reports: List[Dict[str, Any]] = []
    family_reports: Dict[str, List[Dict[str, Any]]] = {family: [] for family in families}

    for fold in split["folds"]:
        fold_train_dates = {pd.to_datetime(value).date() for value in fold["train_dates"]}
        fold_valid_dates = {pd.to_datetime(value).date() for value in fold["validation_dates"]}
        fold_train = work[work["sig_date"].isin(fold_train_dates)].copy()
        fold_valid = work[work["sig_date"].isin(fold_valid_dates)].copy()
        if fold_train.empty or fold_valid.empty:
            continue
        baseline_valid = rank_candidates_by_day(
            fold_valid.assign(baseline_score=_baseline_score_frame(fold_valid, baseline_score_column)),
            score_column="baseline_score",
            top_n=top_n,
        )
        baseline_metrics = summarize_selected_rows(baseline_valid)
        best_family: Dict[str, Any] | None = None
        for family in families:
            family_artifact = fit_family_models(fold_train, family=family)
            if family_artifact is None:
                continue
            selection = select_blend_weight(
                fold_valid,
                family_artifact,
                top_n=top_n,
                baseline_score_column=baseline_score_column,
            )
            candidate = {
                "family": family,
                "blend_weight": selection["blend_weight"],
                "validation_metrics": selection["validation_metrics"],
                "baseline_metrics": baseline_metrics,
                "lift": selection["lift"],
                "composite": _safe_float(selection.get("composite"), float("-inf")),
                "artifact": family_artifact,
            }
            family_reports[family].append(candidate)
            if best_family is None or _safe_float(candidate["composite"], float("-inf")) > _safe_float(best_family["composite"], float("-inf")):
                best_family = candidate
        if best_family is not None:
            fold_reports.append(
                {
                    "train_dates": fold["train_dates"],
                    "validation_dates": fold["validation_dates"],
                    "winner_family": best_family["family"],
                    "blend_weight": best_family["blend_weight"],
                    "validation_metrics": best_family["validation_metrics"],
                    "baseline_metrics": best_family["baseline_metrics"],
                    "lift": best_family["lift"],
                }
            )

    holdout_frame = work[work["sig_date"].isin(holdout_dates)].copy()
    if holdout_frame.empty:
        raise ValueError("No holdout rows available for broad signal model training.")

    # Choose the family and blend that won most validation folds, breaking ties
    # by validation composite performance.
    family_summary: Dict[str, Dict[str, Any]] = {}
    for family, reports in family_reports.items():
        if not reports:
            continue
        wins = sum(1 for report in reports if report["family"] == family)
        best_composite = max(_safe_float(report.get("composite"), float("-inf")) for report in reports)
        family_summary[family] = {
            "validation_runs": int(len(reports)),
            "fold_wins": int(wins),
            "best_composite": float(best_composite),
            "best_report": max(reports, key=lambda report: _safe_float(report.get("composite"), float("-inf"))),
        }

    if not family_summary:
        raise ValueError("Unable to fit any candidate model family.")

    selected_family = max(
        family_summary.items(),
        key=lambda item: (
            _safe_float(item[1].get("best_report", {}).get("validation_metrics", {}).get("avg_open_to_close_pct"), float("-inf")),
            _safe_float(item[1].get("best_composite"), float("-inf")),
        ),
    )[0]
    selected_validation = family_summary[selected_family]["best_report"]

    train_frame = work[work["sig_date"].isin(train_dates)].copy()
    if train_frame.empty:
        train_frame = work[~work["sig_date"].isin(holdout_dates)].copy()

    selected_artifact = fit_family_models(train_frame, family=selected_family)
    if selected_artifact is None:
        raise ValueError(f"Selected family '{selected_family}' could not be refit on training data.")
    selected_artifact["blend_weight"] = _safe_float(selected_validation.get("blend_weight"), 0.75)
    selected_artifact["selected_family"] = selected_family
    selected_artifact["feature_columns"] = list(selected_artifact.get("feature_columns") or FEATURE_COLUMNS)
    selected_artifact["trained_at_utc"] = datetime.utcnow().isoformat()
    selected_artifact["top_n"] = int(top_n)
    selected_artifact["baseline_score_column"] = baseline_score_column

    holdout_summary = build_holdout_summary(
        holdout_frame,
        selected_artifact,
        top_n=top_n,
        baseline_score_column=baseline_score_column,
    )

    selected_validation_metrics = selected_validation.get("validation_metrics", {})
    selected_validation_lift = selected_validation.get("lift", {})
    selection_rows = rank_candidates_by_day(
        score_family_frame(holdout_frame, selected_artifact, blend_weight=selected_artifact["blend_weight"]),
        score_column="model_score",
        top_n=top_n,
    )
    baseline_holdout = rank_candidates_by_day(
        holdout_frame.assign(baseline_score=_baseline_score_frame(holdout_frame, baseline_score_column)),
        score_column="baseline_score",
        top_n=top_n,
    )

    report = {
        "data_summary": {
            "rows": int(len(work)),
            "train_rows": int(len(train_frame)),
            "holdout_rows": int(len(holdout_frame)),
            "unique_dates": int(work["sig_date"].nunique()),
            "unique_tickers": int(work["ticker"].nunique()),
            "winner_rate": round(float(work["winner"].mean()), 4),
            "positive_close_rate": round(float(work["positive_close"].mean()), 4),
            "bad_close_rate": round(float(work["bad_close"].mean()), 4),
            "avg_profit_proxy": round(float(work["profit_proxy"].mean()), 4),
        },
        "split": split,
        "family_summary": family_summary,
        "selected_family": selected_family,
        "selected_blend_weight": float(selected_artifact["blend_weight"]),
        "validation_metrics": selected_validation_metrics,
        "validation_lift": selected_validation_lift,
        "holdout": holdout_summary,
        "baseline_holdout": summarize_selected_rows(baseline_holdout),
        "selected_holdout": summarize_selected_rows(selection_rows),
        "feature_count": int(len(selected_artifact["feature_columns"])),
        "feature_columns": list(selected_artifact["feature_columns"]),
        "fold_reports": fold_reports,
    }

    selected_artifact["report"] = report
    return selected_artifact


def save_model_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    payload = dict(artifact)
    classifier = payload.get("classifier")
    if classifier is not None:
        payload["classifier"] = {
            "serialized_model": _serialize_model(classifier),
        }
    regressor = payload.get("regressor")
    if regressor is not None:
        payload["regressor"] = {
            "serialized_model": _serialize_model(regressor),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_model_artifact(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    classifier = payload.get("classifier")
    if isinstance(classifier, dict) and classifier.get("serialized_model"):
        classifier["serialized_model"] = classifier["serialized_model"]
        try:
            classifier["pipeline"] = _deserialize_model(classifier["serialized_model"])
        except Exception:
            classifier["pipeline"] = None
    regressor = payload.get("regressor")
    if isinstance(regressor, dict) and regressor.get("serialized_model"):
        try:
            regressor["pipeline"] = _deserialize_model(regressor["serialized_model"])
        except Exception:
            regressor["pipeline"] = None
    return payload


def score_candidates_with_artifact(
    rows: pd.DataFrame,
    artifact: Dict[str, Any],
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    work = rows.copy()
    classifier = artifact.get("classifier")
    if isinstance(classifier, dict) and classifier.get("pipeline") is not None:
        artifact_for_scoring = dict(artifact)
        artifact_for_scoring["classifier"] = classifier["pipeline"]
        regressor = artifact.get("regressor")
        if isinstance(regressor, dict) and regressor.get("pipeline") is not None:
            artifact_for_scoring["regressor"] = regressor["pipeline"]
        return score_family_frame(
            work,
            artifact_for_scoring,
            blend_weight=artifact.get("blend_weight"),
        )
    return score_family_frame(work, artifact, blend_weight=artifact.get("blend_weight"))
