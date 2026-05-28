from __future__ import annotations

import base64
import json
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES: List[str] = [
    "atr_percent",
    "volume_ratio",
    "vol_trend_ratio",
    "weekly_return",
    "rsi_14",
    "risk_reward",
    "support_dist_atr",
    "resistance_dist_atr",
    "technical_score",
    "ml_score",
    "sr_score",
    "entry_score",
    "sentiment_score",
    "catalyst_count",
    "fresh_news",
    "discovery_family_count",
    "stale_entry_appearance_streak",
    "fallback_repeat_penalty",
]

TARGET_NAMES: List[str] = [
    "hit_2pct",
    "positive_close",
    "bad_close",
    "trap_risk",
    "fade_risk",
]
REGRESSION_TARGET_NAMES: List[str] = [
    "open_to_high_pct",
    "open_to_close_pct",
    "profit_proxy",
    "close_loss_pct",
    "stickiness_score",
]


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def derive_support_distance_atr(candidate: Dict[str, Any]) -> float:
    raw = candidate.get("support_dist_atr")
    if raw is not None:
        return max(0.0, _safe_float(raw, 0.0))
    entry_price = _safe_float(candidate.get("entry_price"), 0.0)
    atr_percent = _safe_float(candidate.get("atr_percent"), 0.0)
    s1_price = _safe_float(candidate.get("s1_price"), 0.0)
    if entry_price > 0 and atr_percent > 0 and s1_price > 0:
        atr_value = entry_price * (atr_percent / 100.0)
        if atr_value > 0:
            return max(0.0, (entry_price - s1_price) / atr_value)
    return 0.0


def derive_resistance_distance_atr(candidate: Dict[str, Any]) -> float:
    raw = candidate.get("resistance_dist_atr")
    if raw is not None:
        return max(0.0, _safe_float(raw, 0.0))
    entry_price = _safe_float(candidate.get("entry_price"), 0.0)
    atr_percent = _safe_float(candidate.get("atr_percent"), 0.0)
    r1_price = _safe_float(candidate.get("r1_price"), 0.0)
    if entry_price > 0 and atr_percent > 0 and r1_price > 0:
        atr_value = entry_price * (atr_percent / 100.0)
        if atr_value > 0:
            return max(0.0, (r1_price - entry_price) / atr_value)
    return 0.0


def candidate_feature_row(candidate: Dict[str, Any]) -> Dict[str, float]:
    return {
        "atr_percent": _safe_float(candidate.get("atr_percent"), 0.0),
        "volume_ratio": _safe_float(candidate.get("volume_ratio"), 0.0),
        "vol_trend_ratio": _safe_float(candidate.get("vol_trend_ratio"), 1.0),
        "weekly_return": _safe_float(candidate.get("weekly_return"), 0.0),
        "rsi_14": _safe_float(candidate.get("rsi_14"), 50.0),
        "risk_reward": _safe_float(candidate.get("risk_reward"), 0.0),
        "support_dist_atr": derive_support_distance_atr(candidate),
        "resistance_dist_atr": derive_resistance_distance_atr(candidate),
        "technical_score": _safe_float(candidate.get("technical_score"), 50.0),
        "ml_score": _safe_float(candidate.get("ml_score"), 50.0),
        "sr_score": _safe_float(candidate.get("sr_score"), 50.0),
        "entry_score": _safe_float(candidate.get("entry_score"), 50.0),
        "sentiment_score": _safe_float(candidate.get("sentiment_score"), 50.0),
        "catalyst_count": _safe_float(candidate.get("catalyst_count"), 0.0),
        "fresh_news": 1.0 if bool(candidate.get("fresh_news")) else 0.0,
        "discovery_family_count": _safe_float(
            candidate.get("discovery_family_count"), 0.0
        ),
        "stale_entry_appearance_streak": _safe_float(
            candidate.get("stale_entry_appearance_streak"), 0.0
        ),
        "fallback_repeat_penalty": _safe_float(
            candidate.get("fallback_repeat_penalty"), 0.0
        ),
    }


def build_feature_frame(rows: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    records = [candidate_feature_row(row) for row in rows]
    return pd.DataFrame(records, columns=FEATURE_NAMES)


def build_historical_edge_context(rows_df: pd.DataFrame) -> Dict[str, Any]:
    if rows_df.empty:
        return {"strategy": {}, "setup": {}, "symbol": {}}
    df = rows_df.copy()
    if "profit_proxy" not in df.columns:
        return {"strategy": {}, "setup": {}, "symbol": {}}

    def _group_map(column: str, min_count: int) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        if column not in df.columns:
            return result
        for key, group in df.groupby(column):
            if len(group) < min_count:
                continue
            label = str(key or "unknown")
            result[label] = {
                "count": int(len(group)),
                "profit_proxy": round(float(group["profit_proxy"].mean()), 4),
                "open_to_high": round(float(group["open_to_high_pct"].mean()), 4),
                "open_to_close": round(float(group["open_to_close_pct"].mean()), 4),
                "hit_rate_2pct": round(float(group["hit_2pct"].mean()), 4),
                "positive_close_rate": round(float(group["positive_close"].mean()), 4),
                "bad_close_rate": round(float(group["bad_close"].mean()), 4),
            }
        return result

    return {
        "strategy": _group_map("strategy_name", min_count=8),
        "setup": _group_map("setup_type", min_count=8),
        "symbol": _group_map("ticker", min_count=3),
    }


@dataclass
class FittedOvernightModel:
    feature_names: List[str]
    means: List[float]
    scales: List[float]
    coefficients: List[float]
    intercept: float
    sample_count: int
    target_rate: float

    def predict_proba(self, feature_row: Dict[str, Any]) -> float:
        total = float(self.intercept)
        for idx, name in enumerate(self.feature_names):
            raw = _safe_float(feature_row.get(name), 0.0)
            mean = self.means[idx]
            scale = self.scales[idx] if abs(self.scales[idx]) > 1e-9 else 1.0
            total += ((raw - mean) / scale) * self.coefficients[idx]
        return _sigmoid(total)

    def predict_value(self, feature_row: Dict[str, Any]) -> float:
        total = float(self.intercept)
        for idx, name in enumerate(self.feature_names):
            raw = _safe_float(feature_row.get(name), 0.0)
            mean = self.means[idx]
            scale = self.scales[idx] if abs(self.scales[idx]) > 1e-9 else 1.0
            total += ((raw - mean) / scale) * self.coefficients[idx]
        return total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": [round(float(x), 10) for x in self.means],
            "scales": [round(float(x), 10) for x in self.scales],
            "coefficients": [round(float(x), 10) for x in self.coefficients],
            "intercept": round(float(self.intercept), 10),
            "sample_count": int(self.sample_count),
            "target_rate": round(float(self.target_rate), 6),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FittedOvernightModel":
        return cls(
            feature_names=[str(x) for x in payload.get("feature_names", [])],
            means=[float(x) for x in payload.get("means", [])],
            scales=[float(x) for x in payload.get("scales", [])],
            coefficients=[float(x) for x in payload.get("coefficients", [])],
            intercept=float(payload.get("intercept", 0.0) or 0.0),
            sample_count=int(payload.get("sample_count", 0) or 0),
            target_rate=float(payload.get("target_rate", 0.0) or 0.0),
        )


def fit_logistic_model(
    frame: pd.DataFrame, target: pd.Series
) -> Optional[FittedOvernightModel]:
    clean_target = target.fillna(False).astype(int)
    if len(frame) < 40 or clean_target.nunique() < 2:
        return None
    valid_columns = [col for col in frame.columns if frame[col].notna().any()]
    if not valid_columns:
        return None
    fit_frame = frame[valid_columns].copy()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    pipeline.fit(fit_frame, clean_target)

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model: LogisticRegression = pipeline.named_steps["logit"]

    return FittedOvernightModel(
        feature_names=list(fit_frame.columns),
        means=[float(x) for x in scaler.mean_],
        scales=[float(x) if abs(float(x)) > 1e-9 else 1.0 for x in scaler.scale_],
        coefficients=[float(x) for x in model.coef_[0]],
        intercept=float(model.intercept_[0]),
        sample_count=int(len(frame)),
        target_rate=float(clean_target.mean()),
    )


def fit_regression_model(
    frame: pd.DataFrame, target: pd.Series
) -> Optional[FittedOvernightModel]:
    clean_target = pd.to_numeric(target, errors="coerce")
    valid_rows = clean_target.notna()
    if int(valid_rows.sum()) < 50:
        return None
    valid_mask = valid_rows.to_numpy(dtype=bool)
    frame = frame.loc[valid_mask].reset_index(drop=True).copy()
    clean_target = clean_target.loc[valid_rows].reset_index(drop=True)
    valid_columns = [col for col in frame.columns if frame[col].notna().any()]
    if not valid_columns:
        return None
    fit_frame = frame[valid_columns].copy()

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(fit_frame, clean_target)

    scaler: StandardScaler = pipeline.named_steps["scaler"]
    model: Ridge = pipeline.named_steps["ridge"]

    return FittedOvernightModel(
        feature_names=list(fit_frame.columns),
        means=[float(x) for x in scaler.mean_],
        scales=[float(x) if abs(float(x)) > 1e-9 else 1.0 for x in scaler.scale_],
        coefficients=[float(x) for x in np.asarray(model.coef_).reshape(-1)],
        intercept=float(model.intercept_),
        sample_count=int(len(fit_frame)),
        target_rate=float(clean_target.mean()),
    )


def fit_models_from_rows(rows_df: pd.DataFrame) -> Dict[str, Any]:
    if rows_df.empty:
        return {
            "feature_names": list(FEATURE_NAMES),
            "models": {},
            "regression_models": {},
            "rows_used": 0,
        }

    prepared = rows_df.copy()
    if "close_loss_pct" not in prepared.columns:
        prepared.insert(
            len(prepared.columns),
            "close_loss_pct",
            pd.to_numeric(prepared.get("open_to_close_pct"), errors="coerce")
            .fillna(0.0)
            .clip(upper=0.0)
            .abs(),
        )

    feature_frame = build_feature_frame(prepared.to_dict("records"))
    models: Dict[str, Dict[str, Any]] = {}
    for target_name in TARGET_NAMES:
        if target_name not in prepared.columns:
            continue
        fitted = fit_boosted_classifier(feature_frame, prepared[target_name])
        if fitted is None:
            fallback = fit_logistic_model(feature_frame, prepared[target_name])
            fitted = fallback.to_dict() if fallback is not None else None
        if fitted is not None:
            models[target_name] = fitted

    regression_models: Dict[str, Dict[str, Any]] = {}
    for target_name in REGRESSION_TARGET_NAMES:
        if target_name not in prepared.columns:
            continue
        fitted = fit_boosted_regressor(feature_frame, prepared[target_name])
        if fitted is None:
            fallback = fit_regression_model(feature_frame, prepared[target_name])
            fitted = fallback.to_dict() if fallback is not None else None
        if fitted is not None:
            regression_models[target_name] = fitted

    artifact = {
        "version": 2,
        "feature_names": list(FEATURE_NAMES),
        "rows_used": int(len(prepared)),
        "models": models,
        "regression_models": regression_models,
    }
    return artifact


def predict_candidate_probabilities(
    candidate: Dict[str, Any], artifact: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    if not artifact:
        return {}
    feature_row = candidate_feature_row(candidate)
    predictions: Dict[str, float] = {}
    for target_name, payload in (artifact.get("models") or {}).items():
        try:
            if isinstance(payload, dict) and payload.get("serialized_model"):
                predictions[target_name] = round(
                    _predict_serialized_probability(payload, feature_row),
                    6,
                )
            else:
                model = FittedOvernightModel.from_dict(payload)
                predictions[target_name] = round(model.predict_proba(feature_row), 6)
        except Exception:
            continue
    return predictions


def predict_candidate_outcomes(
    candidate: Dict[str, Any], artifact: Optional[Dict[str, Any]]
) -> Dict[str, float]:
    if not artifact:
        return {}
    feature_row = candidate_feature_row(candidate)
    predictions: Dict[str, float] = {}
    for target_name, payload in (artifact.get("regression_models") or {}).items():
        try:
            if isinstance(payload, dict) and payload.get("serialized_model"):
                predictions[target_name] = round(
                    _predict_serialized_regression(payload, feature_row),
                    6,
                )
            else:
                model = FittedOvernightModel.from_dict(payload)
                predictions[target_name] = round(model.predict_value(feature_row), 6)
        except Exception:
            continue
    return predictions


def _serialize_model(model: Any) -> str:
    return base64.b64encode(
        pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")


@lru_cache(maxsize=64)
def _deserialize_model(serialized_model: str) -> Any:
    return pickle.loads(base64.b64decode(serialized_model.encode("ascii")))


def fit_boosted_classifier(
    frame: pd.DataFrame, target: pd.Series
) -> Optional[Dict[str, Any]]:
    clean_target = target.fillna(False).astype(int)
    if len(frame) < 80 or clean_target.nunique() < 2:
        return None
    valid_columns = [col for col in frame.columns if frame[col].notna().any()]
    if not valid_columns:
        return None
    fit_frame = frame[valid_columns].copy()
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.05,
                    max_iter=140,
                    min_samples_leaf=30,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(fit_frame, clean_target)
    return {
        "kind": "sklearn_pipeline",
        "problem_type": "classifier",
        "feature_names": list(fit_frame.columns),
        "serialized_model": _serialize_model(pipeline),
        "sample_count": int(len(fit_frame)),
        "target_rate": round(float(clean_target.mean()), 6),
    }


def fit_boosted_regressor(
    frame: pd.DataFrame, target: pd.Series
) -> Optional[Dict[str, Any]]:
    clean_target = pd.to_numeric(target, errors="coerce")
    valid_rows = clean_target.notna()
    if int(valid_rows.sum()) < 80:
        return None
    valid_mask = valid_rows.to_numpy(dtype=bool)
    fit_source = frame.loc[valid_mask].reset_index(drop=True).copy()
    clean_target = clean_target.loc[valid_rows].reset_index(drop=True)
    valid_columns = [col for col in fit_source.columns if fit_source[col].notna().any()]
    if not valid_columns:
        return None
    fit_frame = fit_source[valid_columns].copy()
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=4,
                    learning_rate=0.05,
                    max_iter=140,
                    min_samples_leaf=30,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(fit_frame, clean_target)
    return {
        "kind": "sklearn_pipeline",
        "problem_type": "regression",
        "feature_names": list(fit_frame.columns),
        "serialized_model": _serialize_model(pipeline),
        "sample_count": int(len(fit_frame)),
        "target_rate": round(float(clean_target.mean()), 6),
    }


def _predict_serialized_probability(
    payload: Dict[str, Any], feature_row: Dict[str, Any]
) -> float:
    model = _deserialize_model(str(payload.get("serialized_model") or ""))
    frame = pd.DataFrame(
        [
            {
                name: _safe_float(feature_row.get(name), 0.0)
                for name in payload.get("feature_names", [])
            }
        ]
    )
    proba = model.predict_proba(frame)
    return float(proba[0][1])


def _predict_serialized_regression(
    payload: Dict[str, Any], feature_row: Dict[str, Any]
) -> float:
    model = _deserialize_model(str(payload.get("serialized_model") or ""))
    frame = pd.DataFrame(
        [
            {
                name: _safe_float(feature_row.get(name), 0.0)
                for name in payload.get("feature_names", [])
            }
        ]
    )
    return float(model.predict(frame)[0])


def load_model_artifact(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_model_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
