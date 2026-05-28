"""
Trend Features Module
=====================

Pure, side-effect-free trend detection transforms.
Includes 3 orthogonal trend definitions:
1. MA slope trend score
2. Directional Movement (ADX) trend score
3. Efficiency Ratio (Kaufman) trend score
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder


def calculate_ma_slope_trend(
    close: pd.Series, ma_window: int = 20, slope_window: int = 5
) -> pd.Series:
    """Calculate trend score based on MA slope (Orthogonal Method 1)."""
    ma = close.rolling(window=ma_window, min_periods=ma_window).mean()
    slope = ma.pct_change(periods=slope_window, fill_method=None)
    return slope * 100


def calculate_directional_movement(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.DataFrame:
    """Calculate Directional Movement components."""
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    # True range must use previous close, not previous high.
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=window, min_periods=window).mean()

    plus_di = (plus_dm.rolling(window=window, min_periods=window).mean() / atr) * 100
    minus_di = (minus_dm.rolling(window=window, min_periods=window).mean() / atr) * 100

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(window=window, min_periods=window).mean()

    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})


def calculate_adx_trend(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Calculate trend score based on ADX (Orthogonal Method 2)."""
    dm = calculate_directional_movement(high, low, close, window)
    return dm["adx"]


def calculate_efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Kaufman Efficiency Ratio (Orthogonal Method 3)."""
    returns = close.diff()
    direction = returns.rolling(window=window, min_periods=window).sum()
    volatility = returns.abs().rolling(window=window, min_periods=window).sum()
    er = direction / volatility.replace(0, np.nan)
    return er


def calculate_trend_strength_composite(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    ma_window: int = 20,
    dmi_window: int = 14,
    er_window: int = 20,
) -> pd.DataFrame:
    """Calculate composite trend strength from all 3 orthogonal methods."""
    ma_slope = calculate_ma_slope_trend(close, ma_window)
    adx = calculate_adx_trend(high, low, close, dmi_window)
    er = calculate_efficiency_ratio(close, er_window)

    ma_slope_norm = (
        ma_slope - ma_slope.rolling(window=60, min_periods=20).mean()
    ) / ma_slope.rolling(window=60, min_periods=20).std()
    adx_norm = adx / 100
    er_norm = er.abs()

    composite = (ma_slope_norm + adx_norm + er_norm) / 3

    return pd.DataFrame(
        {
            "ma_slope_trend": ma_slope,
            "adx_trend": adx,
            "efficiency_ratio": er,
            "trend_strength": composite,
            "trend_direction": np.sign(ma_slope),
        }
    )


def calculate_ma_stacking(
    close: pd.Series, windows: List[int] = [5, 20, 50, 200]
) -> pd.DataFrame:
    """Calculate MA stacking context (which MAs are above/below others)."""
    mas = {}
    for w in windows:
        mas[f"sma{w}"] = close.rolling(window=w, min_periods=w).mean()

    stacking = pd.DataFrame(index=close.index)
    for i, w1 in enumerate(windows):
        for w2 in windows[i + 1 :]:
            stacking[f"sma{w1}_above_sma{w2}"] = (
                mas[f"sma{w1}"] > mas[f"sma{w2}"]
            ).astype(int)

    stack_score = stacking.sum(axis=1) / len(stacking.columns)
    stacking["ma_stack_score"] = stack_score

    return stacking


def calculate_ema_alignment(
    close: pd.Series, windows: List[int] = [8, 21, 55]
) -> pd.Series:
    """Calculate EMA alignment (all EMAs pointing same direction)."""
    emas = {}
    for w in windows:
        emas[f"ema{w}"] = close.ewm(span=w, adjust=False).mean()

    alignment = pd.Series(0.0, index=close.index)
    for i in range(len(windows) - 1):
        alignment += (emas[f"ema{windows[i]}"] > emas[f"ema{windows[i + 1]}"]).astype(
            int
        )

    return alignment / (len(windows) - 1)


class TrendFeatureBuilder:
    """Feature builder for trend indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.TREND

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        ma_window = params.get("ma_slope_window", 20)
        dmi_window = params.get("dmi_window", 14)
        er_window = params.get("efficiency_window", 20)

        if all(col in df.columns for col in ["close", "high", "low"]):
            trend_df = calculate_trend_strength_composite(
                df["close"], df["high"], df["low"], ma_window, dmi_window, er_window
            )
            for col in trend_df.columns:
                result[col] = trend_df[col]

            dm = calculate_directional_movement(
                df["high"], df["low"], df["close"], dmi_window
            )
            result["plus_di"] = dm["plus_di"]
            result["minus_di"] = dm["minus_di"]
            result["adx"] = dm["adx"]

        result["efficiency_ratio"] = calculate_efficiency_ratio(df["close"], er_window)

        ma_windows = params.get("ma_stacking_windows", [5, 20, 50, 200])
        if len(df) >= max(ma_windows):
            stacking = calculate_ma_stacking(df["close"], ma_windows)
            for col in stacking.columns:
                result[col] = stacking[col]

        ema_windows = params.get("ema_alignment_windows", [8, 21, 55])
        if len(df) >= max(ema_windows):
            result["ema_alignment"] = calculate_ema_alignment(df["close"], ema_windows)

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "adx": FeatureMetadata(
                name="adx",
                family=FeatureFamily.TREND,
                description="Average Directional Index",
                unit="index",
                is_leakage_risk=False,
                required_columns=["high", "low", "close"],
                window=14,
            ),
            "efficiency_ratio": FeatureMetadata(
                name="efficiency_ratio",
                family=FeatureFamily.TREND,
                description="Kaufman Efficiency Ratio",
                unit="ratio",
                is_leakage_risk=False,
                required_columns=["close"],
                window=20,
            ),
            "trend_strength": FeatureMetadata(
                name="trend_strength",
                family=FeatureFamily.TREND,
                description="Composite trend strength from 3 orthogonal methods",
                unit="score",
                is_leakage_risk=False,
                required_columns=["close", "high", "low"],
                window=20,
            ),
        }


__all__ = [
    "calculate_ma_slope_trend",
    "calculate_directional_movement",
    "calculate_adx_trend",
    "calculate_efficiency_ratio",
    "calculate_trend_strength_composite",
    "calculate_ma_stacking",
    "calculate_ema_alignment",
    "TrendFeatureBuilder",
]
