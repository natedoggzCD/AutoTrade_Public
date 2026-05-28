"""
Breakout Features Module
=======================

Pure, side-effect-free breakout detection transforms.
Includes Donchian channels, Bollinger squeeze, and volume expansion.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from autotrade.feature_engineering.schemas import FeatureFamily, FeatureMetadata
from autotrade.feature_engineering.interfaces import FeatureBuilder
from autotrade.feature_engineering.technical import calculate_bollinger_bands
from autotrade.feature_engineering.volatility_volume import calculate_volume_ratio


def calculate_donchian_channels(
    high: pd.Series, low: pd.Series, window: int = 20
) -> pd.DataFrame:
    """Calculate Donchian channel (breakout reference levels)."""
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    width = upper - lower

    return pd.DataFrame(
        {
            "donchian_upper": upper,
            "donchian_middle": (upper + lower) / 2,
            "donchian_lower": lower,
            "donchian_width": width,
        }
    )


def calculate_donchian_breakout(
    close: pd.Series, high: pd.Series, low: pd.Series, window: int = 20
) -> pd.DataFrame:
    """Calculate Donchian breakout signals."""
    dc = calculate_donchian_channels(high, low, window)

    upper_breakout = close > dc["donchian_upper"].shift(1)
    lower_breakout = close < dc["donchian_lower"].shift(1)

    return pd.DataFrame(
        {
            "donchian_upper_breakout": upper_breakout.astype(int),
            "donchian_lower_breakout": lower_breakout.astype(int),
            "donchian_breakout_direction": np.where(
                upper_breakout, 1, np.where(lower_breakout, -1, 0)
            ),
        }
    )


def calculate_bollinger_squeeze(
    close: pd.Series,
    window: int = 20,
    kc_window: int = 20,
    kc_mult: float = 1.5,
    percentile_window: int = 120,
    squeeze_ratio_threshold: float = 1.0,
) -> pd.DataFrame:
    """Calculate Bollinger Band squeeze (low volatility expansion potential)."""
    bb = calculate_bollinger_bands(close, window)

    bb_width = bb["bb_width"]

    rolling_std = close.rolling(window=kc_window, min_periods=kc_window).std()
    ma = close.rolling(window=kc_window, min_periods=kc_window).mean()
    kc_upper = ma + (rolling_std * kc_mult)
    kc_lower = ma - (rolling_std * kc_mult)
    kc_width = kc_upper - kc_lower
    kc_width_normalized = kc_width / ma.replace(0, np.nan)

    squeeze = bb_width < (kc_width_normalized * squeeze_ratio_threshold)

    recent_widths = bb_width.rolling(
        window=percentile_window, min_periods=percentile_window
    )
    squeeze_percentile = recent_widths.apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    return pd.DataFrame(
        {
            "squeeze": squeeze.astype(int),
            "squeeze_percentile": squeeze_percentile,
            "bb_width_raw": bb_width,
            "kc_width_raw": kc_width,
        }
    )


def calculate_volume_expansion(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Calculate volume expansion relative to price movement."""
    vol_ratio = calculate_volume_ratio(volume, window)
    price_change = close.pct_change().abs()

    expansion = vol_ratio / (price_change + 0.001)
    return expansion


def calculate_price_momentum_breakout(
    close: pd.Series, window: int = 20, threshold: float = 0.05
) -> pd.Series:
    """Calculate momentum-based breakout (strong move above threshold)."""
    returns = close.pct_change(window)
    breakout = returns > threshold
    return breakout.astype(int)


def calculate_high_low_breakout(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.DataFrame:
    """Calculate N-day high/low breakout."""
    rolling_high = high.rolling(window=window, min_periods=window).max()
    rolling_low = low.rolling(window=window, min_periods=window).min()

    above_high = close > rolling_high.shift(1)
    below_low = close < rolling_low.shift(1)

    return pd.DataFrame(
        {
            f"breakout_above_{window}d_high": above_high.astype(int),
            f"breakout_below_{window}d_low": below_low.astype(int),
        }
    )


class BreakoutFeatureBuilder:
    """Feature builder for breakout indicators."""

    @property
    def family(self) -> FeatureFamily:
        return FeatureFamily.BREAKOUT

    @property
    def required_columns(self) -> List[str]:
        return ["close"]

    def can_compute(self, df: pd.DataFrame) -> bool:
        return all(col in df.columns for col in self.required_columns)

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        result = df.copy()

        donchian_window = params.get("donchian_window", 20)
        if all(col in df.columns for col in ["high", "low"]):
            dc = calculate_donchian_channels(df["high"], df["low"], donchian_window)
            for col in dc.columns:
                result[col] = dc[col]

            dc_breakout = calculate_donchian_breakout(
                df["close"], df["high"], df["low"], donchian_window
            )
            for col in dc_breakout.columns:
                result[col] = dc_breakout[col]

            hl_breakout = calculate_high_low_breakout(
                df["high"], df["low"], df["close"], donchian_window
            )
            for col in hl_breakout.columns:
                result[col] = hl_breakout[col]

        if len(df) >= 20:
            squeeze_window = params.get("squeeze_window", 20)
            squeeze_result = calculate_bollinger_squeeze(
                df["close"],
                squeeze_window,
                params.get("kc_window", 20),
                params.get("kc_mult", 1.5),
                params.get("squeeze_percentile_window", 120),
                params.get("squeeze_ratio_threshold", 1.0),
            )
            for col in squeeze_result.columns:
                result[col] = squeeze_result[col]

        if "volume" in df.columns:
            result["volume_expansion"] = calculate_volume_expansion(
                df["close"], df["volume"], params.get("volume_window", 20)
            )

        breakout_threshold = params.get("breakout_threshold", 0.05)
        result["price_momentum_breakout"] = calculate_price_momentum_breakout(
            df["close"], params.get("breakout_window", 20), breakout_threshold
        )

        return result

    def get_metadata(self) -> Dict[str, FeatureMetadata]:
        return {
            "donchian_upper": FeatureMetadata(
                name="donchian_upper",
                family=FeatureFamily.BREAKOUT,
                description="Donchian channel upper band",
                unit="price",
                is_leakage_risk=False,
                required_columns=["high"],
                window=20,
            ),
            "donchian_lower": FeatureMetadata(
                name="donchian_lower",
                family=FeatureFamily.BREAKOUT,
                description="Donchian channel lower band",
                unit="price",
                is_leakage_risk=False,
                required_columns=["low"],
                window=20,
            ),
            "squeeze": FeatureMetadata(
                name="squeeze",
                family=FeatureFamily.BREAKOUT,
                description="Bollinger Band squeeze indicator",
                unit="boolean",
                is_leakage_risk=False,
                required_columns=["close"],
                window=120,
            ),
        }


__all__ = [
    "calculate_donchian_channels",
    "calculate_donchian_breakout",
    "calculate_bollinger_squeeze",
    "calculate_volume_expansion",
    "calculate_price_momentum_breakout",
    "calculate_high_low_breakout",
    "BreakoutFeatureBuilder",
]
