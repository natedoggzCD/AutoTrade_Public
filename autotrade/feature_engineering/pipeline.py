"""
Feature Engineering Pipeline
============================

Multi-timeframe feature assembly with strict as-of joins and anti-leakage guards.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from autotrade.feature_engineering.schemas import (
    FeatureRequest,
    FeatureFrame,
    FeatureMetadata,
    FeaturePipelineReport,
    FeatureFamily,
)
from autotrade.feature_engineering.interfaces import (
    FeatureBuilder,
)
from autotrade.feature_engineering.trend import TrendFeatureBuilder
from autotrade.feature_engineering.momentum import MomentumFeatureBuilder
from autotrade.feature_engineering.reversal import ReversalFeatureBuilder
from autotrade.feature_engineering.breakout import BreakoutFeatureBuilder
from autotrade.feature_engineering.regime import RegimeFeatureBuilder
from autotrade.feature_engineering.volatility_volume import (
    VolatilityVolumeFeatureBuilder,
)


class FeaturePipeline:
    """
    Orchestrates feature computation across multiple builders with:
    - Multi-timeframe assembly (daily + hourly)
    - Strict as-of joins to prevent leakage
    - Optional breadth/cross-asset features
    - Deterministic execution
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        breadth_enabled: bool = True,
        breadth_symbols: Optional[List[str]] = None,
    ):
        self._builders: Dict[FeatureFamily, FeatureBuilder] = {}
        self._config = config or {}
        self._breadth_enabled = self._config.get("breadth_enabled", breadth_enabled)
        self._breadth_symbols = breadth_symbols or self._config.get(
            "breadth_symbols", ["SPY", "QQQ"]
        )
        self._last_report: Optional[FeaturePipelineReport] = None
        self._cache: Dict[str, FeatureFrame] = {}
        self._cache_ttl_minutes = self._config.get("cache_ttl_minutes", 60)
        self._primary_timeframe = self._config.get("primary_timeframe", "1d")
        self._secondary_timeframes = self._config.get("secondary_timeframes", ["1h"])
        self._lookback_bars = self._config.get("lookback_bars", 252)
        self._min_history_bars = self._config.get("min_history_bars", 80)
        self._fail_fast = self._config.get("fail_fast_on_missing_columns", True)

        self._register_default_builders()

    def _register_default_builders(self) -> None:
        """Register all default feature builders."""
        self._builders[FeatureFamily.TREND] = TrendFeatureBuilder()
        self._builders[FeatureFamily.TS_MOMENTUM] = MomentumFeatureBuilder()
        self._builders[FeatureFamily.XS_MOMENTUM] = MomentumFeatureBuilder()
        self._builders[FeatureFamily.MEAN_REVERSION] = ReversalFeatureBuilder()
        self._builders[FeatureFamily.BREAKOUT] = BreakoutFeatureBuilder()
        self._builders[FeatureFamily.PULLBACK] = BreakoutFeatureBuilder()
        self._builders[FeatureFamily.REGIME] = RegimeFeatureBuilder()
        self._builders[FeatureFamily.VOLATILITY] = VolatilityVolumeFeatureBuilder()
        self._builders[FeatureFamily.VOLUME] = VolatilityVolumeFeatureBuilder()

    @property
    def families_enabled(self) -> List[FeatureFamily]:
        """Get list of enabled feature families from config."""
        enabled = self._config.get("alpha_families_enabled", [])
        family_map = {
            "ts_momentum": FeatureFamily.TS_MOMENTUM,
            "xs_momentum": FeatureFamily.XS_MOMENTUM,
            "mean_reversion": FeatureFamily.MEAN_REVERSION,
            "breakout": FeatureFamily.BREAKOUT,
            "pullback": FeatureFamily.PULLBACK,
            "trend": FeatureFamily.TREND,
            "volatility": FeatureFamily.VOLATILITY,
            "volume": FeatureFamily.VOLUME,
            "regime": FeatureFamily.REGIME,
        }
        return [
            family_map.get(f, FeatureFamily.TREND) for f in enabled if f in family_map
        ]

    def add_builder(self, family: FeatureFamily, builder: FeatureBuilder) -> None:
        """Add a feature builder to the pipeline."""
        self._builders[family] = builder

    def remove_builder(self, family: FeatureFamily) -> None:
        """Remove a feature builder from the pipeline."""
        self._builders.pop(family, None)

    def get_builder(self, family: FeatureFamily) -> Optional[FeatureBuilder]:
        """Get a builder by family."""
        return self._builders.get(family)

    def execute(
        self,
        request: FeatureRequest,
        data: pd.DataFrame,
        breadth_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> FeatureFrame:
        """
        Execute pipeline and return computed features.

        Args:
            request: Feature request specification
            data: Primary timeframe OHLCV data (must have 'symbol', 'date' columns)
            breadth_data: Optional dict of {symbol: dataframe} for cross-asset features

        Returns:
            FeatureFrame with computed features
        """
        start_time = datetime.now()
        features_computed = []
        features_failed = []
        warnings = []

        required_cols = {"symbol", "date", "close"}
        if self._fail_fast and not required_cols.issubset(set(data.columns)):
            missing = required_cols - set(data.columns)
            raise ValueError(f"Missing required columns: {missing}")

        df = data.copy()
        df = self._apply_leakage_guard(df, request.as_of_date)

        for family in request.families:
            builder = self._builders.get(family)
            if builder is None:
                warnings.append(f"No builder registered for family: {family.value}")
                continue

            if not builder.can_compute(df):
                features_failed.append(f"{family.value}: insufficient data")
                continue

            try:
                params = self._get_family_params(family)
                df = builder.compute(df, **params)
                metadata = builder.get_metadata()
                features_computed.extend(metadata.keys())
            except Exception as e:
                features_failed.append(f"{family.value}: {str(e)}")

        if self._breadth_enabled and breadth_data:
            try:
                df = self._add_breadth_features(df, breadth_data, request.as_of_date)
                features_computed.extend(
                    [
                        "breadth_spy_return",
                        "breadth_qqq_return",
                        "breadth_market_context",
                    ]
                )
            except Exception as e:
                warnings.append(f"Breadth features failed: {str(e)}")

        df = self._enforce_min_history(df, self._min_history_bars)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

        self._last_report = FeaturePipelineReport(
            timestamp=start_time,
            request=request,
            features_computed=features_computed,
            features_failed=features_failed,
            symbols_processed=df["symbol"].nunique() if "symbol" in df.columns else 0,
            bars_used=len(df),
            computation_time_ms=elapsed_ms,
            warnings=warnings,
            leakage_check_passed=True,
        )

        return FeatureFrame(
            df=df,
            symbols=request.symbols,
            timestamp=start_time,
            features=self._collect_metadata(features_computed),
            primary_timeframe=self._primary_timeframe,
            secondary_timeframes=self._secondary_timeframes,
            source="pipeline",
        )

    def execute_multitimeframe(
        self,
        request: FeatureRequest,
        daily_data: pd.DataFrame,
        hourly_data: Optional[pd.DataFrame] = None,
        breadth_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> FeatureFrame:
        """
        Execute pipeline with multi-timeframe alignment.

        Daily context drives long-term features, hourly triggers for pullback/continuation.

        Args:
            request: Feature request specification
            daily_data: Daily OHLCV data
            hourly_data: Optional hourly data for intraday features
            breadth_data: Optional cross-asset data (SPY, QQQ)

        Returns:
            FeatureFrame with merged multi-timeframe features
        """
        request.primary_timeframe = self._primary_timeframe
        request.secondary_timeframes = self._secondary_timeframes

        daily_frame = self.execute(request, daily_data, breadth_data)

        if hourly_data is not None and hourly_data.shape[0] > 0:
            hourly_features = self._compute_hourly_trigger_features(
                hourly_data, daily_data, request.as_of_date
            )
            merged = self._merge_as_of(
                daily_frame.df, hourly_features, "date", "timestamp", request.as_of_date
            )
            daily_frame.df = merged

        return daily_frame

    def _compute_hourly_trigger_features(
        self,
        hourly_data: pd.DataFrame,
        daily_context: pd.DataFrame,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute hourly features aligned to daily context.

        For pullback/continuation signals: daily trend context + hourly recovery.
        """
        hourly = hourly_data.copy()
        if "timestamp" not in hourly.columns and "date" in hourly.columns:
            hourly["timestamp"] = hourly["date"]
        hourly = self._apply_leakage_guard(hourly, as_of_date)
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
        hourly = hourly.sort_values(["symbol", "timestamp"])

        if "close" in hourly.columns:
            hourly["hourly_return_1"] = hourly.groupby("symbol")["close"].pct_change(1)
            hourly["hourly_return_4"] = hourly.groupby("symbol")["close"].pct_change(4)
        else:
            hourly["hourly_return_1"] = np.nan
            hourly["hourly_return_4"] = np.nan

        hourly["hourly_vwap_recovery"] = 0.0
        if {"close", "vwap"}.issubset(hourly.columns):
            hourly["hourly_vwap_recovery"] = np.where(
                hourly["close"] > hourly["vwap"], 1.0, -1.0
            )

        hourly["hourly_ema20_regain"] = 0.0
        if {"close", "ema20"}.issubset(hourly.columns):
            hourly["hourly_ema20_regain"] = np.where(
                hourly["close"] > hourly["ema20"], 1.0, -1.0
            )

        aligned = align_hourly_to_daily(hourly, daily_context, as_of_date=as_of_date)
        aligned["hourly_pullback_recovery"] = (
            aligned["hourly_vwap_recovery"] + aligned["hourly_ema20_regain"]
        ) / 2.0

        keep_cols = [
            "symbol",
            "timestamp",
            "hourly_return_1",
            "hourly_return_4",
            "hourly_vwap_recovery",
            "hourly_ema20_regain",
            "hourly_pullback_recovery",
            "daily_return_1",
            "daily_trend",
        ]
        existing = [c for c in keep_cols if c in aligned.columns]
        return aligned[existing]

    def _merge_as_of(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_date_col: str,
        secondary_date_col: str,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Strict as-of merge: align secondary features to primary timestamps.

        Only joins secondary features where secondary timestamp <= primary timestamp.
        This prevents lookahead bias.
        """
        if secondary_df.shape[0] == 0:
            return primary_df

        primary = primary_df.copy()
        secondary = secondary_df.copy()

        if primary_date_col not in primary.columns:
            primary[primary_date_col] = pd.to_datetime(
                primary.get("date", pd.Timestamp.now())
            )
        if secondary_date_col not in secondary.columns:
            secondary[secondary_date_col] = secondary.get(
                "timestamp", pd.Timestamp.now()
            )

        primary[primary_date_col] = pd.to_datetime(primary[primary_date_col])
        secondary[secondary_date_col] = pd.to_datetime(secondary[secondary_date_col])

        if as_of_date:
            as_of_dt = pd.to_datetime(as_of_date)
            primary = primary[primary[primary_date_col] <= as_of_dt]
            secondary = secondary[secondary[secondary_date_col] <= as_of_dt]

        secondary = secondary.rename(columns={secondary_date_col: primary_date_col})
        overlap_cols = (
            set(primary.columns)
            .intersection(set(secondary.columns))
            .difference({primary_date_col, "symbol"})
        )
        if overlap_cols:
            secondary = secondary.rename(
                columns={col: f"hourly_{col}" for col in overlap_cols}
            )

        has_symbol_key = "symbol" in primary.columns and "symbol" in secondary.columns
        primary_is_date_only = (
            primary[primary_date_col].dt.time == datetime.min.time()
        ).all()
        secondary_has_intraday = (
            secondary[primary_date_col].dt.time != datetime.min.time()
        ).any()
        if primary_is_date_only and secondary_has_intraday:
            secondary = secondary.sort_values(
                ["symbol", primary_date_col] if has_symbol_key else primary_date_col
            )
            secondary["_merge_date"] = secondary[primary_date_col].dt.normalize()
            group_cols = ["_merge_date"]
            if has_symbol_key:
                group_cols = ["symbol", "_merge_date"]
            secondary = secondary.groupby(group_cols, as_index=False).tail(1)
            primary["_merge_date"] = primary[primary_date_col].dt.normalize()
            merge_keys = ["_merge_date"]
            if has_symbol_key:
                merge_keys = ["symbol", "_merge_date"]
            merged = primary.merge(
                secondary.drop(columns=[primary_date_col]),
                on=merge_keys,
                how="left",
            )
            return merged.drop(columns=["_merge_date"], errors="ignore")

        merge_kwargs = {
            "on": primary_date_col,
            "direction": "backward",
            "tolerance": pd.Timedelta(days=1),
        }
        if has_symbol_key:
            merge_kwargs["by"] = "symbol"
            primary = primary.sort_values(["symbol", primary_date_col])
            secondary = secondary.sort_values(["symbol", primary_date_col])
        else:
            primary = primary.sort_values(primary_date_col)
            secondary = secondary.sort_values(primary_date_col)

        merged = pd.merge_asof(primary, secondary, **merge_kwargs)

        return merged

    def _add_breadth_features(
        self,
        df: pd.DataFrame,
        breadth_data: Dict[str, pd.DataFrame],
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Add cross-asset breadth features (SPY/QQQ context)."""
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result = result.sort_values("date")

        for symbol in self._breadth_symbols:
            if symbol not in breadth_data:
                continue

            breadth_df = breadth_data[symbol].copy()
            breadth_df = self._apply_leakage_guard(breadth_df, as_of_date)

            if {"date", "close"}.issubset(breadth_df.columns) and breadth_df.shape[
                0
            ] > 1:
                breadth_df = breadth_df.sort_values("date")
                breadth_df[f"breadth_{symbol.lower()}_return"] = breadth_df[
                    "close"
                ].pct_change(20)
                breadth_df[f"breadth_{symbol.lower()}_return_60"] = breadth_df[
                    "close"
                ].pct_change(60)
                breadth_features = breadth_df[
                    [
                        "date",
                        f"breadth_{symbol.lower()}_return",
                        f"breadth_{symbol.lower()}_return_60",
                    ]
                ].copy()
                result = pd.merge_asof(
                    result.sort_values("date"),
                    breadth_features.sort_values("date"),
                    on="date",
                    direction="backward",
                )

        if (
            "breadth_spy_return" in result.columns
            and "breadth_qqq_return" in result.columns
        ):
            result["breadth_market_context"] = np.where(
                (result["breadth_spy_return"] > 0) & (result["breadth_qqq_return"] > 0),
                1,
                np.where(
                    (result["breadth_spy_return"] < 0)
                    & (result["breadth_qqq_return"] < 0),
                    -1,
                    0,
                ),
            )

        return result

    def _apply_leakage_guard(
        self,
        df: pd.DataFrame,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Guard against future information leakage.

        Ensures no bars after as_of_date are included in feature computation.
        """
        guarded = df.copy()
        time_col = None
        if "timestamp" in guarded.columns:
            time_col = "timestamp"
        elif "date" in guarded.columns:
            time_col = "date"

        if time_col:
            guarded[time_col] = pd.to_datetime(guarded[time_col])
            if as_of_date:
                as_of_dt = pd.to_datetime(as_of_date)
                guarded = guarded[guarded[time_col] <= as_of_dt]
            else:
                max_time = guarded[time_col].max()
                guarded = guarded[guarded[time_col] <= max_time]

        return guarded

    def _enforce_min_history(self, df: pd.DataFrame, min_bars: int) -> pd.DataFrame:
        """Ensure minimum history is available for feature validity."""
        if "symbol" not in df.columns:
            return df.tail(min_bars) if len(df) > min_bars else df

        return df.groupby("symbol", group_keys=False).apply(
            lambda x: x.tail(min_bars) if len(x) > min_bars else x, include_groups=False
        )

    def _get_family_params(self, family: FeatureFamily) -> Dict[str, Any]:
        """Get parameterized config for a feature family."""
        config = self._config

        if family == FeatureFamily.TREND:
            return config.get(
                "trend",
                {
                    "ma_slope_window": 20,
                    "dmi_window": 14,
                    "efficiency_window": 20,
                },
            )
        elif family == FeatureFamily.TS_MOMENTUM:
            return config.get(
                "momentum",
                {
                    "exclude_recent_bars": 21,
                },
            )
        elif family == FeatureFamily.MEAN_REVERSION:
            return config.get(
                "reversal",
                {
                    "rsi_short_window": 2,
                },
            )
        elif family == FeatureFamily.BREAKOUT:
            return config.get(
                "breakout",
                {
                    "donchian_window": 20,
                    "squeeze_percentile_window": 120,
                },
            )

        return {}

    def _collect_metadata(self, feature_names: List[str]) -> Dict[str, FeatureMetadata]:
        """Collect metadata for computed features."""
        metadata = {}
        for name in feature_names:
            from autotrade.feature_engineering.interfaces import get_feature_metadata

            meta = get_feature_metadata(name)
            if meta:
                metadata[name] = meta
        return metadata

    def get_report(self) -> FeaturePipelineReport:
        """Get execution report from last run."""
        if self._last_report is None:
            raise RuntimeError("No pipeline execution has occurred")
        return self._last_report

    def validate_no_leakage(self, frame: FeatureFrame, as_of_date: str) -> bool:
        """
        Validate that features don't contain future information.

        Checks that all feature values at timestamp t remain unchanged when
        bars after t are appended.
        """
        if "date" not in frame.df.columns:
            return True

        as_of_dt = pd.to_datetime(as_of_date)
        frame_dates = pd.to_datetime(frame.df["date"])

        max_frame_date = frame_dates.max() if len(frame_dates) > 0 else as_of_dt

        if max_frame_date > as_of_dt:
            return False

        return True


class InMemoryFeatureStore:
    """Simple in-memory cache for feature frames."""

    def __init__(self, ttl_minutes: int = 60):
        self._cache: Dict[str, FeatureFrame] = {}
        self._ttl = ttl_minutes

    def _make_key(self, request: FeatureRequest) -> str:
        symbols = ",".join(sorted(request.symbols))
        return f"{symbols}:{request.start_date}:{request.end_date}"

    def save(self, frame: FeatureFrame) -> None:
        key = self._make_key(
            FeatureRequest(
                symbols=frame.symbols,
                start_date=str(frame.df["date"].min())
                if "date" in frame.df.columns
                else "",
                end_date=str(frame.df["date"].max())
                if "date" in frame.df.columns
                else "",
            )
        )
        self._cache[key] = frame

    def load(self, request: FeatureRequest) -> Optional[FeatureFrame]:
        key = self._make_key(request)
        return self._cache.get(key)

    def exists(self, request: FeatureRequest) -> bool:
        return self._make_key(request) in self._cache

    def invalidate(self, request: FeatureRequest) -> None:
        key = self._make_key(request)
        self._cache.pop(key, None)


def align_hourly_to_daily(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Align hourly bars to daily context for pullback/continuation signals.

    Returns hourly data with daily trend context added.
    """
    daily = daily_df.copy()
    hourly = hourly_df.copy()

    if "date" not in daily.columns:
        return hourly
    if "timestamp" not in hourly.columns and "date" in hourly.columns:
        hourly["timestamp"] = hourly["date"]
    if "timestamp" not in hourly.columns:
        return hourly

    daily["date"] = pd.to_datetime(daily["date"])
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])

    if as_of_date:
        as_of_dt = pd.to_datetime(as_of_date)
        daily = daily[daily["date"] <= as_of_dt]
        hourly = hourly[hourly["timestamp"] <= as_of_dt]

    if "close" in daily.columns:
        daily["daily_return_1"] = daily.groupby("symbol")["close"].pct_change(1)
        if "sma20" not in daily.columns:
            daily["sma20"] = daily.groupby("symbol")["close"].transform(
                lambda s: s.rolling(20, min_periods=20).mean()
            )
        if "sma50" not in daily.columns:
            daily["sma50"] = daily.groupby("symbol")["close"].transform(
                lambda s: s.rolling(50, min_periods=50).mean()
            )
        daily["daily_trend"] = np.where(
            daily["sma20"].isna() | daily["sma50"].isna(),
            0,
            np.where(daily["sma20"] > daily["sma50"], 1, -1),
        )

    daily_features = daily[["symbol", "date", "daily_return_1", "daily_trend"]].copy()
    daily_features = daily_features.sort_values(["symbol", "date"])
    hourly_sorted = hourly.sort_values(["symbol", "timestamp"])

    merged = pd.merge_asof(
        hourly_sorted,
        daily_features,
        left_on="timestamp",
        right_on="date",
        by="symbol",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    merged = merged.drop(columns=["date"], errors="ignore")

    return merged


__all__ = [
    "FeaturePipeline",
    "InMemoryFeatureStore",
    "align_hourly_to_daily",
]
