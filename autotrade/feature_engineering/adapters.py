"""
Feature Engineering Adapters
=============================

Compatibility adapters that wrap the shared feature layer while preserving
existing module signatures and outputs.

These adapters allow existing modules (UniverseScanner, ScreenerV2, Backtest)
to consume the centralized feature layer without breaking their public APIs.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from autotrade.feature_engineering.pipeline import FeaturePipeline
from autotrade.feature_engineering.schemas import (
    FeatureFamily,
    FeatureFrame,
    FeatureRequest,
    FeatureMetadata,
)
from autotrade.feature_engineering.trend import TrendFeatureBuilder
from autotrade.feature_engineering.momentum import MomentumFeatureBuilder
from autotrade.feature_engineering.reversal import ReversalFeatureBuilder
from autotrade.feature_engineering.breakout import BreakoutFeatureBuilder
from autotrade.feature_engineering.volatility_volume import (
    VolatilityVolumeFeatureBuilder,
)
from autotrade.feature_engineering.regime import RegimeFeatureBuilder

logger = logging.getLogger(__name__)


class BaseFeatureAdapter:
    """Base class for feature adapters with common functionality."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._pipeline = None
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the feature pipeline with default builders."""
        try:
            self._pipeline = FeaturePipeline(
                config=self._config,
                breadth_enabled=self._config.get("breadth_enabled", True),
                breadth_symbols=self._config.get("breadth_symbols", ["SPY", "QQQ"]),
            )
        except Exception as e:
            logger.warning(f"Feature pipeline initialization failed: {e}")
            self._pipeline = None

    def compute_features(
        self,
        df: pd.DataFrame,
        families: Optional[List[FeatureFamily]] = None,
    ) -> pd.DataFrame:
        """Compute features using the shared pipeline."""
        if self._pipeline is None:
            logger.warning("Feature pipeline not available, returning input dataframe")
            return df

        try:
            prepared_df, start_date, end_date, symbols = self._prepare_pipeline_input(df)
            if prepared_df.empty:
                return df

            request = FeatureRequest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                families=families or self._pipeline.families_enabled,
                as_of_date=end_date,
            )
            frame = self._pipeline.execute(request, prepared_df)
            if frame.df is None or frame.df.empty:
                return df

            out = frame.df.copy()
            if "symbol" in out.columns and "ticker" not in out.columns:
                out["ticker"] = out["symbol"]

            # Preserve the original column naming expected by callers.
            if "ticker" in df.columns and "symbol" in out.columns:
                out = out.drop(columns=["symbol"], errors="ignore")

            return out
        except Exception as e:
            logger.warning(f"Feature computation failed: {e}")
            return df

    def _prepare_pipeline_input(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, str, str, List[str]]:
        """Normalize caller data into the pipeline's expected symbol/date schema."""
        prepared = df.copy()

        if "ticker" in prepared.columns:
            ticker_series = prepared["ticker"].astype(str).str.upper()
            if "symbol" in prepared.columns:
                has_symbol = prepared["symbol"].notna() & (
                    prepared["symbol"].astype(str).str.strip() != ""
                )
                if not bool(has_symbol.any()):
                    prepared = prepared.drop(columns=["symbol"], errors="ignore")
                    prepared.insert(len(prepared.columns), "symbol", ticker_series)
            else:
                prepared.insert(len(prepared.columns), "symbol", ticker_series)

        if "date" not in prepared.columns:
            for candidate in ("Date", "datetime", "timestamp", "Timestamp"):
                if candidate in prepared.columns:
                    prepared["date"] = prepared[candidate]
                    break

        if "date" not in prepared.columns:
            if isinstance(prepared.index, pd.DatetimeIndex):
                prepared["date"] = prepared.index
            else:
                today = pd.Timestamp(date.today())
                prepared["date"] = today

        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        valid_dates = prepared["date"].dropna()

        if valid_dates.empty:
            today_str = date.today().isoformat()
            prepared["date"] = pd.Timestamp(today_str)
            start_date = today_str
            end_date = today_str
        else:
            start_date = valid_dates.min().date().isoformat()
            end_date = valid_dates.max().date().isoformat()

        symbols: List[str] = []
        if "symbol" in prepared.columns:
            symbols = [
                symbol
                for symbol in prepared["symbol"].dropna().astype(str).str.upper().unique().tolist()
                if symbol and symbol.strip()
            ]
        if not symbols and "ticker" in prepared.columns:
            symbols = [
                symbol
                for symbol in prepared["ticker"].dropna().astype(str).str.upper().unique().tolist()
                if symbol and symbol.strip()
            ]

        return prepared, start_date, end_date, symbols


class UniverseScannerFeatureAdapter(BaseFeatureAdapter):
    """
    Adapter for UniverseScanner to use shared feature layer.

    Preserves existing UniverseScanner.scan() signature while internally
    leveraging the centralized feature transforms.

    Compatible output columns:
    - momentum_score, volume_score, volatility_score, trend_score, support_score
    - s1_price, s1_strength, r1_price, r1_strength, atr_percent_db
    - All existing scan result columns
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_pipeline_features: bool = True,
    ):
        super().__init__(config)
        self._enable_pipeline_features = enable_pipeline_features
        self._feature_column_mapping = self._build_column_mapping()

    def _build_column_mapping(self) -> Dict[str, str]:
        """Build mapping from canonical feature names to scanner expected names."""
        return {
            "return_12_1": "momentum_12_1",
            "return_6_1": "momentum_6_1",
            "atr_percent": "atr_percent_db",
            "rsi_2": "rsi_2",
            "rsi_14": "rsi_14",
            "sma50": "sma50",
            "sma200": "sma200",
            "sma5_slope": "sma_slope",
            "volume_ratio": "volume_ratio",
            "bb_position": "bb_position",
            "rank_6_1": "rank_6_1",
            "rank_12_1": "rank_12_1",
        }

    def enrich_scan_results(
        self,
        df: pd.DataFrame,
        feature_families: Optional[List[FeatureFamily]] = None,
    ) -> pd.DataFrame:
        """
        Enrich DuckDB scan results with centralized features.

        Args:
            df: DataFrame with existing scan results from UniverseScanner
            feature_families: Optional list of feature families to compute

        Returns:
            DataFrame enriched with canonical features while preserving original columns
        """
        if df.empty:
            return df

        if not self._enable_pipeline_features or self._pipeline is None:
            return df

        families = feature_families or [
            FeatureFamily.TS_MOMENTUM,
            FeatureFamily.VOLATILITY,
            FeatureFamily.VOLUME,
            FeatureFamily.MEAN_REVERSION,
        ]

        try:
            enriched = self.compute_features(df, families)

            for canonical, scanner_name in self._feature_column_mapping.items():
                if canonical in enriched.columns and scanner_name not in df.columns:
                    df[scanner_name] = enriched[canonical]

            return df

        except Exception as e:
            logger.warning(f"Feature enrichment failed: {e}, returning original data")
            return df

    def get_features_for_symbols(
        self,
        symbols: List[str],
        as_of_date: str,
        lookback_bars: int = 252,
    ) -> pd.DataFrame:
        """Get canonical features for a list of symbols."""
        if self._pipeline is None:
            return pd.DataFrame()

        try:
            request = FeatureRequest(
                symbols=symbols,
                start_date=as_of_date,
                end_date=as_of_date,
                families=[
                    FeatureFamily.TREND,
                    FeatureFamily.TS_MOMENTUM,
                    FeatureFamily.VOLATILITY,
                    FeatureFamily.VOLUME,
                    FeatureFamily.MEAN_REVERSION,
                ],
                as_of_date=as_of_date,
                lookback_bars=lookback_bars,
            )
            frame = self._pipeline.execute(request, pd.DataFrame())
            return frame.df if frame.df is not None else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Feature fetch failed: {e}")
            return pd.DataFrame()

    def compute_technical_features(
        self,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute technical features from OHLCV data for scanner compatibility."""
        if ohlcv_df.empty:
            return ohlcv_df

        result = ohlcv_df.copy()

        try:
            trend_builder = TrendFeatureBuilder()
            if trend_builder.can_compute(ohlcv_df):
                trend_features = trend_builder.compute(ohlcv_df)
                for col in trend_features.columns:
                    if col not in result.columns:
                        result[col] = trend_features[col]

            vol_builder = VolatilityVolumeFeatureBuilder()
            if vol_builder.can_compute(ohlcv_df):
                vol_features = vol_builder.compute(ohlcv_df)
                for col in vol_features.columns:
                    if col not in result.columns:
                        result[col] = vol_features[col]

            reversal_builder = ReversalFeatureBuilder()
            if reversal_builder.can_compute(ohlcv_df):
                reversal_features = reversal_builder.compute(ohlcv_df)
                for col in reversal_features.columns:
                    if col not in result.columns:
                        result[col] = reversal_features[col]

        except Exception as e:
            logger.warning(f"Technical feature computation failed: {e}")

        return result


class ScreenerV2FeatureAdapter(BaseFeatureAdapter):
    """
    Adapter for ScreenerV2 to use shared feature layer.

    Preserves existing get_entry_candidates() signature while internally
    leveraging centralized feature transforms.

    Compatible output columns (preserved from original):
    - ticker, date, close, volume
    - sma20, ema20, rsi_14, macd, macd_signal, macd_hist
    - bb_mid, bb_upper, bb_lower, stoch_k, stoch_d
    - atr_14, roc_5, roc_10
    - All composite scores and rankings
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_pipeline_features: bool = True,
    ):
        super().__init__(config)
        self._enable_pipeline_features = enable_pipeline_features
        self._feature_column_mapping = self._build_column_mapping()

    def _build_column_mapping(self) -> Dict[str, str]:
        """Build mapping from canonical feature names to screener expected names."""
        return {
            "sma20": "sma20",
            "ema20": "ema20",
            "rsi_14": "rsi_14",
            "macd": "macd",
            "macd_signal": "macd_signal",
            "macd_hist": "macd_hist",
            "bb_mid": "bb_mid",
            "bb_upper": "bb_upper",
            "bb_lower": "bb_lower",
            "atr_14": "atr_14",
            "return_6_1": "return_6_1",
            "return_12_1": "return_12_1",
            "sma50": "sma50",
            "sma200": "sma200",
            "volume_ratio": "volume_ratio",
            "efficiency_ratio": "efficiency_ratio",
            "adx": "adx",
        }

    def enrich_candidates(
        self,
        df: pd.DataFrame,
        feature_families: Optional[List[FeatureFamily]] = None,
    ) -> pd.DataFrame:
        """
        Enrich screener candidates with centralized features.

        Args:
            df: DataFrame with existing screener results
            feature_families: Optional list of feature families to compute

        Returns:
            DataFrame enriched with canonical features while preserving original columns
        """
        if df.empty:
            return df

        if not self._enable_pipeline_features or self._pipeline is None:
            return df

        families = feature_families or [
            FeatureFamily.TREND,
            FeatureFamily.TS_MOMENTUM,
            FeatureFamily.VOLATILITY,
            FeatureFamily.MEAN_REVERSION,
            FeatureFamily.BREAKOUT,
        ]

        try:
            enriched = self.compute_features(df, families)

            for canonical, screener_name in self._feature_column_mapping.items():
                if canonical in enriched.columns and screener_name not in df.columns:
                    df[screener_name] = enriched[canonical]

            for col in enriched.columns:
                if (
                    col not in df.columns
                    and col not in self._feature_column_mapping.values()
                ):
                    df[col] = enriched[col]

            return df

        except Exception as e:
            logger.warning(f"Feature enrichment failed: {e}, returning original data")
            return df

    def compute_candidate_features(
        self,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute features specifically needed for screener candidates."""
        if ohlcv_df.empty:
            return ohlcv_df

        result = ohlcv_df.copy()

        try:
            trend_builder = TrendFeatureBuilder()
            if trend_builder.can_compute(ohlcv_df):
                trend_features = trend_builder.compute(ohlcv_df)
                for col in trend_features.columns:
                    if col not in result.columns:
                        result[col] = trend_features[col]

            momentum_builder = MomentumFeatureBuilder()
            if momentum_builder.can_compute(ohlcv_df):
                momentum_features = momentum_builder.compute(ohlcv_df)
                for col in momentum_features.columns:
                    if col not in result.columns:
                        result[col] = momentum_features[col]

            vol_builder = VolatilityVolumeFeatureBuilder()
            if vol_builder.can_compute(ohlcv_df):
                vol_features = vol_builder.compute(ohlcv_df)
                for col in vol_features.columns:
                    if col not in result.columns:
                        result[col] = vol_features[col]

            reversal_builder = ReversalFeatureBuilder()
            if reversal_builder.can_compute(ohlcv_df):
                reversal_features = reversal_builder.compute(ohlcv_df)
                for col in reversal_features.columns:
                    if col not in result.columns:
                        result[col] = reversal_features[col]

            breakout_builder = BreakoutFeatureBuilder()
            if breakout_builder.can_compute(ohlcv_df):
                breakout_features = breakout_builder.compute(ohlcv_df)
                for col in breakout_features.columns:
                    if col not in result.columns:
                        result[col] = breakout_features[col]

        except Exception as e:
            logger.warning(f"Candidate feature computation failed: {e}")

        return result


class BacktestFeatureAdapter(BaseFeatureAdapter):
    """
    Adapter for backtest modules to use shared feature layer.

    Preserves existing backtest signatures while enabling use of
    centralized feature transforms for OHLCV-derived features.

    Compatible with:
    - strategy_backtester.fetch_ohlcv()
    - duckdb_backtester.DuckDBBacktester.backtest_signal_strategy()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_pipeline_features: bool = True,
    ):
        super().__init__(config)
        self._enable_pipeline_features = enable_pipeline_features
        self._feature_column_mapping = self._build_column_mapping()

    def _build_column_mapping(self) -> Dict[str, str]:
        """Build mapping from canonical feature names to backtest expected names."""
        return {
            "sma20": "sma_20",
            "sma50": "sma_50",
            "sma200": "sma_200",
            "ema20": "ema_20",
            "ema50": "ema_50",
            "rsi_14": "rsi_14",
            "rsi_2": "rsi_2",
            "macd": "macd",
            "macd_signal": "macd_signal",
            "macd_hist": "macd_hist",
            "bb_mid": "bb_mid",
            "bb_upper": "bb_upper",
            "bb_lower": "bb_lower",
            "bb_width": "bb_width",
            "bb_position": "bb_position",
            "atr_14": "atr_14",
            "atr_percent": "atr_percent",
            "return_6_1": "return_6_1",
            "return_12_1": "return_12_1",
            "volume_ratio": "volume_ratio",
            "obv": "obv",
            "vwap": "vwap",
            "efficiency_ratio": "efficiency_ratio",
            "adx": "adx",
        }

    def enrich_backtest_data(
        self,
        df: pd.DataFrame,
        feature_families: Optional[List[FeatureFamily]] = None,
    ) -> pd.DataFrame:
        """
        Enrich backtest OHLCV data with centralized features.

        Args:
            df: DataFrame with OHLCV data for backtesting
            feature_families: Optional list of feature families to compute

        Returns:
            DataFrame enriched with canonical features while preserving OHLCV columns
        """
        if df.empty:
            return df

        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.warning(
                f"Missing required columns for feature enrichment: {missing}"
            )
            return df

        if not self._enable_pipeline_features or self._pipeline is None:
            return df

        families = feature_families or [
            FeatureFamily.TREND,
            FeatureFamily.TS_MOMENTUM,
            FeatureFamily.VOLATILITY,
            FeatureFamily.VOLUME,
            FeatureFamily.MEAN_REVERSION,
            FeatureFamily.BREAKOUT,
        ]

        try:
            enriched = self.compute_features(df, families)

            for canonical, backtest_name in self._feature_column_mapping.items():
                if canonical in enriched.columns and backtest_name not in df.columns:
                    df[backtest_name] = enriched[canonical]

            for col in enriched.columns:
                if (
                    col not in df.columns
                    and col not in self._feature_column_mapping.values()
                ):
                    df[col] = enriched[col]

            return df

        except Exception as e:
            logger.warning(f"Feature enrichment failed: {e}, returning original data")
            return df

    def compute_ohlcv_features(
        self,
        ohlcv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute all features needed for backtesting from OHLCV data."""
        if ohlcv_df.empty:
            return ohlcv_df

        result = ohlcv_df.copy()

        try:
            trend_builder = TrendFeatureBuilder()
            if trend_builder.can_compute(ohlcv_df):
                trend_features = trend_builder.compute(ohlcv_df)
                for col in trend_features.columns:
                    if col not in result.columns:
                        result[col] = trend_features[col]

            momentum_builder = MomentumFeatureBuilder()
            if momentum_builder.can_compute(ohlcv_df):
                momentum_features = momentum_builder.compute(ohlcv_df)
                for col in momentum_features.columns:
                    if col not in result.columns:
                        result[col] = momentum_features[col]

            vol_builder = VolatilityVolumeFeatureBuilder()
            if vol_builder.can_compute(ohlcv_df):
                vol_features = vol_builder.compute(ohlcv_df)
                for col in vol_features.columns:
                    if col not in result.columns:
                        result[col] = vol_features[col]

            reversal_builder = ReversalFeatureBuilder()
            if reversal_builder.can_compute(ohlcv_df):
                reversal_features = reversal_builder.compute(ohlcv_df)
                for col in reversal_features.columns:
                    if col not in result.columns:
                        result[col] = reversal_features[col]

            breakout_builder = BreakoutFeatureBuilder()
            if breakout_builder.can_compute(ohlcv_df):
                breakout_features = breakout_builder.compute(ohlcv_df)
                for col in breakout_features.columns:
                    if col not in result.columns:
                        result[col] = breakout_features[col]

            regime_builder = RegimeFeatureBuilder()
            if regime_builder.can_compute(ohlcv_df):
                regime_features = regime_builder.compute(ohlcv_df)
                for col in regime_features.columns:
                    if col not in result.columns:
                        result[col] = regime_features[col]

        except Exception as e:
            logger.warning(f"OHLCV feature computation failed: {e}")

        return result


class DuckDBBacktestFeatureAdapter(BacktestFeatureAdapter):
    """
    Specialized adapter for DuckDBBacktester.

    Handles DuckDB-specific data flow while using shared features.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_pipeline_features: bool = True,
    ):
        super().__init__(config, enable_pipeline_features)

    def enrich_duckdb_results(
        self,
        df: pd.DataFrame,
        as_of_column: str = "date",
    ) -> pd.DataFrame:
        """
        Enrich DuckDB query results with features, respecting as-of semantics.

        Args:
            df: DataFrame with DuckDB query results
            as_of_column: Name of the date/timestamp column for leakage checks

        Returns:
            DataFrame with features computed up to as-of date
        """
        if df.empty:
            return df

        if as_of_column not in df.columns:
            logger.warning(
                f"As-of column '{as_of_column}' not found, skipping feature enrichment"
            )
            return df

        return self.enrich_backtest_data(df)


def get_universe_scanner_adapter(
    config: Optional[Dict[str, Any]] = None,
) -> UniverseScannerFeatureAdapter:
    """Factory function to get a configured UniverseScanner adapter."""
    return UniverseScannerFeatureAdapter(config=config)


def get_screener_v2_adapter(
    config: Optional[Dict[str, Any]] = None,
) -> ScreenerV2FeatureAdapter:
    """Factory function to get a configured ScreenerV2 adapter."""
    return ScreenerV2FeatureAdapter(config=config)


def get_backtest_adapter(
    config: Optional[Dict[str, Any]] = None,
) -> BacktestFeatureAdapter:
    """Factory function to get a configured Backtest adapter."""
    return BacktestFeatureAdapter(config=config)


__all__ = [
    "BaseFeatureAdapter",
    "UniverseScannerFeatureAdapter",
    "ScreenerV2FeatureAdapter",
    "BacktestFeatureAdapter",
    "DuckDBBacktestFeatureAdapter",
    "get_universe_scanner_adapter",
    "get_screener_v2_adapter",
    "get_backtest_adapter",
]
