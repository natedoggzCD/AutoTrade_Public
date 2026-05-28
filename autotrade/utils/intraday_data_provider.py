"""
Intraday Data Provider - Alpaca primary + yfinance fallback.
=============================================================
Provides a single ``get_intraday_bars`` function that tries Alpaca first
and falls back to yfinance when Alpaca fails or returns no data.

Also provides ``get_current_price_with_fallback`` for price lookups.

Used by the day manager and VWAP universe scanner during market hours.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("AutoTrade.IntradayProvider")


# H1 upstream hardening: wall-clock timeout on every blocking provider
# call (Alpaca SDK get_stock_bars, yfinance history, MCP mcp_alpaca).
# Pre-fix these calls had no explicit socket timeout — a single hung
# call could block the day_manager candidate-scoring executor for 60+
# minutes (the 2026-05-27 09:02 CDT incident). The day_manager-side
# per-symbol budget (autotrade/core/day_manager.py ~L19592) bounds the
# symptom; this wrapper bounds the upstream call itself.
try:
    _PROVIDER_CALL_TIMEOUT_S = float(
        os.environ.get("AUTOTRADE_INTRADAY_PROVIDER_TIMEOUT_S", "10")
    )
except ValueError:
    _PROVIDER_CALL_TIMEOUT_S = 10.0
_PROVIDER_CALL_TIMEOUT_S = max(2.0, _PROVIDER_CALL_TIMEOUT_S)


class ProviderCallTimeout(RuntimeError):
    """Raised when a provider call exceeds its wall-clock timeout."""


def _call_with_timeout(
    func: Callable[..., Any],
    *args,
    _timeout_s: Optional[float] = None,
    _label: str = "provider_call",
    **kwargs,
) -> Any:
    """Run *func* under a wall-clock timeout.

    Uses a short-lived single-worker ThreadPoolExecutor. On timeout we
    raise ProviderCallTimeout and call executor.shutdown(wait=False,
    cancel_futures=True) — the underlying worker thread cannot be
    cancelled mid-call (the SDK is synchronous), but it becomes orphan
    and the caller is no longer blocked. The OS-level TCP timeout
    will eventually reclaim the orphan.
    """
    timeout = float(_timeout_s if _timeout_s is not None else _PROVIDER_CALL_TIMEOUT_S)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="provider-to")
    try:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.warning(
                "[PROVIDER-TIMEOUT] %s exceeded %.1fs wall-clock — abandoning",
                _label,
                timeout,
            )
            raise ProviderCallTimeout(
                f"{_label} timed out after {timeout:.1f}s"
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

# yfinance availability check (done once at import time)
_YF_AVAILABLE = False
try:
    import yfinance as yf

    _YF_AVAILABLE = True
except ImportError:
    logger.info("yfinance not installed - Alpaca-only intraday data")


def get_intraday_bars(
    ticker: str,
    data_client,
    minutes_back: int = 240,
    interval: str = "1m",
    *,
    alpaca_timeframe=None,
    min_bars: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday minute bars for *ticker*.

    Tries Alpaca ``data_client.get_stock_bars`` first. If that fails or
    returns insufficient data, falls back to yfinance ``download()``.

    Args:
        ticker: Stock symbol.
        data_client: Alpaca ``StockHistoricalDataClient`` (can be None).
        minutes_back: How many minutes of history to fetch.
        interval: Bar interval string for yfinance (``"1m"``, ``"5m"``).
        alpaca_timeframe: Optional Alpaca ``TimeFrame`` object.
        min_bars: Minimum bar count required for a successful fetch.

    Returns:
        DataFrame with lowercase columns ``open, high, low, close, volume``
        (index is datetime), or *None* on total failure.
    """
    min_required = max(1, int(min_bars))

    df = _try_alpaca_then_mcp(
        ticker=ticker,
        data_client=data_client,
        minutes_back=minutes_back,
        alpaca_timeframe=alpaca_timeframe,
        interval=interval,
    )
    if df is not None and len(df) >= min_required:
        return df

    # Alpaca failed or insufficient data - try yfinance with requested interval.
    df = _try_yfinance(ticker, minutes_back, interval)
    if df is not None and len(df) >= min_required:
        return df

    # When minute bars are sparse/unavailable, fall back to 5m bars.
    if interval == "1m":
        fallback_minutes = max(minutes_back, 120)
        df = _try_alpaca_then_mcp(
            ticker=ticker,
            data_client=data_client,
            minutes_back=fallback_minutes,
            alpaca_timeframe=_resolve_alpaca_timeframe("5m"),
            interval="5m",
        )
        if df is not None and len(df) >= min_required:
            return df

        df = _try_yfinance(ticker, fallback_minutes, "5m")
        if df is not None and len(df) >= min_required:
            return df

    logger.debug(
        "No intraday bars available for %s (minutes_back=%s, interval=%s, min_bars=%s)",
        ticker,
        minutes_back,
        interval,
        min_required,
    )
    return None


def get_current_price_with_fallback(
    ticker: str,
    data_client=None,
) -> Tuple[Optional[float], str]:
    """
    Return ``(price, source)`` trying Alpaca quote first, yfinance second.

    Source is ``"alpaca"``, ``"yfinance"``, or ``"none"``.
    """
    # --- Alpaca ---
    if data_client is not None:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest

            quote = _call_with_timeout(
                data_client.get_stock_latest_quote,
                StockLatestQuoteRequest(symbol_or_symbols=ticker),
                _label=f"alpaca.get_stock_latest_quote[{ticker}]",
            )

            # SDK response shape
            if isinstance(quote, dict) and ticker in quote and hasattr(quote[ticker], "ask_price"):
                ask = quote[ticker].ask_price
                bid = quote[ticker].bid_price
                if ask and ask > 0:
                    return float(ask), "alpaca"
                if bid and bid > 0:
                    return float(bid), "alpaca"
                if ask and bid:
                    return float((ask + bid) / 2), "alpaca"

            # MCP response shape (best-effort)
            if isinstance(quote, dict):
                ask = quote.get("ask_price") or quote.get("ask")
                bid = quote.get("bid_price") or quote.get("bid")
                if ask and float(ask) > 0:
                    return float(ask), "alpaca"
                if bid and float(bid) > 0:
                    return float(bid), "alpaca"
                if ask and bid:
                    return float((float(ask) + float(bid)) / 2.0), "alpaca"
        except Exception as e:
            logger.debug("Alpaca quote failed for %s: %s", ticker, e)

    # --- yfinance ---
    if _YF_AVAILABLE:
        try:
            def _fetch_yf_quote(symbol: str):
                stock = yf.Ticker(symbol)
                info = stock.fast_info
                last = getattr(info, "last_price", None)
                if last:
                    return float(last)
                prev = getattr(info, "previous_close", None)
                if prev:
                    return float(prev)
                return None

            price = _call_with_timeout(
                _fetch_yf_quote, ticker, _label=f"yfinance.fast_info[{ticker}]"
            )
            if price is not None:
                return price, "yfinance"
        except ProviderCallTimeout:
            pass
        except Exception as e:
            logger.debug("yfinance price failed for %s: %s", ticker, e)

    return None, "none"


def get_intraday_bars_batch(
    tickers: List[str],
    data_client,
    minutes_back: int = 240,
    interval: str = "1m",
    *,
    max_batch: int = 20,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch intraday bars for multiple tickers.

    Uses Alpaca multi-symbol request first, then fills gaps with yfinance.

    Returns:
        Dict mapping ticker -> bars DataFrame.
    """
    result: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []

    # --- Alpaca batch ---
    if data_client is not None and tickers:
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            start = datetime.now() - timedelta(minutes=minutes_back)
            # Alpaca supports multi-symbol but cap batch size
            for i in range(0, len(tickers), max_batch):
                batch = tickers[i : i + max_batch]
                # Batch calls get a proportional timeout (per-symbol-call * batch
                # size) since a batch fetch legitimately takes longer than a
                # single-symbol call.
                batch_timeout = _PROVIDER_CALL_TIMEOUT_S * max(1, len(batch))
                try:
                    bars = _call_with_timeout(
                        data_client.get_stock_bars,
                        StockBarsRequest(
                            symbol_or_symbols=batch,
                            timeframe=TimeFrame.Minute,
                            start=start,
                            end=datetime.now(),
                        ),
                        _timeout_s=batch_timeout,
                        _label=f"alpaca.get_stock_bars.batch[n={len(batch)}]",
                    )
                except ProviderCallTimeout:
                    missing.extend(batch)
                    continue
                if hasattr(bars, "df") and bars.df is not None and not bars.df.empty:
                    for t in batch:
                        try:
                            if t in bars.df.index.get_level_values(0):
                                df = bars.df.loc[t].copy()
                                df.columns = [c.lower() for c in df.columns]
                                if len(df) >= 5:
                                    result[t] = df
                                else:
                                    missing.append(t)
                            else:
                                missing.append(t)
                        except Exception:
                            missing.append(t)
                else:
                    missing.extend(batch)
        except Exception as e:
            logger.warning("Alpaca batch bars failed: %s", e)
            missing = tickers[:]

    else:
        missing = tickers[:]

    # --- yfinance fallback for missing ---
    if missing and _YF_AVAILABLE:
        for t in missing:
            df = _try_yfinance(t, minutes_back, interval)
            if df is not None and len(df) >= 5:
                result[t] = df

    return result


# ------------------------------------------------------------------
#  Internal helpers
# ------------------------------------------------------------------


def _resolve_alpaca_timeframe(interval: str):
    """Map textual interval to Alpaca TimeFrame object."""
    try:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception:
        return None

    if interval == "1m":
        return TimeFrame.Minute
    if interval == "5m":
        return TimeFrame(5, TimeFrameUnit.Minute)
    if interval == "15m":
        return TimeFrame(15, TimeFrameUnit.Minute)
    return None


def _normalize_symbol_bars_df(bars_obj, ticker: str) -> Optional[pd.DataFrame]:
    """Extract a ticker's bars DataFrame from Alpaca SDK response objects."""
    if bars_obj is None or not hasattr(bars_obj, "df") or bars_obj.df is None:
        return None
    if bars_obj.df.empty:
        return None
    if ticker not in bars_obj.df.index.get_level_values(0):
        return None
    df = bars_obj.df.loc[ticker].copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def _is_mcp_proxy_client(data_client) -> bool:
    return str(getattr(data_client, "server_name", "")).lower() == "alpaca"


def _fetch_bars_via_kwargs(
    data_client,
    ticker: str,
    timeframe,
    start_ts: datetime,
    end_ts: datetime,
) -> Optional[pd.DataFrame]:
    """
    MCP proxy compatibility: submit kwargs instead of SDK request object.
    """
    try:
        bars = _call_with_timeout(
            data_client.get_stock_bars,
            symbol=ticker,
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start_ts,
            end=end_ts,
            _label=f"alpaca.get_stock_bars.kwargs[{ticker}]",
        )
    except TypeError:
        return None
    except ProviderCallTimeout:
        return None
    except Exception:
        return None
    return _normalize_symbol_bars_df(bars, ticker)


def _try_alpaca_mcp_fallback(
    ticker: str,
    minutes_back: int,
    interval: str,
) -> Optional[pd.DataFrame]:
    """
    Direct MCP fallback for intraday bars when SDK-style calls fail.
    """
    try:
        from autotrade.utils.mcp_client import mcp_alpaca
    except Exception:
        return None

    tf = "1Min" if interval == "1m" else "5Min"
    start = datetime.now() - timedelta(minutes=minutes_back)
    end = datetime.now()
    try:
        payload = _call_with_timeout(
            mcp_alpaca,
            "get_stock_bars",
            symbol=ticker.upper(),
            timeframe=tf,
            start=start.isoformat(),
            end=end.isoformat(),
            limit=max(20, int(minutes_back)),
            _label=f"mcp_alpaca.get_stock_bars[{ticker}]",
        )
    except ProviderCallTimeout:
        return None

    if not payload:
        return None
    if isinstance(payload, dict) and payload.get("error"):
        return None

    rows = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), dict):
            rows = payload["data"].get(ticker.upper())
        if rows is None:
            rows = payload.get("bars") or payload.get("results") or payload.get("data")
    if not isinstance(rows, list) or not rows:
        return None

    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = row.get("timestamp") or row.get("t") or row.get("time")
        open_px = row.get("open") if "open" in row else row.get("o")
        high_px = row.get("high") if "high" in row else row.get("h")
        low_px = row.get("low") if "low" in row else row.get("l")
        close_px = row.get("close") if "close" in row else row.get("c")
        volume = row.get("volume") if "volume" in row else row.get("v")
        if (
            ts is None
            or open_px is None
            or high_px is None
            or low_px is None
            or close_px is None
        ):
            continue
        normalized_rows.append(
            {
                "timestamp": ts,
                "open": float(open_px),
                "high": float(high_px),
                "low": float(low_px),
                "close": float(close_px),
                "volume": float(volume or 0.0),
            }
        )

    if not normalized_rows:
        return None

    df = pd.DataFrame(normalized_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if df.empty:
        return None
    return df[["open", "high", "low", "close", "volume"]]


def _try_alpaca_then_mcp(
    ticker: str,
    data_client,
    minutes_back: int,
    alpaca_timeframe=None,
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """Try SDK Alpaca request first, then MCP-compatible fallbacks."""
    df = _try_alpaca(ticker, data_client, minutes_back, alpaca_timeframe)
    if df is not None and not df.empty:
        return df

    if not _is_mcp_proxy_client(data_client):
        return None

    start = datetime.now() - timedelta(minutes=minutes_back)
    end = datetime.now()
    tf = alpaca_timeframe or _resolve_alpaca_timeframe(interval)
    kwargs_df = _fetch_bars_via_kwargs(
        data_client=data_client,
        ticker=ticker,
        timeframe=tf,
        start_ts=start,
        end_ts=end,
    )
    if kwargs_df is not None and not kwargs_df.empty:
        return kwargs_df

    return _try_alpaca_mcp_fallback(
        ticker=ticker,
        minutes_back=minutes_back,
        interval=interval,
    )


def _try_alpaca(
    ticker: str,
    data_client,
    minutes_back: int,
    alpaca_timeframe=None,
) -> Optional[pd.DataFrame]:
    """Try fetching bars from Alpaca."""
    if data_client is None:
        return None
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf = alpaca_timeframe or TimeFrame.Minute
        start = datetime.now() - timedelta(minutes=minutes_back)
        bars = _call_with_timeout(
            data_client.get_stock_bars,
            StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=tf,
                start=start,
                end=datetime.now(),
            ),
            _label=f"alpaca.get_stock_bars[{ticker}]",
        )
        return _normalize_symbol_bars_df(bars, ticker)
    except ProviderCallTimeout:
        return None
    except Exception as e:
        logger.debug("Alpaca bars failed for %s: %s", ticker, e)
        return None


def _try_yfinance(
    ticker: str,
    minutes_back: int,
    interval: str = "1m",
) -> Optional[pd.DataFrame]:
    """Fallback: fetch intraday bars from yfinance."""
    if not _YF_AVAILABLE:
        return None
    try:
        # yfinance wants period strings for intraday
        if minutes_back <= 60:
            period = "1d"
        elif minutes_back <= 120:
            period = "1d"
        elif minutes_back <= 390:
            period = "1d"
        elif minutes_back <= 780:
            period = "2d"
        else:
            period = "5d"

        stock = yf.Ticker(ticker)
        try:
            df = _call_with_timeout(
                stock.history,
                period=period,
                interval=interval,
                _label=f"yfinance.history[{ticker}]",
            )
        except ProviderCallTimeout:
            return None
        if df is None or df.empty:
            return None

        # Normalize columns to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Ensure we have the required columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None

        # Keep only market-hours data (filter to today if period > 1d)
        if period in {"2d", "5d"}:
            today = datetime.now().date()
            if hasattr(df.index, "date"):
                df = df[df.index.date == today]

        logger.debug(
            "yfinance intraday: %s -> %d bars (%s, %s)",
            ticker,
            len(df),
            period,
            interval,
        )
        return df[["open", "high", "low", "close", "volume"]]

    except Exception as e:
        logger.debug("yfinance intraday failed for %s: %s", ticker, e)
        return None
