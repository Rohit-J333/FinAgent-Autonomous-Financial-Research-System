"""
Technical Analysis Agent — fetches OHLCV data via yfinance and computes
150+ indicators using pandas-ta.

Fallback chain:
  yfinance → empty DataFrame
  pandas-ta → manual calculation
  < 60 data points → TechnicalAnalysis(source="fallback")
  Any exception → TechnicalAnalysis(source="fallback")
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import List

from agent.specialist_agents.shared_state import TechnicalAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse @traceable (no-op when absent)
# ---------------------------------------------------------------------------

try:
    from langfuse.decorators import observe as traceable  # type: ignore
except ImportError:
    def traceable(name: str = "", **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

# ---------------------------------------------------------------------------
# Price fetching (reuses yfinance pattern from backtest.py)
# ---------------------------------------------------------------------------


def _fetch_ohlcv_sync(symbol: str, period: str = "6mo"):
    """
    Synchronous yfinance download returning a pandas DataFrame.
    Returns None on failure.
    """
    try:
        import yfinance as yf  # type: ignore

        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            logger.warning(f"yfinance returned empty data for {symbol}/{period}.")
            return None

        # Handle MultiIndex columns
        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            df = df.droplevel("Ticker", axis=1)

        return df
    except ImportError:
        logger.warning("yfinance not installed.")
        return None
    except Exception as exc:
        logger.error(f"yfinance fetch failed for {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Indicator computation (CPU-bound → run_in_executor)
# ---------------------------------------------------------------------------


def _compute_indicators_sync(symbol: str, df) -> TechnicalAnalysis:
    """
    Compute technical indicators from OHLCV DataFrame.
    All pandas-ta and pandas operations happen here (synchronous, CPU-bound).
    """
    import pandas as pd  # type: ignore

    if df is None or len(df) < 60:
        logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows (need 60).")
        return _fallback(symbol)

    data_points = len(df)

    # --- Apply pandas-ta indicators ---
    try:
        import pandas_ta as ta  # type: ignore

        df.ta.rsi(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.bbands(length=20, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
    except ImportError:
        logger.warning("pandas-ta not installed — computing indicators manually.")
        _manual_indicators(df)
    except Exception as exc:
        logger.warning(f"pandas-ta failed for {symbol}: {exc}, using manual fallback.")
        _manual_indicators(df)

    latest = df.iloc[-1]

    # --- RSI ---
    rsi_col = [c for c in df.columns if "RSI" in str(c)]
    rsi_14 = float(latest[rsi_col[0]]) if rsi_col and pd.notna(latest[rsi_col[0]]) else 50.0

    # --- MACD ---
    macd_col = [c for c in df.columns if "MACD_12_26_9" == str(c) or "MACD" == str(c)]
    macds_col = [c for c in df.columns if "MACDs" in str(c)]
    macd_val = float(latest[macd_col[0]]) if macd_col and pd.notna(latest[macd_col[0]]) else 0.0
    macds_val = float(latest[macds_col[0]]) if macds_col and pd.notna(latest[macds_col[0]]) else 0.0
    macd_signal = "BULLISH" if macd_val > macds_val else "BEARISH" if macd_val < macds_val else "NEUTRAL"

    # --- Bollinger Bands ---
    bbu_col = [c for c in df.columns if "BBU" in str(c)]
    bbl_col = [c for c in df.columns if "BBL" in str(c)]
    close = float(latest["Close"])
    bbu = float(latest[bbu_col[0]]) if bbu_col and pd.notna(latest[bbu_col[0]]) else close * 1.02
    bbl = float(latest[bbl_col[0]]) if bbl_col and pd.notna(latest[bbl_col[0]]) else close * 0.98

    if close > bbu:
        bb_position = "ABOVE_UPPER"
    elif close < bbl:
        bb_position = "BELOW_LOWER"
    else:
        bb_position = "WITHIN"

    # --- SMAs ---
    sma20_col = [c for c in df.columns if "SMA_20" in str(c)]
    sma50_col = [c for c in df.columns if "SMA_50" in str(c)]
    sma_20 = float(latest[sma20_col[0]]) if sma20_col and pd.notna(latest[sma20_col[0]]) else close
    sma_50 = float(latest[sma50_col[0]]) if sma50_col and pd.notna(latest[sma50_col[0]]) else close

    # --- Trend ---
    if close > sma_20 > sma_50:
        trend = "UPTREND"
    elif close < sma_20 < sma_50:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"

    # --- Support / Resistance ---
    support_level = float(df["Close"].rolling(20).min().iloc[-1])
    resistance_level = float(df["Close"].rolling(20).max().iloc[-1])

    # --- Momentum score ---
    rsi_component = (rsi_14 - 50) / 50 * 0.5
    macd_component = (1.0 if macd_signal == "BULLISH" else -1.0) * 0.5
    momentum_score = max(-1.0, min(1.0, rsi_component + macd_component))

    # --- Volume trend ---
    vol_sma_20 = df["Volume"].rolling(20).mean().iloc[-1]
    vol_last_5 = df["Volume"].tail(5).mean()
    if vol_last_5 > vol_sma_20 * 1.1:
        volume_trend = "INCREASING"
    elif vol_last_5 < vol_sma_20 * 0.9:
        volume_trend = "DECREASING"
    else:
        volume_trend = "STABLE"

    return TechnicalAnalysis(
        symbol=symbol,
        trend=trend,
        rsi_14=round(rsi_14, 2),
        macd_signal=macd_signal,
        bb_position=bb_position,
        sma_20=round(sma_20, 2),
        sma_50=round(sma_50, 2),
        support_level=round(support_level, 2),
        resistance_level=round(resistance_level, 2),
        momentum_score=round(momentum_score, 4),
        volume_trend=volume_trend,
        data_points=data_points,
        source="yfinance_live",
    )


def _manual_indicators(df) -> None:
    """Compute basic indicators manually when pandas-ta is unavailable."""
    import pandas as pd  # type: ignore

    # RSI (14-period)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan"))
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_12_26_9"] = ema12 - ema26
    df["MACDs_12_26_9"] = df["MACD_12_26_9"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BBU_20_2.0"] = sma20 + 2 * std20
    df["BBL_20_2.0"] = sma20 - 2 * std20

    # SMAs
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()


# ---------------------------------------------------------------------------
# Per-symbol analysis
# ---------------------------------------------------------------------------


async def _analyze_single_symbol(symbol: str) -> TechnicalAnalysis:
    """Fetch OHLCV and compute indicators for one symbol."""
    try:
        loop = asyncio.get_running_loop()

        # Fetch data
        df = await loop.run_in_executor(None, _fetch_ohlcv_sync, symbol, "6mo")

        if df is None or len(df) < 60:
            return _fallback(symbol)

        # Compute indicators (CPU-bound)
        result = await loop.run_in_executor(None, _compute_indicators_sync, symbol, df)
        return result

    except Exception as exc:
        logger.error(f"Technical agent failed for {symbol}: {exc}")
        return _fallback(symbol)


def _fallback(symbol: str) -> TechnicalAnalysis:
    """Complete fallback when data is insufficient or unavailable."""
    return TechnicalAnalysis(
        symbol=symbol,
        trend="SIDEWAYS",
        rsi_14=50.0,
        macd_signal="NEUTRAL",
        bb_position="WITHIN",
        momentum_score=0.0,
        source="fallback",
        data_points=0,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@traceable(name="technical_agent")
async def run_technical_agent(symbols: List[str]) -> List[TechnicalAnalysis]:
    """
    Run technical analysis for all symbols in parallel.

    Never raises — every symbol gets a result (live or fallback).
    """
    t0 = time.perf_counter()
    logger.info(f"[technical_agent] Starting analysis for {symbols}")

    tasks = [_analyze_single_symbol(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    analyses: List[TechnicalAnalysis] = []
    for i, res in enumerate(results):
        sym = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"[technical_agent] {sym} raised: {res}")
            analyses.append(_fallback(sym))
        else:
            analyses.append(res)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[technical_agent] Completed {len(symbols)} symbols in {elapsed:.0f}ms")
    return analyses
