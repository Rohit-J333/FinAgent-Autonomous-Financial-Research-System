"""
Risk Agent — fetches 1-year OHLCV for the symbol and SPY benchmark,
computes risk metrics using quantstats, and produces a RiskAnalysis.

Fallback chain:
  yfinance → empty DataFrame
  quantstats → manual numpy/pandas calculations
  < 100 data points → RiskAnalysis(source="fallback")
  Any exception → RiskAnalysis(source="fallback")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Tuple

from agent.specialist_agents.shared_state import RiskAnalysis

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
# Price fetching (same yfinance pattern)
# ---------------------------------------------------------------------------


def _fetch_returns_sync(symbol: str, period: str = "1y"):
    """
    Fetch OHLCV and compute daily returns.
    Returns a pandas Series of daily returns, or None.
    """
    try:
        import yfinance as yf  # type: ignore
        import pandas as pd  # type: ignore

        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            logger.warning(f"yfinance returned empty data for {symbol}/{period}.")
            return None

        # Handle MultiIndex columns
        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            df = df.droplevel("Ticker", axis=1)

        returns = df["Close"].pct_change().dropna()
        returns.name = symbol
        return returns
    except ImportError:
        logger.warning("yfinance not installed.")
        return None
    except Exception as exc:
        logger.error(f"yfinance fetch failed for {symbol}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Risk metric computation (CPU-bound → run_in_executor)
# ---------------------------------------------------------------------------


def _compute_risk_sync(symbol: str, sym_returns, spy_returns) -> RiskAnalysis:
    """
    Compute risk metrics from daily return series.
    Uses quantstats if available, falls back to manual numpy calculations.
    """
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    if sym_returns is None or len(sym_returns) < 100:
        return _fallback(symbol)

    # --- Try quantstats first ---
    sharpe = sortino = max_dd = None
    try:
        import quantstats as qs  # type: ignore

        sharpe = float(qs.stats.sharpe(sym_returns))
        sortino = float(qs.stats.sortino(sym_returns))
        max_dd = float(qs.stats.max_drawdown(sym_returns)) * 100
    except ImportError:
        logger.info("quantstats not installed — using manual calculations.")
    except Exception as exc:
        logger.warning(f"quantstats failed for {symbol}: {exc}, using manual calcs.")

    # --- Manual fallback for missing values ---
    returns_arr = sym_returns.values.astype(float)

    if sharpe is None:
        mean_r = np.mean(returns_arr)
        std_r = np.std(returns_arr, ddof=1)
        sharpe = float((mean_r / std_r) * (252 ** 0.5)) if std_r > 0 else 0.0

    if sortino is None:
        mean_r = np.mean(returns_arr)
        downside = returns_arr[returns_arr < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
        sortino = float((mean_r / down_std) * (252 ** 0.5))

    if max_dd is None:
        cumulative = (1 + sym_returns).cumprod()
        running_max = cumulative.cummax()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(drawdowns.min()) * 100

    # --- VaR / CVaR ---
    var_95 = float(np.percentile(returns_arr, 5)) * 100
    cvar_mask = returns_arr <= np.percentile(returns_arr, 5)
    cvar_95 = float(np.mean(returns_arr[cvar_mask])) * 100 if cvar_mask.any() else var_95

    # --- Volatility ---
    volatility_annualized = float(np.std(returns_arr, ddof=1) * (252 ** 0.5) * 100)

    # --- Beta vs SPY ---
    beta = 1.0
    if spy_returns is not None and len(spy_returns) > 0:
        try:
            # Align both series on date index
            aligned = pd.concat([sym_returns, spy_returns], axis=1, join="inner").dropna()
            if len(aligned) > 50:
                sym_arr = aligned.iloc[:, 0].values.astype(float)
                spy_arr = aligned.iloc[:, 1].values.astype(float)
                cov = np.cov(sym_arr, spy_arr)[0][1]
                var_spy = np.var(spy_arr, ddof=1)
                beta = float(cov / var_spy) if var_spy > 0 else 1.0
        except Exception as exc:
            logger.warning(f"Beta calculation failed for {symbol}: {exc}")

    # --- Risk rating and position sizing ---
    vol = volatility_annualized
    dd = abs(max_dd)

    if vol < 15 and dd < 20:
        risk_rating = "LOW"
        position_size_pct = 10.0
    elif vol < 25 and dd < 35:
        risk_rating = "MEDIUM"
        position_size_pct = 7.0
    elif vol < 40:
        risk_rating = "HIGH"
        position_size_pct = 4.0
    else:
        risk_rating = "VERY_HIGH"
        position_size_pct = 2.0

    # Handle NaN/inf
    def safe(v: float, default: float = 0.0) -> float:
        if v != v or abs(v) == float("inf"):  # NaN or inf check
            return default
        return round(v, 4)

    return RiskAnalysis(
        symbol=symbol,
        sharpe_ratio=safe(sharpe),
        sortino_ratio=safe(sortino),
        max_drawdown_pct=safe(max_dd),
        var_95=safe(var_95),
        cvar_95=safe(cvar_95),
        volatility_annualized=safe(volatility_annualized, 25.0),
        beta_vs_spy=safe(beta, 1.0),
        risk_rating=risk_rating,
        position_size_pct=position_size_pct,
        source="quantstats_live",
    )


# ---------------------------------------------------------------------------
# Per-symbol analysis
# ---------------------------------------------------------------------------


async def _analyze_single_symbol(symbol: str) -> RiskAnalysis:
    """Fetch returns and compute risk metrics for one symbol."""
    try:
        loop = asyncio.get_running_loop()

        # Fetch symbol and SPY returns in parallel
        sym_ret, spy_ret = await asyncio.gather(
            loop.run_in_executor(None, _fetch_returns_sync, symbol, "1y"),
            loop.run_in_executor(None, _fetch_returns_sync, "SPY", "1y"),
        )

        if sym_ret is None or len(sym_ret) < 100:
            logger.warning(f"Insufficient return data for {symbol}.")
            return _fallback(symbol)

        # Compute metrics (CPU-bound)
        result = await loop.run_in_executor(None, _compute_risk_sync, symbol, sym_ret, spy_ret)
        return result

    except Exception as exc:
        logger.error(f"Risk agent failed for {symbol}: {exc}")
        return _fallback(symbol)


def _fallback(symbol: str) -> RiskAnalysis:
    """Complete fallback when data is insufficient or unavailable."""
    return RiskAnalysis(
        symbol=symbol,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown_pct=0.0,
        var_95=0.0,
        cvar_95=0.0,
        volatility_annualized=25.0,
        beta_vs_spy=1.0,
        risk_rating="MEDIUM",
        position_size_pct=5.0,
        source="fallback",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@traceable(name="risk_agent")
async def run_risk_agent(symbols: List[str]) -> List[RiskAnalysis]:
    """
    Run risk analysis for all symbols in parallel.

    Never raises — every symbol gets a result (live or fallback).
    """
    t0 = time.perf_counter()
    logger.info(f"[risk_agent] Starting analysis for {symbols}")

    tasks = [_analyze_single_symbol(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    analyses: List[RiskAnalysis] = []
    for i, res in enumerate(results):
        sym = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"[risk_agent] {sym} raised: {res}")
            analyses.append(_fallback(sym))
        else:
            analyses.append(res)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[risk_agent] Completed {len(symbols)} symbols in {elapsed:.0f}ms")
    return analyses
