"""
Backtest tool – calls the C++ strategy engine binary via subprocess.
Falls back to a Python-based mock backtester when the binary is absent.

Fix 6: Feeds real OHLCV data from yfinance into the C++ binary, with
optional Redis/Upstash caching (TTL 300s). Graceful fallback to mock
presets when yfinance or the binary is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BACKTESTER_BIN = os.getenv("BACKTESTER_BIN", "./strategy_engine/bin/backtest")

# ---------------------------------------------------------------------------
# Redis helper (optional – skip all caching if UPSTASH_REDIS_URL is unset)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_checked = False


def _get_redis():
    """Lazy-init Redis client. Returns None if unavailable."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    url = os.getenv("UPSTASH_REDIS_URL")
    if not url:
        logger.info("UPSTASH_REDIS_URL not set – Redis caching disabled.")
        return None
    try:
        import redis as redis_lib  # type: ignore

        _redis_client = redis_lib.from_url(url, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis connection established for price caching.")
    except Exception as exc:
        logger.warning(f"Redis connection failed ({exc}) – caching disabled.")
        _redis_client = None
    return _redis_client


# ---------------------------------------------------------------------------
# Price data (yfinance + Redis cache)
# ---------------------------------------------------------------------------


def _fetch_prices_sync(symbol: str, period: str = "6mo") -> List[Dict]:
    """
    Synchronous yfinance download.  Called via run_in_executor so it
    never blocks the event loop.

    Returns list of {date, open, high, low, close, volume} dicts.
    """
    try:
        import yfinance as yf  # type: ignore

        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            logger.warning(f"yfinance returned empty data for {symbol}/{period}.")
            return []

        # Handle MultiIndex columns from yfinance
        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            df = df.droplevel("Ticker", axis=1)

        records: List[Dict] = []
        for ts, row in df.iterrows():
            records.append(
                {
                    "date": ts.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": int(row["Volume"]),
                }
            )
        return records
    except ImportError:
        logger.warning("yfinance not installed – cannot fetch real prices.")
        return []
    except Exception as exc:
        logger.error(f"yfinance fetch failed for {symbol}: {exc}")
        return []


async def get_price_data(symbol: str, period: str = "6mo") -> List[Dict]:
    """
    Get OHLCV price data for *symbol*.

    1. Check Redis cache  (key ``prices:{symbol}:{period}``, TTL 300 s)
    2. On miss → yfinance download (in executor)
    3. Cache result if Redis is available
    4. Return ``[]`` on any failure — never raise.
    """
    cache_key = f"prices:{symbol}:{period}"

    # --- cache read ---
    r = _get_redis()
    if r is not None:
        try:
            cached = r.get(cache_key)
            if cached:
                logger.debug(f"Redis cache HIT for {cache_key}")
                return json.loads(cached)
        except Exception as exc:
            logger.warning(f"Redis read error: {exc}")

    # --- fetch ---
    loop = asyncio.get_running_loop()
    prices = await loop.run_in_executor(None, _fetch_prices_sync, symbol, period)

    # --- cache write ---
    if prices and r is not None:
        try:
            r.setex(cache_key, 300, json.dumps(prices))
        except Exception as exc:
            logger.warning(f"Redis write error: {exc}")

    return prices


# ---------------------------------------------------------------------------
# Current quote helper
# ---------------------------------------------------------------------------


def _fetch_quote_sync(symbol: str) -> Dict:
    """Synchronous fast quote via yfinance Ticker.fast_info."""
    try:
        import yfinance as yf  # type: ignore

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = float(info.get("lastPrice", 0) or info.get("last_price", 0))
        prev = float(info.get("previousClose", 0) or info.get("previous_close", 0))
        change_pct = ((price - prev) / prev * 100) if prev else 0
        volume = int(info.get("lastVolume", 0) or info.get("last_volume", 0))
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
        }
    except Exception as exc:
        logger.error(f"Quote fetch failed for {symbol}: {exc}")
        return {"symbol": symbol, "price": 0, "change_pct": 0, "volume": 0}


async def get_current_quote(symbol: str) -> Dict:
    """
    Current price quote with Redis cache (TTL 300 s).
    Falls back to zeroed dict on any error.
    """
    cache_key = f"quote:{symbol}"
    r = _get_redis()
    if r is not None:
        try:
            cached = r.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    quote = await loop.run_in_executor(None, _fetch_quote_sync, symbol)

    if r is not None and quote.get("price", 0) > 0:
        try:
            r.setex(cache_key, 300, json.dumps(quote))
        except Exception:
            pass

    return quote


# ---------------------------------------------------------------------------
# execute_backtest  (updated for real OHLCV data)
# ---------------------------------------------------------------------------


async def execute_backtest(
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
    params: Dict | None = None,
) -> Dict:
    """
    Execute a backtest via the C++ binary with real OHLCV data.

    Falls back to mock presets when:
    - yfinance returns no data
    - C++ binary not found
    - subprocess fails
    """
    params = params or {}

    # --- Attempt to fetch real prices ---
    prices = await get_price_data(symbol, period="6mo")

    if not os.path.isfile(BACKTESTER_BIN):
        logger.warning(
            f"C++ backtester binary not found at '{BACKTESTER_BIN}'. "
            "Using Python mock backtester."
        )
        return _mock_backtest(strategy, symbol, start_date, end_date, params)

    if not prices:
        logger.warning(
            f"No price data for {symbol} – falling back to mock backtest."
        )
        return _mock_backtest(strategy, symbol, start_date, end_date, params)

    # --- Write input + price data to temp files ---
    backtest_input = {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "parameters": params,
    }

    input_file = None
    data_file = None
    output_file = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(backtest_input, f)
            input_file = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_data.json", delete=False
        ) as f:
            json.dump(prices, f)
            data_file = f.name

        output_file = input_file.replace(".json", "_output.json")

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                [BACKTESTER_BIN, input_file, output_file, "--data", data_file],
                timeout=30,
                capture_output=True,
            ),
        )

        if result.returncode != 0:
            err = result.stderr.decode()
            logger.error(f"Backtester error: {err}")
            return _mock_backtest(strategy, symbol, start_date, end_date, params)

        with open(output_file, "r") as f:
            results = json.load(f)

        return {
            "strategy": strategy,
            "symbol": symbol,
            "total_return": results.get("total_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "total_trades": results.get("total_trades", 0),
            "data_points": len(prices),
            "mock": False,
        }

    except subprocess.TimeoutExpired:
        logger.error("Backtester timed out after 30 seconds.")
        return _mock_backtest(strategy, symbol, start_date, end_date, params)
    except Exception as exc:
        logger.error(f"Backtest execution error: {exc}")
        return _mock_backtest(strategy, symbol, start_date, end_date, params)
    finally:
        for path in [input_file, data_file, output_file]:
            if path and os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Python mock backtester (deterministic, used when binary/data is absent)
# ---------------------------------------------------------------------------


def _mock_backtest(
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
    params: Dict,
) -> Dict:
    """
    Deterministic mock backtest results keyed by strategy type.
    Values are plausible but synthetic – for demo / development only.
    """
    presets = {
        "momentum": {
            "total_return": 0.23,
            "sharpe_ratio": 1.42,
            "max_drawdown": -0.08,
            "win_rate": 0.61,
            "total_trades": 48,
        },
        "mean_reversion": {
            "total_return": 0.15,
            "sharpe_ratio": 1.10,
            "max_drawdown": -0.06,
            "win_rate": 0.57,
            "total_trades": 72,
        },
        "macd": {
            "total_return": 0.19,
            "sharpe_ratio": 1.25,
            "max_drawdown": -0.10,
            "win_rate": 0.55,
            "total_trades": 35,
        },
    }

    base = presets.get(strategy, presets["momentum"])
    return {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        **base,
        "mock": True,
    }
