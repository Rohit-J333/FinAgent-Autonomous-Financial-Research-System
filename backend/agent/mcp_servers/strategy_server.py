"""
MCP Strategy Server for FinAgent.

Exposes two tools via the Model Context Protocol:
  1. run_momentum_backtest – execute a momentum backtest for a symbol
  2. get_technical_signals  – compute SMA/momentum indicators from price data

Runnable standalone: python -m agent.mcp_servers.strategy_server
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core tool functions (usable directly OR via FastMCP)
# ---------------------------------------------------------------------------


async def run_momentum_backtest(symbol: str, lookback_days: int = 90) -> Dict:
    """
    Run a momentum backtest for *symbol* using the C++ binary or mock fallback.
    """
    from agent.tools.backtest import execute_backtest

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%d"
    )
    return await execute_backtest(
        strategy="momentum",
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        params={"lookback_period": 20},
    )


async def get_technical_signals(symbol: str) -> Dict:
    """
    Compute basic technical signals from 3-month price data.

    Returns {symbol, sma_20, sma_50, trend, momentum_pct, data_points}.
    Falls back to error dict if insufficient data (< 50 points).
    """
    from agent.tools.backtest import get_price_data

    prices = await get_price_data(symbol, period="3mo")

    if len(prices) < 50:
        return {"symbol": symbol, "error": "insufficient data"}

    closes = [p["close"] for p in prices]

    # 20-day SMA
    sma_20 = sum(closes[-20:]) / 20
    # 50-day SMA
    sma_50 = sum(closes[-50:]) / 50

    latest = closes[-1]

    # Trend classification
    if latest > sma_20 and latest > sma_50:
        trend = "ABOVE_BOTH"
    elif latest < sma_20 and latest < sma_50:
        trend = "BELOW_BOTH"
    else:
        trend = "MIXED"

    # Simple momentum: % change over 20 days
    close_20_ago = closes[-21] if len(closes) > 20 else closes[0]
    momentum_pct = ((latest / close_20_ago) - 1) * 100 if close_20_ago else 0

    return {
        "symbol": symbol,
        "sma_20": round(sma_20, 2),
        "sma_50": round(sma_50, 2),
        "trend": trend,
        "momentum_pct": round(momentum_pct, 2),
        "data_points": len(prices),
    }


# ---------------------------------------------------------------------------
# FastMCP server (standalone mode)
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore

    mcp = FastMCP("finagent-strategy-server")

    @mcp.tool()
    async def mcp_run_momentum_backtest(
        symbol: str, lookback_days: int = 90
    ) -> dict:
        """Run a momentum backtest for a stock symbol."""
        return await run_momentum_backtest(symbol, lookback_days)

    @mcp.tool()
    async def mcp_get_technical_signals(symbol: str) -> dict:
        """Get SMA, trend, and momentum signals for a stock symbol."""
        return await get_technical_signals(symbol)

except ImportError:
    mcp = None
    logger.debug("mcp package not installed – FastMCP server unavailable.")


# Legacy singleton for backward compatibility
class StrategyMCPServer:
    name = "strategy-mcp-server"
    backtest_strategy = staticmethod(run_momentum_backtest)
    get_signals = staticmethod(get_technical_signals)


strategy_server = StrategyMCPServer()

if __name__ == "__main__":
    if mcp is not None:
        mcp.run(transport="stdio")
    else:
        print("mcp package not installed. pip install mcp")
