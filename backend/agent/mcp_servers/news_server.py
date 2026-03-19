"""
MCP News Server for FinAgent.

Exposes two tools via the Model Context Protocol:
  1. get_financial_news   – fetch & return recent articles from NewsAPI
  2. get_market_sentiment_summary – quick headline-based sentiment snapshot

Runnable standalone: python -m agent.mcp_servers.news_server
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2/everything"

# Keyword lists for lightweight headline sentiment (no model needed)
_BULLISH = {
    "surge", "gain", "rise", "rally", "beat", "profit", "growth", "bullish",
    "upgrade", "outperform", "record", "strong", "soar", "boom", "jump",
}
_BEARISH = {
    "drop", "fall", "loss", "decline", "miss", "bearish", "downgrade",
    "underperform", "weak", "crash", "plunge", "warning", "slump", "sink",
}


# ---------------------------------------------------------------------------
# Core tool functions (usable directly OR via FastMCP)
# ---------------------------------------------------------------------------


async def get_financial_news(symbol: str, limit: int = 10) -> List[Dict]:
    """
    Fetch recent financial news articles for *symbol* from NewsAPI.

    Returns list of {title, description, publishedAt, source} dicts.
    Falls back to empty list if API key missing or request fails.
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        logger.warning("NEWS_API_KEY not set – returning empty news list.")
        return []

    params = {
        "q": f"{symbol} stock earnings",
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": min(limit, 100),
        "apiKey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(NEWS_API_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.error(f"NewsAPI request failed for {symbol}: {exc}")
        return []

    articles = data.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "publishedAt": a.get("publishedAt", ""),
            "source": a.get("source", {}).get("name", "Unknown"),
        }
        for a in articles[:limit]
    ]


async def get_market_sentiment_summary(symbol: str) -> Dict:
    """
    Quick headline-based sentiment snapshot for *symbol*.

    Fetches the 5 most recent articles and classifies sentiment using
    keyword matching (no ML model needed).

    Returns {symbol, article_count, avg_sentiment_hint, latest_headline}.
    """
    articles = await get_financial_news(symbol, limit=5)

    if not articles:
        return {
            "symbol": symbol,
            "article_count": 0,
            "avg_sentiment_hint": "neutral",
            "latest_headline": "",
        }

    bullish_count = 0
    bearish_count = 0

    for a in articles:
        text = (a.get("title", "") + " " + a.get("description", "")).lower()
        b = sum(1 for w in _BULLISH if w in text)
        s = sum(1 for w in _BEARISH if w in text)
        if b > s:
            bullish_count += 1
        elif s > b:
            bearish_count += 1

    if bullish_count > bearish_count:
        hint = "positive"
    elif bearish_count > bullish_count:
        hint = "negative"
    else:
        hint = "neutral"

    return {
        "symbol": symbol,
        "article_count": len(articles),
        "avg_sentiment_hint": hint,
        "latest_headline": articles[0].get("title", ""),
    }


# ---------------------------------------------------------------------------
# FastMCP server (standalone mode)
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore

    mcp = FastMCP("finagent-news-server")

    @mcp.tool()
    async def mcp_get_financial_news(symbol: str, limit: int = 10) -> list[dict]:
        """Fetch recent financial news for a stock symbol."""
        return await get_financial_news(symbol, limit)

    @mcp.tool()
    async def mcp_get_market_sentiment(symbol: str) -> dict:
        """Get a quick headline-based sentiment summary for a stock symbol."""
        return await get_market_sentiment_summary(symbol)

except ImportError:
    mcp = None
    logger.debug("mcp package not installed – FastMCP server unavailable.")


# Legacy singleton for backward compatibility
class NewsMCPServer:
    name = "news-mcp-server"
    scrape_market_news = staticmethod(get_financial_news)
    get_sentiment_summary = staticmethod(get_market_sentiment_summary)


news_server = NewsMCPServer()

if __name__ == "__main__":
    if mcp is not None:
        mcp.run(transport="stdio")
    else:
        print("mcp package not installed. pip install mcp")
