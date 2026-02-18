"""
MCP-style News Server.

Exposes a `scrape_market_news` tool that the LangGraph agent can call
via the MCP protocol. This module also works as a standalone FastAPI
sub-application for testing.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from agent.tools.sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2/everything"


class NewsMCPServer:
    """
    Lightweight MCP-compatible news server.

    In a full MCP deployment this would be launched as a separate process
    and communicate over stdio/SSE. Here we expose the core logic as
    async methods that the orchestrator can call directly.
    """

    name = "news-mcp-server"

    async def scrape_market_news(
        self,
        symbol: str,
        sentiment_filter: Optional[str] = None,
        page_size: int = 50,
    ) -> dict:
        """
        Fetch financial news for *symbol* and enrich with sentiment scores.

        Args:
            symbol:           Stock ticker, e.g. "AAPL"
            sentiment_filter: "positive" | "negative" | None (all)
            page_size:        Articles to request from NewsAPI

        Returns:
            {
                "symbol": str,
                "total_articles": int,
                "articles": [...],
                "aggregate_sentiment": {...},
                "timestamp": str
            }
        """
        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key:
            logger.warning("NEWS_API_KEY not set – returning mock data.")
            return self._mock_response(symbol)

        params = {
            "q": f"{symbol} stock market",
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": page_size,
            "apiKey": api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(NEWS_API_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.error(f"NewsAPI error: {exc}")
            return self._mock_response(symbol)

        articles = data.get("articles", [])
        enriched = []
        aggregate = {"positive": 0, "negative": 0, "neutral": 0}

        for article in articles:
            text = article.get("description") or article.get("title") or ""
            sentiment = analyze_sentiment(text)
            label = sentiment["label"]

            if sentiment_filter and label != sentiment_filter:
                continue

            aggregate[label] = aggregate.get(label, 0) + 1
            enriched.append(
                {
                    "title": article.get("title", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                    "sentiment": label,
                    "confidence": sentiment["score"],
                }
            )

        return {
            "symbol": symbol,
            "total_articles": len(enriched),
            "articles": enriched[:20],
            "aggregate_sentiment": aggregate,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @staticmethod
    def _mock_response(symbol: str) -> dict:
        return {
            "symbol": symbol,
            "total_articles": 3,
            "articles": [
                {
                    "title": f"{symbol} beats earnings expectations",
                    "source": "Reuters",
                    "published_at": datetime.utcnow().isoformat(),
                    "url": "",
                    "sentiment": "positive",
                    "confidence": 0.91,
                },
                {
                    "title": f"Analysts raise {symbol} price target",
                    "source": "Bloomberg",
                    "published_at": datetime.utcnow().isoformat(),
                    "url": "",
                    "sentiment": "positive",
                    "confidence": 0.87,
                },
                {
                    "title": f"Macro headwinds could pressure {symbol}",
                    "source": "CNBC",
                    "published_at": datetime.utcnow().isoformat(),
                    "url": "",
                    "sentiment": "negative",
                    "confidence": 0.72,
                },
            ],
            "aggregate_sentiment": {"positive": 2, "negative": 1, "neutral": 0},
            "timestamp": datetime.utcnow().isoformat(),
        }


# Singleton
news_server = NewsMCPServer()
