"""
News scraping tool – fetches financial news from NewsAPI.org
and enriches each article with a sentiment score.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

import httpx

from .sentiment import analyze_sentiment

logger = logging.getLogger(__name__)

NEWS_API_BASE = "https://newsapi.org/v2/everything"


async def scrape_financial_news(
    symbol: str,
    sentiment_filter: Optional[str] = None,
    page_size: int = 50,
) -> Dict:
    """
    Fetch and enrich financial news for a given stock symbol.

    Args:
        symbol:           Stock ticker, e.g. "AAPL"
        sentiment_filter: Optional – "positive", "negative", or None (all)
        page_size:        Number of articles to request from NewsAPI

    Returns:
        {
            "symbol": str,
            "total_articles": int,
            "articles": [...],
            "aggregate_sentiment": {"positive": int, "negative": int, "neutral": int},
            "timestamp": str
        }
    """
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        logger.warning("NEWS_API_KEY not set – returning mock news data.")
        return _mock_news(symbol)

    params = {
        "q": f"{symbol} stock market",
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(NEWS_API_BASE, params=params)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.error(f"NewsAPI request failed: {exc}")
        return _mock_news(symbol)

    articles = data.get("articles", [])
    enriched: List[Dict] = []
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


# ---------------------------------------------------------------------------
# Mock data – used when NEWS_API_KEY is absent (demo / CI)
# ---------------------------------------------------------------------------

def _mock_news(symbol: str) -> Dict:
    mock_articles = [
        {
            "title": f"{symbol} reports record quarterly earnings",
            "source": "Reuters",
            "published_at": datetime.utcnow().isoformat(),
            "url": "https://reuters.com",
            "sentiment": "positive",
            "confidence": 0.92,
        },
        {
            "title": f"Analysts upgrade {symbol} to Buy amid strong guidance",
            "source": "Bloomberg",
            "published_at": datetime.utcnow().isoformat(),
            "url": "https://bloomberg.com",
            "sentiment": "positive",
            "confidence": 0.88,
        },
        {
            "title": f"Market volatility weighs on {symbol} shares",
            "source": "CNBC",
            "published_at": datetime.utcnow().isoformat(),
            "url": "https://cnbc.com",
            "sentiment": "negative",
            "confidence": 0.75,
        },
    ]
    return {
        "symbol": symbol,
        "total_articles": len(mock_articles),
        "articles": mock_articles,
        "aggregate_sentiment": {"positive": 2, "negative": 1, "neutral": 0},
        "timestamp": datetime.utcnow().isoformat(),
    }
