"""
Sentiment analysis tool using Hugging Face Transformers (DistilBERT).
Converts news text → sentiment scores (bullish / bearish).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

logger = logging.getLogger(__name__)

# Lazy-load the pipeline so the module can be imported without GPU/model
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline  # type: ignore

            _pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,
            )
            logger.info("FinBERT sentiment pipeline loaded.")
        except Exception as exc:
            logger.warning(f"Could not load FinBERT ({exc}). Falling back to heuristic.")
            _pipeline = "heuristic"
    return _pipeline


def _heuristic_sentiment(text: str) -> Dict:
    """Simple keyword-based fallback when model is unavailable."""
    text_lower = text.lower()
    positive_words = {
        "surge", "gain", "rise", "rally", "beat", "profit", "growth",
        "bullish", "upgrade", "outperform", "record", "strong",
    }
    negative_words = {
        "drop", "fall", "loss", "decline", "miss", "bearish", "downgrade",
        "underperform", "weak", "crash", "plunge", "warning",
    }
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    total = pos + neg or 1
    if pos > neg:
        return {"label": "positive", "score": round(pos / total, 2)}
    elif neg > pos:
        return {"label": "negative", "score": round(neg / total, 2)}
    return {"label": "neutral", "score": 0.5}


def analyze_sentiment(text: str) -> Dict:
    """
    Analyze sentiment of a text snippet.

    Returns:
        {
            "label": "positive" | "negative" | "neutral",
            "score": float  # confidence 0-1
        }
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.5}

    pipe = _get_pipeline()

    if pipe == "heuristic":
        return _heuristic_sentiment(text)

    try:
        # FinBERT returns labels: positive / negative / neutral
        results = pipe(text[:512], truncation=True)
        # results is a list of lists when top_k=None
        scores = results[0] if isinstance(results[0], list) else results
        best = max(scores, key=lambda x: x["score"])
        return {"label": best["label"].lower(), "score": round(best["score"], 4)}
    except Exception as exc:
        logger.error(f"Sentiment inference failed: {exc}")
        return _heuristic_sentiment(text)
