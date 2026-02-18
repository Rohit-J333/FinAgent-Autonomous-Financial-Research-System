"""
RAG Retriever – high-level interface for the FastAPI backend.
Combines vector search with LLM-based answer generation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from rag.vectordb import FinancialVectorDB

logger = logging.getLogger(__name__)


class FinancialRetriever:
    """
    Retrieves relevant document chunks for a query and formats them
    for use as LLM context.
    """

    def __init__(self, vectordb: FinancialVectorDB):
        self.vectordb = vectordb

    def retrieve(
        self,
        query: str,
        symbol: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict:
        """
        Retrieve the most relevant chunks for *query*.

        Returns:
            {
                "query": str,
                "symbol": str | None,
                "results": [...],
                "context": str   # concatenated text for LLM prompt
            }
        """
        results = self.vectordb.hybrid_search(query, symbol=symbol, top_k=top_k)

        context = "\n\n---\n\n".join(
            f"[Source: {r['symbol']} chunk {r['chunk_index']}]\n{r['text']}"
            for r in results
        )

        return {
            "query": query,
            "symbol": symbol,
            "results": results,
            "context": context,
        }
