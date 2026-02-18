"""
FinAgent FastAPI Backend

Endpoints:
  POST /api/analyze          – Trigger autonomous market analysis
  WS   /ws/analysis          – Real-time streaming via WebSocket
  GET  /api/fundamentals/{s} – Retrieve company fundamentals via RAG
  GET  /health               – Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.orchestrator import AgentState, agent
from rag.retriever import FinancialRetriever
from rag.vectordb import FinancialVectorDB
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

vectordb: Optional[FinancialVectorDB] = None
retriever: Optional[FinancialRetriever] = None

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectordb, retriever
    logger.info("Initializing FinAgent backend …")

    try:
        vectordb = FinancialVectorDB(Config.QDRANT_URL)
        retriever = FinancialRetriever(vectordb)
        logger.info("Indexing 10-K filings …")
        await vectordb.index_10k_filings(DEFAULT_SYMBOLS)
        logger.info("Vector DB ready.")
    except Exception as exc:
        logger.warning(f"Vector DB init failed (Qdrant may not be running): {exc}")
        vectordb = None
        retriever = None

    yield

    logger.info("Shutting down FinAgent backend.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FinAgent – Autonomous Financial Research System",
    description="LangGraph-powered autonomous agent for financial market analysis.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    symbols: List[str] = ["AAPL", "MSFT"]


class AnalyzeResponse(BaseModel):
    status: str
    symbols: List[str]
    decision: str
    confidence: float
    sentiment: dict
    backtest_results: dict
    reflection: str
    steps: int
    timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_markets(request: AnalyzeRequest):
    """
    Trigger the full autonomous analysis pipeline:
    research → backtest → reflect → decide
    """
    symbols = request.symbols
    logger.info(f"Starting analysis for {symbols}")

    initial_state: AgentState = {
        "messages": [],
        "symbols": symbols,
        "market_data": {},
        "sentiment_scores": {},
        "backtest_results": {},
        "fundamentals": {},
        "decision": "",
        "confidence": 0.0,
        "reflection": "",
        "step": 0,
    }

    try:
        final_state = await agent.ainvoke(
            initial_state, {"recursion_limit": Config.MAX_AGENT_STEPS}
        )

        return AnalyzeResponse(
            status="success",
            symbols=symbols,
            decision=final_state["decision"],
            confidence=final_state.get("confidence", 0.0),
            sentiment=final_state["sentiment_scores"],
            backtest_results=final_state["backtest_results"],
            reflection=final_state["reflection"],
            steps=final_state["step"],
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as exc:
        logger.error(f"Analysis failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/ws/analysis")
async def websocket_analysis(websocket: WebSocket):
    """
    Real-time analysis streaming.

    Client sends: {"symbols": ["AAPL", "MSFT"]}
    Server streams step-by-step updates, then final result.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    try:
        while True:
            data = await websocket.receive_json()
            symbols = data.get("symbols", ["AAPL"])

            initial_state: AgentState = {
                "messages": [],
                "symbols": symbols,
                "market_data": {},
                "sentiment_scores": {},
                "backtest_results": {},
                "fundamentals": {},
                "decision": "",
                "confidence": 0.0,
                "reflection": "",
                "step": 0,
            }

            await websocket.send_json(
                {"type": "status", "message": f"Starting analysis for {symbols} …"}
            )

            try:
                async for chunk in agent.astream(initial_state):
                    node_name = list(chunk.keys())[0]
                    node_output = chunk[node_name]

                    await websocket.send_json(
                        {
                            "type": "step",
                            "node": node_name,
                            "data": _serialize(node_output),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    await asyncio.sleep(0.05)

                await websocket.send_json({"type": "complete", "status": "success"})

            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")


@app.get("/api/fundamentals/{symbol}")
async def get_fundamentals(symbol: str, query: str = "key financial metrics"):
    """Retrieve company fundamentals from 10-K via RAG hybrid search."""
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Vector DB not available. Start Qdrant and restart the server.",
        )

    full_query = f"{query} for {symbol}"
    result = retriever.retrieve(full_query, symbol=symbol, top_k=3)

    return {
        "symbol": symbol,
        "query": full_query,
        "fundamentals": result["results"],
        "context": result["context"],
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vectordb": "connected" if vectordb else "unavailable",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(obj):
    """Make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
