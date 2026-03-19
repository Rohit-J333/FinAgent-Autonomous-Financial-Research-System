"""
LangGraph Agent Orchestrator for FinAgent.

Workflow:
  START → research → backtest → reflect ─┬─► decide → END
                                          └─► research (max loops)

Each node is a pure function that receives the full AgentState
and returns a partial update dict.

Phase 2 additions:
  Fix 7  – parallel symbol processing via asyncio.gather
  Fix 8  – MCP server integration (optional, fallback to direct calls)
  Fix 9  – CONFIDENCE_THRESHOLD / REFLECTION_THRESHOLD wired into routing
  Fix 10 – @traceable decorators for Langfuse per-node observability
"""

from __future__ import annotations

import asyncio
import logging
import operator
import time
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Literal, Sequence, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_core.messages import BaseMessage  # type: ignore
from langgraph.graph import END, START, StateGraph  # type: ignore
from pydantic import BaseModel, Field

from agent.tools.backtest import execute_backtest
from agent.tools.news_scraper import scrape_financial_news
from agent.tools.sentiment import analyze_sentiment
from utils.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse @traceable decorator (Fix 10)
# Wraps to a no-op if langfuse is not installed.
# ---------------------------------------------------------------------------

try:
    from langfuse.decorators import observe as traceable  # type: ignore
except ImportError:
    # No-op decorator when langfuse is absent
    def traceable(name: str = "", **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output (Fix 2)
# ---------------------------------------------------------------------------


class SymbolDecision(BaseModel):
    """Structured decision for a single symbol."""

    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class DecisionOutput(BaseModel):
    """Complete structured output from the decision node."""

    decisions: list[SymbolDecision]
    overall_market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"]
    data_quality_score: float = Field(ge=0.0, le=1.0)


class ReflectionOutput(BaseModel):
    """Structured output from the reflection node."""

    assessment: str
    should_gather_more: bool
    missing_data_types: list[str]
    confidence_in_current_data: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    symbols: List[str]
    market_data: Dict
    sentiment_scores: Dict
    backtest_results: Dict
    fundamentals: Dict
    decision: str
    confidence: float
    reflection: str
    step: int
    reflection_step_count: int
    structured_decisions: List[Dict]
    routing_reason: str


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model=Config.GEMINI_MODEL,
    google_api_key=Config.GEMINI_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True,
)

# Structured-output wrappers
decision_llm = llm.with_structured_output(DecisionOutput)
reflection_llm = llm.with_structured_output(ReflectionOutput)


# ---------------------------------------------------------------------------
# MCP helpers (Fix 8) — prefer MCP tools when available, fall back to direct
# ---------------------------------------------------------------------------

_mcp_available = False

try:
    if Config.USE_MCP:
        from agent.mcp_servers.news_server import (
            get_financial_news as mcp_get_news,
            get_market_sentiment_summary as mcp_get_sentiment_summary,
        )
        from agent.mcp_servers.strategy_server import (
            run_momentum_backtest as mcp_run_backtest,
            get_technical_signals as mcp_get_signals,
        )
        _mcp_available = True
        logger.info("MCP server tools loaded and available.")
except ImportError as exc:
    logger.info(f"MCP tools not available ({exc}) – using direct function calls.")
except Exception as exc:
    logger.warning(f"MCP import error: {exc}")


# ---------------------------------------------------------------------------
# Routing function (Fix 1 + Fix 9: thresholds wired in)
# ---------------------------------------------------------------------------


def should_gather_more(state: AgentState) -> str:
    """
    Route after reflection: loop back to research or proceed to decision.

    Decision logic (evaluated top-to-bottom, first match wins):
      1. reflection_step_count >= REFLECTION_THRESHOLD  → decide (hard cap)
      2. "GATHER_MORE_DATA" in reflection text           → research
      3. confidence < CONFIDENCE_THRESHOLD               → research
      4. otherwise                                        → decide
    """
    reflection_count = state.get("reflection_step_count", 0)
    reflection_text = state.get("reflection", "")
    confidence = state.get("confidence", 0.5)
    symbols = state.get("symbols", [])

    max_reflections = Config.REFLECTION_THRESHOLD  # default 2
    conf_threshold = Config.CONFIDENCE_THRESHOLD   # default 0.6

    # Rule 1 — hard cap on reflection iterations
    if reflection_count >= max_reflections:
        reason = (
            f"Max reflection iterations reached ({reflection_count}/{max_reflections}) "
            f"for symbols {symbols}. Forcing decision."
        )
        logger.warning(f"[routing] {reason}")
        return "decide"

    # Rule 2 — explicit GATHER_MORE_DATA signal from LLM
    if "GATHER_MORE_DATA" in reflection_text:
        reason = (
            f"LLM requested GATHER_MORE_DATA (loop {reflection_count}/{max_reflections}). "
            f"confidence={confidence:.2f}, threshold={conf_threshold}"
        )
        logger.info(f"[routing] {reason}")
        return "research"

    # Rule 3 — confidence below threshold
    if confidence < conf_threshold:
        reason = (
            f"Confidence {confidence:.2f} < threshold {conf_threshold} "
            f"(loop {reflection_count}/{max_reflections}). Looping back."
        )
        logger.info(f"[routing] {reason}")
        return "research"

    # Rule 4 — all good, proceed
    reason = (
        f"Proceeding to decision. confidence={confidence:.2f} >= {conf_threshold}, "
        f"reflections={reflection_count}/{max_reflections}"
    )
    logger.info(f"[routing] {reason}")
    return "decide"


# ---------------------------------------------------------------------------
# Per-symbol parallel helper (Fix 7)
# ---------------------------------------------------------------------------


async def _process_single_symbol(symbol: str) -> Dict:
    """
    Process one symbol completely: news fetch + sentiment aggregation.
    MCP tools are used when available, otherwise direct function calls.
    """
    t0 = time.perf_counter()

    # --- Fetch news ---
    news = await scrape_financial_news(symbol)

    # --- Enrich with MCP sentiment summary if available ---
    if _mcp_available:
        try:
            mcp_summary = await mcp_get_sentiment_summary(symbol)
            news["mcp_sentiment_hint"] = mcp_summary.get("avg_sentiment_hint", "neutral")
        except Exception as exc:
            logger.debug(f"MCP sentiment summary failed for {symbol}: {exc}")

    # --- Aggregate sentiment ---
    articles = news.get("articles", [])
    if articles:
        scores = [a["confidence"] for a in articles if a["sentiment"] == "positive"]
        neg_scores = [a["confidence"] for a in articles if a["sentiment"] == "negative"]
        pos_ratio = len(scores) / len(articles)
        avg_pos = sum(scores) / len(scores) if scores else 0
        avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
        sentiment = {
            "positive_ratio": round(pos_ratio, 3),
            "avg_positive_confidence": round(avg_pos, 3),
            "avg_negative_confidence": round(avg_neg, 3),
            "total_articles": len(articles),
        }
    else:
        sentiment = {"positive_ratio": 0.5, "total_articles": 0}

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[research] {symbol} completed in {elapsed:.0f}ms")

    return {"symbol": symbol, "news": news, "sentiment": sentiment}


async def _backtest_single_symbol(
    symbol: str, strategy: str, start_date: str, end_date: str
) -> Dict:
    """Run backtest for one symbol. MCP fallback → direct call."""
    t0 = time.perf_counter()

    # Prefer MCP backtest if available
    if _mcp_available:
        try:
            result = await mcp_run_backtest(symbol, lookback_days=365)
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[backtest/mcp] {symbol} completed in {elapsed:.0f}ms")
            return {"symbol": symbol, "result": result}
        except Exception as exc:
            logger.debug(f"MCP backtest failed for {symbol}: {exc}, falling back.")

    # Direct call
    result = await execute_backtest(
        strategy=strategy,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        params={"lookback_period": 20},
    )
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[backtest] {symbol} completed in {elapsed:.0f}ms")
    return {"symbol": symbol, "result": result}


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


@traceable(name="research_node")
async def research_node(state: AgentState) -> Dict:
    """
    Node 1 – Research: scrape news and compute aggregate sentiment per symbol.
    Symbols are processed in parallel via asyncio.gather (Fix 7).
    """
    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[research_node] Fetching news for {symbols} (parallel)")
    t0 = time.perf_counter()

    # --- Parallel fetch ---
    tasks = [_process_single_symbol(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_news: Dict = {}
    all_sentiment: Dict = {}

    for i, res in enumerate(results):
        sym = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"[research_node] {sym} failed: {res}")
            all_news[sym] = {"symbol": sym, "articles": [], "error": str(res)}
            all_sentiment[sym] = {"positive_ratio": 0.5, "total_articles": 0}
        else:
            all_news[sym] = res["news"]
            all_sentiment[sym] = res["sentiment"]

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[research_node] Total: {elapsed:.0f}ms for {len(symbols)} symbols")

    return {
        "market_data": all_news,
        "sentiment_scores": all_sentiment,
        "step": state.get("step", 0) + 1,
    }


@traceable(name="backtest_node")
async def backtest_node(state: AgentState) -> Dict:
    """
    Node 2 – Backtest: run strategy simulations in parallel (Fix 7).
    """
    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[backtest_node] Running backtests for {symbols} (parallel)")
    t0 = time.perf_counter()

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Build tasks
    tasks = []
    for symbol in symbols:
        sentiment = state.get("sentiment_scores", {}).get(symbol, {})
        pos_ratio = sentiment.get("positive_ratio", 0.5)
        strategy = "momentum" if pos_ratio > 0.55 else "mean_reversion"
        tasks.append(_backtest_single_symbol(symbol, strategy, start_date, end_date))

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: Dict = {}
    for i, res in enumerate(raw_results):
        sym = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"[backtest_node] {sym} failed: {res}")
            results[sym] = {"error": str(res), "mock": True}
        else:
            results[sym] = res["result"]

    # Enrich with MCP technical signals when available
    if _mcp_available:
        for sym in symbols:
            try:
                signals = await mcp_get_signals(sym)
                results[sym]["technical_signals"] = signals
            except Exception as exc:
                logger.debug(f"MCP technical signals failed for {sym}: {exc}")

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[backtest_node] Total: {elapsed:.0f}ms for {len(symbols)} symbols")

    return {
        "backtest_results": results,
        "step": state.get("step", 0) + 1,
    }


@traceable(name="reflection_node")
async def reflection_node(state: AgentState) -> Dict:
    """
    Node 3 – Reflect: LLM evaluates research + backtest quality.

    Uses structured output (ReflectionOutput) for reliable parsing.
    Sets confidence from the LLM's self-assessment (Fix 9).
    """
    logger.info("[reflection_node] Reflecting on research and backtest results")

    sentiment_summary = "\n".join(
        f"  {sym}: positive_ratio={v.get('positive_ratio', 'N/A')}"
        for sym, v in state.get("sentiment_scores", {}).items()
    )
    backtest_summary = "\n".join(
        f"  {sym}: return={v.get('total_return', 'N/A')}, sharpe={v.get('sharpe_ratio', 'N/A')}, win_rate={v.get('win_rate', 'N/A')}"
        for sym, v in state.get("backtest_results", {}).items()
    )

    reflection_count = state.get("reflection_step_count", 0)

    prompt = f"""You are a senior quantitative analyst reviewing an AI agent's market research.
This is reflection iteration {reflection_count + 1} of a maximum of {Config.REFLECTION_THRESHOLD}.

Sentiment Analysis Results:
{sentiment_summary}

Backtest Results (1-year lookback):
{backtest_summary}

Evaluate the quality of this research:
1. Is the sentiment data sufficient to make a trading decision?
2. Are the backtest metrics (Sharpe ratio, win rate) strong enough?
3. What risks or uncertainties should be flagged?
4. Should the agent proceed to a final decision or gather more data?

Respond with your structured assessment. Set should_gather_more to true ONLY
if the data is clearly insufficient and another research pass would help.
Set confidence_in_current_data between 0.0 and 1.0 — this determines whether
the agent loops back for more research (threshold: {Config.CONFIDENCE_THRESHOLD})."""

    try:
        result: ReflectionOutput = await reflection_llm.ainvoke([("user", prompt)])
        reflection_text = result.assessment
        confidence = result.confidence_in_current_data
        if result.should_gather_more:
            reflection_text += " GATHER_MORE_DATA"
    except Exception as exc:
        logger.error(f"LLM reflection failed: {exc}")
        reflection_text = (
            "Reflection unavailable. Proceeding with available data. PROCEED"
        )
        confidence = 0.5

    return {
        "reflection": reflection_text,
        "confidence": round(confidence, 3),
        "reflection_step_count": reflection_count + 1,
        "step": state.get("step", 0) + 1,
    }


@traceable(name="decision_node")
async def decision_node(state: AgentState) -> Dict:
    """
    Node 4 – Decide: structured BUY / SELL / HOLD output via Pydantic model.
    """
    logger.info("[decision_node] Making final trading decision")

    symbols = state.get("symbols", ["AAPL"])

    sentiment_summary = "\n".join(
        f"  {sym}: {v}" for sym, v in state.get("sentiment_scores", {}).items()
    )
    backtest_summary = "\n".join(
        f"  {sym}: {v}" for sym, v in state.get("backtest_results", {}).items()
    )

    prompt = f"""You are an autonomous financial research agent making a final trading recommendation.

Symbols to evaluate: {symbols}

Sentiment Analysis:
{sentiment_summary}

Backtest Results:
{backtest_summary}

Agent Reflection:
{state.get('reflection', 'No reflection available.')}

Based on all available data, provide a trading decision (BUY, SELL, or HOLD)
for each symbol with a confidence score between 0.0 and 1.0 and a brief
rationale (2-3 sentences). Also assess the overall market sentiment and
rate the quality of the data you had available."""

    try:
        result: DecisionOutput = await decision_llm.ainvoke([("user", prompt)])

        structured = [d.model_dump() for d in result.decisions]

        decision_text = "\n".join(
            f"{d.symbol}: {d.action} (confidence: {int(d.confidence * 100)}%) - {d.reasoning}"
            for d in result.decisions
        )

        avg_confidence = (
            sum(d.confidence for d in result.decisions) / len(result.decisions)
            if result.decisions
            else 0.5
        )

    except Exception as exc:
        logger.error(f"LLM decision failed: {exc}")
        structured = [
            {
                "symbol": s,
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": "Insufficient data to make a confident recommendation.",
            }
            for s in symbols
        ]
        decision_text = "HOLD – insufficient data to make a confident decision."
        avg_confidence = 0.5

    return {
        "decision": decision_text,
        "confidence": round(avg_confidence, 3),
        "structured_decisions": structured,
        "step": state.get("step", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("research", research_node)
    workflow.add_node("backtest", backtest_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("decide", decision_node)

    workflow.add_edge(START, "research")
    workflow.add_edge("research", "backtest")
    workflow.add_edge("backtest", "reflect")

    # Conditional routing: loop back or proceed (Fix 1 + Fix 9)
    workflow.add_conditional_edges(
        "reflect",
        should_gather_more,
        {"research": "research", "decide": "decide"},
    )

    workflow.add_edge("decide", END)

    return workflow.compile()


# Singleton agent instance
agent = build_agent()
