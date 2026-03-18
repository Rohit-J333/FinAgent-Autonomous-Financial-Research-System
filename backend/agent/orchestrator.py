"""
LangGraph Agent Orchestrator for FinAgent.

Workflow:
  START → research → backtest → reflect ─┬─► decide → END
                                          └─► research (max 2 loops)

Each node is a pure function that receives the full AgentState
and returns a partial update dict.
"""

from __future__ import annotations

import logging
import operator
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
# Routing function (Fix 1)
# ---------------------------------------------------------------------------


def should_gather_more(state: AgentState) -> str:
    """
    Route after reflection: loop back to research or proceed to decision.

    Returns "research" if the reflection says GATHER_MORE_DATA **and** we
    have not yet looped more than 2 times (step guard prevents infinite loops).
    """
    reflection_count = state.get("reflection_step_count", 0)
    reflection_text = state.get("reflection", "")

    if "GATHER_MORE_DATA" in reflection_text and reflection_count < 2:
        logger.info(
            f"[routing] Gathering more data (reflection loop {reflection_count}/2)"
        )
        return "research"

    logger.info("[routing] Proceeding to decision")
    return "decide"


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


async def research_node(state: AgentState) -> Dict:
    """
    Node 1 – Research: scrape news and compute aggregate sentiment per symbol.
    """
    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[research_node] Fetching news for {symbols}")

    all_news: Dict = {}
    all_sentiment: Dict = {}

    for symbol in symbols:
        news = await scrape_financial_news(symbol)
        all_news[symbol] = news

        # Aggregate sentiment across articles
        articles = news.get("articles", [])
        if articles:
            scores = [a["confidence"] for a in articles if a["sentiment"] == "positive"]
            neg_scores = [a["confidence"] for a in articles if a["sentiment"] == "negative"]
            pos_ratio = len(scores) / len(articles)
            avg_pos = sum(scores) / len(scores) if scores else 0
            avg_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0
            all_sentiment[symbol] = {
                "positive_ratio": round(pos_ratio, 3),
                "avg_positive_confidence": round(avg_pos, 3),
                "avg_negative_confidence": round(avg_neg, 3),
                "total_articles": len(articles),
            }
        else:
            all_sentiment[symbol] = {"positive_ratio": 0.5, "total_articles": 0}

    return {
        "market_data": all_news,
        "sentiment_scores": all_sentiment,
        "step": state.get("step", 0) + 1,
    }


def backtest_node(state: AgentState) -> Dict:
    """
    Node 2 – Backtest: run strategy simulations for each symbol.
    """
    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[backtest_node] Running backtests for {symbols}")

    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    results: Dict = {}
    for symbol in symbols:
        # Choose strategy based on sentiment
        sentiment = state.get("sentiment_scores", {}).get(symbol, {})
        pos_ratio = sentiment.get("positive_ratio", 0.5)
        strategy = "momentum" if pos_ratio > 0.55 else "mean_reversion"

        result = execute_backtest(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            params={"lookback_period": 20},
        )
        results[symbol] = result

    return {
        "backtest_results": results,
        "step": state.get("step", 0) + 1,
    }


async def reflection_node(state: AgentState) -> Dict:
    """
    Node 3 – Reflect: LLM evaluates its own research + backtest quality.

    Uses structured output (ReflectionOutput) for reliable parsing.
    Appends GATHER_MORE_DATA to the text if the LLM recommends it,
    so the routing function can branch back to research.
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
This is reflection iteration {reflection_count + 1} of a maximum of 2.

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
if the data is clearly insufficient and another research pass would help."""

    try:
        result: ReflectionOutput = await reflection_llm.ainvoke([("user", prompt)])
        reflection_text = result.assessment
        # Embed the signal so the routing function can do a simple string check
        if result.should_gather_more:
            reflection_text += " GATHER_MORE_DATA"
    except Exception as exc:
        logger.error(f"LLM reflection failed: {exc}")
        reflection_text = (
            "Reflection unavailable. Proceeding with available data. PROCEED"
        )

    return {
        "reflection": reflection_text,
        "reflection_step_count": reflection_count + 1,
        "step": state.get("step", 0) + 1,
    }


async def decision_node(state: AgentState) -> Dict:
    """
    Node 4 – Decide: structured BUY / SELL / HOLD output via Pydantic model.

    No regex parsing — Gemini returns a DecisionOutput directly.
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

        # Build a human-readable decision string for backward compatibility
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
        # Deterministic fallback — HOLD everything at 50 % confidence
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

    # Fix 1: conditional routing — loop back to research or proceed to decide
    workflow.add_conditional_edges(
        "reflect",
        should_gather_more,
        {"research": "research", "decide": "decide"},
    )

    workflow.add_edge("decide", END)

    return workflow.compile()


# Singleton agent instance
agent = build_agent()
