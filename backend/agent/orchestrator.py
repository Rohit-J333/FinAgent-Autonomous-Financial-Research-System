"""
LangGraph Agent Orchestrator for FinAgent.

Workflow:
  START → research → backtest → reflect → decide → END

Each node is a pure function that receives the full AgentState
and returns a partial update dict.
"""

from __future__ import annotations

import logging
import operator
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Sequence, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from langchain_core.messages import BaseMessage  # type: ignore
from langgraph.graph import END, START, StateGraph  # type: ignore

from agent.tools.backtest import execute_backtest
from agent.tools.news_scraper import scrape_financial_news
from agent.tools.sentiment import analyze_sentiment
from utils.config import Config

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model=Config.GEMINI_MODEL,
    google_api_key=Config.GEMINI_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True,
)

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

    prompt = f"""You are a senior quantitative analyst reviewing an AI agent's market research.

Sentiment Analysis Results:
{sentiment_summary}

Backtest Results (1-year lookback):
{backtest_summary}

Reflect on the quality of this research:
1. Is the sentiment data sufficient to make a trading decision?
2. Are the backtest metrics (Sharpe ratio, win rate) strong enough?
3. What risks or uncertainties should be flagged?
4. Should the agent proceed to a final decision or gather more data?

Be concise (3-5 sentences). End with PROCEED or GATHER_MORE_DATA."""

    try:
        response = await llm.ainvoke([("user", prompt)])
        reflection_text = response.content
    except Exception as exc:
        logger.error(f"LLM reflection failed: {exc}")
        reflection_text = "Reflection unavailable. Proceeding with available data. PROCEED"

    return {
        "reflection": reflection_text,
        "step": state.get("step", 0) + 1,
    }


async def decision_node(state: AgentState) -> Dict:
    """
    Node 4 – Decide: make final BUY / SELL / HOLD decision with confidence.
    """
    logger.info("[decision_node] Making final trading decision")

    sentiment_summary = "\n".join(
        f"  {sym}: {v}"
        for sym, v in state.get("sentiment_scores", {}).items()
    )
    backtest_summary = "\n".join(
        f"  {sym}: {v}"
        for sym, v in state.get("backtest_results", {}).items()
    )

    prompt = f"""You are an autonomous financial research agent making a final trading recommendation.

Sentiment Analysis:
{sentiment_summary}

Backtest Results:
{backtest_summary}

Agent Reflection:
{state.get('reflection', 'No reflection available.')}

Based on all available data, provide:
1. A trading decision: BUY, SELL, or HOLD for each symbol
2. A confidence score (0-100) for each decision
3. A brief rationale (2-3 sentences)

Format your response as:
SYMBOL: DECISION (confidence: XX%) - rationale"""

    try:
        response = await llm.ainvoke([("user", prompt)])
        decision_text = response.content

        # Extract average confidence from text (simple heuristic)
        import re
        confidences = re.findall(r"confidence:\s*(\d+)%", decision_text)
        avg_confidence = (
            sum(int(c) for c in confidences) / len(confidences) / 100
            if confidences
            else 0.7
        )
    except Exception as exc:
        logger.error(f"LLM decision failed: {exc}")
        decision_text = "HOLD – insufficient data to make a confident decision."
        avg_confidence = 0.5

    return {
        "decision": decision_text,
        "confidence": round(avg_confidence, 3),
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
    workflow.add_edge("reflect", "decide")
    workflow.add_edge("decide", END)

    return workflow.compile()


# Singleton agent instance
agent = build_agent()
