"""
LangGraph Agent Orchestrator for FinAgent.

Phase 1-2 (single agent):
  START → research → backtest → reflect ─┬─► decide → END
                                          └─► research (max loops)

Phase 3 (multi-agent):
  START → director_init → run_specialists → run_synthesis → format_report → END

The multi-agent graph runs 3 specialist agents in parallel (SEC, Technical,
Risk), then synthesises results. Config.MULTI_AGENT_MODE selects which graph.

Each node is a pure function that receives the full state and returns a
partial update dict.
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
# Agent State (single-agent, Phase 1-2)
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

try:
    llm = ChatGoogleGenerativeAI(
        model=Config.GEMINI_MODEL,
        google_api_key=Config.GEMINI_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True,
    )
    # Structured-output wrappers
    decision_llm = llm.with_structured_output(DecisionOutput)
    reflection_llm = llm.with_structured_output(ReflectionOutput)
    logger.info("Gemini LLM initialised.")
except Exception as _llm_init_err:
    logger.warning(
        f"Gemini LLM unavailable ({_llm_init_err}). "
        "All LLM nodes will use fallback responses."
    )
    llm = None  # type: ignore[assignment]
    decision_llm = None  # type: ignore[assignment]
    reflection_llm = None  # type: ignore[assignment]


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


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE-AGENT GRAPH (Phase 1-2 fallback)
# ═══════════════════════════════════════════════════════════════════════════


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
        if reflection_llm is None:
            raise RuntimeError("LLM not initialised (no API key)")
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
        confidence = 0.7

    # Pre-compute why routing will fire so it ends up in state
    reflection_count_new = reflection_count + 1
    if reflection_count_new >= Config.REFLECTION_THRESHOLD:
        routing_reason = f"Hard cap reached ({reflection_count_new}/{Config.REFLECTION_THRESHOLD}). Forcing decide."
    elif "GATHER_MORE_DATA" in reflection_text:
        routing_reason = f"LLM requested GATHER_MORE_DATA (loop {reflection_count_new}/{Config.REFLECTION_THRESHOLD})."
    elif confidence < Config.CONFIDENCE_THRESHOLD:
        routing_reason = f"Confidence {confidence:.2f} < threshold {Config.CONFIDENCE_THRESHOLD}. Looping back."
    else:
        routing_reason = f"Confidence {confidence:.2f} >= threshold {Config.CONFIDENCE_THRESHOLD}. Proceeding to decide."

    logger.info(f"[reflection_node] routing_reason: {routing_reason}")

    return {
        "reflection": reflection_text,
        "confidence": round(confidence, 3),
        "reflection_step_count": reflection_count_new,
        "routing_reason": routing_reason,
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
        if decision_llm is None:
            raise RuntimeError("LLM not initialised (no API key)")
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
# Build the single-agent LangGraph (Phase 1-2)
# ---------------------------------------------------------------------------


def build_single_agent_graph():
    """Original 4-node graph — used as fallback when MULTI_AGENT_MODE=false."""
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


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-AGENT GRAPH (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════

from agent.specialist_agents.shared_state import (
    FinalRecommendation,
    FundamentalAnalysis,
    MultiAgentState,
    RiskAnalysis,
    TechnicalAnalysis,
)


@traceable(name="director_init")
async def director_init_node(state: MultiAgentState) -> Dict:
    """
    Research Director: validate symbols, fetch news + sentiment in parallel.
    Reuses the Phase 2 parallel news pipeline.
    """
    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[director_init] Starting multi-agent analysis for {symbols}")
    t0 = time.perf_counter()

    # Parallel news + sentiment fetch (reuse Phase 2 pattern)
    tasks = [_process_single_symbol(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_news: Dict = {}
    all_sentiment: Dict = {}

    for i, res in enumerate(results):
        sym = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"[director_init] {sym} failed: {res}")
            all_news[sym] = {"symbol": sym, "articles": [], "error": str(res)}
            all_sentiment[sym] = {"positive_ratio": 0.5, "total_articles": 0}
        else:
            all_news[sym] = res["news"]
            all_sentiment[sym] = res["sentiment"]

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[director_init] News fetched in {elapsed:.0f}ms")

    return {
        "news_data": all_news,
        "sentiment": all_sentiment,
        "step": 1,
        "latency_ms": {"director_init": round(elapsed, 1)},
    }


@traceable(name="run_specialists")
async def run_specialists_node(state: MultiAgentState) -> Dict:
    """
    Run SEC, Technical, and Risk agents ALL IN PARALLEL.
    """
    from agent.specialist_agents.sec_agent import run_sec_agent
    from agent.specialist_agents.technical_agent import run_technical_agent
    from agent.specialist_agents.risk_agent import run_risk_agent

    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[run_specialists] Launching 3 agents in parallel for {symbols}")
    t0 = time.perf_counter()

    t_sec = time.perf_counter()
    t_tech = time.perf_counter()
    t_risk = time.perf_counter()

    # All three specialist agents run concurrently
    fundamental_results, technical_results, risk_results = await asyncio.gather(
        run_sec_agent(symbols),
        run_technical_agent(symbols),
        run_risk_agent(symbols),
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[run_specialists] All 3 agents completed in {elapsed:.0f}ms")

    # Merge latency
    existing_latency = state.get("latency_ms", {})
    existing_latency["run_specialists"] = round(elapsed, 1)

    return {
        "fundamental_analyses": fundamental_results,
        "technical_analyses": technical_results,
        "risk_analyses": risk_results,
        "step": state.get("step", 0) + 1,
        "latency_ms": existing_latency,
    }


@traceable(name="run_synthesis")
async def run_synthesis_node(state: MultiAgentState) -> Dict:
    """
    For each symbol: match analyses and run synthesis agent.
    Synthesis runs in parallel across symbols.
    """
    from agent.specialist_agents.synthesis_agent import run_synthesis_agent

    symbols = state.get("symbols", ["AAPL"])
    logger.info(f"[run_synthesis] Synthesising results for {symbols}")
    t0 = time.perf_counter()

    # Build lookup maps
    fund_map: Dict[str, FundamentalAnalysis] = {}
    for fa in state.get("fundamental_analyses", []):
        fund_map[fa.symbol] = fa

    tech_map: Dict[str, TechnicalAnalysis] = {}
    for ta in state.get("technical_analyses", []):
        tech_map[ta.symbol] = ta

    risk_map: Dict[str, RiskAnalysis] = {}
    for ra in state.get("risk_analyses", []):
        risk_map[ra.symbol] = ra

    # Build synthesis tasks
    tasks = []
    for sym in symbols:
        fund = fund_map.get(sym, FundamentalAnalysis(symbol=sym))
        tech = tech_map.get(sym, TechnicalAnalysis(symbol=sym))
        risk = risk_map.get(sym, RiskAnalysis(symbol=sym))

        # Compute sentiment score from Phase 2 data (-1 to +1)
        sent_data = state.get("sentiment", {}).get(sym, {})
        pos_ratio = sent_data.get("positive_ratio", 0.5)
        sentiment_score = (pos_ratio - 0.5) * 2  # map [0,1] → [-1,+1]

        # Top 3 news headlines
        news_data = state.get("news_data", {}).get(sym, {})
        headlines = [a.get("title", "") for a in news_data.get("articles", [])[:3]]

        tasks.append(run_synthesis_agent(
            symbol=sym,
            fundamental=fund,
            technical=tech,
            risk=risk,
            sentiment_score=sentiment_score,
            news_headlines=headlines,
        ))

    recommendations = await asyncio.gather(*tasks, return_exceptions=True)

    final_recs: List[FinalRecommendation] = []
    for i, rec in enumerate(recommendations):
        sym = symbols[i]
        if isinstance(rec, Exception):
            logger.error(f"[run_synthesis] {sym} synthesis failed: {rec}")
            final_recs.append(FinalRecommendation(
                symbol=sym,
                action="HOLD",
                confidence=0.3,
                thesis=f"Synthesis failed for {sym}.",
                key_risks=["Synthesis error"],
            ))
        else:
            final_recs.append(rec)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[run_synthesis] Completed in {elapsed:.0f}ms")

    existing_latency = state.get("latency_ms", {})
    existing_latency["run_synthesis"] = round(elapsed, 1)

    return {
        "recommendations": final_recs,
        "step": state.get("step", 0) + 1,
        "latency_ms": existing_latency,
    }


@traceable(name="format_report")
async def format_report_node(state: MultiAgentState) -> Dict:
    """
    Convert FinalRecommendation list into the AnalyzeResponse-compatible
    format so the REST API response schema does NOT change.
    """
    recommendations = state.get("recommendations", [])
    symbols = state.get("symbols", ["AAPL"])

    # Map STRONG_BUY/STRONG_SELL → BUY/SELL for backward compat
    _action_map = {
        "STRONG_BUY": "BUY",
        "BUY": "BUY",
        "HOLD": "HOLD",
        "SELL": "SELL",
        "STRONG_SELL": "SELL",
    }

    # Build structured_decisions in the Phase 1-2 format
    structured_decisions = []
    decision_lines = []
    total_confidence = 0.0

    for rec in recommendations:
        compat_action = _action_map.get(rec.action, "HOLD")
        structured_decisions.append({
            "symbol": rec.symbol,
            "action": compat_action,
            "confidence": rec.confidence,
            "reasoning": rec.thesis,
            # Phase 3 extras
            "full_action": rec.action,
            "composite_score": rec.composite_score,
            "time_horizon": rec.time_horizon,
            "conflicting_signals": [c.model_dump() for c in rec.conflicting_signals],
            "bull_case": rec.bull_case,
            "bear_case": rec.bear_case,
            "key_catalysts": rec.key_catalysts,
            "key_risks": rec.key_risks,
            "price_target_6m": rec.price_target_6m,
        })
        decision_lines.append(
            f"{rec.symbol}: {rec.action} (confidence: {int(rec.confidence * 100)}%) - {rec.thesis}"
        )
        total_confidence += rec.confidence

    avg_confidence = total_confidence / len(recommendations) if recommendations else 0.5

    # Build backtest_results stub from risk analyses (so existing fields are populated)
    backtest_results = {}
    for ra in state.get("risk_analyses", []):
        backtest_results[ra.symbol] = {
            "sharpe_ratio": ra.sharpe_ratio,
            "max_drawdown": ra.max_drawdown_pct,
            "volatility": ra.volatility_annualized,
            "beta": ra.beta_vs_spy,
            "risk_rating": ra.risk_rating,
            "mock": ra.source == "fallback",
        }

    # Build final report text
    report_lines = ["# FinAgent Multi-Agent Analysis Report\n"]
    for rec in recommendations:
        report_lines.append(f"## {rec.symbol}: {rec.action}")
        report_lines.append(f"**Confidence:** {rec.confidence:.0%}")
        report_lines.append(f"**Thesis:** {rec.thesis}")
        if rec.conflicting_signals:
            report_lines.append("**Conflicts:**")
            for c in rec.conflicting_signals:
                report_lines.append(f"  - [{c.severity}] {c.description}: {c.resolution}")
        report_lines.append("")

    return {
        "decision": "\n".join(decision_lines),
        "confidence": round(avg_confidence, 3),
        "structured_decisions": structured_decisions,
        "backtest_results": backtest_results,
        "final_report": "\n".join(report_lines),
        "reflection": "Multi-agent synthesis complete.",
        "routing_reason": "Multi-agent mode — no reflection loop.",
        "step": state.get("step", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Build the multi-agent LangGraph (Phase 3)
# ---------------------------------------------------------------------------


def build_multi_agent_graph():
    """
    Research Director supervisor graph:
      director_init → run_specialists → run_synthesis → format_report → END
    """
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("director_init", director_init_node)
    workflow.add_node("run_specialists", run_specialists_node)
    workflow.add_node("run_synthesis", run_synthesis_node)
    workflow.add_node("format_report", format_report_node)

    workflow.add_edge(START, "director_init")
    workflow.add_edge("director_init", "run_specialists")
    workflow.add_edge("run_specialists", "run_synthesis")
    workflow.add_edge("run_synthesis", "format_report")
    workflow.add_edge("format_report", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Graph selection based on config
# ---------------------------------------------------------------------------


def build_agent():
    """Select the appropriate graph based on MULTI_AGENT_MODE config."""
    if Config.MULTI_AGENT_MODE:
        logger.info("Building multi-agent graph (Phase 3).")
        return build_multi_agent_graph()
    else:
        logger.info("Building single-agent graph (Phase 1-2 fallback).")
        return build_single_agent_graph()


# Singleton agent instance
agent = build_agent()
