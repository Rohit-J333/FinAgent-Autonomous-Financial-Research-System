"""
Shared Pydantic models and LangGraph state for the multi-agent system.

All specialist agents produce one of the analysis models below.
The Synthesis Agent consumes all three plus sentiment to produce
FinalRecommendation objects.

MultiAgentState uses Annotated[list, operator.add] so parallel
node writes merge correctly in LangGraph.
"""

from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Per-agent output models
# ---------------------------------------------------------------------------


class FundamentalAnalysis(BaseModel):
    """Structured output from the SEC Fundamental Agent."""

    symbol: str
    revenue_growth_yoy: Optional[float] = None      # % YoY change
    gross_margin: Optional[float] = None             # 0.0-1.0
    debt_to_equity: Optional[float] = None
    pe_ratio: Optional[float] = None
    key_risks: List[str] = Field(default_factory=list)  # top 3 from Item 1A
    competitive_moat: str = ""                       # one sentence
    analyst_summary: str = ""                        # 2-3 sentence LLM summary
    data_quality: Literal["high", "medium", "low"] = "low"
    source: str = "fallback"                         # "edgar_live" or "fallback"


class TechnicalAnalysis(BaseModel):
    """Structured output from the Technical Analysis Agent."""

    symbol: str
    trend: Literal["UPTREND", "DOWNTREND", "SIDEWAYS"] = "SIDEWAYS"
    rsi_14: float = 50.0                             # 0-100
    macd_signal: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "NEUTRAL"
    bb_position: Literal["ABOVE_UPPER", "BELOW_LOWER", "WITHIN"] = "WITHIN"
    sma_20: float = 0.0
    sma_50: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    momentum_score: float = 0.0                      # -1.0 to +1.0
    volume_trend: Literal["INCREASING", "DECREASING", "STABLE"] = "STABLE"
    data_points: int = 0                             # how many OHLCV rows used
    source: str = "fallback"                         # "yfinance_live" or "fallback"


class RiskAnalysis(BaseModel):
    """Structured output from the Risk Agent."""

    symbol: str
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0                    # negative, e.g. -23.4
    var_95: float = 0.0                              # Value at Risk 95% as %
    cvar_95: float = 0.0                             # Conditional VaR as %
    volatility_annualized: float = 25.0              # as %
    beta_vs_spy: float = 1.0
    risk_rating: Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH"] = "MEDIUM"
    position_size_pct: float = 5.0                   # recommended % of portfolio
    source: str = "fallback"                         # "quantstats_live" or "fallback"


# ---------------------------------------------------------------------------
# Synthesis models
# ---------------------------------------------------------------------------


class SignalConflict(BaseModel):
    """A detected conflict between two or more analysis signals."""

    description: str
    severity: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    resolution: str = ""


class FinalRecommendation(BaseModel):
    """Complete recommendation produced by the Synthesis Agent."""

    symbol: str
    action: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"] = "HOLD"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    price_target_6m: Optional[float] = None
    thesis: str = ""                                 # 2-3 sentence investment thesis
    bull_case: str = ""                              # one sentence
    bear_case: str = ""                              # one sentence
    key_catalysts: List[str] = Field(default_factory=list)   # max 3
    key_risks: List[str] = Field(default_factory=list)       # max 3
    time_horizon: Literal["SHORT", "MEDIUM", "LONG"] = "MEDIUM"
    conflicting_signals: List[SignalConflict] = Field(default_factory=list)
    composite_score: float = Field(default=0.0, ge=-1.0, le=1.0)


# ---------------------------------------------------------------------------
# LangGraph state for the multi-agent supervisor
# ---------------------------------------------------------------------------


class MultiAgentState(TypedDict):
    """
    State shared across all nodes in the multi-agent graph.

    Annotated[list[X], operator.add] allows parallel nodes to each
    return partial lists that LangGraph merges automatically.
    """

    # Inputs
    symbols: List[str]
    mode: str

    # Per-agent outputs (merge-friendly for parallel writes)
    fundamental_analyses: Annotated[List[FundamentalAnalysis], operator.add]
    technical_analyses: Annotated[List[TechnicalAnalysis], operator.add]
    risk_analyses: Annotated[List[RiskAnalysis], operator.add]

    # Sentiment from existing Phase 2 tools (reused)
    sentiment: Dict
    news_data: Dict

    # Synthesis output
    recommendations: List[FinalRecommendation]
    final_report: str

    # Routing and observability
    routing_reason: str
    reflection_step_count: int
    confidence: float
    latency_ms: Dict                                 # per-agent timing

    # Compat fields for AnalyzeResponse mapping
    decision: str
    step: int
    structured_decisions: List[Dict]
    backtest_results: Dict
