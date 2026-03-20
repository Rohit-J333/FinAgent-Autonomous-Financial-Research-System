"""
Synthesis Agent — reconciles fundamental, technical, risk, and sentiment
signals to produce a FinalRecommendation for each symbol.

1. Computes weighted composite score from all signals
2. Detects conflicts between signals
3. Maps composite to action (STRONG_BUY → STRONG_SELL)
4. Uses Gemini 1.5 Pro for thesis/bull/bear generation
5. Falls back to computed scores with template text if LLM unavailable
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from pydantic import BaseModel

from agent.specialist_agents.shared_state import (
    FinalRecommendation,
    FundamentalAnalysis,
    RiskAnalysis,
    SignalConflict,
    TechnicalAnalysis,
)
from utils.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Langfuse @traceable (no-op when absent)
# ---------------------------------------------------------------------------

try:
    from langfuse.decorators import observe as traceable  # type: ignore
except ImportError:
    def traceable(name: str = "", **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

# ---------------------------------------------------------------------------
# LLM for narrative generation
# ---------------------------------------------------------------------------

_synthesis_llm = None

try:
    if Config.GEMINI_API_KEY:
        _synthesis_llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.3,
            convert_system_message_to_human=True,
        )
        logger.info("Synthesis agent LLM initialised.")
except Exception as exc:
    logger.warning(f"Synthesis agent LLM unavailable ({exc}).")

# ---------------------------------------------------------------------------
# LLM response model (subset of fields the LLM fills)
# ---------------------------------------------------------------------------


class _LLMNarrative(BaseModel):
    thesis: str = ""
    bull_case: str = ""
    bear_case: str = ""
    key_catalysts: List[str] = []
    key_risks: List[str] = []
    price_target_6m: Optional[float] = None


# ---------------------------------------------------------------------------
# Signal reconciliation helpers
# ---------------------------------------------------------------------------

_RISK_SCORE_MAP = {"LOW": 0.3, "MEDIUM": 0.0, "HIGH": -0.3, "VERY_HIGH": -0.5}

_WEIGHTS = {"sentiment": 0.15, "technical": 0.35, "fundamental": 0.30, "risk": 0.20}


def _compute_fundamental_score(f: FundamentalAnalysis) -> float:
    """Score fundamental analysis on -1.0 to +1.0 scale."""
    if f.source == "fallback":
        return 0.0
    if f.data_quality == "high" and f.pe_ratio is not None:
        pe = f.pe_ratio
        if pe < 15:
            return 0.5
        elif pe < 25:
            return 0.2
        elif pe < 40:
            return -0.1
        else:
            return -0.4
    if f.data_quality == "medium":
        return 0.1
    return 0.0


def _compute_composite(
    sentiment_score: float,
    technical: TechnicalAnalysis,
    fundamental: FundamentalAnalysis,
    risk: RiskAnalysis,
) -> float:
    """Weighted composite score, clamped to [-1.0, +1.0]."""
    tech_score = technical.momentum_score
    fund_score = _compute_fundamental_score(fundamental)
    risk_score = _RISK_SCORE_MAP.get(risk.risk_rating, 0.0)

    composite = (
        sentiment_score * _WEIGHTS["sentiment"]
        + tech_score * _WEIGHTS["technical"]
        + fund_score * _WEIGHTS["fundamental"]
        + risk_score * _WEIGHTS["risk"]
    )
    return max(-1.0, min(1.0, composite))


def _detect_conflicts(
    sentiment_score: float,
    technical: TechnicalAnalysis,
    fundamental: FundamentalAnalysis,
    risk: RiskAnalysis,
) -> List[SignalConflict]:
    """Detect conflicts between signals."""
    conflicts: List[SignalConflict] = []

    # Conflict A: tech bullish but sentiment bearish
    tech_bullish = technical.momentum_score > 0.2
    if tech_bullish and sentiment_score < -0.3:
        conflicts.append(SignalConflict(
            description="Technical momentum bullish but news sentiment bearish",
            severity="MEDIUM",
            resolution="Weight technicals for short-term; sentiment may lag",
        ))

    # Conflict B: uptrend but high risk
    if technical.trend == "UPTREND" and risk.risk_rating in ("HIGH", "VERY_HIGH"):
        conflicts.append(SignalConflict(
            description="Price trending up but risk metrics elevated",
            severity="HIGH",
            resolution="Consider reduced position size per risk recommendation",
        ))

    # Conflict C: strong momentum but high valuation
    if fundamental.pe_ratio and fundamental.pe_ratio > 40 and technical.momentum_score > 0.3:
        conflicts.append(SignalConflict(
            description="Strong momentum but elevated valuation (P/E > 40)",
            severity="LOW",
            resolution="Momentum may continue short-term; valuation risk is long-term",
        ))

    # Conflict D: both fundamental and technical data unavailable
    if fundamental.source == "fallback" and technical.source == "fallback":
        conflicts.append(SignalConflict(
            description="Both fundamental and technical data unavailable",
            severity="HIGH",
            resolution="Recommendation based primarily on sentiment; treat with caution",
        ))

    return conflicts


def _composite_to_action(composite: float) -> str:
    """Map composite score to action string."""
    if composite >= 0.5:
        return "STRONG_BUY"
    elif composite >= 0.2:
        return "BUY"
    elif composite >= -0.2:
        return "HOLD"
    elif composite >= -0.5:
        return "SELL"
    else:
        return "STRONG_SELL"


def _compute_time_horizon(
    technical: TechnicalAnalysis,
    fundamental: FundamentalAnalysis,
) -> str:
    """Determine recommended time horizon."""
    if abs(technical.momentum_score) > 0.4:
        return "SHORT"
    elif fundamental.data_quality == "high":
        return "LONG"
    return "MEDIUM"


def _compute_confidence(composite: float, num_conflicts: int) -> float:
    """Confidence from composite score and conflict count."""
    raw = abs(composite) + (1 - num_conflicts * 0.15)
    return max(0.1, min(0.95, raw))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@traceable(name="synthesis_agent")
async def run_synthesis_agent(
    symbol: str,
    fundamental: FundamentalAnalysis,
    technical: TechnicalAnalysis,
    risk: RiskAnalysis,
    sentiment_score: float,
    news_headlines: List[str],
) -> FinalRecommendation:
    """
    Produce a final recommendation for one symbol by reconciling all signals.

    Never raises — returns a computed recommendation even without LLM.
    """
    try:
        # Step 1: Composite score
        composite = _compute_composite(sentiment_score, technical, fundamental, risk)

        # Step 2: Detect conflicts
        conflicts = _detect_conflicts(sentiment_score, technical, fundamental, risk)

        # Step 3: Map to action
        action = _composite_to_action(composite)

        # Step 4: Time horizon
        time_horizon = _compute_time_horizon(technical, fundamental)

        # Step 5: Confidence
        confidence = _compute_confidence(composite, len(conflicts))

        # Step 6: LLM narrative generation
        thesis = bull_case = bear_case = ""
        key_catalysts: List[str] = []
        key_risks: List[str] = []
        price_target_6m: Optional[float] = None

        if _synthesis_llm is not None:
            try:
                narrative_llm = _synthesis_llm.with_structured_output(_LLMNarrative)

                prompt = f"""You are a senior equity research analyst writing for institutional investors.
Be specific, cite numbers, acknowledge uncertainty. Never hallucinate financial data.

Symbol: {symbol}
Computed Action: {action} (composite score: {composite:.3f})
Confidence: {confidence:.2f}

FUNDAMENTAL ANALYSIS:
{fundamental.model_dump_json(indent=2)}

TECHNICAL ANALYSIS:
{technical.model_dump_json(indent=2)}

RISK ANALYSIS:
{risk.model_dump_json(indent=2)}

Sentiment Score: {sentiment_score:.3f} (-1 bearish to +1 bullish)
Recent Headlines: {news_headlines[:3]}

Detected Conflicts:
{[c.model_dump() for c in conflicts]}

Generate:
- thesis: 2-3 sentence investment thesis for {action}
- bull_case: one sentence best-case scenario
- bear_case: one sentence worst-case scenario
- key_catalysts: up to 3 specific catalysts
- key_risks: up to 3 specific risks
- price_target_6m: 6-month price target if enough data, else null"""

                narrative: _LLMNarrative = await narrative_llm.ainvoke([("user", prompt)])
                thesis = narrative.thesis
                bull_case = narrative.bull_case
                bear_case = narrative.bear_case
                key_catalysts = narrative.key_catalysts[:3]
                key_risks = narrative.key_risks[:3]
                price_target_6m = narrative.price_target_6m

            except Exception as exc:
                logger.error(f"Synthesis LLM failed for {symbol}: {exc}")

        # Fallback text if LLM produced nothing
        if not thesis:
            thesis = f"{action} based on composite score {composite:.2f}."
        if not bull_case:
            bull_case = "Technical and fundamental analysis pending manual review."
        if not bear_case:
            bear_case = "Fallback mode — LLM unavailable for detailed bear case."
        if not key_risks:
            key_risks = ["LLM synthesis unavailable — manual review recommended"]

        return FinalRecommendation(
            symbol=symbol,
            action=action,
            confidence=round(confidence, 3),
            price_target_6m=price_target_6m,
            thesis=thesis,
            bull_case=bull_case,
            bear_case=bear_case,
            key_catalysts=key_catalysts,
            key_risks=key_risks,
            time_horizon=time_horizon,
            conflicting_signals=conflicts,
            composite_score=round(composite, 4),
        )

    except Exception as exc:
        logger.error(f"Synthesis agent failed for {symbol}: {exc}")
        return FinalRecommendation(
            symbol=symbol,
            action="HOLD",
            confidence=0.3,
            thesis=f"Error during synthesis for {symbol}: {exc}",
            bull_case="Unable to generate.",
            bear_case="Unable to generate.",
            key_risks=["Synthesis error — manual review required"],
            composite_score=0.0,
        )
