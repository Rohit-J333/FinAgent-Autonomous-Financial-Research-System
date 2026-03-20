"""
SEC Fundamental Agent — fetches real 10-K filings via edgartools and
produces structured FundamentalAnalysis objects using Gemini.

Fallback chain:
  edgartools → placeholder text
  Gemini structured output → conservative defaults
  Any exception → FundamentalAnalysis(source="fallback")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

from agent.specialist_agents.shared_state import FundamentalAnalysis
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
# LLM for structured extraction (gemini-1.5-flash for speed/cost)
# ---------------------------------------------------------------------------

_sec_llm = None

try:
    if Config.GEMINI_API_KEY:
        _sec_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True,
        ).with_structured_output(FundamentalAnalysis)
        logger.info("SEC agent LLM (gemini-1.5-flash) initialised.")
except Exception as exc:
    logger.warning(f"SEC agent LLM unavailable ({exc}).")

# ---------------------------------------------------------------------------
# 10-K fetching (edgartools, synchronous → run_in_executor)
# ---------------------------------------------------------------------------


def _fetch_10k_sync(symbol: str) -> dict:
    """
    Fetch real 10-K data using edgartools.

    Returns dict with keys: business, risks, facts (dict or None).
    """
    try:
        from edgar import Company, set_identity  # type: ignore

        set_identity("FinAgent rohit.janbandhu25@gmail.com")

        company = Company(symbol)
        filings = company.get_filings(form="10-K")
        latest_filings = filings.latest(1)

        if not latest_filings:
            logger.warning(f"No 10-K filings found for {symbol}.")
            return {"business": "", "risks": "", "facts": None}

        filing = (
            latest_filings[0]
            if hasattr(latest_filings, "__getitem__")
            else latest_filings
        )

        ten_k = filing.obj()

        # Item 1 — Business Description
        try:
            business = str(ten_k["Item 1"])[:3000]
        except Exception:
            business = "Item 1 (Business Description) not available."

        # Item 1A — Risk Factors
        try:
            risks = str(ten_k["Item 1A"])[:2000]
        except Exception:
            risks = "Item 1A (Risk Factors) not available."

        # Try to get financial facts for revenue/margins
        facts = None
        try:
            facts_obj = company.get_facts()
            if facts_obj is not None:
                facts = {"available": True}
        except Exception:
            pass

        logger.info(f"Fetched real 10-K for {symbol} ({len(business) + len(risks)} chars).")
        return {"business": business, "risks": risks, "facts": facts}

    except ImportError:
        logger.warning("edgartools not installed — using placeholder.")
        return {"business": "", "risks": "", "facts": None}
    except Exception as exc:
        logger.error(f"edgartools fetch failed for {symbol}: {exc}")
        return {"business": "", "risks": "", "facts": None}


# ---------------------------------------------------------------------------
# Per-symbol analysis
# ---------------------------------------------------------------------------


async def _analyze_single_symbol(symbol: str) -> FundamentalAnalysis:
    """Fetch 10-K data and produce structured analysis for one symbol."""
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _fetch_10k_sync, symbol)

        business = data["business"]
        risks = data["risks"]

        if not business and not risks:
            logger.warning(f"No 10-K text for {symbol}, returning fallback.")
            return _fallback(symbol)

        # Use LLM to produce structured analysis
        if _sec_llm is None:
            return _fallback_from_text(symbol, business, risks)

        prompt = f"""You are a CFA analyst. Extract structured data from this SEC 10-K filing excerpt.
Be conservative — if data is not clearly stated, use null. Never hallucinate financial figures.

Symbol: {symbol}

BUSINESS DESCRIPTION (Item 1):
{business}

RISK FACTORS (Item 1A):
{risks}

Provide a FundamentalAnalysis with:
- revenue_growth_yoy: percentage if mentioned, else null
- gross_margin: decimal 0-1 if mentioned, else null
- debt_to_equity: ratio if mentioned, else null
- pe_ratio: if mentioned, else null
- key_risks: top 3 specific risks from Item 1A (short phrases)
- competitive_moat: one sentence about competitive advantage
- analyst_summary: 2-3 sentence summary of financial position
- data_quality: "high" if revenue/margin numbers found, "medium" if only qualitative, "low" if sparse
- source: "edgar_live"
- symbol: "{symbol}"
"""

        try:
            result: FundamentalAnalysis = await _sec_llm.ainvoke([("user", prompt)])
            # Ensure symbol and source are correct
            result.symbol = symbol
            result.source = "edgar_live"
            return result
        except Exception as exc:
            logger.error(f"LLM extraction failed for {symbol}: {exc}")
            return _fallback_from_text(symbol, business, risks)

    except Exception as exc:
        logger.error(f"SEC agent failed for {symbol}: {exc}")
        return _fallback(symbol)


def _fallback_from_text(symbol: str, business: str, risks: str) -> FundamentalAnalysis:
    """Build a fallback from raw filing text without LLM."""
    risk_lines = [line.strip() for line in risks.split(".")[:3] if len(line.strip()) > 20]
    return FundamentalAnalysis(
        symbol=symbol,
        key_risks=risk_lines[:3] if risk_lines else ["Data unavailable — manual review required"],
        competitive_moat="See Item 1 in SEC filing for details.",
        analyst_summary=f"10-K data retrieved for {symbol} but LLM unavailable for structured extraction.",
        data_quality="medium" if business else "low",
        source="edgar_live" if business else "fallback",
    )


def _fallback(symbol: str) -> FundamentalAnalysis:
    """Complete fallback when no data is available."""
    return FundamentalAnalysis(
        symbol=symbol,
        key_risks=["Data unavailable — manual review required"],
        competitive_moat="Unable to determine — no filing data.",
        analyst_summary=f"Could not fetch 10-K data for {symbol}.",
        data_quality="low",
        source="fallback",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@traceable(name="sec_agent")
async def run_sec_agent(symbols: List[str]) -> List[FundamentalAnalysis]:
    """
    Run SEC fundamental analysis for all symbols in parallel.

    Never raises — every symbol gets a result (live or fallback).
    """
    t0 = time.perf_counter()
    logger.info(f"[sec_agent] Starting analysis for {symbols}")

    tasks = [_analyze_single_symbol(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    analyses: List[FundamentalAnalysis] = []
    for i, res in enumerate(results):
        sym = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"[sec_agent] {sym} raised: {res}")
            analyses.append(_fallback(sym))
        else:
            analyses.append(res)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[sec_agent] Completed {len(symbols)} symbols in {elapsed:.0f}ms")
    return analyses
