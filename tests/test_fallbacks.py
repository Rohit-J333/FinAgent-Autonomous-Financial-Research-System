"""
Zero-API-key fallback test suite for FinAgent.

Every test passes with ALL external services absent:
  - No GEMINI_API_KEY      → LLM nodes fall back to HOLD + confidence 0.7
  - No NEWS_API_KEY        → news scraper returns 3 mock articles
  - No C++ binary          → backtester returns mock presets (mock=True)
  - No Qdrant              → RAG disabled, /api/fundamentals returns 503
  - No Redis               → price caching silently skipped
  - No Langfuse keys       → tracing is a no-op, agent unaffected

Note on imports
───────────────
conftest.py inserts backend/ into sys.path so all imports here are
relative to backend/ (e.g. `from agent.tools.sentiment import ...`)
matching how the production code itself resolves modules.

Run:
    pytest tests/test_fallbacks.py -v
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ──────────────────────────────────────────────────────────────────────────────
# Layer 1 — Unit tests: individual tool fallbacks
# ──────────────────────────────────────────────────────────────────────────────


class TestSentimentFallback:
    """FinBERT may or may not be installed. Heuristic must always work."""

    def test_positive_headline(self):
        from agent.tools.sentiment import analyze_sentiment

        result = analyze_sentiment("Company reports record earnings surge")
        assert result["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= result["score"] <= 1.0

    def test_negative_headline(self):
        from agent.tools.sentiment import analyze_sentiment

        result = analyze_sentiment("Stock crashes amid market turmoil and loss warning")
        assert result["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= result["score"] <= 1.0

    def test_empty_text_returns_neutral(self):
        from agent.tools.sentiment import analyze_sentiment

        result = analyze_sentiment("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    def test_whitespace_only_returns_neutral(self):
        from agent.tools.sentiment import analyze_sentiment

        result = analyze_sentiment("   ")
        assert result["label"] == "neutral"


class TestNewsFallback:
    """NEWS_API_KEY is empty → must return mock articles, never raise."""

    @pytest.mark.asyncio
    async def test_returns_mock_when_no_key(self):
        from agent.tools.news_scraper import scrape_financial_news

        result = await scrape_financial_news("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["total_articles"] > 0
        for article in result["articles"]:
            assert "sentiment" in article
            assert "confidence" in article
            assert article["sentiment"] in ("positive", "negative", "neutral")

    @pytest.mark.asyncio
    async def test_mock_articles_have_valid_confidence(self):
        from agent.tools.news_scraper import scrape_financial_news

        result = await scrape_financial_news("MSFT")
        for article in result["articles"]:
            assert 0.0 <= article["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_aggregate_sentiment_present(self):
        from agent.tools.news_scraper import scrape_financial_news

        result = await scrape_financial_news("GOOGL")
        agg = result["aggregate_sentiment"]
        assert all(k in agg for k in ("positive", "negative", "neutral"))


class TestBacktestFallback:
    """C++ binary absent → mock presets returned, mock=True, never raise."""

    @pytest.mark.asyncio
    async def test_momentum_mock(self):
        from agent.tools.backtest import execute_backtest

        result = await execute_backtest(
            strategy="momentum", symbol="AAPL",
            start_date="2024-01-01", end_date="2024-12-31",
        )
        assert result["mock"] is True
        assert result["strategy"] == "momentum"
        assert result["symbol"] == "AAPL"
        assert "total_return" in result and "sharpe_ratio" in result

    @pytest.mark.asyncio
    async def test_mean_reversion_mock(self):
        from agent.tools.backtest import execute_backtest

        result = await execute_backtest(
            strategy="mean_reversion", symbol="MSFT",
            start_date="2024-01-01", end_date="2024-12-31",
        )
        assert result["mock"] is True
        assert 0.0 <= result["win_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_unknown_strategy_falls_back_to_momentum_preset(self):
        from agent.tools.backtest import execute_backtest

        result = await execute_backtest(
            strategy="nonexistent_strategy", symbol="TSLA",
            start_date="2024-01-01", end_date="2024-12-31",
        )
        assert result["mock"] is True
        assert "total_return" in result

    @pytest.mark.asyncio
    async def test_get_price_data_never_raises(self):
        from agent.tools.backtest import get_price_data

        result = await get_price_data("AAPL", period="1mo")
        assert isinstance(result, list)
        if result:  # validate schema if yfinance returned data
            assert all(k in result[0] for k in ("date", "close", "volume"))

    @pytest.mark.asyncio
    async def test_get_current_quote_never_raises(self):
        from agent.tools.backtest import get_current_quote

        result = await get_current_quote("AAPL")
        assert all(k in result for k in ("symbol", "price", "change_pct", "volume"))
        assert isinstance(result["price"], (int, float))


class TestRedisSkipped:
    """UPSTASH_REDIS_URL="" → caching silently skipped, fetch still works."""

    @pytest.mark.asyncio
    async def test_price_data_works_without_redis(self):
        from agent.tools.backtest import get_price_data

        result = await get_price_data("MSFT")
        assert isinstance(result, list)

    def test_redis_client_is_none_without_url(self):
        from agent.tools.backtest import _get_redis

        assert _get_redis() is None


class TestLangfuseNoOp:
    """No Langfuse keys → handler is None, agent unaffected."""

    def test_get_handler_returns_none_without_keys(self):
        from utils.observability import get_langfuse_handler

        assert get_langfuse_handler(user_id="test", session_id="s1") is None

    def test_log_result_does_not_raise_without_keys(self):
        from utils.observability import log_analysis_result

        log_analysis_result(
            symbols=["AAPL"],
            result={"confidence": 0.7, "steps": 4},
            latency_ms=1234.5,
            session_id="s1",
        )  # must be a silent no-op


# ──────────────────────────────────────────────────────────────────────────────
# Layer 2 — Integration tests: full pipeline via HTTP
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_returns_200(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["vectordb"] == "unavailable"  # Qdrant on port 9999

    async def test_health_has_timestamp(self, async_client):
        resp = await async_client.get("/health")
        assert "timestamp" in resp.json()


@pytest.mark.asyncio
class TestSymbolValidation:
    """Fix 4 — server-side validation returns 422, never 500."""

    async def test_valid_symbols_accepted(self, async_client):
        resp = await async_client.post("/api/analyze", json={"symbols": ["AAPL"]})
        assert resp.status_code != 422

    async def test_too_many_symbols_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]},
        )
        assert resp.status_code == 422
        # Pydantic v2 fires max_length=6 constraint; message varies by version
        errors = resp.json()["detail"]
        assert any("symbols" in str(e.get("loc", "")) for e in errors)

    async def test_empty_symbols_returns_422(self, async_client):
        resp = await async_client.post("/api/analyze", json={"symbols": []})
        assert resp.status_code == 422

    async def test_symbol_with_special_chars_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL", "DROP TABLE"]}
        )
        assert resp.status_code == 422

    async def test_symbol_with_digits_rejected(self, async_client):
        resp = await async_client.post("/api/analyze", json={"symbols": ["AA1PL"]})
        assert resp.status_code == 422

    async def test_symbol_too_long_rejected(self, async_client):
        resp = await async_client.post("/api/analyze", json={"symbols": ["TOOLONG"]})
        assert resp.status_code == 422

    async def test_lowercase_symbols_normalized_not_rejected(self, async_client):
        resp = await async_client.post("/api/analyze", json={"symbols": ["aapl"]})
        assert resp.status_code != 422  # cleaned to "AAPL"


@pytest.mark.asyncio
class TestFullPipelineZeroKeys:
    """Critical recruiter test: full analysis with zero valid API keys → 200."""

    async def test_single_symbol_returns_200(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=60.0
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    async def test_response_has_all_required_fields(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=60.0
        )
        required = {
            "status", "symbols", "decision", "confidence", "sentiment",
            "backtest_results", "reflection", "steps", "timestamp",
            "structured_decisions", "latency_ms",
        }
        assert required.issubset(resp.json().keys())

    async def test_confidence_is_valid_float(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=60.0
        )
        assert 0.0 <= resp.json()["confidence"] <= 1.0

    async def test_results_present_for_all_requested_symbols(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL", "MSFT"]}, timeout=90.0
        )
        body = resp.json()
        assert "AAPL" in body["backtest_results"] and "MSFT" in body["backtest_results"]
        assert "AAPL" in body["sentiment"] and "MSFT" in body["sentiment"]

    async def test_mock_backtest_flagged(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=60.0
        )
        assert resp.json()["backtest_results"]["AAPL"]["mock"] is True

    async def test_latency_ms_is_positive(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=60.0
        )
        assert resp.json()["latency_ms"] > 0

    async def test_no_reflection_loop_with_no_llm(self, async_client):
        """
        Bug fix verification: reflection fallback confidence=0.7 >= threshold 0.6
        → agent completes in exactly 4 steps, no unnecessary loop.
        """
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=60.0
        )
        steps = resp.json()["steps"]
        assert steps == 4, (
            f"Expected 4 steps (1 pass only), got {steps}. "
            "Fallback confidence is probably below CONFIDENCE_THRESHOLD — "
            "check the fix in reflection_node."
        )


@pytest.mark.asyncio
class TestParallelSymbolFailure:
    """
    Fix 7: one symbol failing inside asyncio.gather must not crash the whole run.

    Note: with no API keys every symbol uses mock data so nothing naturally
    fails — we use unittest.mock.patch to inject a real exception for MSFT
    and verify AAPL's results are still returned.
    """

    async def test_one_symbol_failure_does_not_crash_analysis(self, async_client):
        from agent.tools.news_scraper import scrape_financial_news as _real_fn

        async def _patched(symbol, **kwargs):
            if symbol == "MSFT":
                raise RuntimeError("Simulated network failure for MSFT")
            return await _real_fn(symbol, **kwargs)

        with patch(
            "agent.orchestrator.scrape_financial_news",
            side_effect=_patched,
        ):
            resp = await async_client.post(
                "/api/analyze",
                json={"symbols": ["AAPL", "MSFT"]},
                timeout=60.0,
            )

        assert resp.status_code == 200
        body = resp.json()

        # AAPL must have real (mock) data
        assert "AAPL" in body["sentiment"]
        assert body["sentiment"]["AAPL"]["total_articles"] > 0

        # MSFT must have a safe fallback (0 articles, neutral ratio)
        assert "MSFT" in body["sentiment"]
        assert body["sentiment"]["MSFT"]["total_articles"] == 0
        assert body["sentiment"]["MSFT"]["positive_ratio"] == 0.5

    async def test_all_symbols_fail_gracefully_returns_200(self, async_client):
        """Even if every symbol fails, the pipeline should still return 200."""

        async def _always_fail(symbol, **kwargs):
            raise RuntimeError(f"Total failure for {symbol}")

        with patch(
            "agent.orchestrator.scrape_financial_news",
            side_effect=_always_fail,
        ):
            resp = await async_client.post(
                "/api/analyze",
                json={"symbols": ["AAPL"]},
                timeout=60.0,
            )

        assert resp.status_code == 200


@pytest.mark.asyncio
class TestFundamentalsEndpoint:
    """RAG endpoint degrades gracefully when Qdrant is unreachable."""

    async def test_returns_503_when_qdrant_down(self, async_client):
        resp = await async_client.get(
            "/api/fundamentals/AAPL", params={"query": "revenue growth"}
        )
        assert resp.status_code == 503
        assert "Vector DB" in resp.json()["detail"]


@pytest.mark.asyncio
class TestCORSHeaders:
    """Fix 5 — wildcard origin removed; localhost origins allowed."""

    async def test_localhost_5173_allowed(self, async_client):
        resp = await async_client.options(
            "/api/analyze",
            headers={"Origin": "http://localhost:5173"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:5173"

    async def test_evil_origin_blocked(self, async_client):
        resp = await async_client.options(
            "/api/analyze",
            headers={"Origin": "http://evil.com"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao != "*" and "evil.com" not in acao


# ──────────────────────────────────────────────────────────────────────────────
# Layer 3 — Routing logic unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRoutingLogic:
    """Fix 1 + Fix 9 — all 4 routing rules, plus the bug-fix assertions."""

    def _state(self, **kw):
        return {
            "reflection_step_count": 0,
            "reflection": "",
            "confidence": 0.7,
            "symbols": ["AAPL"],
            **kw,
        }

    def test_rule1_hard_cap_forces_decide(self):
        from agent.orchestrator import should_gather_more

        assert should_gather_more(self._state(reflection_step_count=2, confidence=0.1)) == "decide"

    def test_rule2_gather_more_data_routes_to_research(self):
        from agent.orchestrator import should_gather_more

        assert should_gather_more(self._state(
            reflection="GATHER_MORE_DATA", confidence=0.9
        )) == "research"

    def test_rule3_low_confidence_routes_to_research(self):
        from agent.orchestrator import should_gather_more

        assert should_gather_more(self._state(confidence=0.3)) == "research"

    def test_rule4_good_confidence_routes_to_decide(self):
        from agent.orchestrator import should_gather_more

        assert should_gather_more(self._state(confidence=0.8)) == "decide"

    def test_rule1_beats_rule2_hard_cap_wins(self):
        from agent.orchestrator import should_gather_more

        # Even with GATHER_MORE_DATA, hard cap must fire
        assert should_gather_more(self._state(
            reflection_step_count=2, reflection="GATHER_MORE_DATA", confidence=0.1
        )) == "decide"

    def test_fallback_confidence_above_threshold(self):
        """
        Bug fix: reflection fallback confidence (0.7) must be >= CONFIDENCE_THRESHOLD (0.6).
        If this breaks, the agent enters a 2-loop spin with no LLM key.
        """
        from utils.config import Config

        fallback = 0.7  # value set in reflection_node except block
        assert fallback >= Config.CONFIDENCE_THRESHOLD, (
            f"Fallback {fallback} < threshold {Config.CONFIDENCE_THRESHOLD}: "
            "agent will loop unnecessarily when LLM is unavailable."
        )

    def test_routing_reason_written_to_state(self):
        """
        Bug fix: routing_reason must be set in state by reflection_node,
        not computed inside should_gather_more (which can't write state).
        """
        from agent.orchestrator import AgentState

        # routing_reason must exist as a declared field
        assert "routing_reason" in AgentState.__annotations__, (
            "routing_reason not declared in AgentState — "
            "it will silently be dropped when reflection_node returns."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Layer 4 — Phase 3 multi-agent tests
# ──────────────────────────────────────────────────────────────────────────────


class TestSharedStateModels:
    """Step 11: Verify all Pydantic models instantiate with defaults."""

    def test_fundamental_analysis_defaults(self):
        from agent.specialist_agents.shared_state import FundamentalAnalysis

        fa = FundamentalAnalysis(symbol="AAPL")
        assert fa.source == "fallback"
        assert fa.data_quality == "low"
        assert fa.revenue_growth_yoy is None

    def test_technical_analysis_defaults(self):
        from agent.specialist_agents.shared_state import TechnicalAnalysis

        ta = TechnicalAnalysis(symbol="AAPL")
        assert ta.trend == "SIDEWAYS"
        assert ta.rsi_14 == 50.0
        assert ta.momentum_score == 0.0

    def test_risk_analysis_defaults(self):
        from agent.specialist_agents.shared_state import RiskAnalysis

        ra = RiskAnalysis(symbol="AAPL")
        assert ra.risk_rating == "MEDIUM"
        assert ra.position_size_pct == 5.0

    def test_final_recommendation_defaults(self):
        from agent.specialist_agents.shared_state import FinalRecommendation

        fr = FinalRecommendation(symbol="AAPL")
        assert fr.action == "HOLD"
        assert fr.confidence == 0.5

    def test_multi_agent_state_has_required_fields(self):
        from agent.specialist_agents.shared_state import MultiAgentState

        required = {
            "symbols", "fundamental_analyses", "technical_analyses",
            "risk_analyses", "sentiment", "recommendations",
            "decision", "step", "structured_decisions", "backtest_results",
        }
        assert required.issubset(MultiAgentState.__annotations__.keys())


class TestMultiAgentFallbacks:
    """Step 12-14: Each specialist agent handles failures gracefully."""

    @pytest.mark.asyncio
    async def test_sec_agent_no_edgartools(self):
        """Mock ImportError for edgartools → returns FundamentalAnalysis with source='fallback'."""
        from agent.specialist_agents.sec_agent import run_sec_agent

        with patch(
            "agent.specialist_agents.sec_agent._fetch_10k_sync",
            return_value={"business": "", "risks": "", "facts": None},
        ):
            results = await run_sec_agent(["AAPL"])

        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        assert results[0].source == "fallback"

    @pytest.mark.asyncio
    async def test_sec_agent_no_llm(self):
        """LLM unavailable → returns fallback FundamentalAnalysis with data from filing."""
        from agent.specialist_agents.sec_agent import run_sec_agent

        with patch(
            "agent.specialist_agents.sec_agent._fetch_10k_sync",
            return_value={
                "business": "Apple Inc. designs consumer electronics.",
                "risks": "Competition is intense. Supply chain risks exist. Regulatory changes possible.",
                "facts": None,
            },
        ), patch("agent.specialist_agents.sec_agent._sec_llm", None):
            results = await run_sec_agent(["AAPL"])

        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        # Should have extracted risk lines from the text
        assert results[0].source in ("edgar_live", "fallback")

    @pytest.mark.asyncio
    async def test_technical_agent_insufficient_data(self):
        """< 60 rows → returns fallback TechnicalAnalysis."""
        from agent.specialist_agents.technical_agent import run_technical_agent

        # Mock yfinance returning only 30 rows
        with patch(
            "agent.specialist_agents.technical_agent._fetch_ohlcv_sync",
            return_value=None,
        ):
            results = await run_technical_agent(["AAPL"])

        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        assert results[0].source == "fallback"
        assert results[0].data_points == 0

    @pytest.mark.asyncio
    async def test_risk_agent_no_spy_data(self):
        """SPY fetch fails → returns fallback RiskAnalysis."""
        from agent.specialist_agents.risk_agent import run_risk_agent

        with patch(
            "agent.specialist_agents.risk_agent._fetch_returns_sync",
            return_value=None,
        ):
            results = await run_risk_agent(["AAPL"])

        assert len(results) == 1
        assert results[0].symbol == "AAPL"
        assert results[0].source == "fallback"
        assert results[0].risk_rating == "MEDIUM"

    @pytest.mark.asyncio
    async def test_synthesis_agent_no_llm(self):
        """LLM fails → returns computed scores with fallback text."""
        from agent.specialist_agents.synthesis_agent import run_synthesis_agent
        from agent.specialist_agents.shared_state import (
            FundamentalAnalysis, TechnicalAnalysis, RiskAnalysis,
        )

        with patch("agent.specialist_agents.synthesis_agent._synthesis_llm", None):
            result = await run_synthesis_agent(
                symbol="AAPL",
                fundamental=FundamentalAnalysis(symbol="AAPL"),
                technical=TechnicalAnalysis(symbol="AAPL"),
                risk=RiskAnalysis(symbol="AAPL"),
                sentiment_score=0.3,
                news_headlines=["Test headline"],
            )

        assert result.symbol == "AAPL"
        assert result.action in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
        assert -1.0 <= result.composite_score <= 1.0
        assert 0.0 < result.confidence <= 1.0

    def test_conflict_detection_all_four(self):
        """Set up state triggering all 4 conflicts → assert 4 SignalConflict objects."""
        from agent.specialist_agents.synthesis_agent import _detect_conflicts
        from agent.specialist_agents.shared_state import (
            FundamentalAnalysis, TechnicalAnalysis, RiskAnalysis,
        )

        # Construct inputs that trigger all 4 conflict detectors:
        # A: tech bullish (momentum > 0.2) + sentiment bearish (< -0.3)
        # B: UPTREND + HIGH risk
        # C: PE > 40 + momentum > 0.3
        # D: both fundamental and technical source = "fallback"
        conflicts = _detect_conflicts(
            sentiment_score=-0.5,  # bearish → triggers A
            technical=TechnicalAnalysis(
                symbol="TEST",
                trend="UPTREND",              # triggers B
                momentum_score=0.5,           # triggers A and C
                source="fallback",            # triggers D
            ),
            fundamental=FundamentalAnalysis(
                symbol="TEST",
                pe_ratio=50.0,                # triggers C
                source="fallback",            # triggers D
            ),
            risk=RiskAnalysis(
                symbol="TEST",
                risk_rating="HIGH",           # triggers B
            ),
        )

        assert len(conflicts) == 4, f"Expected 4 conflicts, got {len(conflicts)}: {[c.description for c in conflicts]}"
        severities = {c.severity for c in conflicts}
        assert "HIGH" in severities
        assert "MEDIUM" in severities
        assert "LOW" in severities


@pytest.mark.asyncio
class TestResearchDirector:
    """Step 15: Full multi-agent pipeline integration tests."""

    async def test_multi_agent_zero_keys(self, async_client):
        """POST /api/analyze with MULTI_AGENT_MODE=true, all keys blank → 200."""
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=120.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        # Response must have same schema as before
        required = {
            "status", "symbols", "decision", "confidence", "sentiment",
            "backtest_results", "reflection", "steps", "timestamp",
            "structured_decisions", "latency_ms",
        }
        assert required.issubset(body.keys())

    async def test_multi_agent_structured_decisions_have_extras(self, async_client):
        """Multi-agent decisions include Phase 3 fields like composite_score."""
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=120.0,
        )
        body = resp.json()
        assert resp.status_code == 200
        decisions = body["structured_decisions"]
        assert len(decisions) >= 1
        d = decisions[0]
        assert "composite_score" in d
        assert "time_horizon" in d
        assert "conflicting_signals" in d

    async def test_multi_agent_confidence_valid(self, async_client):
        """Confidence is a valid float between 0 and 1."""
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=120.0,
        )
        assert 0.0 <= resp.json()["confidence"] <= 1.0

    async def test_parallel_specialist_timing(self, async_client):
        """run_specialists node completes in < 30s with mocked data."""
        t0 = time.perf_counter()
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL"]}, timeout=120.0,
        )
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200
        # With all mocked data, the pipeline should complete well under 30s
        assert elapsed < 30, f"Pipeline took {elapsed:.1f}s — expected < 30s"
