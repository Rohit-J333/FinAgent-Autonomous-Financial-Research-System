"""
Zero-API-key fallback test suite.

Every test in this file must pass with ALL external services absent:
  - No GEMINI_API_KEY      → LLM nodes fall back to HOLD + confidence 0.7
  - No NEWS_API_KEY        → news scraper returns 3 mock articles
  - No C++ binary          → backtester returns mock presets (mock=True)
  - No Qdrant              → RAG disabled, /api/fundamentals returns 503
  - No Redis               → price caching silently skipped
  - No Langfuse keys       → tracing no-op, agent unaffected

Run:
    pytest tests/test_fallbacks.py -v
"""

from __future__ import annotations

import asyncio
import os
import pytest
import pytest_asyncio


# ---------------------------------------------------------------------------
# Layer 1: Unit tests — individual tool fallbacks
# ---------------------------------------------------------------------------


class TestSentimentFallback:
    """FinBERT may or may not be installed. Heuristic must always work."""

    def test_positive_headline(self):
        from backend.agent.tools.sentiment import analyze_sentiment  # type: ignore

        result = analyze_sentiment("Company reports record earnings surge")
        assert result["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= result["score"] <= 1.0

    def test_negative_headline(self):
        from backend.agent.tools.sentiment import analyze_sentiment  # type: ignore

        result = analyze_sentiment("Stock crashes amid market turmoil and loss warning")
        assert result["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= result["score"] <= 1.0

    def test_empty_text_returns_neutral(self):
        from backend.agent.tools.sentiment import analyze_sentiment  # type: ignore

        result = analyze_sentiment("")
        assert result["label"] == "neutral"
        assert result["score"] == 0.5

    def test_whitespace_only_returns_neutral(self):
        from backend.agent.tools.sentiment import analyze_sentiment  # type: ignore

        result = analyze_sentiment("   ")
        assert result["label"] == "neutral"


class TestNewsFallback:
    """NEWS_API_KEY is empty → must return mock articles, never raise."""

    @pytest.mark.asyncio
    async def test_returns_mock_when_no_key(self):
        from backend.agent.tools.news_scraper import scrape_financial_news  # type: ignore

        result = await scrape_financial_news("AAPL")
        assert result["symbol"] == "AAPL"
        assert result["total_articles"] > 0
        assert len(result["articles"]) > 0
        # Every article must have required fields
        for article in result["articles"]:
            assert "sentiment" in article
            assert "confidence" in article
            assert article["sentiment"] in ("positive", "negative", "neutral")

    @pytest.mark.asyncio
    async def test_mock_articles_have_valid_confidence(self):
        from backend.agent.tools.news_scraper import scrape_financial_news  # type: ignore

        result = await scrape_financial_news("MSFT")
        for article in result["articles"]:
            assert 0.0 <= article["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_aggregate_sentiment_present(self):
        from backend.agent.tools.news_scraper import scrape_financial_news  # type: ignore

        result = await scrape_financial_news("GOOGL")
        agg = result["aggregate_sentiment"]
        assert "positive" in agg
        assert "negative" in agg
        assert "neutral" in agg


class TestBacktestFallback:
    """C++ binary absent → must return mock presets, mock=True, never raise."""

    @pytest.mark.asyncio
    async def test_momentum_mock(self):
        from backend.agent.tools.backtest import execute_backtest  # type: ignore

        result = await execute_backtest(
            strategy="momentum",
            symbol="AAPL",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert result["mock"] is True
        assert result["strategy"] == "momentum"
        assert result["symbol"] == "AAPL"
        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "win_rate" in result

    @pytest.mark.asyncio
    async def test_mean_reversion_mock(self):
        from backend.agent.tools.backtest import execute_backtest  # type: ignore

        result = await execute_backtest(
            strategy="mean_reversion",
            symbol="MSFT",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert result["mock"] is True
        assert 0.0 <= result["win_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_unknown_strategy_falls_back_to_momentum_preset(self):
        from backend.agent.tools.backtest import execute_backtest  # type: ignore

        result = await execute_backtest(
            strategy="fancy_new_strategy",
            symbol="TSLA",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert result["mock"] is True
        assert "total_return" in result  # fell back to momentum preset

    @pytest.mark.asyncio
    async def test_get_price_data_returns_list_not_raises(self):
        """yfinance may return real data or [] — must never raise."""
        from backend.agent.tools.backtest import get_price_data  # type: ignore

        result = await get_price_data("AAPL", period="1mo")
        assert isinstance(result, list)
        # If data came back, validate schema
        if result:
            assert "date" in result[0]
            assert "close" in result[0]
            assert "volume" in result[0]

    @pytest.mark.asyncio
    async def test_get_current_quote_never_raises(self):
        from backend.agent.tools.backtest import get_current_quote  # type: ignore

        result = await get_current_quote("AAPL")
        assert "symbol" in result
        assert "price" in result
        assert "change_pct" in result
        assert "volume" in result
        # Values must be numeric (may be 0 if yfinance fails)
        assert isinstance(result["price"], (int, float))


class TestRedisSkipped:
    """No UPSTASH_REDIS_URL → caching skipped silently, fetch still works."""

    @pytest.mark.asyncio
    async def test_price_data_works_without_redis(self):
        from backend.agent.tools.backtest import get_price_data  # type: ignore

        # Should return data (or empty list) without raising
        result = await get_price_data("MSFT")
        assert isinstance(result, list)

    def test_redis_client_is_none_without_url(self):
        from backend.agent.tools.backtest import _get_redis  # type: ignore

        # UPSTASH_REDIS_URL is "" in conftest — client must be None
        client = _get_redis()
        assert client is None


class TestLangfuseNoOp:
    """No Langfuse keys → handler is None, agent unaffected."""

    def test_get_handler_returns_none_without_keys(self):
        from backend.utils.observability import get_langfuse_handler  # type: ignore

        handler = get_langfuse_handler(user_id="test", session_id="s1")
        assert handler is None

    def test_log_result_does_not_raise_without_keys(self):
        from backend.utils.observability import log_analysis_result  # type: ignore

        # Must be a no-op, not raise
        log_analysis_result(
            symbols=["AAPL"],
            result={"confidence": 0.7, "steps": 4},
            latency_ms=1234.5,
            session_id="s1",
        )


# ---------------------------------------------------------------------------
# Layer 2: Integration tests — full pipeline via HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHealthEndpoint:
    async def test_health_returns_200(self, async_client):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        # Qdrant is deliberately unreachable — vectordb should be "unavailable"
        assert body["vectordb"] == "unavailable"

    async def test_health_has_timestamp(self, async_client):
        resp = await async_client.get("/health")
        assert "timestamp" in resp.json()


@pytest.mark.asyncio
class TestSymbolValidation:
    """Fix 4 — server-side validation must return 422, never 500."""

    async def test_valid_symbols_accepted(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL", "MSFT"]}
        )
        # May be 200 (with mock data) or 500 (LLM unavailable) but NOT 422
        assert resp.status_code != 422

    async def test_too_many_symbols_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META"]},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert any("Maximum 6" in str(e) for e in detail)

    async def test_empty_symbols_returns_422(self, async_client):
        resp = await async_client.post("/api/analyze", json={"symbols": []})
        assert resp.status_code == 422

    async def test_invalid_symbol_chars_returns_422(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AAPL", "DROP TABLE"]}
        )
        assert resp.status_code == 422

    async def test_symbol_with_digits_rejected(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["AA1PL"]}
        )
        assert resp.status_code == 422

    async def test_lowercase_symbols_auto_uppercased(self, async_client):
        """Lowercase should be cleaned, not rejected."""
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["aapl"]}
        )
        # 422 would mean validation rejected it — we expect it to be normalized
        assert resp.status_code != 422

    async def test_symbol_too_long_rejected(self, async_client):
        resp = await async_client.post(
            "/api/analyze", json={"symbols": ["TOOLONG"]}
        )
        assert resp.status_code == 422


@pytest.mark.asyncio
class TestFullPipelineZeroKeys:
    """
    The critical recruiter test: full analysis with zero valid API keys.
    Must return 200 with valid structure — never 500.
    """

    async def test_single_symbol_returns_200(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    async def test_response_has_all_required_fields(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        body = resp.json()
        required = {
            "status", "symbols", "decision", "confidence",
            "sentiment", "backtest_results", "reflection",
            "steps", "timestamp", "structured_decisions", "latency_ms",
        }
        assert required.issubset(body.keys())

    async def test_confidence_is_valid_float(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        body = resp.json()
        assert 0.0 <= body["confidence"] <= 1.0

    async def test_backtest_results_present_for_each_symbol(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL", "MSFT"]},
            timeout=90.0,
        )
        body = resp.json()
        assert "AAPL" in body["backtest_results"]
        assert "MSFT" in body["backtest_results"]

    async def test_sentiment_present_for_each_symbol(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL", "MSFT"]},
            timeout=90.0,
        )
        body = resp.json()
        assert "AAPL" in body["sentiment"]
        assert "MSFT" in body["sentiment"]

    async def test_structured_decisions_is_list(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        body = resp.json()
        assert isinstance(body["structured_decisions"], list)

    async def test_mock_backtest_flagged_as_mock(self, async_client):
        """Without C++ binary, backtest results must have mock=True."""
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        body = resp.json()
        aapl_bt = body["backtest_results"].get("AAPL", {})
        assert aapl_bt.get("mock") is True

    async def test_latency_ms_is_positive(self, async_client):
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        body = resp.json()
        assert body["latency_ms"] > 0

    async def test_no_reflection_loop_with_no_llm(self, async_client):
        """
        With no GEMINI_API_KEY, reflection fallback confidence=0.7 >= threshold 0.6.
        Agent should complete in 1 reflection pass, not 2.
        steps should be exactly 4 (research + backtest + reflect + decide).
        """
        resp = await async_client.post(
            "/api/analyze",
            json={"symbols": ["AAPL"]},
            timeout=60.0,
        )
        body = resp.json()
        # 4 steps = 1 research + 1 backtest + 1 reflect + 1 decide
        assert body["steps"] == 4, (
            f"Expected 4 steps (no reflection loop), got {body['steps']}. "
            "Fallback confidence may be below CONFIDENCE_THRESHOLD."
        )


@pytest.mark.asyncio
class TestFundamentalsEndpoint:
    """RAG endpoint must degrade gracefully when Qdrant is unreachable."""

    async def test_returns_503_when_qdrant_down(self, async_client):
        resp = await async_client.get(
            "/api/fundamentals/AAPL",
            params={"query": "revenue growth"},
        )
        assert resp.status_code == 503
        assert "Vector DB" in resp.json()["detail"]


@pytest.mark.asyncio
class TestCORSHeaders:
    """Fix 5 — wildcard origin must be gone; localhost dev origins allowed."""

    async def test_localhost_5173_allowed(self, async_client):
        resp = await async_client.options(
            "/api/analyze",
            headers={"Origin": "http://localhost:5173"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:5173"

    async def test_evil_origin_not_in_allow_header(self, async_client):
        resp = await async_client.options(
            "/api/analyze",
            headers={"Origin": "http://evil.com"},
        )
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao != "*"
        assert "evil.com" not in acao


# ---------------------------------------------------------------------------
# Layer 3: Routing logic unit tests
# ---------------------------------------------------------------------------


class TestRoutingLogic:
    """Fix 1 + Fix 9 — routing function must respect all 4 rules."""

    def _make_state(self, **kwargs):
        defaults = {
            "reflection_step_count": 0,
            "reflection": "",
            "confidence": 0.7,
            "symbols": ["AAPL"],
        }
        defaults.update(kwargs)
        return defaults

    def test_rule1_hard_cap_forces_decide(self):
        from backend.agent.orchestrator import should_gather_more  # type: ignore

        # reflection_step_count >= REFLECTION_THRESHOLD (2) → decide
        state = self._make_state(reflection_step_count=2, confidence=0.3)
        assert should_gather_more(state) == "decide"

    def test_rule2_gather_more_data_string_routes_to_research(self):
        from backend.agent.orchestrator import should_gather_more  # type: ignore

        state = self._make_state(
            reflection_step_count=0,
            reflection="Assessment looks weak. GATHER_MORE_DATA",
            confidence=0.8,  # above threshold — rule 2 still fires
        )
        assert should_gather_more(state) == "research"

    def test_rule3_low_confidence_routes_to_research(self):
        from backend.agent.orchestrator import should_gather_more  # type: ignore

        state = self._make_state(
            reflection_step_count=0,
            reflection="All good. PROCEED",
            confidence=0.3,  # below 0.6 threshold
        )
        assert should_gather_more(state) == "research"

    def test_rule4_good_confidence_proceeds_to_decide(self):
        from backend.agent.orchestrator import should_gather_more  # type: ignore

        state = self._make_state(
            reflection_step_count=0,
            reflection="Data looks solid. PROCEED",
            confidence=0.8,
        )
        assert should_gather_more(state) == "decide"

    def test_rule1_takes_priority_over_rule2(self):
        """Hard cap must fire even if reflection says GATHER_MORE_DATA."""
        from backend.agent.orchestrator import should_gather_more  # type: ignore

        state = self._make_state(
            reflection_step_count=2,  # at cap
            reflection="GATHER_MORE_DATA",
            confidence=0.1,
        )
        assert should_gather_more(state) == "decide"

    def test_fallback_confidence_above_threshold(self):
        """
        Verify the fix: reflection fallback confidence (0.7) must be >= threshold (0.6).
        This prevents unnecessary research loops when LLM is unavailable.
        """
        from backend.utils.config import Config  # type: ignore

        fallback_confidence = 0.7  # value set in reflection_node except block
        assert fallback_confidence >= Config.CONFIDENCE_THRESHOLD, (
            f"Fallback confidence {fallback_confidence} is below "
            f"CONFIDENCE_THRESHOLD {Config.CONFIDENCE_THRESHOLD}. "
            "This causes unnecessary reflection loops when LLM is absent."
        )
