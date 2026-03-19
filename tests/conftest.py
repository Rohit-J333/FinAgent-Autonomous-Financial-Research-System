"""
Pytest configuration for FinAgent fallback tests.

Sets up a FastAPI test client with all external API keys explicitly empty,
simulating exactly what a recruiter sees on first clone + run.
"""

import os
import pytest

# Wipe all external API keys BEFORE any app code is imported.
# This is the recruiter scenario: git clone → cp .env.example .env → run.
_EMPTY_KEYS = {
    "GEMINI_API_KEY": "",
    "NEWS_API_KEY": "",
    "ALPHA_VANTAGE_KEY": "",
    "UPSTASH_REDIS_URL": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "QDRANT_URL": "http://localhost:9999",   # deliberately unreachable
    "BACKTESTER_BIN": "/nonexistent/backtest",
    "USE_MCP": "true",
    "OBSERVABILITY_ENABLED": "false",
    "CONFIDENCE_THRESHOLD": "0.6",
    "REFLECTION_THRESHOLD": "2",
}

for k, v in _EMPTY_KEYS.items():
    os.environ[k] = v


# Now import the app (env vars are already patched above)
from fastapi.testclient import TestClient  # noqa: E402
from httpx import AsyncClient, ASGITransport  # noqa: E402
import pytest_asyncio  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """
    Synchronous test client — Qdrant will fail on startup (port 9999),
    vectordb is set to None, RAG disabled. Everything else runs normally.
    """
    from backend.main import app  # type: ignore
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
async def async_client():
    from backend.main import app  # type: ignore
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
