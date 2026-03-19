"""
Pytest configuration for FinAgent fallback tests.

Sets up a FastAPI test client with all external API keys explicitly empty,
simulating exactly what a recruiter sees on first clone + run.

sys.path note
─────────────
backend/main.py uses bare imports like `from agent.orchestrator import ...`
(relative to the backend/ directory, not the repo root).  We add backend/
to sys.path BEFORE importing anything so those internal imports resolve
correctly in every environment: local, CI, Docker.
"""

from __future__ import annotations

import os
import sys

# ── path setup (must happen before any app import) ───────────────────────────
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ── wipe all external API keys ───────────────────────────────────────────────
# This is the recruiter scenario: git clone → cp .env.example .env → run.
_EMPTY_KEYS = {
    "GEMINI_API_KEY": "",
    "NEWS_API_KEY": "",
    "ALPHA_VANTAGE_KEY": "",
    "UPSTASH_REDIS_URL": "",
    "LANGFUSE_PUBLIC_KEY": "",
    "LANGFUSE_SECRET_KEY": "",
    "QDRANT_URL": "http://localhost:9999",  # deliberately unreachable
    "BACKTESTER_BIN": "/nonexistent/backtest",
    "USE_MCP": "true",
    "OBSERVABILITY_ENABLED": "false",
    "CONFIDENCE_THRESHOLD": "0.6",
    "REFLECTION_THRESHOLD": "2",
}

for _k, _v in _EMPTY_KEYS.items():
    os.environ[_k] = _v

# ── fixtures ─────────────────────────────────────────────────────────────────
import pytest  # noqa: E402
from httpx import AsyncClient, ASGITransport  # noqa: E402


@pytest.fixture(scope="session")
def app():
    """Import the FastAPI app once per session after env vars are patched."""
    from main import app as _app  # noqa: PLC0415  (backend/ is in sys.path)
    return _app


@pytest.fixture
async def async_client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
