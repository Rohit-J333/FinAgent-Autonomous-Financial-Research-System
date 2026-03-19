"""
Langfuse observability integration for FinAgent.

All functions degrade gracefully to no-ops when:
  - Langfuse keys are not configured
  - The langfuse package is not installed
  - Any Langfuse API call fails

A tracing failure must NEVER crash the agent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from utils.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Langfuse client
# ---------------------------------------------------------------------------

_langfuse = None
_langfuse_checked = False


def _get_langfuse():
    """Return a Langfuse client singleton, or None if unavailable."""
    global _langfuse, _langfuse_checked
    if _langfuse_checked:
        return _langfuse
    _langfuse_checked = True

    if not Config.LANGFUSE_PUBLIC_KEY or not Config.LANGFUSE_SECRET_KEY:
        logger.info("Langfuse keys not set – observability disabled.")
        return None

    if not Config.OBSERVABILITY_ENABLED:
        logger.info("OBSERVABILITY_ENABLED=false – tracing disabled.")
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _langfuse = Langfuse(
            public_key=Config.LANGFUSE_PUBLIC_KEY,
            secret_key=Config.LANGFUSE_SECRET_KEY,
            host=Config.LANGFUSE_HOST,
        )
        logger.info(f"Langfuse client initialized (host={Config.LANGFUSE_HOST}).")
    except ImportError:
        logger.info("langfuse package not installed – tracing disabled.")
    except Exception as exc:
        logger.warning(f"Langfuse init failed: {exc}")

    return _langfuse


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_langfuse_handler(
    user_id: str = "anonymous",
    session_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Build a LangChain CallbackHandler for Langfuse tracing.

    Returns ``None`` when Langfuse is not available, so callers can do::

        handler = get_langfuse_handler(...)
        config = {"callbacks": [handler]} if handler else {}
        await agent.ainvoke(state, config)
    """
    lf = _get_langfuse()
    if lf is None:
        return None

    try:
        from langfuse.callback import CallbackHandler  # type: ignore

        return CallbackHandler(
            public_key=Config.LANGFUSE_PUBLIC_KEY,
            secret_key=Config.LANGFUSE_SECRET_KEY,
            host=Config.LANGFUSE_HOST,
            trace_name="finagent_analysis",
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
    except ImportError:
        logger.debug("langfuse.callback not available.")
        return None
    except Exception as exc:
        logger.warning(f"Failed to create Langfuse handler: {exc}")
        return None


def log_analysis_result(
    symbols: list[str],
    result: Dict[str, Any],
    latency_ms: float,
    session_id: str = "",
) -> None:
    """
    Log a score and metadata to Langfuse for the completed analysis trace.
    No-op if observability is disabled.
    """
    lf = _get_langfuse()
    if lf is None:
        return

    try:
        trace = lf.trace(
            name="finagent_analysis_result",
            session_id=session_id,
            metadata={
                "symbols": symbols,
                "latency_ms": round(latency_ms, 1),
                "steps": result.get("steps", 0),
            },
        )
        trace.score(
            name="confidence",
            value=result.get("confidence", 0.0),
            comment=f"Symbols: {symbols}, latency: {latency_ms:.0f}ms",
        )
    except Exception as exc:
        logger.warning(f"Langfuse log_analysis_result failed: {exc}")
