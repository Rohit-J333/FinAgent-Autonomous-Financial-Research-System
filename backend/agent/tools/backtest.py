"""
Backtest tool – calls the C++ strategy engine binary via subprocess.
Falls back to a Python-based mock backtester when the binary is absent.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, List

logger = logging.getLogger(__name__)

BACKTESTER_BIN = os.getenv("BACKTESTER_BIN", "./strategy_engine/bin/backtest")


def execute_backtest(
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
    params: Dict | None = None,
) -> Dict:
    """
    Execute a backtest via the C++ binary.

    Args:
        strategy:   "momentum" | "mean_reversion" | "macd"
        symbol:     Stock ticker
        start_date: YYYY-MM-DD
        end_date:   YYYY-MM-DD
        params:     Strategy-specific parameters

    Returns:
        {
            "strategy": str,
            "symbol": str,
            "total_return": float,
            "sharpe_ratio": float,
            "max_drawdown": float,
            "win_rate": float,
            "total_trades": int,
        }
    """
    params = params or {}

    if not os.path.isfile(BACKTESTER_BIN):
        logger.warning(
            f"C++ backtester binary not found at '{BACKTESTER_BIN}'. "
            "Using Python mock backtester."
        )
        return _mock_backtest(strategy, symbol, start_date, end_date, params)

    backtest_input = {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "parameters": params,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        json.dump(backtest_input, f)
        input_file = f.name

    output_file = input_file.replace(".json", "_output.json")

    try:
        result = subprocess.run(
            [BACKTESTER_BIN, input_file, output_file],
            timeout=30,
            capture_output=True,
        )

        if result.returncode != 0:
            err = result.stderr.decode()
            logger.error(f"Backtester error: {err}")
            return {"error": err}

        with open(output_file, "r") as f:
            results = json.load(f)

        return {
            "strategy": strategy,
            "symbol": symbol,
            "total_return": results.get("total_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "total_trades": results.get("total_trades", 0),
        }

    except subprocess.TimeoutExpired:
        logger.error("Backtester timed out after 30 seconds.")
        return {"error": "timeout"}
    finally:
        for path in [input_file, output_file]:
            if os.path.exists(path):
                os.remove(path)


# ---------------------------------------------------------------------------
# Python mock backtester (deterministic, used when binary is absent)
# ---------------------------------------------------------------------------

def _mock_backtest(
    strategy: str,
    symbol: str,
    start_date: str,
    end_date: str,
    params: Dict,
) -> Dict:
    """
    Deterministic mock backtest results keyed by strategy type.
    Values are plausible but synthetic – for demo / development only.
    """
    presets = {
        "momentum": {
            "total_return": 0.23,
            "sharpe_ratio": 1.42,
            "max_drawdown": -0.08,
            "win_rate": 0.61,
            "total_trades": 48,
        },
        "mean_reversion": {
            "total_return": 0.15,
            "sharpe_ratio": 1.10,
            "max_drawdown": -0.06,
            "win_rate": 0.57,
            "total_trades": 72,
        },
        "macd": {
            "total_return": 0.19,
            "sharpe_ratio": 1.25,
            "max_drawdown": -0.10,
            "win_rate": 0.55,
            "total_trades": 35,
        },
    }

    base = presets.get(strategy, presets["momentum"])
    return {
        "strategy": strategy,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        **base,
        "mock": True,
    }
