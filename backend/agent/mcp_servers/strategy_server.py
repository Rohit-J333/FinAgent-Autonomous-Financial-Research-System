"""
MCP-style Strategy Server.

Wraps the C++ backtest engine (or Python mock) and exposes it as an
MCP-compatible tool that the LangGraph agent can call.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Optional

logger = logging.getLogger(__name__)

BACKTESTER_BIN = os.getenv("BACKTESTER_BIN", "./strategy_engine/bin/backtest")


class StrategyMCPServer:
    """
    MCP-compatible strategy engine server.

    Calls the C++ binary for deterministic backtesting.
    Falls back to a Python mock when the binary is absent.
    """

    name = "strategy-mcp-server"

    async def backtest_strategy(
        self,
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute a backtest and return performance metrics.

        Args:
            strategy_name: "momentum" | "mean_reversion" | "macd"
            symbol:        Stock ticker
            start_date:    YYYY-MM-DD
            end_date:      YYYY-MM-DD
            params:        Strategy-specific parameters

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
                f"Binary not found at '{BACKTESTER_BIN}'. Using Python mock."
            )
            return self._mock_backtest(strategy_name, symbol, start_date, end_date)

        backtest_input = {
            "strategy": strategy_name,
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
                "strategy": strategy_name,
                "symbol": symbol,
                "total_return": results.get("total_return", 0),
                "sharpe_ratio": results.get("sharpe_ratio", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "win_rate": results.get("win_rate", 0),
                "total_trades": results.get("total_trades", 0),
            }

        except subprocess.TimeoutExpired:
            logger.error("Backtester timed out.")
            return {"error": "timeout"}
        finally:
            for path in [input_file, output_file]:
                if os.path.exists(path):
                    os.remove(path)

    @staticmethod
    def _mock_backtest(
        strategy_name: str, symbol: str, start_date: str, end_date: str
    ) -> Dict:
        presets = {
            "momentum":       {"total_return": 0.23, "sharpe_ratio": 1.42, "max_drawdown": -0.08, "win_rate": 0.61, "total_trades": 48},
            "mean_reversion": {"total_return": 0.15, "sharpe_ratio": 1.10, "max_drawdown": -0.06, "win_rate": 0.57, "total_trades": 72},
            "macd":           {"total_return": 0.19, "sharpe_ratio": 1.25, "max_drawdown": -0.10, "win_rate": 0.55, "total_trades": 35},
        }
        base = presets.get(strategy_name, presets["momentum"])
        return {
            "strategy": strategy_name,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            **base,
            "mock": True,
        }


# Singleton
strategy_server = StrategyMCPServer()
