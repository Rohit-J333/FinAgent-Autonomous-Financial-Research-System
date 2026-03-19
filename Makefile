# FinAgent — developer convenience commands
# Requires: uv (https://docs.astral.sh/uv/getting-started/installation/)
#
# Quick install of uv:
#   curl -LsSf https://astral.sh/uv/install.sh | sh   (macOS / Linux)
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  (Windows)

.PHONY: help install sync lock upgrade test test-unit test-integration \
        docker-up docker-down docker-test backend frontend build-cpp clean

# ── default ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  make install        Create .venv and install all deps (uv sync)"
	@echo "  make sync           Re-sync venv to uv.lock without changing lock"
	@echo "  make lock           Regenerate uv.lock from pyproject.toml"
	@echo "  make upgrade        Upgrade all deps to latest allowed versions"
	@echo ""
	@echo "  make test           Run the full zero-key fallback test suite"
	@echo "  make test-unit      Unit tests only (fast, no HTTP)"
	@echo "  make test-integration  Integration tests (inline FastAPI)"
	@echo ""
	@echo "  make docker-up      docker-compose up --build (full stack)"
	@echo "  make docker-down    docker-compose down"
	@echo "  make docker-test    Zero-key smoke test against running stack"
	@echo ""
	@echo "  make backend        Run backend locally (uvicorn, hot-reload)"
	@echo "  make frontend       Run frontend locally (vite)"
	@echo "  make build-cpp      Build the C++ backtest binary"
	@echo ""

# ── environment management ───────────────────────────────────────────────────

# First-time setup: creates .venv and installs everything including dev extras
install:
	uv sync --group dev
	cd frontend && npm install

# Re-sync without touching the lock file (safe for CI)
sync:
	uv sync --group dev

# Rebuild uv.lock from scratch (run after editing pyproject.toml)
lock:
	uv lock

# Upgrade all packages to their latest allowed versions and re-lock
upgrade:
	uv lock --upgrade
	uv sync --extra dev

# ── tests (zero API keys) ────────────────────────────────────────────────────
test:
	@echo "Running full fallback test suite (zero API keys)…"
	uv run --extra dev pytest tests/test_fallbacks.py -v

test-unit:
	uv run --extra dev pytest tests/test_fallbacks.py -v \
	    -k "TestSentimentFallback or TestNewsFallback or TestBacktestFallback \
	        or TestRedisSkipped or TestLangfuseNoOp or TestRoutingLogic"

test-integration:
	uv run --extra dev pytest tests/test_fallbacks.py -v \
	    -k "TestHealthEndpoint or TestSymbolValidation or TestFullPipeline \
	        or TestFundamentals or TestCORS"

# ── docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down -v

docker-test:
	@echo "── /health ──────────────────────────────────────────────────────"
	curl -s http://localhost:8000/health | python -m json.tool
	@echo ""
	@echo "── POST /api/analyze (AAPL only) ───────────────────────────────"
	curl -s -X POST http://localhost:8000/api/analyze \
	    -H "Content-Type: application/json" \
	    -d '{"symbols": ["AAPL"]}' | python -m json.tool
	@echo ""
	@echo "── /api/fundamentals/AAPL (expect 503 if Qdrant down) ──────────"
	curl -s http://localhost:8000/api/fundamentals/AAPL | python -m json.tool
	@echo ""
	@echo "── 422: too many symbols ────────────────────────────────────────"
	curl -s -X POST http://localhost:8000/api/analyze \
	    -H "Content-Type: application/json" \
	    -d '{"symbols":["A","B","C","D","E","F","G"]}' | python -m json.tool
	@echo ""
	@echo "── 422: invalid symbol ──────────────────────────────────────────"
	curl -s -X POST http://localhost:8000/api/analyze \
	    -H "Content-Type: application/json" \
	    -d '{"symbols":["DROP TABLE"]}' | python -m json.tool

# ── local dev ────────────────────────────────────────────────────────────────
backend:
	uv run --directory backend uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

# ── C++ backtest binary ──────────────────────────────────────────────────────
build-cpp:
	cd strategy_engine && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
	@echo "Binary at: strategy_engine/bin/backtest"

# ── cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .venv .pytest_cache tests/.pytest_cache backend/.pytest_cache
