# FinAgent — All 10 Fixes: What Changed from the Original

> **Phase 1** (Fixes 1–5): API reliability, input safety, structured output, real SEC data, CORS hardening
> **Phase 2** (Fixes 6–10): Real market data, parallel execution, MCP integration, routing logic, observability
> **Bug fixes** (post-phase): Reflection fallback confidence, routing_reason state field

## Quick Start — Zero Keys

```bash
pip install -r requirements-dev.txt   # includes pytest + httpx
make test                              # runs full fallback suite, no API keys needed
# OR: test a live stack
docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build
make docker-test                       # curl smoke tests against running stack
```

---

## Table of Contents

- [Fix 1 — Reflection Loop Wired](#fix-1--reflection-loop-wired)
- [Fix 2 — Structured LLM Output](#fix-2--structured-llm-output)
- [Fix 3 — Real SEC 10-K Fetching](#fix-3--real-sec-10-k-fetching)
- [Fix 4 — Server-Side Symbol Validation](#fix-4--server-side-symbol-validation)
- [Fix 5 — CORS Hardened](#fix-5--cors-hardened)
- [Fix 6 — Real OHLCV Prices + Redis Caching](#fix-6--real-ohlcv-prices--redis-caching)
- [Fix 7 — Parallel Symbol Processing](#fix-7--parallel-symbol-processing)
- [Fix 8 — MCP Servers Wired In](#fix-8--mcp-servers-wired-in)
- [Fix 9 — Thresholds Wired into Routing](#fix-9--thresholds-wired-into-routing)
- [Fix 10 — Langfuse Observability](#fix-10--langfuse-observability)
- [Files Changed Summary](#files-changed-summary)
- [New Dependencies](#new-dependencies)

---

## Fix 1 — Reflection Loop Wired

**File:** `backend/agent/orchestrator.py`

### Problem (Original)
Node 3 (`reflection_node`) could return `"GATHER_MORE_DATA"` in its text but the graph had a **fixed edge** from `reflect → decide`. The signal was written to state and immediately ignored — Node 4 always fired regardless.

```python
# BEFORE — fixed, unconditional edge
workflow.add_edge("reflect", "decide")
```

### What Changed
- Added `reflection_step_count: int` to `AgentState` to track how many reflection loops have run.
- Added `should_gather_more(state)` routing function that checks the reflection text **and** the step counter.
- Replaced the fixed edge with `add_conditional_edges()` so the graph can branch back to `research`.
- `reflection_node` increments `reflection_step_count` on every pass.
- Hard cap: maximum **2** reflection iterations before forcing `decide` regardless.

```python
# AFTER — conditional routing
def should_gather_more(state: AgentState) -> str:
    if "GATHER_MORE_DATA" in state["reflection"] and state["reflection_step_count"] < 2:
        return "research"
    return "decide"

workflow.add_conditional_edges("reflect", should_gather_more, {"research": "research", "decide": "decide"})
```

---

## Fix 2 — Structured LLM Output

**File:** `backend/agent/orchestrator.py`

### Problem (Original)
`decision_node` extracted BUY/SELL/HOLD and confidence from the LLM's free-form text using **regex**. If Gemini changed its phrasing, parsing would silently break and return wrong confidence values.

```python
# BEFORE — brittle regex in decision_node
confidences = re.findall(r"confidence:\s*(\d+)%", decision_text)
avg_confidence = sum(int(c) for c in confidences) / len(confidences) / 100
```

### What Changed
Three new Pydantic models added:

| Model | Fields |
|---|---|
| `SymbolDecision` | `symbol`, `action: Literal["BUY","SELL","HOLD"]`, `confidence: float`, `reasoning: str` |
| `DecisionOutput` | `decisions: list[SymbolDecision]`, `overall_market_sentiment`, `data_quality_score` |
| `ReflectionOutput` | `assessment: str`, `should_gather_more: bool`, `missing_data_types`, `confidence_in_current_data` |

Both nodes use `llm.with_structured_output()` — zero regex:

```python
# AFTER — Gemini returns typed objects
decision_llm = llm.with_structured_output(DecisionOutput)
result: DecisionOutput = await decision_llm.ainvoke([("user", prompt)])
structured = [d.model_dump() for d in result.decisions]
```

`AgentState` gains `structured_decisions: List[Dict]` and `AnalyzeResponse` exposes it. Fallback on LLM failure produces valid typed dicts at 50% HOLD confidence.

---

## Fix 3 — Real SEC 10-K Fetching

**File:** `backend/rag/vectordb.py`

### Problem (Original)
`_fetch_10k()` returned a single hardcoded placeholder string regardless of symbol:

```python
# BEFORE — fake data
return f"SEC 10-K filing data for {symbol}. Revenue grew 12% YoY..."
```

### What Changed
- Added `edgartools>=5.23` dependency.
- `_fetch_10k_sync(symbol)` uses real `edgartools` library: calls `set_identity()`, fetches `Company(symbol).get_filings(form="10-K").latest(1)`, extracts **Item 1** (Business, ≤3000 chars) and **Item 1A** (Risk Factors, ≤2000 chars).
- Wrapped in `asyncio.get_running_loop().run_in_executor()` so the synchronous edgartools library never blocks the event loop.
- On `ImportError` or any exception → falls back to rich per-symbol placeholder text (AAPL/MSFT/GOOGL have realistic content; other tickers get a generic template).

```python
# AFTER — real SEC EDGAR via edgartools
async def _fetch_10k(self, symbol: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, self._fetch_10k_sync, symbol)
```

---

## Fix 4 — Server-Side Symbol Validation

**File:** `backend/main.py`

### Problem (Original)
`AnalyzeRequest` accepted any string list with no server-side enforcement. Only the frontend enforced a 6-symbol cap; the backend would happily pass 100 symbols or `"DROP TABLE;"` into the LLM prompt.

```python
# BEFORE — no validation
class AnalyzeRequest(BaseModel):
    symbols: List[str] = ["AAPL", "MSFT"]
```

### What Changed
`@field_validator("symbols")` (Pydantic v2 syntax) enforces four rules:

| Rule | Error |
|---|---|
| Empty list | `"At least one symbol is required"` |
| `len > 6` | `"Maximum 6 symbols per request"` |
| Doesn't match `^[A-Z]{1,5}$` | `"Invalid symbol: '...'"` |
| Leading/trailing whitespace | Auto-stripped and uppercased |

Returns standard **422 Unprocessable Entity** — no custom exception handler needed.

```python
# AFTER
class AnalyzeRequest(BaseModel):
    symbols: List[str] = Field(default=["AAPL", "MSFT"], max_length=6)

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, v):
        ...
        if not re.match(r"^[A-Z]{1,5}$", s):
            raise ValueError(f"Invalid symbol: '{raw}'")
        ...
```

---

## Fix 5 — CORS Hardened

**Files:** `backend/main.py`, `backend/utils/config.py`

### Problem (Original)
```python
# BEFORE — allows any origin, any method
allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
allow_methods=["*"],
```
The wildcard `"*"` makes the earlier explicit entries irrelevant — any origin is allowed.

### What Changed
- `Config` gains `FRONTEND_URL: Optional[str]` (default `http://localhost:5173`) and `VERCEL_URL: Optional[str]`.
- Allowed origins list is built dynamically from env vars at startup; `"*"` is removed entirely.
- `allow_methods` restricted to `["GET", "POST"]`.

```python
# AFTER
_allowed_origins = []
if Config.FRONTEND_URL:    _allowed_origins.append(Config.FRONTEND_URL)
if Config.VERCEL_URL:      _allowed_origins.append(f"https://{Config.VERCEL_URL}")
_allowed_origins += ["http://localhost:5173", "http://localhost:3000"]

app.add_middleware(CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"], ...)
```

---

## Fix 6 — Real OHLCV Prices + Redis Caching

**File:** `backend/agent/tools/backtest.py`

### Problem (Original)
The C++ backtester operated on **synthetic, deterministically-generated** price data inside the binary. Real performance metrics were not based on actual market history. The Python function was also synchronous.

### What Changed

**Two new public functions:**

```python
async def get_price_data(symbol: str, period: str = "6mo") -> List[Dict]:
    # 1. Check Redis cache  (key "prices:{symbol}:{period}", TTL 300s)
    # 2. On miss: yfinance.download() via run_in_executor
    # 3. Write back to cache
    # Returns [{date, open, high, low, close, volume}, ...]

async def get_current_quote(symbol: str) -> Dict:
    # Redis cache (key "quote:{symbol}", TTL 300s)
    # yfinance Ticker.fast_info
    # Returns {symbol, price, change_pct, volume}
```

**`execute_backtest` is now `async`:**
- Fetches real OHLCV first via `get_price_data()`
- Writes prices to a separate temp file, passes `--data {file}` to the C++ binary
- Fallback chain: no prices → mock | binary missing → mock | subprocess error → mock
- All temp files cleaned up in `finally` block

```
BEFORE: execute_backtest(strategy, symbol, start, end, params) → Dict  [sync]
AFTER:  execute_backtest(strategy, symbol, start, end, params) → Dict  [async, real data]
```

> **Note:** Because `execute_backtest` is now `async`, `backtest_node` in the orchestrator is also now `async`.

---

## Fix 7 — Parallel Symbol Processing

**File:** `backend/agent/orchestrator.py`

### Problem (Original)
Both `research_node` and `backtest_node` processed symbols in a **sequential `for` loop**. Each extra symbol added ~3 seconds of latency.

```python
# BEFORE — sequential, ~3s per symbol
for symbol in symbols:
    news = await scrape_financial_news(symbol)
    ...
```

### What Changed
Two private async helpers added:
- `_process_single_symbol(symbol)` — full news fetch + sentiment aggregation for one symbol
- `_backtest_single_symbol(symbol, strategy, start, end)` — full backtest for one symbol

Both nodes replaced their loops with `asyncio.gather`:

```python
# AFTER — parallel, ~3s total regardless of symbol count
results = await asyncio.gather(
    *[_process_single_symbol(s) for s in symbols],
    return_exceptions=True
)
for i, res in enumerate(results):
    if isinstance(res, Exception):
        logger.error(f"{symbols[i]} failed: {res}")  # log, don't crash
        all_sentiment[symbols[i]] = {"positive_ratio": 0.5, "total_articles": 0}
```

Per-symbol and total wall-clock times are logged via `time.perf_counter()`.

**Latency improvement:**

| Symbols | Before | After |
|---|---|---|
| 1 | ~3s | ~3s |
| 2 | ~6s | ~3s |
| 4 | ~12s | ~3–4s |
| 6 | ~18s | ~4–5s |

---

## Fix 8 — MCP Servers Wired In

**Files:** `backend/agent/mcp_servers/news_server.py`, `backend/agent/mcp_servers/strategy_server.py`, `backend/agent/orchestrator.py`

### Problem (Original)
Both MCP server files existed as class-based stubs with no actual MCP protocol implementation and were never called by the orchestrator.

### What Changed

**`news_server.py`** — fully rewritten:
- `get_financial_news(symbol, limit)` — real NewsAPI fetch, returns `[{title, description, publishedAt, source}]`
- `get_market_sentiment_summary(symbol)` — keyword-based headline sentiment (no ML model), returns `{symbol, article_count, avg_sentiment_hint, latest_headline}`
- Both wrapped as `@mcp.tool()` via `FastMCP` when the `mcp` package is installed
- `if __name__ == "__main__": mcp.run(transport="stdio")` for standalone use

**`strategy_server.py`** — fully rewritten:
- `run_momentum_backtest(symbol, lookback_days)` — calls `execute_backtest` with real prices
- `get_technical_signals(symbol)` — computes 20-day SMA, 50-day SMA, trend (`ABOVE_BOTH`/`BELOW_BOTH`/`MIXED`), and 20-day momentum % from `get_price_data(symbol, "3mo")`; returns error dict if < 50 data points
- Both wrapped as `@mcp.tool()` + `mcp.run(transport="stdio")`

**`orchestrator.py`** — MCP integration:
```python
# Prefer MCP tools when available; fall back to direct calls on ImportError
if Config.USE_MCP:
    from agent.mcp_servers.news_server import get_financial_news as mcp_get_news, ...
    _mcp_available = True
```
`_process_single_symbol` enriches news results with MCP sentiment hints; `_backtest_single_symbol` prefers MCP backtest when available. Toggle with `USE_MCP=false`.

---

## Fix 9 — Thresholds Wired into Routing

**Files:** `backend/agent/orchestrator.py`, `backend/utils/config.py`

### Problem (Original)
`REFLECTION_THRESHOLD` and `CONFIDENCE_THRESHOLD` were defined in `Config` and documented but had **zero effect** on agent behavior. The routing function only checked for a string and a hardcoded `< 2` step counter.

```python
# BEFORE — thresholds defined but never read
REFLECTION_THRESHOLD: float = float(os.getenv("REFLECTION_THRESHOLD", "0.7"))
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

def should_gather_more(state):
    if "GATHER_MORE_DATA" in reflection_text and reflection_count < 2:  # hardcoded 2
        return "research"
    return "decide"
```

### What Changed

`Config` types corrected and defaults updated:

```python
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
REFLECTION_THRESHOLD: int   = int(os.getenv("REFLECTION_THRESHOLD", "2"))
```

`should_gather_more` now evaluates **4 rules in priority order**:

| Priority | Condition | Route | Log level |
|---|---|---|---|
| 1 (hard cap) | `reflection_step_count >= REFLECTION_THRESHOLD` | `decide` | WARNING |
| 2 | `"GATHER_MORE_DATA" in reflection` | `research` | INFO |
| 3 | `confidence < CONFIDENCE_THRESHOLD` | `research` | INFO |
| 4 (default) | everything else | `decide` | INFO |

`reflection_node` now always sets `state["confidence"]` from `ReflectionOutput.confidence_in_current_data` (default 0.5 on failure). This means confidence is **set by the LLM's self-assessment** and then tested by the router.

New `routing_reason: str` field added to `AgentState` for audit logging.

---

## Fix 10 — Langfuse Observability

**Files:** `backend/utils/observability.py` *(new)*, `backend/agent/orchestrator.py`, `backend/main.py`, `backend/utils/config.py`

### Problem (Original)
No tracing existed. There was no way to see per-node latency, token usage, failure rates, or confidence trends across runs.

### What Changed

**New `backend/utils/observability.py`:**
```python
def get_langfuse_handler(user_id, session_id, metadata) -> CallbackHandler | None
def log_analysis_result(symbols, result, latency_ms, session_id) -> None
```
Both return `None` / do nothing if:
- `LANGFUSE_PUBLIC_KEY` is not set
- `langfuse` package is not installed
- Any Langfuse API call raises an exception

**`orchestrator.py`** — all 4 nodes decorated:
```python
from langfuse.decorators import observe as traceable  # no-op if not installed

@traceable(name="research_node")
async def research_node(state): ...

@traceable(name="backtest_node")
async def backtest_node(state): ...

@traceable(name="reflection_node")
async def reflection_node(state): ...

@traceable(name="decision_node")
async def decision_node(state): ...
```

**`main.py`** — handler injected per request:
```python
handler = get_langfuse_handler(user_id="api", session_id=session_id, metadata={"symbols": symbols})
invoke_config = {"recursion_limit": Config.MAX_AGENT_STEPS}
if handler:
    invoke_config["callbacks"] = [handler]

t0 = time.perf_counter()
final_state = await agent.ainvoke(_initial_state(symbols), invoke_config)
latency_ms = (time.perf_counter() - t0) * 1000
```

`AnalyzeResponse` gains `latency_ms: float` field. WebSocket endpoint also passes the handler to `astream`.

**New `Config` fields:**
```python
LANGFUSE_PUBLIC_KEY: Optional[str]
LANGFUSE_SECRET_KEY: Optional[str]
LANGFUSE_HOST: str = "https://cloud.langfuse.com"
OBSERVABILITY_ENABLED: bool = True
```

---

## Files Changed Summary

| File | Phase 1 | Phase 2 | What changed |
|---|---|---|---|
| `backend/agent/orchestrator.py` | Fix 1, 2 | Fix 7, 8, 9, 10 | Conditional routing, structured output, parallel gather, MCP integration, threshold routing, `@traceable` decorators, `routing_reason` state field |
| `backend/main.py` | Fix 4, 5 | Fix 10 | Symbol validator, CORS whitelist, Langfuse handler injection, latency timing, `routing_reason` in initial state |
| `backend/utils/config.py` | Fix 5 | Fix 6, 8, 9, 10 | `FRONTEND_URL`, `VERCEL_URL`, `UPSTASH_REDIS_URL`, `USE_MCP`, `LANGFUSE_*`, `OBSERVABILITY_ENABLED`, threshold type corrections |
| `backend/rag/vectordb.py` | Fix 3 | — | Real `edgartools` 10-K fetch, `run_in_executor` wrapper, per-symbol fallback placeholders |
| `backend/agent/tools/backtest.py` | — | Fix 6 | Now `async`; `get_price_data`, `get_current_quote`, Redis caching, real OHLCV → C++ binary |
| `backend/agent/mcp_servers/news_server.py` | — | Fix 8 | Full rewrite: `get_financial_news`, `get_market_sentiment_summary`, FastMCP wrappers |
| `backend/agent/mcp_servers/strategy_server.py` | — | Fix 8 | Full rewrite: `run_momentum_backtest`, `get_technical_signals`, FastMCP wrappers |
| `backend/utils/observability.py` | — | Fix 10 | **New file**: Langfuse client, `get_langfuse_handler`, `log_analysis_result` |
| `requirements.txt` | Fix 3 | Fix 6, 8, 10 | Added `edgartools`, `yfinance`, `redis`, `upstash-redis`, `langchain-mcp-adapters`, `mcp`, `langfuse` |
| `.env.example` | — | Fix 6, 10 | Added `UPSTASH_REDIS_URL`, `LANGFUSE_*`, `OBSERVABILITY_ENABLED`, `USE_MCP` |
| `docker-compose.yml` | — | Fix 6, 8, 9, 10 | Pass all new env vars to backend container |

---

## New Dependencies

| Package | Version | Fix | Purpose |
|---|---|---|---|
| `edgartools` | ≥5.23 | Fix 3 | Real SEC 10-K fetching |
| `yfinance` | ≥0.2.36 | Fix 6 | Real OHLCV price data |
| `redis` | ≥5.0.0 | Fix 6 | Redis client for price caching |
| `upstash-redis` | ≥1.0.0 | Fix 6 | Upstash Redis (serverless) |
| `langchain-mcp-adapters` | ≥0.1.8 | Fix 8 | LangChain ↔ MCP bridge |
| `mcp` | ≥1.0.0 | Fix 8 | Model Context Protocol server |
| `langfuse` | ≥2.0.0 | Fix 10 | Observability / tracing |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Zero-API-Key Fallback Matrix

Every fix preserves the ability to run the system with zero API keys:

| Dependency | Missing condition | Fallback |
|---|---|---|
| `GEMINI_API_KEY` | Not set | HOLD decision text |
| `NEWS_API_KEY` | Not set | 3 hardcoded mock articles |
| `UPSTASH_REDIS_URL` | Not set | Skip caching, fetch fresh |
| `yfinance` not installed | ImportError | Mock backtest presets |
| C++ binary | Not built | Mock backtest presets |
| `edgartools` not installed | ImportError | Placeholder 10-K text |
| `mcp` not installed | ImportError | Direct function calls |
| `langfuse` not installed | ImportError | No-op `@traceable` decorator |
| `LANGFUSE_PUBLIC_KEY` | Not set | No tracing, agent unaffected |
| Qdrant not running | Connection error | RAG disabled, 503 on fundamentals endpoint |
