# SPRINT.md — Week 5 (Platform Foundation: API, Runner, Deploy)

> Theme: Turn the engine into a running service. FastAPI wraps the engine, bots run continuously, PostgreSQL goes to production, CI/CD ensures quality.
> Start: May 5, 2026
> Target: May 9, 2026
> Tasks: 7 (ordered by dependency)
> Foundation: ~757+ tests. Core engine COMPLETE: full pipeline, Executor, ReflectionAgent, Sentinel, TraderBot lifecycle, ExecutionCostModel. Engine can analyze and trade on Hyperliquid testnet.

---

## Task 1: FastAPI Web Layer — Core Endpoints [L, High]

**Status:** [ ] Not Started
**Depends on:** Pipeline (Week 3), Executor (Week 4), TraderBot (Week 4)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] FastAPI app in `api/app.py` with versioned routes (`/v1/...`)
- [ ] Bot management: POST /v1/bots (create), GET /v1/bots (list), GET /v1/bots/{id} (detail), DELETE /v1/bots/{id} (stop)
- [ ] Cycle trigger: POST /v1/bots/{id}/analyze (trigger one cycle manually)
- [ ] Health: GET /v1/health (system health from HealthTracker)
- [ ] Positions: GET /v1/positions (all open positions across bots)
- [ ] Trades: GET /v1/trades (recent trades, filterable by bot/symbol)
- [ ] Rules: GET /v1/rules (active reflection rules)
- [ ] All endpoints require API key auth (simple header check for now)
- [ ] Proper error handling with structured JSON responses
- [ ] Unit tests with TestClient
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to HIGH for this task.

Build the FastAPI web layer. This wraps the engine — no engine logic lives here.
The API layer is a thin translation between HTTP and engine calls.

IMPORTANT: No FastAPI imports in engine/. The API imports FROM engine, never the reverse.

FILE 1: api/app.py
- FastAPI app with lifespan (startup: init repos, shutdown: cleanup)
- Include routers from api/routes/

FILE 2: api/auth.py
- Simple API key auth via X-API-Key header
- Keys stored in env var API_KEYS (comma-separated)
- Dependency: get_current_user() raises 401 if invalid

FILE 3: api/routes/bots.py
- POST /v1/bots — create bot config (symbol, timeframe, exchange)
  Returns bot_id. Does NOT start the bot (that's the runner's job).
- GET /v1/bots — list all bots for this user
- GET /v1/bots/{bot_id} — bot detail + health + last cycle
- DELETE /v1/bots/{bot_id} — mark bot as stopped
- POST /v1/bots/{bot_id}/analyze — trigger one manual analysis cycle
  Returns: conviction, action, signals summary

FILE 4: api/routes/trades.py
- GET /v1/trades — recent trades, query params: bot_id, symbol, limit
- GET /v1/trades/{trade_id} — full trade detail with cycle data

FILE 5: api/routes/health.py
- GET /v1/health — system health snapshot from HealthTracker
  Returns: uptime, cycles total/success/error, active bots, db status

FILE 6: api/routes/positions.py
- GET /v1/positions — all open positions across all bots

FILE 7: api/routes/rules.py
- GET /v1/rules — active reflection rules, query params: symbol, timeframe

FILE 8: api/schemas.py
- Pydantic request/response models for all endpoints
- BotCreateRequest, BotResponse, TradeResponse, HealthResponse, etc.

Write tests in tests/api/test_endpoints.py:
- Use FastAPI TestClient
- Test bot CRUD operations
- Test trades endpoint with query filters
- Test health endpoint returns valid snapshot
- Test auth: valid key passes, missing/invalid key returns 401
- Test 404 for nonexistent bot/trade

Update PROJECT_CONTEXT.md sections 2, 3, 14. Update CHANGELOG.md.
```

---

## Task 2: Continuous Bot Runner [L, High]

**Status:** [ ] Not Started
**Depends on:** Sentinel (Week 4), BotManager (Week 4), FastAPI (Task 1)
**Estimated time:** 90 minutes

**Acceptance criteria:**
- [ ] `BotRunner` manages lifecycle of all active bots as a long-running service
- [ ] Starts Sentinel monitors for each active bot's symbol
- [ ] Scheduled fallback timer fires analysis at configured interval per bot
- [ ] BotManager spawns TraderBots on SetupDetected or schedule
- [ ] Graceful shutdown: stops all Sentinels, waits for active TraderBots to finish
- [ ] Auto-restart crashed Sentinels (with exponential backoff)
- [ ] Integrates with FastAPI: bot creation in API triggers runner to start monitoring
- [ ] Can run standalone (without API) via `python -m quantagent run`
- [ ] Unit tests with mock components
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to HIGH for this task.

The BotRunner is the production service — it keeps everything alive.

FILE 1: quantagent/runner.py

class BotRunner:
    def __init__(self, repos, adapter_factory, llm_provider, event_bus, config):
        self._sentinels: dict[str, SentinelMonitor] = {}
        self._bot_manager: BotManager
        self._scheduled_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start():
        # Load active bots from DB, start Sentinel per symbol, schedule fallbacks

    async def stop():
        # Graceful: stop sentinels, cancel tasks, wait for active bots

    async def add_bot(bot_config):
        # Save to DB, start sentinel + schedule (called by API)

    async def remove_bot(bot_id):
        # Stop sentinel if no other bots use that symbol, cancel schedule

    async def _run_sentinel_safe(sentinel):
        # Auto-restart with exponential backoff on crash

    async def _scheduled_loop(bot, interval):
        # Fallback: trigger analysis on schedule

FILE 2: quantagent/main.py (update CLI)
- Add 'run' command: python -m quantagent run
- Starts BotRunner + FastAPI (uvicorn) together
- Ctrl+C triggers graceful shutdown via signal handlers

Write tests in tests/unit/test_runner.py:
- Test start loads bots from DB and creates sentinels
- Test add_bot creates sentinel and scheduled task
- Test remove_bot cleans up
- Test graceful shutdown stops all sentinels
- Test sentinel auto-restart on crash (mock crash then recover)
- Test scheduled fallback fires at interval

Update PROJECT_CONTEXT.md sections 2, 3, 14. Update CHANGELOG.md.
```

---

## Task 3: WebSocket Streams for Dashboard [M, Medium-High]

**Status:** [ ] Not Started
**Depends on:** FastAPI (Task 1), Event Bus (Week 1)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] WebSocket endpoint: `/v1/ws` streams live events to dashboard
- [ ] Events streamed: CycleCompleted, TradeOpened, TradeClosed, SetupDetected, PositionUpdated
- [ ] Connection-scoped: each client receives only their user's events
- [ ] Heartbeat ping/pong every 30s to detect stale connections
- [ ] Reconnection-friendly (client can reconnect and get current state)
- [ ] Unit tests with WebSocket test client
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM-HIGH for this task.

WebSocket streams power the real-time dashboard. SolidJS will consume these later.

FILE 1: api/websocket.py
class ConnectionManager:
    - connect(websocket, user_id): accept + register
    - disconnect(websocket, user_id): remove
    - broadcast_to_user(user_id, event): send to all user's connections

FILE 2: api/routes/websocket.py
@router.websocket("/v1/ws")
- Authenticate via query param or first message
- Subscribe to Event Bus for user's events
- Forward events as JSON
- Heartbeat every 30s

Bridge: Event Bus events → ConnectionManager.broadcast_to_user()

Write tests in tests/api/test_websocket.py:
- Test connect and receive event
- Test disconnect cleanup
- Test user isolation (user A doesn't see user B events)
- Test heartbeat

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 4: PostgreSQL Production Migration [M, Medium-High]

**Status:** [ ] Not Started
**Depends on:** Repository Pattern (Week 3)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] Alembic migration setup with initial migration creating all tables
- [ ] Migration tested against real PostgreSQL instance
- [ ] Connection pooling configured (asyncpg pool, min=2, max=10)
- [ ] Health check endpoint verifies DB connection
- [ ] Environment-based switching: dev=SQLite, prod=PostgreSQL
- [ ] Migration CLI: `python -m quantagent migrate`
- [ ] Seed data script for dev
- [ ] Unit tests for migration (create/rollback)
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM-HIGH for this task.

Move from raw CREATE TABLE to proper Alembic migrations.

FILE 1: alembic.ini — standard config, reads DATABASE_URL from env
FILE 2: alembic/env.py — async migration support for asyncpg
FILE 3: alembic/versions/001_initial.py — create all tables with indexes
FILE 4: Update storage/repositories/postgres.py — connection pool lifecycle, health_check()
FILE 5: quantagent/main.py — add 'migrate' command
FILE 6: scripts/seed_dev.py — create test bot config

Write tests in tests/integration/test_migrations.py

Update PROJECT_CONTEXT.md sections 2, 7, 14. Update CHANGELOG.md.
```

---

## Task 5: CI/CD Pipeline — GitHub Actions [M, Medium]

**Status:** [ ] Not Started
**Depends on:** All tests passing
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] GitHub Actions workflow: `.github/workflows/ci.yml`
- [ ] Every push: run unit tests
- [ ] PR merge to main: run full suite including integration tests
- [ ] Import violation check in CI
- [ ] Cache pip deps for speed
- [ ] CI badge in README
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM for this task.

Standard CI/CD. Tests run on every push, import violations checked.

FILE 1: .github/workflows/ci.yml
- Python 3.12, pip cache, install deps
- Unit tests on every push
- Integration tests on main merge only
- Import violation grep check
- Fail build on any failure

FILE 2: README.md (create or update)
- CI badge, project description, setup instructions

No new tests. Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 6: Caching Layer [M, Medium-High]

**Status:** [ ] Not Started
**Depends on:** Config (Week 1), Data Layer (Week 2)
**Estimated time:** 45 minutes

**Acceptance criteria:**
- [ ] CacheManager with get_or_fetch() pattern
- [ ] MemoryCache backend (cachetools with TTL)
- [ ] Cache OHLCV per symbol/timeframe (TTL = one candle period)
- [ ] Cache flow data per symbol (TTL = 30 seconds)
- [ ] Cache external API responses (TTL = 1-4 hours)
- [ ] Cache asset metadata for ExecutionCostModel (TTL = 24 hours)
- [ ] Cache metrics: hit rate, miss rate exposed in health endpoint
- [ ] LLM responses NEVER cached
- [ ] Unit tests
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM-HIGH for this task.

Multi-layer caching reduces data fetches ~10x and cuts API costs.
LLM responses are NEVER cached — market context changes even when inputs look similar.

FILE 1: storage/cache/base.py — CacheBackend ABC (get, set, delete, clear)
FILE 2: storage/cache/memory.py — MemoryCache using cachetools.TTLCache
FILE 3: storage/cache/__init__.py — CacheManager with get_or_fetch(), invalidate(), flush_all()
FILE 4: storage/cache/metrics.py — CacheMetrics (hits, misses, hit_rate)

Cache TTL constants:
    OHLCV: one candle period (15m=900, 1h=3600, 4h=14400)
    Flow: 30 seconds
    External API: 3600 seconds
    Asset metadata: 86400 seconds (24h)

Integrate into OHLCVFetcher, FlowAgent, HyperliquidCostModel.

Write tests in tests/unit/test_cache.py

Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## Task 7: Deployment Script + Server Setup [M, Medium]

**Status:** [ ] Not Started
**Depends on:** FastAPI (Task 1), Runner (Task 2), PostgreSQL (Task 4)
**Estimated time:** 30 minutes

**Acceptance criteria:**
- [ ] Deployment script: `scripts/deploy.sh` (SSH, git pull, install, migrate, restart)
- [ ] Systemd service file: `deploy/quantagent.service`
- [ ] Nginx config: `deploy/nginx.conf` (reverse proxy + WebSocket support)
- [ ] Environment template: `deploy/.env.production.template`
- [ ] Health check script: `scripts/health_check.sh`
- [ ] Deploy docs: `deploy/README.md`
- [ ] PROJECT_CONTEXT.md + CHANGELOG.md updated

**Claude Code Instructions:**

```
Read CLAUDE.md and PROJECT_CONTEXT.md first.

Set effort to MEDIUM for this task.

Production deployment to Hetzner. Simple, reliable, no Kubernetes.

FILE 1: deploy/quantagent.service — systemd unit (auto-restart, env file)
FILE 2: deploy/nginx.conf — reverse proxy to :8000, WebSocket /v1/ws, SSL placeholder
FILE 3: deploy/.env.production.template — all env vars with comments
FILE 4: scripts/deploy.sh — one-command deploy (ssh, pull, install, migrate, restart, health check)
FILE 5: scripts/health_check.sh — curl /v1/health, pretty print
FILE 6: deploy/README.md — server setup, first-time, deploy, logs, restart

No automated tests — infrastructure config.
Update PROJECT_CONTEXT.md sections 2, 14. Update CHANGELOG.md.
```

---

## End of Week 5 Checklist

- [ ] pytest passes all tests (target: ~850+)
- [ ] FastAPI serves all endpoints with API key auth
- [ ] Bot CRUD works via API
- [ ] Manual cycle trigger via POST /v1/bots/{id}/analyze works
- [ ] BotRunner starts/stops bots, manages Sentinels continuously
- [ ] Sentinels auto-restart on crash with backoff
- [ ] Scheduled fallback fires at timeframe interval
- [ ] WebSocket streams live events to connected clients
- [ ] Alembic migrations create all tables in PostgreSQL
- [ ] Connection pooling configured and health-checked
- [ ] CI/CD runs pytest on every push, blocks on failure
- [ ] Import violations checked in CI
- [ ] CacheManager reduces redundant data fetches
- [ ] LLM responses never cached
- [ ] Deploy script works end-to-end to Hetzner
- [ ] Systemd keeps engine running after deploy
- [ ] Health check confirms service is up
- [ ] No SQL imports in engine/
- [ ] No FastAPI imports in engine/

**If all pass:** QuantAgent runs as a production service. Deploy to Hetzner, create bots via API, they trade autonomously with Sentinel monitoring. The platform is live.

---

## Week 6 Preview

- Backtesting framework (offline replay with historical data)
- Signal distribution: Discord webhook + Telegram Bot API
- Landing page (static, with live performance metrics)
- Overnight Quant MCP agent (nightly cron → alpha_factors.json)
- Macro Regime Manager MCP agent (daily → macro_regime.json)
