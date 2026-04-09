"""PostgreSQL repository implementations using asyncpg.

Standard backend for production. Uses connection pooling for concurrent access.
All queries use parameterized $1, $2 syntax — never string interpolation.
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg

from storage.repositories.base import (
    BotRepository,
    CrossBotRepository,
    CycleRepository,
    RuleRepository,
    TradeRepository,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Schema DDL
# ──────────────────────────────────────────────────────────────────────

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price DOUBLE PRECISION,
    exit_price DOUBLE PRECISION,
    size DOUBLE PRECISION,
    pnl DOUBLE PRECISION,
    r_multiple DOUBLE PRECISION,
    entry_time TIMESTAMPTZ,
    exit_time TIMESTAMPTZ,
    exit_reason TEXT,
    conviction_score DOUBLE PRECISION,
    engine_version TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    forward_max_r DOUBLE PRECISION,
    is_shadow BOOLEAN NOT NULL DEFAULT FALSE
);
"""

_CREATE_CYCLES = """
CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicators_json JSONB,
    signals_json JSONB,
    conviction_json JSONB,
    action TEXT,
    conviction_score DOUBLE PRECISION,
    is_shadow BOOLEAN NOT NULL DEFAULT FALSE
);
"""

_CREATE_RULES = """
CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    score INTEGER NOT NULL DEFAULT 0,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL
);
"""

_CREATE_BOTS = """
CREATE TABLE IF NOT EXISTS bots (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    exchange TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    config_json JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    last_health JSONB,
    is_shadow BOOLEAN NOT NULL DEFAULT FALSE,
    mode VARCHAR(10) NOT NULL DEFAULT 'live'
);
"""

_CREATE_CROSS_BOT_SIGNALS = """
CREATE TABLE IF NOT EXISTS cross_bot_signals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    conviction DOUBLE PRECISION NOT NULL,
    bot_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL
);
"""

_ALL_TABLES = [
    _CREATE_TRADES,
    _CREATE_CYCLES,
    _CREATE_RULES,
    _CREATE_BOTS,
    _CREATE_CROSS_BOT_SIGNALS,
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_to_dict(record: asyncpg.Record) -> dict:
    """Convert asyncpg Record to plain dict."""
    return dict(record)


# ──────────────────────────────────────────────────────────────────────
# Trade Repository
# ──────────────────────────────────────────────────────────────────────


class PostgresTradeRepository(TradeRepository):

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_trade(self, trade: dict) -> str:
        trade_id = trade.get("id") or str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO trades
                   (id, user_id, bot_id, symbol, timeframe, direction,
                    entry_price, exit_price, size, pnl, r_multiple,
                    entry_time, exit_time, exit_reason, conviction_score,
                    engine_version, status, is_shadow)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                           $12, $13, $14, $15, $16, $17, $18)""",
                trade_id,
                trade["user_id"],
                trade["bot_id"],
                trade["symbol"],
                trade["timeframe"],
                trade["direction"],
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("size"),
                trade.get("pnl"),
                trade.get("r_multiple"),
                trade.get("entry_time", _now_iso()),
                trade.get("exit_time"),
                trade.get("exit_reason"),
                trade.get("conviction_score"),
                trade.get("engine_version"),
                trade.get("status", "open"),
                bool(trade.get("is_shadow", False)),
            )
        return trade_id

    async def get_trade(self, trade_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM trades WHERE id = $1", trade_id
            )
            return _record_to_dict(row) if row else None

    async def get_open_positions(
        self, user_id: str, bot_id: str, *, include_shadow: bool = False
    ) -> list[dict]:
        shadow_clause = "" if include_shadow else " AND is_shadow = FALSE"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM trades WHERE user_id = $1 AND bot_id = $2 "
                f"AND status = 'open'{shadow_clause}",
                user_id, bot_id,
            )
            return [_record_to_dict(r) for r in rows]

    async def get_trades_by_bot(
        self, bot_id: str, limit: int = 50, *, include_shadow: bool = False
    ) -> list[dict]:
        shadow_clause = "" if include_shadow else " AND is_shadow = FALSE"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM trades WHERE bot_id = $1{shadow_clause} "
                f"ORDER BY entry_time DESC LIMIT $2",
                bot_id, limit,
            )
            return [_record_to_dict(r) for r in rows]

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        if not updates:
            return False
        set_parts = []
        values = []
        for i, (k, v) in enumerate(updates.items(), start=1):
            set_parts.append(f"{k} = ${i}")
            values.append(v)
        values.append(trade_id)
        query = f"UPDATE trades SET {', '.join(set_parts)} WHERE id = ${len(values)}"
        async with self._pool.acquire() as conn:
            result = await conn.execute(query, *values)
            return result != "UPDATE 0"


# ──────────────────────────────────────────────────────────────────────
# Cycle Repository
# ──────────────────────────────────────────────────────────────────────


class PostgresCycleRepository(CycleRepository):

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_cycle(self, cycle: dict) -> str:
        cycle_id = cycle.get("id") or str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO cycles
                   (id, bot_id, symbol, timeframe, timestamp,
                    indicators_json, signals_json, conviction_json,
                    action, conviction_score, is_shadow)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                cycle_id,
                cycle["bot_id"],
                cycle["symbol"],
                cycle["timeframe"],
                cycle.get("timestamp", _now_iso()),
                json.dumps(cycle.get("indicators")) if cycle.get("indicators") else None,
                json.dumps(cycle.get("signals")) if cycle.get("signals") else None,
                json.dumps(cycle.get("conviction")) if cycle.get("conviction") else None,
                cycle.get("action"),
                cycle.get("conviction_score"),
                bool(cycle.get("is_shadow", False)),
            )
        return cycle_id

    async def get_recent_cycles(
        self, bot_id: str, limit: int = 5, *, include_shadow: bool = False
    ) -> list[dict]:
        shadow_clause = "" if include_shadow else " AND is_shadow = FALSE"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM cycles WHERE bot_id = $1{shadow_clause} "
                f"ORDER BY timestamp DESC LIMIT $2",
                bot_id, limit,
            )
        results = [_record_to_dict(r) for r in rows]
        for row in results:
            for col in ("indicators_json", "signals_json", "conviction_json"):
                if row.get(col) and isinstance(row[col], str):
                    row[col] = json.loads(row[col])
        return results


# ──────────────────────────────────────────────────────────────────────
# Rule Repository
# ──────────────────────────────────────────────────────────────────────


class PostgresRuleRepository(RuleRepository):

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_rule(self, rule: dict) -> str:
        rule_id = rule.get("id") or str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO rules (id, symbol, timeframe, rule_text, score, active, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                rule_id,
                rule["symbol"],
                rule["timeframe"],
                rule["rule_text"],
                rule.get("score", 0),
                rule.get("active", True),
                rule.get("created_at", _now_iso()),
            )
        return rule_id

    async def get_rules(self, symbol: str, timeframe: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM rules WHERE symbol = $1 AND timeframe = $2 AND active = TRUE",
                symbol, timeframe,
            )
            return [_record_to_dict(r) for r in rows]

    async def update_rule_score(self, rule_id: str, delta: int) -> bool:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "UPDATE rules SET score = score + $1 WHERE id = $2",
                    delta, rule_id,
                )
                if result == "UPDATE 0":
                    return False
                # Auto-deactivate if score drops below -2
                await conn.execute(
                    "UPDATE rules SET active = FALSE WHERE id = $1 AND score < -2",
                    rule_id,
                )
                return True

    async def deactivate_rule(self, rule_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE rules SET active = FALSE WHERE id = $1",
                rule_id,
            )
            return result != "UPDATE 0"


# ──────────────────────────────────────────────────────────────────────
# Bot Repository
# ──────────────────────────────────────────────────────────────────────


class PostgresBotRepository(BotRepository):

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_bot(self, bot: dict) -> str:
        bot_id = bot.get("id") or str(uuid4())
        mode = bot.get("mode", "live")
        is_shadow = bool(bot.get("is_shadow") or mode == "shadow")
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO bots
                   (id, user_id, symbol, timeframe, exchange, status,
                    config_json, created_at, last_health, is_shadow, mode)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                bot_id,
                bot["user_id"],
                bot["symbol"],
                bot["timeframe"],
                bot["exchange"],
                bot.get("status", "active"),
                json.dumps(bot.get("config")) if bot.get("config") else None,
                bot.get("created_at", _now_iso()),
                None,
                is_shadow,
                mode,
            )
        return bot_id

    async def get_bot(self, bot_id: str) -> dict | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM bots WHERE id = $1", bot_id
            )
            if not row:
                return None
            result = _record_to_dict(row)
        if result.get("config_json") and isinstance(result["config_json"], str):
            result["config_json"] = json.loads(result["config_json"])
        if result.get("last_health") and isinstance(result["last_health"], str):
            result["last_health"] = json.loads(result["last_health"])
        return result

    async def get_bots_by_user(self, user_id: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM bots WHERE user_id = $1", user_id
            )
        results = [_record_to_dict(r) for r in rows]
        for row in results:
            if row.get("config_json") and isinstance(row["config_json"], str):
                row["config_json"] = json.loads(row["config_json"])
            if row.get("last_health") and isinstance(row["last_health"], str):
                row["last_health"] = json.loads(row["last_health"])
        return results

    async def get_active_bots(self, *, include_shadow: bool = False) -> list[dict]:
        shadow_clause = "" if include_shadow else " AND is_shadow = FALSE"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM bots WHERE status = $1{shadow_clause}", "active"
            )
        results = [_record_to_dict(r) for r in rows]
        for row in results:
            if row.get("config_json") and isinstance(row["config_json"], str):
                row["config_json"] = json.loads(row["config_json"])
            if row.get("last_health") and isinstance(row["last_health"], str):
                row["last_health"] = json.loads(row["last_health"])
        return results

    async def get_active_bots_by_mode(self, mode: str) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM bots WHERE status = $1 AND mode = $2",
                "active", mode,
            )
        results = [_record_to_dict(r) for r in rows]
        for row in results:
            if row.get("config_json") and isinstance(row["config_json"], str):
                row["config_json"] = json.loads(row["config_json"])
            if row.get("last_health") and isinstance(row["last_health"], str):
                row["last_health"] = json.loads(row["last_health"])
        return results

    async def update_bot_health(self, bot_id: str, health: dict) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE bots SET last_health = $1 WHERE id = $2",
                json.dumps(health), bot_id,
            )
            return result != "UPDATE 0"


# ──────────────────────────────────────────────────────────────────────
# Cross-Bot Repository
# ──────────────────────────────────────────────────────────────────────


class PostgresCrossBotRepository(CrossBotRepository):

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def save_signal(self, signal: dict) -> None:
        signal_id = signal.get("id") or str(uuid4())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO cross_bot_signals
                   (id, user_id, symbol, direction, conviction, bot_id, timestamp)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                signal_id,
                signal["user_id"],
                signal["symbol"],
                signal["direction"],
                signal["conviction"],
                signal["bot_id"],
                signal.get("timestamp", _now_iso()),
            )

    async def get_recent_signals(
        self, symbol: str, user_id: str, limit: int = 10
    ) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM cross_bot_signals
                   WHERE symbol = $1 AND user_id = $2
                   ORDER BY timestamp DESC LIMIT $3""",
                symbol, user_id, limit,
            )
            return [_record_to_dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


class PostgresRepositories:
    """Container that holds all PostgreSQL repository implementations.

    Connection pool lifecycle:
        repos = PostgresRepositories(dsn)
        await repos.init_db()       # creates pool + tables
        ...
        await repos.health_check()  # verify connectivity
        await repos.close()         # drain pool
    """

    def __init__(
        self,
        dsn: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

        # Cache repo instances to avoid re-creating on every property access
        self._trades: PostgresTradeRepository | None = None
        self._cycles: PostgresCycleRepository | None = None
        self._rules: PostgresRuleRepository | None = None
        self._bots: PostgresBotRepository | None = None
        self._cross_bot: PostgresCrossBotRepository | None = None

    async def init_db(self) -> None:
        """Create connection pool and all tables if they don't exist.

        Tables are created with CREATE TABLE IF NOT EXISTS as a fallback
        for environments not using Alembic (dev, testing). Production
        should use `quantagent migrate` (Alembic) instead.
        """
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )
        async with self._pool.acquire() as conn:
            for ddl in _ALL_TABLES:
                await conn.execute(ddl)
        logger.info(
            f"PostgreSQL initialized (pool: {self._min_pool_size}-{self._max_pool_size})"
        )

    async def close(self) -> None:
        """Close the connection pool gracefully."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._trades = None
            self._cycles = None
            self._rules = None
            self._bots = None
            self._cross_bot = None
            logger.info("PostgreSQL connection pool closed")

    async def health_check(self) -> dict:
        """Check database connectivity and pool stats.

        Returns:
            Dict with status ("ok" or "error"), pool size, free connections,
            and PostgreSQL server version.
        """
        if self._pool is None:
            return {"status": "error", "error": "Pool not initialized"}

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("SELECT version(), now()")
                return {
                    "status": "ok",
                    "pool_size": self._pool.get_size(),
                    "pool_free": self._pool.get_idle_size(),
                    "pool_min": self._pool.get_min_size(),
                    "pool_max": self._pool.get_max_size(),
                    "server_version": str(row["version"]).split(",")[0] if row else "unknown",
                    "server_time": str(row["now"]) if row else "unknown",
                }
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def _ensure_pool(self) -> asyncpg.Pool:
        """Assert pool is initialized, return it."""
        if self._pool is None:
            raise RuntimeError("Call init_db() before using repositories")
        return self._pool

    @property
    def trades(self) -> PostgresTradeRepository:
        if self._trades is None:
            self._trades = PostgresTradeRepository(self._ensure_pool())
        return self._trades

    @property
    def cycles(self) -> PostgresCycleRepository:
        if self._cycles is None:
            self._cycles = PostgresCycleRepository(self._ensure_pool())
        return self._cycles

    @property
    def rules(self) -> PostgresRuleRepository:
        if self._rules is None:
            self._rules = PostgresRuleRepository(self._ensure_pool())
        return self._rules

    @property
    def bots(self) -> PostgresBotRepository:
        if self._bots is None:
            self._bots = PostgresBotRepository(self._ensure_pool())
        return self._bots

    @property
    def cross_bot(self) -> PostgresCrossBotRepository:
        if self._cross_bot is None:
            self._cross_bot = PostgresCrossBotRepository(self._ensure_pool())
        return self._cross_bot
