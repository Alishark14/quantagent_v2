"""SQLite repository implementations for local development only.

Uses aiosqlite for async access. All queries are parameterized — never string interpolation.
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

import aiosqlite

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
# ���──────────────────────────��──────────────────────────────────────────

_CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL,
    exit_price REAL,
    size REAL,
    pnl REAL,
    r_multiple REAL,
    entry_time TEXT,
    exit_time TEXT,
    exit_reason TEXT,
    conviction_score REAL,
    engine_version TEXT,
    status TEXT NOT NULL DEFAULT 'open'
);
"""

_CREATE_CYCLES = """
CREATE TABLE IF NOT EXISTS cycles (
    id TEXT PRIMARY KEY,
    bot_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    indicators_json TEXT,
    signals_json TEXT,
    conviction_json TEXT,
    action TEXT,
    conviction_score REAL
);
"""

_CREATE_RULES = """
CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    score INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
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
    config_json TEXT,
    created_at TEXT NOT NULL,
    last_health TEXT
);
"""

_CREATE_CROSS_BOT_SIGNALS = """
CREATE TABLE IF NOT EXISTS cross_bot_signals (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    conviction REAL NOT NULL,
    bot_id TEXT NOT NULL,
    timestamp TEXT NOT NULL
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


# ───���──────────────────────────────────────────────────────────────────
# Trade Repository
# ��─────────────────────────────────────────────────────────────────────


class SQLiteTradeRepository(TradeRepository):

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def save_trade(self, trade: dict) -> str:
        trade_id = trade.get("id") or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO trades
                   (id, user_id, bot_id, symbol, timeframe, direction,
                    entry_price, exit_price, size, pnl, r_multiple,
                    entry_time, exit_time, exit_reason, conviction_score,
                    engine_version, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
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
                ),
            )
            await db.commit()
        return trade_id

    async def get_trade(self, trade_id: str) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_open_positions(self, user_id: str, bot_id: str) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trades WHERE user_id = ? AND bot_id = ? AND status = 'open'",
                (user_id, bot_id),
            ) as cursor:
                return [dict(row) async for row in cursor]

    async def get_trades_by_bot(self, bot_id: str, limit: int = 50) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trades WHERE bot_id = ? ORDER BY entry_time DESC LIMIT ?",
                (bot_id, limit),
            ) as cursor:
                return [dict(row) async for row in cursor]

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        if not updates:
            return False
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [trade_id]
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                f"UPDATE trades SET {set_clauses} WHERE id = ?",
                values,
            )
            await db.commit()
            return cursor.rowcount > 0


# ─────────��─────────────────��──────────────────────────────────────────
# Cycle Repository
# ──────��──────────────���────────────────────────────────────────────────


class SQLiteCycleRepository(CycleRepository):

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def save_cycle(self, cycle: dict) -> str:
        cycle_id = cycle.get("id") or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO cycles
                   (id, bot_id, symbol, timeframe, timestamp,
                    indicators_json, signals_json, conviction_json,
                    action, conviction_score)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
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
                ),
            )
            await db.commit()
        return cycle_id

    async def get_recent_cycles(self, bot_id: str, limit: int = 5) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM cycles WHERE bot_id = ? ORDER BY timestamp DESC LIMIT ?",
                (bot_id, limit),
            ) as cursor:
                rows = [dict(row) async for row in cursor]
        for row in rows:
            for col in ("indicators_json", "signals_json", "conviction_json"):
                if row.get(col):
                    row[col] = json.loads(row[col])
        return rows


# ─���───────────────────���──────────────────────────────��─────────────────
# Rule Repository
# ───────────────��──────────────────────────────────���───────────────────


class SQLiteRuleRepository(RuleRepository):

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def save_rule(self, rule: dict) -> str:
        rule_id = rule.get("id") or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO rules (id, symbol, timeframe, rule_text, score, active, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    rule_id,
                    rule["symbol"],
                    rule["timeframe"],
                    rule["rule_text"],
                    rule.get("score", 0),
                    1 if rule.get("active", True) else 0,
                    rule.get("created_at", _now_iso()),
                ),
            )
            await db.commit()
        return rule_id

    async def get_rules(self, symbol: str, timeframe: str) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM rules WHERE symbol = ? AND timeframe = ? AND active = 1",
                (symbol, timeframe),
            ) as cursor:
                rows = [dict(row) async for row in cursor]
        for row in rows:
            row["active"] = bool(row["active"])
        return rows

    async def update_rule_score(self, rule_id: str, delta: int) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "UPDATE rules SET score = score + ? WHERE id = ?",
                (delta, rule_id),
            )
            if cursor.rowcount == 0:
                await db.commit()
                return False
            # Auto-deactivate if score drops below -2
            await db.execute(
                "UPDATE rules SET active = 0 WHERE id = ? AND score < -2",
                (rule_id,),
            )
            await db.commit()
            return True

    async def deactivate_rule(self, rule_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "UPDATE rules SET active = 0 WHERE id = ?",
                (rule_id,),
            )
            await db.commit()
            return cursor.rowcount > 0


# ────────────────────────────────────────────────────────────────────��─
# Bot Repository
# ───────────────────────────��────────────────────────────────���─────────


class SQLiteBotRepository(BotRepository):

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def save_bot(self, bot: dict) -> str:
        bot_id = bot.get("id") or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO bots
                   (id, user_id, symbol, timeframe, exchange, status,
                    config_json, created_at, last_health)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    bot_id,
                    bot["user_id"],
                    bot["symbol"],
                    bot["timeframe"],
                    bot["exchange"],
                    bot.get("status", "active"),
                    json.dumps(bot.get("config")) if bot.get("config") else None,
                    bot.get("created_at", _now_iso()),
                    None,
                ),
            )
            await db.commit()
        return bot_id

    async def get_bot(self, bot_id: str) -> dict | None:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM bots WHERE id = ?", (bot_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                result = dict(row)
        if result.get("config_json"):
            result["config_json"] = json.loads(result["config_json"])
        if result.get("last_health"):
            result["last_health"] = json.loads(result["last_health"])
        return result

    async def get_bots_by_user(self, user_id: str) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM bots WHERE user_id = ?", (user_id,)
            ) as cursor:
                rows = [dict(row) async for row in cursor]
        for row in rows:
            if row.get("config_json"):
                row["config_json"] = json.loads(row["config_json"])
            if row.get("last_health"):
                row["last_health"] = json.loads(row["last_health"])
        return rows

    async def update_bot_health(self, bot_id: str, health: dict) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(
                "UPDATE bots SET last_health = ? WHERE id = ?",
                (json.dumps(health), bot_id),
            )
            await db.commit()
            return cursor.rowcount > 0


# ────���─────────────────────────────��───────────────────────────────────
# Cross-Bot Repository
# ───────────────────────��──────────────────────────────────────────────


class SQLiteCrossBotRepository(CrossBotRepository):

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def save_signal(self, signal: dict) -> None:
        signal_id = signal.get("id") or str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """INSERT INTO cross_bot_signals
                   (id, user_id, symbol, direction, conviction, bot_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    signal_id,
                    signal["user_id"],
                    signal["symbol"],
                    signal["direction"],
                    signal["conviction"],
                    signal["bot_id"],
                    signal.get("timestamp", _now_iso()),
                ),
            )
            await db.commit()

    async def get_recent_signals(
        self, symbol: str, user_id: str, limit: int = 10
    ) -> list[dict]:
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM cross_bot_signals
                   WHERE symbol = ? AND user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (symbol, user_id, limit),
            ) as cursor:
                return [dict(row) async for row in cursor]


# ─────────────��────────────────────────────────���───────────────────────
# Container
# ─────────────────────────────────���────────────────────────────────────


class SQLiteRepositories:
    """Container that holds all SQLite repository implementations."""

    def __init__(self, db_path: str = "quantagent_dev.db") -> None:
        self._db_path = db_path
        self._trades = SQLiteTradeRepository(db_path)
        self._cycles = SQLiteCycleRepository(db_path)
        self._rules = SQLiteRuleRepository(db_path)
        self._bots = SQLiteBotRepository(db_path)
        self._cross_bot = SQLiteCrossBotRepository(db_path)

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            for ddl in _ALL_TABLES:
                await db.execute(ddl)
            await db.commit()
        logger.info(f"SQLite database initialized at {self._db_path}")

    @property
    def trades(self) -> SQLiteTradeRepository:
        return self._trades

    @property
    def cycles(self) -> SQLiteCycleRepository:
        return self._cycles

    @property
    def rules(self) -> SQLiteRuleRepository:
        return self._rules

    @property
    def bots(self) -> SQLiteBotRepository:
        return self._bots

    @property
    def cross_bot(self) -> SQLiteCrossBotRepository:
        return self._cross_bot
