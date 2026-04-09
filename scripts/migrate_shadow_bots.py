"""One-shot migration: copy bots from `quantagent_shadow` → `quantagent`.

The shadow-mode redesign collapses the separate `quantagent_shadow`
PostgreSQL database into the shared `quantagent` database, with each
bot tagged via the new `is_shadow` / `mode` columns (added by Alembic
revision 003).

This script reads every bot row from the legacy shadow DB, then inserts
it into the production DB with `is_shadow=true` and `mode='shadow'`.
Bot IDs are preserved so that existing trades, cycles, and Sentinel
escalation state remain joinable. Duplicates (matched by primary key)
are skipped silently — re-running the script is safe.

Usage::

    export DATABASE_URL=postgresql://user:pass@host/quantagent
    export SHADOW_DATABASE_URL=postgresql://user:pass@host/quantagent_shadow
    python scripts/migrate_shadow_bots.py

After this script reports success and the new server has been verified
in production, the old `quantagent_shadow` database can be dropped::

    sudo -u postgres dropdb quantagent_shadow
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_shadow_bots")


def _normalise_dsn(dsn: str) -> str:
    """asyncpg understands `postgresql://`, not `postgresql+asyncpg://`."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql://", 1)
    return dsn


async def _fetch_shadow_bots(shadow_dsn: str) -> list[asyncpg.Record]:
    conn = await asyncpg.connect(shadow_dsn)
    try:
        rows = await conn.fetch("SELECT * FROM bots")
    finally:
        await conn.close()
    return rows


async def _insert_shadow_bot(conn: asyncpg.Connection, bot: asyncpg.Record) -> str:
    """Insert one shadow bot into production. Returns 'inserted' or 'skipped'."""
    existing = await conn.fetchval("SELECT 1 FROM bots WHERE id = $1", bot["id"])
    if existing:
        return "skipped"

    config_json = bot.get("config_json") if "config_json" in bot.keys() else None
    if config_json is not None and not isinstance(config_json, str):
        config_json = json.dumps(config_json)

    last_health = bot.get("last_health") if "last_health" in bot.keys() else None
    if last_health is not None and not isinstance(last_health, str):
        last_health = json.dumps(last_health)

    await conn.execute(
        """INSERT INTO bots
           (id, user_id, symbol, timeframe, exchange, status,
            config_json, created_at, last_health, is_shadow, mode)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, true, 'shadow')""",
        bot["id"],
        bot["user_id"],
        bot["symbol"],
        bot["timeframe"],
        bot["exchange"],
        bot.get("status", "active") if hasattr(bot, "get") else (bot["status"] or "active"),
        config_json,
        bot["created_at"],
        last_health,
    )
    return "inserted"


async def main() -> int:
    prod_dsn = os.environ.get("DATABASE_URL", "")
    shadow_dsn = os.environ.get(
        "SHADOW_DATABASE_URL",
        prod_dsn.replace("/quantagent", "/quantagent_shadow") if prod_dsn else "",
    )

    if not prod_dsn:
        logger.error("DATABASE_URL is required (production DB)")
        return 1
    if not shadow_dsn:
        logger.error("SHADOW_DATABASE_URL is required (legacy shadow DB)")
        return 1

    prod_dsn = _normalise_dsn(prod_dsn)
    shadow_dsn = _normalise_dsn(shadow_dsn)

    logger.info("Reading bots from shadow DB...")
    try:
        shadow_bots = await _fetch_shadow_bots(shadow_dsn)
    except Exception as exc:
        logger.error(f"Failed to read shadow DB: {exc}")
        return 1
    logger.info(f"Found {len(shadow_bots)} bots in shadow DB")

    if not shadow_bots:
        logger.info("Nothing to migrate.")
        return 0

    inserted = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    prod_conn = await asyncpg.connect(prod_dsn)
    try:
        for bot in shadow_bots:
            bot_id = bot["id"]
            try:
                result = await _insert_shadow_bot(prod_conn, bot)
            except Exception as exc:
                logger.error(f"  ✗ {bot_id}: {exc}")
                failed.append((bot_id, str(exc)))
                continue
            if result == "inserted":
                inserted += 1
                logger.info(f"  ✓ inserted {bot_id} ({bot['symbol']} {bot['timeframe']})")
            else:
                skipped += 1
                logger.info(f"  · skipped {bot_id} (already present)")
    finally:
        await prod_conn.close()

    logger.info("─" * 60)
    logger.info(f"Migration summary: {inserted} inserted, {skipped} skipped, {len(failed)} failed")
    if failed:
        logger.error("Failed bot IDs:")
        for bot_id, err in failed:
            logger.error(f"  - {bot_id}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
