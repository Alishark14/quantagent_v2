"""Seed development database with test bot configurations.

Usage:
    python -m quantagent seed
    # or directly:
    python scripts/seed_dev.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from storage.repositories import get_repositories

logger = logging.getLogger(__name__)

# Test bot configs for local development
_DEV_BOTS = [
    {
        "id": "dev-btc-1h",
        "user_id": "dev-user",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "exchange": "hyperliquid",
        "status": "active",
        "config": {
            "account_balance": 10000.0,
            "conviction_threshold": 0.5,
            "max_position_pct": 0.5,
        },
    },
    {
        "id": "dev-eth-4h",
        "user_id": "dev-user",
        "symbol": "ETH-USDC",
        "timeframe": "4h",
        "exchange": "hyperliquid",
        "status": "active",
        "config": {
            "account_balance": 5000.0,
            "conviction_threshold": 0.6,
            "max_position_pct": 0.3,
        },
    },
    {
        "id": "dev-sol-15m",
        "user_id": "dev-user",
        "symbol": "SOL-USDC",
        "timeframe": "15m",
        "exchange": "hyperliquid",
        "status": "active",
        "config": {
            "account_balance": 2000.0,
            "conviction_threshold": 0.55,
            "max_position_pct": 0.4,
        },
    },
]

# Sample reflection rules
_DEV_RULES = [
    {
        "id": "rule-rsi-overbought",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "rule_text": "Avoid LONG entries when RSI > 78 — high reversal probability in ranging markets",
        "score": 3,
        "active": True,
    },
    {
        "id": "rule-volume-confirm",
        "symbol": "BTC-USDC",
        "timeframe": "1h",
        "rule_text": "Require volume > 1.5x 20-period average for breakout confirmation",
        "score": 2,
        "active": True,
    },
]


async def seed_dev_data() -> None:
    """Seed the database with test data for local development."""
    repos = await get_repositories()
    now = datetime.now(timezone.utc).isoformat()

    bot_count = 0
    for bot_cfg in _DEV_BOTS:
        existing = await repos.bots.get_bot(bot_cfg["id"])
        if existing is not None:
            logger.info(f"Bot {bot_cfg['id']} already exists, skipping")
            continue

        bot = {**bot_cfg, "created_at": now}
        await repos.bots.save_bot(bot)
        bot_count += 1
        logger.info(f"Created bot: {bot_cfg['id']} ({bot_cfg['symbol']}/{bot_cfg['timeframe']})")

    rule_count = 0
    for rule_cfg in _DEV_RULES:
        rule = {**rule_cfg, "created_at": now}
        try:
            await repos.rules.save_rule(rule)
            rule_count += 1
            logger.info(f"Created rule: {rule_cfg['id']}")
        except Exception:
            logger.info(f"Rule {rule_cfg['id']} already exists, skipping")

    print(f"Seeded {bot_count} bots and {rule_count} rules.")

    if hasattr(repos, "close"):
        await repos.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(seed_dev_data())
