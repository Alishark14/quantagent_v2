"""Bot management API endpoints.

CRUD for bot configuration. The /analyze endpoint triggers a single
manual analysis cycle — it does NOT start a persistent bot.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import get_current_user
from api.dependencies import get_bot_repo, get_cycle_repo
from api.schemas import (
    AnalyzeResponse,
    BotCreateRequest,
    BotListResponse,
    BotResponse,
    ErrorResponse,
)
from storage.repositories.base import BotRepository, CycleRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/bots", tags=["bots"])


def _bot_to_response(bot: dict, last_cycle: dict | None = None) -> BotResponse:
    """Convert a bot dict from the repository to a BotResponse."""
    config_raw = bot.get("config_json", "{}")
    if isinstance(config_raw, str):
        try:
            config = json.loads(config_raw)
        except (json.JSONDecodeError, TypeError):
            config = {}
    else:
        config = config_raw or {}

    health_raw = bot.get("last_health", None)
    if isinstance(health_raw, str):
        try:
            health = json.loads(health_raw)
        except (json.JSONDecodeError, TypeError):
            health = None
    else:
        health = health_raw

    # ``created_at`` may arrive as a datetime (from asyncpg / new code paths)
    # or a string (from SQLite / legacy callers). Normalize to ISO-8601 so
    # the Pydantic ``str`` field accepts it.
    created_at_raw = bot.get("created_at", "")
    if isinstance(created_at_raw, datetime):
        created_at_str = created_at_raw.isoformat()
    else:
        created_at_str = created_at_raw or ""

    return BotResponse(
        id=bot["id"],
        user_id=bot.get("user_id", ""),
        symbol=bot.get("symbol", ""),
        timeframe=bot.get("timeframe", ""),
        exchange=bot.get("exchange", ""),
        status=bot.get("status", "unknown"),
        config=config,
        created_at=created_at_str,
        last_health=health,
        last_cycle=last_cycle,
    )


@router.post(
    "",
    response_model=BotResponse,
    status_code=status.HTTP_201_CREATED,
    responses={401: {"model": ErrorResponse}},
)
async def create_bot(
    req: BotCreateRequest,
    user_id: str = Depends(get_current_user),
    bot_repo: BotRepository = Depends(get_bot_repo),
) -> BotResponse:
    """Create a new bot configuration. Does NOT start the bot."""
    # Raw datetime — postgres TIMESTAMPTZ via asyncpg refuses ISO strings.
    now = datetime.now(timezone.utc)
    bot_id = str(uuid4())

    config = {
        "account_balance": req.account_balance,
        "conviction_threshold": req.conviction_threshold,
        "max_position_pct": req.max_position_pct,
    }

    bot = {
        "id": bot_id,
        "user_id": user_id,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "exchange": req.exchange,
        "status": "created",
        "config": config,
        "config_json": config,  # pre-parsed for response converter
        "created_at": now,
        "last_health": None,
    }

    await bot_repo.save_bot(bot)
    logger.info(f"Bot created: {bot_id} ({req.symbol}/{req.timeframe})")

    return _bot_to_response(bot)


@router.get(
    "",
    response_model=BotListResponse,
    responses={401: {"model": ErrorResponse}},
)
async def list_bots(
    user_id: str = Depends(get_current_user),
    bot_repo: BotRepository = Depends(get_bot_repo),
) -> BotListResponse:
    """List all bots for the authenticated user."""
    bots = await bot_repo.get_bots_by_user(user_id)
    return BotListResponse(
        bots=[_bot_to_response(b) for b in bots],
        count=len(bots),
    )


@router.get(
    "/{bot_id}",
    response_model=BotResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_bot(
    bot_id: str,
    user_id: str = Depends(get_current_user),
    bot_repo: BotRepository = Depends(get_bot_repo),
    cycle_repo: CycleRepository = Depends(get_cycle_repo),
) -> BotResponse:
    """Get bot detail including health and last cycle."""
    bot = await bot_repo.get_bot(bot_id)
    if bot is None or bot.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot {bot_id} not found.",
        )

    # Fetch last cycle for this bot
    last_cycle = None
    try:
        cycles = await cycle_repo.get_recent_cycles(bot_id, limit=1)
        if cycles:
            last_cycle = cycles[0]
    except Exception:
        logger.warning(f"Failed to fetch last cycle for bot {bot_id}", exc_info=True)

    return _bot_to_response(bot, last_cycle=last_cycle)


@router.delete(
    "/{bot_id}",
    response_model=BotResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def delete_bot(
    bot_id: str,
    user_id: str = Depends(get_current_user),
    bot_repo: BotRepository = Depends(get_bot_repo),
) -> BotResponse:
    """Mark a bot as stopped. Does not delete data."""
    bot = await bot_repo.get_bot(bot_id)
    if bot is None or bot.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot {bot_id} not found.",
        )

    await bot_repo.update_bot_health(bot_id, {"status": "stopped"})

    # Reflect the updated status in the response
    bot["status"] = "stopped"
    return _bot_to_response(bot)


@router.post(
    "/{bot_id}/analyze",
    response_model=AnalyzeResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def analyze_bot(
    bot_id: str,
    user_id: str = Depends(get_current_user),
    bot_repo: BotRepository = Depends(get_bot_repo),
    cycle_repo: CycleRepository = Depends(get_cycle_repo),
) -> AnalyzeResponse:
    """Trigger one manual analysis cycle for this bot.

    This is a lightweight endpoint that runs the pipeline once and returns
    the result. It does NOT execute trades — it only returns the analysis.

    NOTE: Full pipeline execution requires engine dependencies (LLM, exchange).
    This endpoint returns the most recent cycle data if available, or triggers
    a new cycle when the full engine is wired up.
    """
    bot = await bot_repo.get_bot(bot_id)
    if bot is None or bot.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Bot {bot_id} not found.",
        )

    # Return most recent cycle as the analysis result
    cycles = await cycle_repo.get_recent_cycles(bot_id, limit=1)
    if cycles:
        cycle = cycles[0]
        signals_json = cycle.get("signals_json", "[]")
        if isinstance(signals_json, str):
            try:
                signals = json.loads(signals_json)
            except (json.JSONDecodeError, TypeError):
                signals = []
        else:
            signals = signals_json or []

        return AnalyzeResponse(
            bot_id=bot_id,
            status="OK",
            action=cycle.get("action", "SKIP"),
            conviction_score=cycle.get("conviction_score", 0.0) or 0.0,
            reasoning="Analysis from most recent cycle",
            signals_summary=signals,
        )

    # No cycle data yet
    return AnalyzeResponse(
        bot_id=bot_id,
        status="NO_DATA",
        action="SKIP",
        conviction_score=0.0,
        reasoning="No analysis cycles recorded for this bot yet.",
    )
