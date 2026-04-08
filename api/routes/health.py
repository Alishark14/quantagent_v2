"""System health API endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from api.dependencies import get_bot_repo, get_health_tracker
from api.schemas import HealthResponse
from quantagent.version import ENGINE_VERSION
from storage.repositories.base import BotRepository
from tracking.health import HealthTracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def get_health(
    health_tracker: HealthTracker = Depends(get_health_tracker),
    bot_repo: BotRepository = Depends(get_bot_repo),
) -> HealthResponse:
    """System health snapshot. No auth required — used for monitoring."""
    summary = health_tracker.summary()

    # Count active bots
    active_bots = 0
    try:
        # Get all bots across all users (health check is global)
        # BotRepository doesn't have a get_all method, so we report 0
        # until we can query the DB directly. This is fine for health checks.
        active_bots = summary.get("active_bots", 0)
    except Exception:
        logger.warning("Failed to count active bots", exc_info=True)

    # DB status
    db_status = "ok"
    try:
        # Quick connectivity check — try to fetch a non-existent bot
        await bot_repo.get_bot("__health_check__")
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="healthy" if summary["error_count"] == 0 else "degraded",
        engine_version=ENGINE_VERSION,
        uptime_seconds=summary["uptime_seconds"],
        total_events=summary["total_events"],
        event_counts=summary["event_counts"],
        error_count=summary["error_count"],
        recent_errors=summary["recent_errors"],
        active_bots=active_bots,
        db_status=db_status,
    )
