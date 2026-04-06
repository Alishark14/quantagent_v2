"""FastAPI dependency injection providers.

All shared state (repos, health tracker) is initialized at startup via lifespan
and stored in app.state. Dependencies read from there — no global mutable state.
"""

from __future__ import annotations

from fastapi import Depends, Request

from storage.repositories.base import (
    BotRepository,
    CycleRepository,
    RuleRepository,
    TradeRepository,
)
from tracking.health import HealthTracker


def get_repos(request: Request):
    """Return the repository container from app state."""
    return request.app.state.repos


def get_trade_repo(request: Request) -> TradeRepository:
    """Return the trade repository."""
    return request.app.state.repos.trades


def get_bot_repo(request: Request) -> BotRepository:
    """Return the bot repository."""
    return request.app.state.repos.bots


def get_cycle_repo(request: Request) -> CycleRepository:
    """Return the cycle repository."""
    return request.app.state.repos.cycles


def get_rule_repo(request: Request) -> RuleRepository:
    """Return the rule repository."""
    return request.app.state.repos.rules


def get_health_tracker(request: Request) -> HealthTracker:
    """Return the health tracker from app state."""
    return request.app.state.health_tracker
