"""FastAPI application factory with lifespan management.

Startup: initialize repos + health tracker.
Shutdown: cleanup (close DB connections).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from storage.repositories import get_repositories
from tracking.health import HealthTracker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the FastAPI app."""
    # ── Startup ──
    logger.info("QuantAgent API starting up")

    repos = await get_repositories()
    app.state.repos = repos

    health_tracker = HealthTracker()
    app.state.health_tracker = health_tracker

    logger.info("QuantAgent API ready")
    yield

    # ── Shutdown ──
    logger.info("QuantAgent API shutting down")
    # Close DB connections if the container exposes a close method
    if hasattr(repos, "close"):
        await repos.close()
    logger.info("QuantAgent API stopped")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="QuantAgent API",
        description="AI-powered trading engine API",
        version="2026.04.2.0.0-alpha.1",
        lifespan=lifespan,
    )

    # Import and include routers
    from api.routes.bots import router as bots_router
    from api.routes.health import router as health_router
    from api.routes.positions import router as positions_router
    from api.routes.rules import router as rules_router
    from api.routes.trades import router as trades_router

    app.include_router(bots_router)
    app.include_router(trades_router)
    app.include_router(health_router)
    app.include_router(positions_router)
    app.include_router(rules_router)

    return app
