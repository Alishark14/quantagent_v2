"""QuantAgent CLI entry point.

Commands:
    python -m quantagent run       — Start BotRunner + FastAPI together
    python -m quantagent run --shadow — Same runner, sim exchange + shadow DB
    python -m quantagent migrate   — Run Alembic migrations (upgrade to head)
    python -m quantagent seed      — Seed dev database with test data
    python -m quantagent           — Show help
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys


def _setup_logging() -> None:
    """Configure structured logging."""
    level = logging.DEBUG if os.environ.get("VERBOSE") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def _run_server() -> None:
    """Start the BotRunner and FastAPI server together.

    Ctrl+C triggers graceful shutdown via signal handlers.
    """
    import uvicorn

    from api.app import create_app
    from backtesting.shadow import (
        ensure_shadow_db,
        is_shadow_mode,
    )
    from engine.bot_manager import BotManager
    from engine.events import create_event_bus
    from storage.repositories import get_repositories

    logger = logging.getLogger("quantagent")

    from quantagent.version import ENGINE_VERSION
    logger.info(f"QuantAgent {ENGINE_VERSION} starting...")

    # ── Shadow-mode setup (if active) ──
    # CLI parsing in main() has already flipped QUANTAGENT_SHADOW and
    # mutated DATABASE_URL via configure_shadow(). All we need to do
    # here is ensure the shadow DB exists and migrate it before any
    # repository connection is made.
    if is_shadow_mode():
        logger.warning(
            "⚠️ SHADOW MODE — no real trades will be placed. "
            "All writes go to the shadow database."
        )
        shadow_url = os.environ.get("DATABASE_URL", "")
        if shadow_url:
            await ensure_shadow_db(shadow_url)

    # ── Initialize infrastructure ──
    logger.info("Initializing infrastructure...")
    repos = await get_repositories()
    event_bus = create_event_bus("memory")

    # Bot factory placeholder — wires engine deps per symbol
    def bot_factory(symbol: str, bot_id: str):
        raise NotImplementedError(
            "Full bot_factory requires LLM + exchange adapter wiring. "
            "Use run_trade.py for manual cycles or wire the factory "
            "in a deployment-specific entrypoint."
        )

    bot_manager = BotManager(
        event_bus=event_bus,
        bot_factory=bot_factory,
    )

    # ── Initialize BotRunner ──
    from quantagent.runner import BotRunner

    def adapter_factory(exchange: str):
        from exchanges.factory import ExchangeFactory
        return ExchangeFactory.get_adapter(exchange)

    runner = BotRunner(
        repos=repos,
        adapter_factory=adapter_factory,
        llm_provider=None,  # wired by bot_factory per bot
        event_bus=event_bus,
        bot_manager=bot_manager,
    )

    # Load active bots from DB
    # Since BotRepository.get_bots_by_user requires a user_id,
    # we start with no bots — they're added via API at runtime.
    await runner.start()

    # ── Create FastAPI app ──
    app = create_app()
    app.state.repos = repos
    app.state.runner = runner

    # ── Setup signal handlers for graceful shutdown ──
    shutdown_event = asyncio.Event()

    def _signal_handler(sig, frame):
        logger.info(f"Received {signal.Signals(sig).name}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Start uvicorn ──
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    # Run server and wait for shutdown signal
    server_task = asyncio.create_task(server.serve())

    logger.info(f"QuantAgent running on {host}:{port}")

    # Wait for shutdown signal
    await shutdown_event.wait()

    # ── Graceful shutdown ──
    logger.info("Graceful shutdown initiated...")
    await runner.stop()
    server.should_exit = True
    await server_task

    logger.info("QuantAgent stopped.")


def run() -> None:
    """Run the full server (BotRunner + FastAPI)."""
    _setup_logging()
    try:
        asyncio.run(_run_server())
    except KeyboardInterrupt:
        pass


def migrate(revision: str = "head") -> None:
    """Run Alembic migrations to the specified revision."""
    _setup_logging()
    logger = logging.getLogger("quantagent")

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable is required.")
        sys.exit(1)

    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config("alembic.ini")

    logger.info(f"Running migrations to: {revision}")
    command.upgrade(alembic_cfg, revision)
    logger.info("Migrations complete.")


def seed() -> None:
    """Seed the dev database with test data."""
    _setup_logging()

    async def _seed():
        from scripts.seed_dev import seed_dev_data
        await seed_dev_data()

    asyncio.run(_seed())


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    # ``--shadow`` is global: it applies to any subcommand and is
    # stripped from argv before subcommand dispatch so the existing
    # positional logic keeps working.
    shadow_requested = "--shadow" in args
    if shadow_requested:
        args = [a for a in args if a != "--shadow"]
        from backtesting.shadow import configure_shadow

        # Build a tiny config holder so configure_shadow() has something
        # to mutate. The runtime config in this CLI is env-var based, so
        # the env-var side-effects are what actually matter — but we
        # capture the snapshot for log clarity.
        class _RuntimeConfig:
            database_url = os.environ.get("DATABASE_URL", "")
            shadow_mode = False
            use_simulated_exchange = False

        configure_shadow(_RuntimeConfig)

    if not args or args[0] in ("--help", "-h"):
        print("QuantAgent v2 — AI-powered trading engine")
        print()
        print("Usage:")
        print("  python -m quantagent run            Start BotRunner + API server")
        print("  python -m quantagent run --shadow   Same runner, sim exchange + shadow DB")
        print("  python -m quantagent migrate        Run database migrations (Alembic)")
        print("  python -m quantagent seed           Seed dev database with test data")
        print("  python -m quantagent --help         Show this help")
        return

    if args[0] == "run":
        run()
    elif args[0] == "migrate":
        revision = args[1] if len(args) > 1 else "head"
        migrate(revision)
    elif args[0] == "seed":
        seed()
    else:
        print(f"Unknown command: {args[0]}")
        print("Use --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
