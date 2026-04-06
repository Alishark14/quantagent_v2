"""Repository factory — returns the correct backend based on configuration.

Usage:
    repos = await get_repositories()          # reads DATABASE_BACKEND env var
    repos = await get_repositories("sqlite")  # explicit backend

    trade_id = await repos.trades.save_trade({...})
    bot = await repos.bots.get_bot(bot_id)
"""

import os
import logging

logger = logging.getLogger(__name__)


async def get_repositories(backend: str | None = None):
    """Create and initialize a repository container for the given backend.

    Args:
        backend: "sqlite" or "postgresql". Defaults to DATABASE_BACKEND env var,
                 falling back to "sqlite" if unset.

    Returns:
        SQLiteRepositories or PostgresRepositories with all tables initialized.
    """
    backend = backend or os.getenv("DATABASE_BACKEND", "sqlite")

    if backend == "sqlite":
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = os.getenv("SQLITE_DB_PATH", "quantagent_dev.db")
        repos = SQLiteRepositories(db_path=db_path)
    elif backend == "postgresql":
        from storage.repositories.postgres import PostgresRepositories

        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise ValueError("DATABASE_URL environment variable is required for PostgreSQL backend")
        repos = PostgresRepositories(dsn=dsn)
    else:
        raise ValueError(f"Unknown database backend: {backend!r}. Use 'sqlite' or 'postgresql'.")

    await repos.init_db()
    logger.info(f"Repositories initialized with {backend} backend")
    return repos
