"""Alembic environment configuration for async PostgreSQL migrations.

Reads DATABASE_URL from environment. Supports both online (asyncpg)
and offline (SQL generation) modes.

For SQLite dev: set DATABASE_BACKEND=sqlite to skip Alembic entirely
(SQLite repos use CREATE TABLE IF NOT EXISTS on init_db).
"""

from __future__ import annotations

import asyncio
import os
import logging
from logging.config import fileConfig

from alembic import context

from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import async_engine_from_config, create_async_engine

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# No SQLAlchemy MetaData — we use raw SQL migrations
target_metadata = None


def get_database_url() -> str:
    """Get database URL from environment, converting to async driver."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is required for migrations. "
            "Set it to a PostgreSQL connection string."
        )
    # Convert postgresql:// to postgresql+asyncpg:// for async support
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode — generates SQL without a DB connection."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations within a connection context."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = create_async_engine(get_database_url(), poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
