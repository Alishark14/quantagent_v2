"""Tests for Alembic migration infrastructure and database schema.

Tests the migration files, SQLite schema parity, seed script,
and PostgresRepositories pool lifecycle (mocked — no real PG needed).
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Migration file structure tests
# ---------------------------------------------------------------------------


class TestMigrationStructure:
    """Verify that the Alembic migration infrastructure is correctly set up."""

    def test_alembic_ini_exists(self):
        ini_path = Path("alembic.ini")
        assert ini_path.exists(), "alembic.ini should exist in project root"

    def test_alembic_ini_references_alembic_dir(self):
        content = Path("alembic.ini").read_text()
        assert "script_location = alembic" in content

    def test_alembic_env_exists(self):
        env_path = Path("alembic/env.py")
        assert env_path.exists()

    def test_alembic_env_reads_database_url(self):
        content = Path("alembic/env.py").read_text()
        assert "DATABASE_URL" in content
        assert "asyncpg" in content

    def test_initial_migration_exists(self):
        migration_path = Path("alembic/versions/001_initial.py")
        assert migration_path.exists()

    def test_initial_migration_creates_all_tables(self):
        content = Path("alembic/versions/001_initial.py").read_text()
        expected_tables = ["bots", "trades", "cycles", "rules", "cross_bot_signals"]
        for table in expected_tables:
            assert f'"{table}"' in content, f"Migration should create {table} table"

    def test_initial_migration_has_downgrade(self):
        content = Path("alembic/versions/001_initial.py").read_text()
        assert "def downgrade" in content
        assert "drop_table" in content

    def test_initial_migration_has_indexes(self):
        content = Path("alembic/versions/001_initial.py").read_text()
        assert "create_index" in content
        # Key indexes
        assert "ix_bots_user_id" in content
        assert "ix_trades_bot_id" in content
        assert "ix_trades_status" in content
        assert "ix_cycles_bot_timestamp" in content
        assert "ix_rules_symbol_tf" in content
        assert "ix_cross_bot_user_symbol" in content

    def test_script_mako_template_exists(self):
        mako_path = Path("alembic/script.py.mako")
        assert mako_path.exists()

    def test_migration_revision_chain(self):
        """001 is the initial migration with no parent."""
        content = Path("alembic/versions/001_initial.py").read_text()
        assert 'revision: str = "001"' in content
        assert "down_revision" in content


# ---------------------------------------------------------------------------
# SQLite schema parity tests — ensure DDL matches migration
# ---------------------------------------------------------------------------


class TestSQLiteSchemaParity:
    """Verify SQLite CREATE TABLE DDL matches the Alembic migration columns."""

    @pytest.mark.asyncio
    async def test_sqlite_creates_all_tables(self, tmp_path):
        """init_db should create all 5 tables in SQLite."""
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ) as cursor:
                tables = [row[0] async for row in cursor]

        expected = ["bots", "cross_bot_signals", "cycles", "rules", "trades"]
        assert tables == expected

    @pytest.mark.asyncio
    async def test_bots_table_columns(self, tmp_path):
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA table_info(bots)") as cursor:
                columns = {row[1] async for row in cursor}

        expected = {
            "id", "user_id", "symbol", "timeframe", "exchange",
            "status", "config_json", "created_at", "last_health",
        }
        assert columns == expected

    @pytest.mark.asyncio
    async def test_trades_table_columns(self, tmp_path):
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA table_info(trades)") as cursor:
                columns = {row[1] async for row in cursor}

        expected = {
            "id", "user_id", "bot_id", "symbol", "timeframe", "direction",
            "entry_price", "exit_price", "size", "pnl", "r_multiple",
            "entry_time", "exit_time", "exit_reason", "conviction_score",
            "engine_version", "status", "forward_max_r",
        }
        assert columns == expected

    @pytest.mark.asyncio
    async def test_cycles_table_columns(self, tmp_path):
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA table_info(cycles)") as cursor:
                columns = {row[1] async for row in cursor}

        expected = {
            "id", "bot_id", "symbol", "timeframe", "timestamp",
            "indicators_json", "signals_json", "conviction_json",
            "action", "conviction_score",
        }
        assert columns == expected

    @pytest.mark.asyncio
    async def test_rules_table_columns(self, tmp_path):
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA table_info(rules)") as cursor:
                columns = {row[1] async for row in cursor}

        expected = {"id", "symbol", "timeframe", "rule_text", "score", "active", "created_at"}
        assert columns == expected

    @pytest.mark.asyncio
    async def test_cross_bot_signals_table_columns(self, tmp_path):
        from storage.repositories.sqlite import SQLiteRepositories

        db_path = str(tmp_path / "test.db")
        repos = SQLiteRepositories(db_path=db_path)
        await repos.init_db()

        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA table_info(cross_bot_signals)") as cursor:
                columns = {row[1] async for row in cursor}

        expected = {"id", "user_id", "symbol", "direction", "conviction", "bot_id", "timestamp"}
        assert columns == expected


# ---------------------------------------------------------------------------
# PostgresRepositories pool lifecycle tests (mocked)
# ---------------------------------------------------------------------------


class TestPostgresPoolLifecycle:
    """Test PostgresRepositories pool creation, health_check, and close."""

    def test_pool_not_initialized_raises(self):
        from storage.repositories.postgres import PostgresRepositories

        repos = PostgresRepositories(dsn="postgresql://localhost/test")
        with pytest.raises(RuntimeError, match="init_db"):
            _ = repos.trades

    def test_pool_params_stored(self):
        from storage.repositories.postgres import PostgresRepositories

        repos = PostgresRepositories(
            dsn="postgresql://localhost/test",
            min_pool_size=3,
            max_pool_size=20,
        )
        assert repos._min_pool_size == 3
        assert repos._max_pool_size == 20

    @pytest.mark.asyncio
    async def test_health_check_returns_error_when_no_pool(self):
        from storage.repositories.postgres import PostgresRepositories

        repos = PostgresRepositories(dsn="postgresql://localhost/test")
        result = await repos.health_check()
        assert result["status"] == "error"
        assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_close_without_init_is_safe(self):
        from storage.repositories.postgres import PostgresRepositories

        repos = PostgresRepositories(dsn="postgresql://localhost/test")
        await repos.close()  # should not raise

    def test_repo_properties_cache_instances(self):
        """Accessing .trades twice should return the same instance."""
        from storage.repositories.postgres import PostgresRepositories

        repos = PostgresRepositories(dsn="postgresql://localhost/test")
        repos._pool = MagicMock()  # fake pool

        trades1 = repos.trades
        trades2 = repos.trades
        assert trades1 is trades2

        bots1 = repos.bots
        bots2 = repos.bots
        assert bots1 is bots2


# ---------------------------------------------------------------------------
# Seed script tests
# ---------------------------------------------------------------------------


class TestSeedScript:
    """Test the dev seed script with SQLite backend."""

    @pytest.mark.asyncio
    async def test_seed_creates_bots(self, tmp_path):
        db_path = str(tmp_path / "seed_test.db")

        with patch.dict(os.environ, {
            "DATABASE_BACKEND": "sqlite",
            "SQLITE_DB_PATH": db_path,
        }):
            from scripts.seed_dev import seed_dev_data
            await seed_dev_data()

        # Verify bots were created
        from storage.repositories.sqlite import SQLiteRepositories
        repos = SQLiteRepositories(db_path=db_path)

        bot = await repos.bots.get_bot("dev-btc-1h")
        assert bot is not None
        assert bot["symbol"] == "BTC-USDC"
        assert bot["timeframe"] == "1h"

        bot2 = await repos.bots.get_bot("dev-eth-4h")
        assert bot2 is not None
        assert bot2["symbol"] == "ETH-USDC"

    @pytest.mark.asyncio
    async def test_seed_creates_rules(self, tmp_path):
        db_path = str(tmp_path / "seed_test.db")

        with patch.dict(os.environ, {
            "DATABASE_BACKEND": "sqlite",
            "SQLITE_DB_PATH": db_path,
        }):
            from scripts.seed_dev import seed_dev_data
            await seed_dev_data()

        from storage.repositories.sqlite import SQLiteRepositories
        repos = SQLiteRepositories(db_path=db_path)

        rules = await repos.rules.get_rules("BTC-USDC", "1h")
        assert len(rules) >= 2

    @pytest.mark.asyncio
    async def test_seed_is_idempotent(self, tmp_path):
        """Running seed twice should not crash or duplicate bots."""
        db_path = str(tmp_path / "seed_test.db")

        with patch.dict(os.environ, {
            "DATABASE_BACKEND": "sqlite",
            "SQLITE_DB_PATH": db_path,
        }):
            from scripts.seed_dev import seed_dev_data
            await seed_dev_data()
            # Second run — bots already exist
            await seed_dev_data()

        from storage.repositories.sqlite import SQLiteRepositories
        repos = SQLiteRepositories(db_path=db_path)

        bots = await repos.bots.get_bots_by_user("dev-user")
        assert len(bots) == 3  # not 6


# ---------------------------------------------------------------------------
# CLI migrate command tests
# ---------------------------------------------------------------------------


class TestCLIMigrate:
    """Test the CLI migrate command registration."""

    def test_help_shows_migrate(self, capsys):
        sys.argv = ["quantagent", "--help"]
        from quantagent.main import main
        main()
        captured = capsys.readouterr()
        assert "migrate" in captured.out
        assert "seed" in captured.out

    def test_migrate_without_database_url_exits(self):
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False):
            from quantagent.main import migrate
            with pytest.raises(SystemExit):
                migrate()


# ---------------------------------------------------------------------------
# Alembic env.py tests
# ---------------------------------------------------------------------------


class TestAlembicEnv:
    """Test the alembic env.py get_database_url logic.

    alembic/env.py can't be imported directly (it requires Alembic runtime
    context). Instead we test the same URL conversion logic inline.
    """

    @staticmethod
    def _get_database_url() -> str:
        """Reproduce the get_database_url logic from alembic/env.py."""
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            raise ValueError("DATABASE_URL environment variable is required")
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url

    def test_env_py_contains_get_database_url(self):
        """Verify the function exists in alembic/env.py source."""
        content = Path("alembic/env.py").read_text()
        assert "def get_database_url" in content
        assert "postgresql+asyncpg://" in content

    def test_url_converts_postgresql_to_asyncpg(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db"}):
            url = self._get_database_url()
            assert url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_url_converts_postgres_to_asyncpg(self):
        with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@localhost/db"}):
            url = self._get_database_url()
            assert url == "postgresql+asyncpg://user:pass@localhost/db"

    def test_url_raises_without_env(self):
        with patch.dict(os.environ, {"DATABASE_URL": ""}):
            with pytest.raises(ValueError, match="DATABASE_URL"):
                self._get_database_url()
