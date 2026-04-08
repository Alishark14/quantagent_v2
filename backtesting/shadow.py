"""Shadow mode — same engine, fake exchange + isolated database.

Per ARCHITECTURE.md §31.3.4, Tier 4 backtesting runs the **exact same**
``BotRunner`` code path against live data, but with two swaps:

1. The PostgreSQL connection points to a dedicated ``shadow_db``.
2. The ``SimulatedExchangeAdapter`` is injected instead of the real
   adapter.

Critically, this is **not a separate runner**. Code drift between
shadow and live runners would invalidate every test result. Instead the
swap happens via dependency injection: ``configure_shadow()`` mutates
the runtime config in place, and downstream consumers (the exchange
factory, repository factory) read from the same global state they
always do.

The shadow data MUST NOT mix with live data — otherwise the Quant Data
Scientist would mine alpha factors from fake fills and the tracking
system would report incorrect metrics. Database isolation is enforced
at the URL level; ``ensure_shadow_db()`` creates the shadow database
on first run and applies Alembic migrations.

Detection precedence (highest first):

1. ``QUANTAGENT_SHADOW`` environment variable set to a truthy value
2. CLI ``--shadow`` flag (which sets the env var before anything else)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)


_SHADOW_ENV_VAR = "QUANTAGENT_SHADOW"
_SHADOW_SUFFIX = "_shadow"
_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def is_shadow_mode() -> bool:
    """Return True if shadow mode is active for this process."""
    return os.environ.get(_SHADOW_ENV_VAR, "").lower() in _TRUTHY


def enable_shadow_mode() -> None:
    """Set the shadow env var so ``is_shadow_mode()`` returns True everywhere
    downstream in this process."""
    os.environ[_SHADOW_ENV_VAR] = "1"


def disable_shadow_mode() -> None:
    """Clear the shadow env var. Used in tests and shutdown paths."""
    os.environ.pop(_SHADOW_ENV_VAR, None)


# ---------------------------------------------------------------------------
# Database URL transformation
# ---------------------------------------------------------------------------


def get_shadow_db_url(base_url: str) -> str:
    """Return the shadow-database URL for ``base_url``.

    Postgres example::

        postgresql://u:p@host:5432/quantagent
            → postgresql://u:p@host:5432/quantagent_shadow

    Query strings (e.g. ``?sslmode=require``) are preserved::

        postgresql://u:p@host/quantagent?sslmode=require
            → postgresql://u:p@host/quantagent_shadow?sslmode=require

    SQLite example (file URL)::

        sqlite+aiosqlite:///./dev.db
            → sqlite+aiosqlite:///./dev_shadow.db

    Idempotent: passing an already-shadowed URL returns it unchanged.

    Raises:
        ValueError: ``base_url`` is empty or has no parseable path.
    """
    if not base_url:
        raise ValueError("base_url must be non-empty")

    if base_url.startswith("sqlite"):
        # SQLite URLs use the triple-slash convention
        # (sqlite:///path or sqlite+aiosqlite:///path) which urlparse
        # round-trips lossily when netloc is empty. Handle by string ops
        # so the canonical form survives.
        return _shadow_sqlite_url_string(base_url)

    parsed = urlparse(base_url)
    return _shadow_network_db_url(parsed)


def _shadow_network_db_url(parsed) -> str:
    """Postgres / MySQL / etc. — the database name is the path component."""
    path = parsed.path or ""
    if not path or path == "/":
        raise ValueError(
            f"Cannot derive shadow DB from URL with no database name: "
            f"{urlunparse(parsed)}"
        )
    db_name = path.lstrip("/")
    if db_name.endswith(_SHADOW_SUFFIX):
        return urlunparse(parsed)  # already shadow — idempotent
    new_path = "/" + db_name + _SHADOW_SUFFIX
    return urlunparse(parsed._replace(path=new_path))


def _shadow_sqlite_url_string(url: str) -> str:
    """Suffix the filename stem of a SQLite URL.

    Examples:
        sqlite:///./dev.db          → sqlite:///./dev_shadow.db
        sqlite+aiosqlite:///dev.db  → sqlite+aiosqlite:///dev_shadow.db
        sqlite:///path/to/dev.db    → sqlite:///path/to/dev_shadow.db

    Idempotent: passing an already-shadowed URL returns it unchanged.
    """
    # Split on the first ":" only — keeps any "+driver" suffix attached.
    if ":" not in url:
        raise ValueError(f"Invalid SQLite URL (no scheme separator): {url}")
    scheme, _, after = url.partition(":")
    # The body should start with "//" (sqlite:// or sqlite:///path)
    if not after.startswith("//"):
        raise ValueError(f"Invalid SQLite URL (missing '//' after scheme): {url}")

    # Find the last "/" in the body — that's the start of the filename.
    body = after  # starts with "//"
    last_slash = body.rfind("/")
    if last_slash < 2 or last_slash == len(body) - 1:
        raise ValueError(
            f"Cannot derive shadow DB from SQLite URL with no filename: {url}"
        )

    prefix = body[: last_slash + 1]  # everything up to and including the last "/"
    filename = body[last_slash + 1 :]

    if "." in filename:
        stem, _, ext = filename.rpartition(".")
        if stem.endswith(_SHADOW_SUFFIX):
            return url  # idempotent
        new_filename = f"{stem}{_SHADOW_SUFFIX}.{ext}"
    else:
        if filename.endswith(_SHADOW_SUFFIX):
            return url
        new_filename = filename + _SHADOW_SUFFIX

    return f"{scheme}:{prefix}{new_filename}"


# ---------------------------------------------------------------------------
# Config mutation
# ---------------------------------------------------------------------------


@dataclass
class ShadowConfig:
    """Captures the state introduced by enabling shadow mode.

    The CLI / runtime layer can use this for logging and tests can use
    it to assert exactly what was changed.
    """

    enabled: bool = False
    original_database_url: str | None = None
    shadow_database_url: str | None = None
    initial_balance: float = 10_000.0
    extras: dict = field(default_factory=dict)


def configure_shadow(config) -> ShadowConfig:
    """Mutate ``config`` in place to enable shadow mode.

    The function does three things:

    1. Sets ``config.shadow_mode = True`` (always — the spec marker).
    2. Swaps ``config.database_url`` (if present) to the shadow URL,
       and writes the shadow URL into ``DATABASE_URL`` env var so any
       downstream code that reads from env (the repository factory)
       sees the swap.
    3. Sets ``config.use_simulated_exchange = True`` so the exchange
       factory routes through the sim adapter.

    The function also flips the ``QUANTAGENT_SHADOW`` env var so that
    ``is_shadow_mode()`` returns ``True`` everywhere — this is what
    makes the swap work for code that doesn't see ``config`` directly.

    Returns a :class:`ShadowConfig` snapshot describing the changes.

    The function is duck-typed: ``config`` can be any object on which
    you can ``setattr``. A bare dict-like object works too.
    """
    snapshot = ShadowConfig(enabled=True)

    enable_shadow_mode()
    _set(config, "shadow_mode", True)
    _set(config, "use_simulated_exchange", True)

    # Swap DB URL if the config or environment has one.
    base_url = _get(config, "database_url") or os.environ.get("DATABASE_URL", "")
    if base_url:
        try:
            shadow_url = get_shadow_db_url(base_url)
        except ValueError as e:
            logger.warning(f"Cannot derive shadow DB URL: {e}; leaving DB unchanged")
            shadow_url = base_url
        snapshot.original_database_url = base_url
        snapshot.shadow_database_url = shadow_url
        os.environ["DATABASE_URL"] = shadow_url
        _set(config, "database_url", shadow_url)
        _set(config, "original_database_url", base_url)

    # Carry through the simulated initial balance, if the config provides one.
    initial_balance = _get(config, "initial_balance")
    if initial_balance is not None:
        snapshot.initial_balance = float(initial_balance)

    logger.warning(
        "⚠️ SHADOW MODE — no real trades, writing to "
        f"{snapshot.shadow_database_url or 'shadow_db'}"
    )
    return snapshot


def _set(obj, key: str, value) -> None:
    """Set an attribute or dict key, depending on the object type."""
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _get(obj, key: str):
    """Get an attribute or dict key, returning None if absent."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# ---------------------------------------------------------------------------
# Database creation + migration
# ---------------------------------------------------------------------------


async def ensure_shadow_db(db_url: str) -> None:
    """Create the shadow database if it doesn't exist, then run migrations.

    For PostgreSQL: connects to the server's ``postgres`` maintenance
    database and issues ``CREATE DATABASE`` if the target doesn't exist.
    Then runs Alembic ``upgrade head`` against the shadow URL.

    For SQLite: nothing to create (the file is opened on first connect).
    Migrations still run.

    Failures are logged but do not raise — the shadow DB may already
    exist and be migrated, in which case we want to proceed silently.
    Operators reviewing logs after a failed setup will see the warning.
    """
    if not db_url:
        logger.warning("ensure_shadow_db: empty db_url, skipping")
        return

    parsed = urlparse(db_url)
    if parsed.scheme.startswith("sqlite"):
        # SQLite creates files lazily; just run migrations.
        _run_alembic_upgrade(db_url)
        return

    db_name = (parsed.path or "").lstrip("/")
    if not db_name:
        logger.warning(f"ensure_shadow_db: no database name in {db_url}")
        return

    try:
        await _create_postgres_db_if_missing(parsed, db_name)
    except Exception:
        # Already exists, no permission, or asyncpg unavailable — log
        # and continue. Migrations may still succeed.
        logger.exception(f"ensure_shadow_db: create-database step failed for {db_name}")

    try:
        _run_alembic_upgrade(db_url)
    except Exception:
        logger.exception(f"ensure_shadow_db: Alembic upgrade failed for {db_name}")


async def _create_postgres_db_if_missing(parsed, db_name: str) -> None:
    """Connect to the ``postgres`` maintenance DB and CREATE if missing."""
    try:
        import asyncpg  # type: ignore
    except ImportError:  # pragma: no cover - asyncpg is a project dep
        logger.warning("asyncpg not installed; cannot pre-create shadow DB")
        return

    maintenance_url = urlunparse(parsed._replace(path="/postgres"))
    # asyncpg accepts the postgresql:// scheme directly
    if "+" in parsed.scheme:
        bare_scheme = parsed.scheme.split("+", 1)[0]
        maintenance_url = urlunparse(
            parsed._replace(scheme=bare_scheme, path="/postgres")
        )
    conn = await asyncpg.connect(maintenance_url)
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        if exists:
            logger.info(f"ensure_shadow_db: {db_name} already exists")
            return
        # Identifier quoting — the spec restricts shadow names to a known
        # suffix so this is safe, but quote anyway for defensive coding.
        await conn.execute(f'CREATE DATABASE "{db_name}"')
        logger.info(f"ensure_shadow_db: created database {db_name}")
    finally:
        await conn.close()


def _run_alembic_upgrade(db_url: str) -> None:
    """Run ``alembic upgrade head`` against the given URL."""
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(cfg, "head")
    logger.info(f"ensure_shadow_db: migrations up to head on {db_url}")
