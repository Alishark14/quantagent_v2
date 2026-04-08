"""Unit tests for shadow mode (Tier 4 backtesting)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from backtesting.shadow import (
    ShadowConfig,
    configure_shadow,
    disable_shadow_mode,
    enable_shadow_mode,
    get_shadow_db_url,
    is_shadow_mode,
)
from backtesting.sim_exchange import SimulatedExchangeAdapter
from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from exchanges.factory import ExchangeFactory


# ---------------------------------------------------------------------------
# Test fixtures + isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_environment():
    """Snapshot env vars and factory state; restore after every test.

    Without this, a test that flips QUANTAGENT_SHADOW or mutates the
    ExchangeFactory cache would poison every subsequent test in the
    suite (factory reset clears _registry, so the real Hyperliquid
    adapter would silently disappear).
    """
    saved_env = {
        k: os.environ.get(k)
        for k in ("QUANTAGENT_SHADOW", "DATABASE_URL", "QUANTAGENT_SHADOW_BALANCE")
    }
    saved_instances = dict(ExchangeFactory._instances)
    saved_shadow = dict(ExchangeFactory._shadow_instances)
    saved_registry = dict(ExchangeFactory._registry)

    yield

    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    ExchangeFactory._instances.clear()
    ExchangeFactory._instances.update(saved_instances)
    ExchangeFactory._shadow_instances.clear()
    ExchangeFactory._shadow_instances.update(saved_shadow)
    ExchangeFactory._registry.clear()
    ExchangeFactory._registry.update(saved_registry)


class _LiveAdapter(ExchangeAdapter):
    """Minimal real-adapter stand-in for the factory tests."""

    def name(self) -> str:
        return "live-stub"

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_sl_tp=False, supports_short=False, market_hours=None,
            asset_types=["spot"], margin_type="cash", has_funding_rate=False,
            has_oi_data=False, max_leverage=1.0, order_types=["market"],
            supports_partial_close=False,
        )

    async def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
        return []

    async def get_ticker(self, symbol):
        return {}

    async def get_balance(self):
        return 0.0

    async def get_positions(self, symbol=None):
        return []

    async def place_market_order(self, symbol, side, size):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")

    async def place_limit_order(self, symbol, side, size, price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")

    async def place_sl_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")

    async def place_tp_order(self, symbol, side, size, trigger_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")

    async def cancel_order(self, symbol, order_id):
        return False

    async def cancel_all_orders(self, symbol):
        return 0

    async def close_position(self, symbol):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")

    async def modify_sl(self, symbol, new_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")

    async def modify_tp(self, symbol, new_price):
        return OrderResult(success=False, order_id=None, fill_price=None, fill_size=None, error="live")


# ---------------------------------------------------------------------------
# is_shadow_mode / enable / disable
# ---------------------------------------------------------------------------


def test_shadow_mode_off_by_default():
    disable_shadow_mode()
    assert is_shadow_mode() is False


def test_shadow_mode_on_when_env_var_set():
    os.environ["QUANTAGENT_SHADOW"] = "1"
    assert is_shadow_mode() is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "Y", "t"])
def test_truthy_env_values_enable_shadow(value):
    os.environ["QUANTAGENT_SHADOW"] = value
    assert is_shadow_mode() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "anything else"])
def test_falsy_env_values_keep_shadow_off(value):
    os.environ["QUANTAGENT_SHADOW"] = value
    assert is_shadow_mode() is False


def test_enable_and_disable_shadow():
    disable_shadow_mode()
    assert is_shadow_mode() is False
    enable_shadow_mode()
    assert is_shadow_mode() is True
    disable_shadow_mode()
    assert is_shadow_mode() is False


# ---------------------------------------------------------------------------
# get_shadow_db_url
# ---------------------------------------------------------------------------


def test_postgres_url_appends_shadow_suffix():
    url = "postgresql://user:pass@host:5432/quantagent"
    assert (
        get_shadow_db_url(url)
        == "postgresql://user:pass@host:5432/quantagent_shadow"
    )


def test_postgres_url_preserves_query_string():
    url = "postgresql://user:pass@host/quantagent?sslmode=require"
    assert (
        get_shadow_db_url(url)
        == "postgresql://user:pass@host/quantagent_shadow?sslmode=require"
    )


def test_postgres_url_with_asyncpg_driver_suffix():
    url = "postgresql+asyncpg://user:pass@host:5432/quantagent"
    assert (
        get_shadow_db_url(url)
        == "postgresql+asyncpg://user:pass@host:5432/quantagent_shadow"
    )


def test_sqlite_file_url_suffixes_filename():
    url = "sqlite:///./dev.db"
    out = get_shadow_db_url(url)
    assert out.endswith("dev_shadow.db")


def test_sqlite_aiosqlite_url():
    url = "sqlite+aiosqlite:///dev.db"
    out = get_shadow_db_url(url)
    assert out.endswith("dev_shadow.db")
    assert out.startswith("sqlite+aiosqlite:")


def test_get_shadow_db_url_idempotent_for_postgres():
    url = "postgresql://u:p@h:5432/quantagent_shadow"
    assert get_shadow_db_url(url) == url


def test_get_shadow_db_url_idempotent_for_sqlite():
    url = "sqlite:///dev_shadow.db"
    assert get_shadow_db_url(url) == url


def test_get_shadow_db_url_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        get_shadow_db_url("")


def test_get_shadow_db_url_no_db_name_raises():
    with pytest.raises(ValueError, match="no database name"):
        get_shadow_db_url("postgresql://user:pass@host:5432/")


# ---------------------------------------------------------------------------
# configure_shadow
# ---------------------------------------------------------------------------


@dataclass
class _Cfg:
    database_url: str = "postgresql://u:p@h:5432/quantagent"
    shadow_mode: bool = False
    use_simulated_exchange: bool = False
    initial_balance: float = 25_000.0


def test_configure_shadow_mutates_object_and_env():
    cfg = _Cfg()
    snapshot = configure_shadow(cfg)

    # Object mutated in place
    assert cfg.shadow_mode is True
    assert cfg.use_simulated_exchange is True
    assert cfg.database_url == "postgresql://u:p@h:5432/quantagent_shadow"
    # Original URL preserved on the config for rollback / logging
    assert getattr(cfg, "original_database_url") == "postgresql://u:p@h:5432/quantagent"

    # Env var side-effects
    assert is_shadow_mode() is True
    assert os.environ["DATABASE_URL"] == "postgresql://u:p@h:5432/quantagent_shadow"

    # Snapshot
    assert isinstance(snapshot, ShadowConfig)
    assert snapshot.enabled is True
    assert snapshot.original_database_url == "postgresql://u:p@h:5432/quantagent"
    assert snapshot.shadow_database_url == "postgresql://u:p@h:5432/quantagent_shadow"
    assert snapshot.initial_balance == 25_000.0


def test_configure_shadow_works_with_dict_config():
    cfg = {"database_url": "postgresql://u:p@h/quantagent"}
    snapshot = configure_shadow(cfg)
    assert cfg["shadow_mode"] is True
    assert cfg["use_simulated_exchange"] is True
    assert cfg["database_url"] == "postgresql://u:p@h/quantagent_shadow"
    assert snapshot.enabled is True


def test_configure_shadow_falls_back_to_env_db_url_when_config_has_none():
    os.environ["DATABASE_URL"] = "postgresql://u:p@h/from_env"

    class _Bare:
        shadow_mode = False
        use_simulated_exchange = False

    snapshot = configure_shadow(_Bare)
    assert snapshot.original_database_url == "postgresql://u:p@h/from_env"
    assert os.environ["DATABASE_URL"] == "postgresql://u:p@h/from_env_shadow"


def test_configure_shadow_handles_unparseable_url_gracefully(caplog):
    """If the URL can't be transformed, configure_shadow leaves it
    untouched but still flips the env var. The operator sees a warning."""
    import logging

    cfg = _Cfg(database_url="postgresql://u:p@h:5432/")
    with caplog.at_level(logging.WARNING):
        snapshot = configure_shadow(cfg)
    assert is_shadow_mode() is True
    assert cfg.shadow_mode is True
    assert any("Cannot derive shadow DB URL" in r.message for r in caplog.records)
    assert snapshot.enabled is True


# ---------------------------------------------------------------------------
# ExchangeFactory shadow swap
# ---------------------------------------------------------------------------


def test_factory_returns_simulated_adapter_in_shadow_mode():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)
    enable_shadow_mode()

    adapter = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(adapter, SimulatedExchangeAdapter)
    # Name carries the requested exchange so consumers can still tell
    # which venue they were *supposed* to be using
    assert adapter.name() == "shadow-hyperliquid"


def test_factory_returns_real_adapter_when_not_shadow():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)
    disable_shadow_mode()

    adapter = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(adapter, _LiveAdapter)
    assert not isinstance(adapter, SimulatedExchangeAdapter)


def test_factory_shadow_singleton_per_exchange_name():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)
    enable_shadow_mode()

    a1 = ExchangeFactory.get_adapter("hyperliquid")
    a2 = ExchangeFactory.get_adapter("hyperliquid")
    assert a1 is a2  # cached


def test_factory_shadow_separate_instance_per_exchange():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)
    ExchangeFactory.register("binance", _LiveAdapter)
    enable_shadow_mode()

    hl = ExchangeFactory.get_adapter("hyperliquid")
    bn = ExchangeFactory.get_adapter("binance")
    assert hl is not bn
    assert hl.name() == "shadow-hyperliquid"
    assert bn.name() == "shadow-binance"


def test_factory_shadow_unknown_exchange_still_works():
    """Shadow mode bypasses the registry — useful for testing
    against exchanges that haven't been added yet."""
    ExchangeFactory.reset()
    enable_shadow_mode()

    adapter = ExchangeFactory.get_adapter("ibkr")
    assert isinstance(adapter, SimulatedExchangeAdapter)
    assert adapter.name() == "shadow-ibkr"


def test_factory_shadow_balance_from_env():
    ExchangeFactory.reset()
    enable_shadow_mode()
    os.environ["QUANTAGENT_SHADOW_BALANCE"] = "50000"

    adapter = ExchangeFactory.get_adapter("hyperliquid")
    assert adapter.balance == 50_000.0


def test_factory_reset_shadow_cache_only():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)

    # Live cache populated
    disable_shadow_mode()
    live = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(live, _LiveAdapter)

    # Shadow cache populated
    enable_shadow_mode()
    shadow = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(shadow, SimulatedExchangeAdapter)

    # Reset only the shadow cache; live registry/instances survive
    ExchangeFactory.reset_shadow_cache()
    assert ExchangeFactory._shadow_instances == {}
    assert "hyperliquid" in ExchangeFactory._registry
    assert "hyperliquid" in ExchangeFactory._instances


# ---------------------------------------------------------------------------
# Integration: configure_shadow → factory swap
# ---------------------------------------------------------------------------


def test_end_to_end_configure_then_factory_returns_sim():
    """The full happy path: a fresh config gets configured for shadow,
    then any code that asks the factory for an adapter gets a sim."""
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)
    disable_shadow_mode()

    cfg = _Cfg()
    configure_shadow(cfg)

    adapter = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(adapter, SimulatedExchangeAdapter)
