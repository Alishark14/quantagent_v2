"""Unit tests for the ExchangeFactory shadow-mode swap.

The legacy shadow infrastructure (`configure_shadow`, `ensure_shadow_db`,
`is_shadow_mode`, `get_shadow_db_url`, `ShadowConfig`) was deleted in
Task 4 of the Shadow Redesign sprint. The remaining tests cover the
explicit `mode="shadow"` path on `ExchangeFactory.get_adapter` and the
defense-in-depth signing-key scrubbing on the live adapter that gets
wrapped as the shadow sim's data delegate.
"""

from __future__ import annotations

import os

import pytest

from backtesting.sim_exchange import SimulatedExchangeAdapter
from engine.types import AdapterCapabilities, OrderResult, Position
from exchanges.base import ExchangeAdapter
from exchanges.factory import ExchangeFactory


# ---------------------------------------------------------------------------
# Test fixtures + isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def restore_environment():
    """Snapshot factory state and shadow-balance env var; restore after each test.

    The factory `reset()` calls inside individual tests would otherwise
    wipe the real Hyperliquid adapter registration that conftest set up
    for the rest of the suite. Snapshotting + restoring keeps each test
    isolated.
    """
    saved_env = {
        k: os.environ.get(k)
        for k in ("QUANTAGENT_SHADOW_BALANCE",)
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
# ExchangeFactory shadow swap
# ---------------------------------------------------------------------------


def test_factory_returns_simulated_adapter_in_shadow_mode():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)

    adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    assert isinstance(adapter, SimulatedExchangeAdapter)
    # Name carries the requested exchange so consumers can still tell
    # which venue they were *supposed* to be using
    assert adapter.name() == "shadow-hyperliquid"


def test_factory_returns_real_adapter_when_not_shadow():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)

    adapter = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(adapter, _LiveAdapter)
    assert not isinstance(adapter, SimulatedExchangeAdapter)


def test_factory_shadow_singleton_per_exchange_name():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)

    a1 = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    a2 = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    assert a1 is a2  # cached


def test_factory_shadow_separate_instance_per_exchange():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)
    ExchangeFactory.register("binance", _LiveAdapter)

    hl = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    bn = ExchangeFactory.get_adapter("binance", mode="shadow")
    assert hl is not bn
    assert hl.name() == "shadow-hyperliquid"
    assert bn.name() == "shadow-binance"


def test_factory_shadow_unknown_exchange_still_works():
    """Shadow mode bypasses the registry — useful for testing
    against exchanges that haven't been added yet."""
    ExchangeFactory.reset()

    adapter = ExchangeFactory.get_adapter("ibkr", mode="shadow")
    assert isinstance(adapter, SimulatedExchangeAdapter)
    assert adapter.name() == "shadow-ibkr"
    # No registered live adapter → data_adapter is None; the sim raises
    # the explicit RuntimeError on first fetch_ohlcv (correct failure mode).
    assert adapter._data_adapter is None


def test_factory_shadow_wires_real_adapter_as_data_delegate():
    """Shadow sim must delegate read-only methods to a real live adapter
    so Sentinel + signals see real market data while orders stay virtual."""
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)

    adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    assert isinstance(adapter, SimulatedExchangeAdapter)
    assert isinstance(adapter._data_adapter, _LiveAdapter)


def test_factory_shadow_data_delegate_construct_failure_is_nonfatal(caplog):
    """If the live adapter ctor blows up (bad credentials, etc.) the
    factory must still return a sim — just without a data delegate."""
    class _BoomAdapter(_LiveAdapter):
        def __init__(self):
            raise RuntimeError("missing credentials")

    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _BoomAdapter)

    with caplog.at_level("ERROR"):
        adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    assert isinstance(adapter, SimulatedExchangeAdapter)
    assert adapter._data_adapter is None
    assert any("data delegate" in r.message for r in caplog.records)


def test_factory_shadow_balance_from_env():
    ExchangeFactory.reset()
    os.environ["QUANTAGENT_SHADOW_BALANCE"] = "50000"

    adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    assert adapter.balance == 50_000.0


def test_factory_reset_shadow_cache_only():
    ExchangeFactory.reset()
    ExchangeFactory.register("hyperliquid", _LiveAdapter)

    # Live cache populated
    live = ExchangeFactory.get_adapter("hyperliquid")
    assert isinstance(live, _LiveAdapter)

    # Shadow cache populated
    shadow = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
    assert isinstance(shadow, SimulatedExchangeAdapter)

    # Reset only the shadow cache; live registry/instances survive
    ExchangeFactory.reset_shadow_cache()
    assert ExchangeFactory._shadow_instances == {}
    assert "hyperliquid" in ExchangeFactory._registry
    assert "hyperliquid" in ExchangeFactory._instances


# ---------------------------------------------------------------------------
# Task 3 — explicit mode parameter + signing-key scrubbing
# ---------------------------------------------------------------------------


class _CredentialedAdapter(_LiveAdapter):
    """A live-adapter stand-in that owns both layers of credentials.

    Mirrors the structure of ``HyperliquidAdapter``: an inner ccxt-like
    object on ``self._exchange`` carrying ``privateKey`` + ``walletAddress``,
    plus a wrapper-level ``_private_key`` so the scrubber can be tested
    against both surfaces.
    """

    def __init__(
        self,
        wallet_address: str = "0xWALLET",
        private_key: str = "deadbeef" * 8,
        secret: str = "topsecret",
    ) -> None:
        self._private_key = private_key

        class _InnerCcxt:
            pass

        inner = _InnerCcxt()
        inner.privateKey = private_key
        inner.secret = secret
        inner.walletAddress = wallet_address
        inner.apiKey = "public-id-not-secret"
        self._exchange = inner

    async def place_market_order(self, symbol, side, size):
        """Real implementation would sign with self._exchange.privateKey
        and POST to the venue. With a scrubbed key it raises so that
        any code path that bypasses the sim's order layer surfaces a
        loud failure instead of placing a real trade."""
        if (
            getattr(self._exchange, "privateKey", None) is None
            and getattr(self, "_private_key", None) is None
        ):
            raise RuntimeError(
                "auth: no signing key configured for credentialed adapter"
            )
        return OrderResult(
            success=True, order_id="LIVE-1", fill_price=100.0,
            fill_size=size, error=None,
        )


class TestShadowModeParameter:
    """Task 3 acceptance criteria — six explicit cases."""

    def test_1_shadow_mode_returns_simulated_exchange_adapter(self):
        """get_adapter(name, mode='shadow') returns a SimulatedExchangeAdapter."""
        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _CredentialedAdapter)

        adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
        assert isinstance(adapter, SimulatedExchangeAdapter)
        assert adapter.name() == "shadow-hyperliquid"

    def test_2_shadow_data_adapter_has_nulled_private_key(self):
        """The data delegate's signing key is scrubbed to None.

        Both the inner ccxt object's `privateKey` / `secret` and the
        adapter wrapper's `_private_key` must be None after the factory
        wraps it.
        """
        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _CredentialedAdapter)

        adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
        delegate = adapter._data_adapter
        assert isinstance(delegate, _CredentialedAdapter)

        # Inner ccxt-style object scrubbed
        assert delegate._exchange.privateKey is None
        assert delegate._exchange.secret is None

        # Wrapper-level signing attribute scrubbed
        assert delegate._private_key is None

    def test_3_shadow_data_adapter_wallet_address_is_preserved(self):
        """Wallet address must survive scrubbing — it's needed for
        read-only authenticated metadata calls (`fetch_user_fees`)."""
        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _CredentialedAdapter)

        adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
        delegate = adapter._data_adapter
        assert delegate._exchange.walletAddress == "0xWALLET"
        # apiKey is intentionally NOT scrubbed (read-only identifier on
        # most venues; signing requires the paired secret which IS gone).
        assert delegate._exchange.apiKey == "public-id-not-secret"

    def test_4_shadow_fetch_ohlcv_delegates_to_real_adapter(self):
        """Read-only data calls go to the data_adapter, not the sim's
        own (offline) data loader. We capture the delegate's call list
        to prove it was reached."""

        class _SpyAdapter(_CredentialedAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple] = []

            async def fetch_ohlcv(self, symbol, timeframe, limit=100, since=None):
                self.calls.append(("fetch_ohlcv", symbol, timeframe, limit))
                return [
                    [1700000000000, 100.0, 101.0, 99.0, 100.5, 1234.5],
                ]

        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _SpyAdapter)

        adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
        spy = adapter._data_adapter

        import asyncio
        candles = asyncio.run(adapter.fetch_ohlcv("BTC-USDC", "1h", limit=1))
        assert spy.calls == [("fetch_ohlcv", "BTC-USDC", "1h", 1)]
        assert candles == [
            [1700000000000, 100.0, 101.0, 99.0, 100.5, 1234.5],
        ]

    def test_5_shadow_place_order_uses_virtual_portfolio_not_data_adapter(self):
        """ORDER methods stay 100% on the virtual portfolio. Even if
        the delegate had a working signing key, the sim's order layer
        must intercept first. We track delegate calls to prove it was
        never reached on the order path."""

        class _OrderSpyAdapter(_CredentialedAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.order_calls: list[str] = []

            async def place_market_order(self, symbol, side, size):
                self.order_calls.append("place_market_order")
                return await super().place_market_order(symbol, side, size)

        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _OrderSpyAdapter)

        adapter = ExchangeFactory.get_adapter("hyperliquid", mode="shadow")
        spy = adapter._data_adapter

        # Push a current candle so the virtual portfolio can fill the order
        adapter.set_current_candle(
            "BTC-USDC",
            {
                "timestamp": 1700000000000,
                "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.5, "volume": 10.0,
            },
        )

        import asyncio
        result = asyncio.run(adapter.place_market_order("BTC-USDC", "buy", 0.1))

        # Sim filled it on the virtual portfolio
        assert result.success is True
        # Delegate was never called for the order path
        assert spy.order_calls == []
        # And if the delegate WERE called, the scrubbed key would have
        # caused a RuntimeError instead of a real fill — verify by
        # calling its order method directly:
        with pytest.raises(RuntimeError, match="no signing key"):
            asyncio.run(spy.place_market_order("BTC-USDC", "buy", 0.1))

    def test_6_live_mode_returns_real_adapter_unchanged(self):
        """Default mode is live — registered adapter returned with
        credentials INTACT. Live path must be byte-for-byte identical
        to the pre-Task-3 behaviour."""
        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _CredentialedAdapter)

        adapter = ExchangeFactory.get_adapter("hyperliquid")
        assert isinstance(adapter, _CredentialedAdapter)
        # Credentials NOT scrubbed in live mode
        assert adapter._exchange.privateKey == "deadbeef" * 8
        assert adapter._exchange.secret == "topsecret"
        assert adapter._private_key == "deadbeef" * 8
        assert adapter._exchange.walletAddress == "0xWALLET"

        # Same call without an explicit mode also returns live
        a2 = ExchangeFactory.get_adapter("hyperliquid", mode="live")
        assert a2 is adapter  # cached

    def test_invalid_mode_raises(self):
        """Belt-and-braces: an unknown mode value should fail loudly,
        not silently fall through to live."""
        ExchangeFactory.reset()
        ExchangeFactory.register("hyperliquid", _CredentialedAdapter)

        with pytest.raises(ValueError, match="Unknown adapter mode"):
            ExchangeFactory.get_adapter("hyperliquid", mode="paper")
