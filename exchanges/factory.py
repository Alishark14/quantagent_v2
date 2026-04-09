"""Exchange adapter factory: get_adapter(name) with singleton cache.

When shadow mode is active (see ``backtesting.shadow.is_shadow_mode``),
``get_adapter`` returns a ``SimulatedExchangeAdapter`` regardless of
which exchange name was requested. This is the central swap point that
makes Tier 4 shadow mode transparent to the rest of the codebase: every
existing call site (``BotRunner``, executor, sentinel, etc.) keeps
asking for ``get_adapter("hyperliquid")`` and gets back a fake exchange
without any code changes.

Shadow instances live in their own cache namespace so toggling shadow
mode on / off in tests doesn't poison the live singleton cache.
"""

from __future__ import annotations

from typing import Any

from exchanges.base import ExchangeAdapter


# Default starting balance for the shadow simulated exchange. Can be
# overridden by the caller via ``get_adapter(..., initial_balance=N)`` or
# by the QUANTAGENT_SHADOW_BALANCE env var.
_DEFAULT_SHADOW_BALANCE = 10_000.0


class ExchangeFactory:
    """Singleton-cached factory for exchange adapters."""

    _instances: dict[str, ExchangeAdapter] = {}
    _shadow_instances: dict[str, ExchangeAdapter] = {}
    _registry: dict[str, type[ExchangeAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[ExchangeAdapter]) -> None:
        """Register an adapter class by name."""
        cls._registry[name] = adapter_class

    @classmethod
    def get_adapter(cls, name: str, **kwargs: Any) -> ExchangeAdapter:
        """Get or create a singleton adapter instance.

        In shadow mode, returns a cached ``SimulatedExchangeAdapter``
        keyed by ``name`` so different requested exchanges still get
        their own sim instances (matters when a portfolio backtest runs
        the same engine against multiple venues simultaneously).

        ``kwargs`` are forwarded to the real-adapter constructor only;
        the shadow path ignores them except for ``initial_balance``.
        """
        # ── Shadow path ──
        # Imported lazily to avoid a hard dependency cycle
        # (shadow.py uses no exchanges code).
        from backtesting.shadow import is_shadow_mode

        if is_shadow_mode():
            return cls._get_shadow_adapter(name, **kwargs)

        # ── Live path ──
        if name in cls._instances:
            return cls._instances[name]
        if name not in cls._registry:
            raise ValueError(f"Unknown exchange: {name}")
        instance = cls._registry[name](**kwargs)
        cls._instances[name] = instance
        return instance

    @classmethod
    def _get_shadow_adapter(cls, name: str, **kwargs: Any) -> ExchangeAdapter:
        if name in cls._shadow_instances:
            return cls._shadow_instances[name]
        # Lazy import — backtesting depends on engine, but engine
        # imports the factory. Pulling sim_exchange in at module load
        # would create a hot import cycle.
        from backtesting.sim_exchange import SimulatedExchangeAdapter

        import os
        balance = float(
            kwargs.get("initial_balance")
            or os.environ.get("QUANTAGENT_SHADOW_BALANCE")
            or _DEFAULT_SHADOW_BALANCE
        )

        # Build a real adapter to delegate read-only data calls (ohlcv,
        # ticker, orderbook, funding, oi, meta) to a live venue. Sentinel
        # + signal agents need real market state in shadow mode; only the
        # ORDER methods should be virtualised. If the requested exchange
        # isn't in the registry, fall back to a sim with no data delegate
        # — the caller will hit the explicit RuntimeError on first
        # fetch_ohlcv, which is the correct failure mode.
        data_adapter: ExchangeAdapter | None = None
        if name in cls._registry:
            try:
                data_adapter = cls._registry[name](**kwargs)
            except Exception:
                # Credential / config error constructing the real adapter
                # in shadow mode is non-fatal: log and continue with a
                # data-less sim. The runner will surface the exact failure
                # when something tries to fetch data.
                import logging
                logging.getLogger(__name__).exception(
                    f"shadow factory: failed to construct data delegate "
                    f"for {name}; sim adapter will have no data source"
                )

        instance = SimulatedExchangeAdapter(
            initial_balance=balance,
            data_adapter=data_adapter,
            name=f"shadow-{name}",
        )
        cls._shadow_instances[name] = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances and registrations."""
        cls._instances.clear()
        cls._shadow_instances.clear()
        cls._registry.clear()

    @classmethod
    def reset_shadow_cache(cls) -> None:
        """Clear only the shadow-instance cache. Used in tests when
        toggling shadow mode within a single process."""
        cls._shadow_instances.clear()
