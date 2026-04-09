"""Exchange adapter factory: ``get_adapter(name, mode="live")`` with singleton cache.

The factory is the central swap point that lets the rest of the
codebase ask for ``get_adapter("hyperliquid")`` and never know whether
it got back a real venue, a virtualised shadow exchange, or a real
adapter pointed at the venue's testnet. As of the shadow-mode
redesign (Task 3 of the Shadow Redesign sprint), the swap is driven
by an explicit ``mode`` argument instead of a process-global
``is_shadow_mode()`` env-var check — this lets a single BotRunner
manage live and shadow bots side-by-side from one process.

Per-mode behaviour:

* ``mode="live"``  → returns the registered adapter from
  ``cls._instances`` (cached singleton). Identical to the pre-redesign
  behaviour. Construction errors propagate to the caller.

* ``mode="shadow"`` → returns a ``SimulatedExchangeAdapter`` from
  ``cls._shadow_instances``. The sim is constructed with a real,
  fully-credentialed live adapter as its ``data_adapter`` so that
  Sentinel + signal agents see real OHLCV / orderbook / funding /
  open-interest data; ORDER methods stay 100 % on the virtual portfolio.
  Before the live adapter is handed to the sim, its **signing key**
  (private key + ccxt secret) is scrubbed to ``None`` as defense in
  depth — even if some untested code path leaks an order through to the
  data delegate, the request fails with an auth error rather than
  placing a real trade. The wallet address is intentionally **preserved**
  so that read-only authenticated methods (``fetch_user_fees``, account
  metadata) still work.

* ``mode="paper"`` → returns the registered adapter from
  ``cls._paper_instances`` constructed with ``testnet=True`` forwarded
  into the ctor. Real signing capability, real order routing, but the
  venue's testnet endpoint — fills come from the real testnet
  orderbook with fake money. **NO key scrubbing** (we need to sign
  real testnet orders) and **NO sim wrapping** (we want real fills).
  Paper mode is the cheapest way to validate the production execution
  path against a real venue without risking mainnet funds.

Live, shadow, and paper caches are kept in separate dicts so
toggling between modes (or constructing all three kinds of adapter for
different bots within the same process) doesn't poison the other
namespaces.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from exchanges.base import ExchangeAdapter

logger = logging.getLogger(__name__)


# Default starting balance for the shadow simulated exchange. Can be
# overridden by the caller via ``get_adapter(..., initial_balance=N)`` or
# by the QUANTAGENT_SHADOW_BALANCE env var.
_DEFAULT_SHADOW_BALANCE = 10_000.0


# Attribute names that may hold a live signing key on the underlying
# ccxt.Exchange instance. ``privateKey`` is what Hyperliquid uses;
# ``secret`` is the standard ccxt name on most other venues. Both are
# nulled in shadow mode. ``apiKey`` is intentionally NOT scrubbed —
# without a paired ``secret`` no signed request can succeed, and some
# venues use ``apiKey`` alone for read-only metadata.
_CCXT_SIGNING_ATTRS: tuple[str, ...] = ("privateKey", "secret")

# Attribute names that may hold a signing key directly on the
# ExchangeAdapter wrapper class itself, in case a future adapter stores
# its credentials outside the inner ccxt object.
_ADAPTER_SIGNING_ATTRS: tuple[str, ...] = (
    "_private_key",
    "private_key",
    "_secret",
)


def _scrub_signing_keys(adapter: ExchangeAdapter) -> None:
    """Null out the signing key on a live adapter so it cannot place orders.

    This is the defense-in-depth half of shadow mode: ``ExchangeFactory``
    already wraps the live adapter in a ``SimulatedExchangeAdapter`` that
    keeps every order method on the virtual portfolio, but if some
    future refactor accidentally lets an order method through to the
    inner data delegate, the scrubbed key forces an auth failure
    instead of letting a real trade through.

    Both layers are scrubbed:

    * The inner ``ccxt.Exchange`` instance (``adapter._exchange``) — its
      ``privateKey`` and ``secret`` attributes are set to ``None``.
    * The ``ExchangeAdapter`` wrapper itself — any attribute matching
      ``_private_key`` / ``private_key`` / ``_secret`` is also nulled.

    The wallet address (``walletAddress`` on the ccxt object, plus any
    adapter-level wallet attribute) is **never** touched, because it
    remains a load-bearing input to authenticated read-only methods
    like ``fetch_user_fees`` and to position-metadata lookups.
    """
    inner = getattr(adapter, "_exchange", None)
    if inner is not None:
        for attr in _CCXT_SIGNING_ATTRS:
            if getattr(inner, attr, None) is not None:
                try:
                    setattr(inner, attr, None)
                except Exception:
                    # Some ccxt classes use __slots__ or properties; if
                    # we can't write the attribute we just skip it. The
                    # SimulatedExchangeAdapter still owns every order
                    # method, so this is best-effort hardening.
                    logger.debug(
                        f"shadow factory: could not scrub {attr} on inner ccxt "
                        f"object for {type(adapter).__name__}"
                    )

    for attr in _ADAPTER_SIGNING_ATTRS:
        if getattr(adapter, attr, None) is not None:
            try:
                setattr(adapter, attr, None)
            except Exception:
                logger.debug(
                    f"shadow factory: could not scrub {attr} on "
                    f"{type(adapter).__name__}"
                )


class ExchangeFactory:
    """Singleton-cached factory for exchange adapters.

    Live, shadow, and paper instances are cached in separate dicts
    keyed by ``name``. ``reset()`` clears everything;
    ``reset_shadow_cache()`` clears only the shadow namespace.
    Paper mode shares its cache with neither — a process can hold
    a live, a shadow, and a paper adapter for the same exchange
    name simultaneously without interference.
    """

    _instances: dict[str, ExchangeAdapter] = {}
    _shadow_instances: dict[str, ExchangeAdapter] = {}
    _paper_instances: dict[str, ExchangeAdapter] = {}
    _registry: dict[str, type[ExchangeAdapter]] = {}

    @classmethod
    def register(cls, name: str, adapter_class: type[ExchangeAdapter]) -> None:
        """Register an adapter class by name."""
        cls._registry[name] = adapter_class

    @classmethod
    def get_adapter(
        cls, name: str, mode: str = "live", **kwargs: Any
    ) -> ExchangeAdapter:
        """Get or create a singleton adapter instance.

        Args:
            name: Registered exchange name (e.g. ``"hyperliquid"``).
            mode: ``"live"`` (default), ``"shadow"``, or ``"paper"``.

                * ``"shadow"`` returns a ``SimulatedExchangeAdapter``
                  whose ``data_adapter`` is a real adapter with its
                  signing key scrubbed (virtual fills, real data).
                * ``"paper"`` returns the real registered adapter
                  constructed with ``testnet=True`` — real signing,
                  real order routing, testnet endpoint.

            **kwargs: Forwarded to the real-adapter constructor (live
                AND paper modes, plus the data delegate built in shadow
                mode). ``initial_balance`` is consumed by the shadow
                path and NOT forwarded to the live constructor. In
                paper mode, ``testnet=True`` is injected before the
                kwargs are forwarded — if the caller also passed an
                explicit ``testnet`` value it is overridden to ``True``
                so paper mode can never accidentally hit mainnet.

        Returns:
            The cached adapter instance for ``(name, mode)``.

        Raises:
            ValueError: in live or paper mode when ``name`` isn't
                registered. (Shadow mode falls back to a sim with
                ``data_adapter=None`` so unregistered exchanges still
                get a usable testbed.)
        """
        if mode == "shadow":
            return cls._build_shadow_adapter(name, **kwargs)

        if mode == "paper":
            return cls._build_paper_adapter(name, **kwargs)

        if mode != "live":
            raise ValueError(
                f"Unknown adapter mode: {mode!r} "
                "(expected 'live', 'shadow', or 'paper')"
            )

        # ── Live path ──
        if name in cls._instances:
            return cls._instances[name]
        if name not in cls._registry:
            raise ValueError(f"Unknown exchange: {name}")
        instance = cls._registry[name](**kwargs)
        cls._instances[name] = instance
        return instance

    @classmethod
    def _build_shadow_adapter(
        cls, name: str, **kwargs: Any
    ) -> ExchangeAdapter:
        """Construct (or fetch from cache) a shadow adapter for ``name``.

        Builds the real live adapter, scrubs its signing key, and wraps
        it as the ``data_adapter`` of a ``SimulatedExchangeAdapter`` so
        that read-only data calls go to the real venue while order
        methods stay on the virtual portfolio. If the live adapter
        constructor raises (typically a credentials problem in dev),
        the failure is logged at ERROR level and a sim with
        ``data_adapter=None`` is returned — the operator will then see
        an explicit ``RuntimeError`` on the first ``fetch_ohlcv`` call,
        which is the correct loud failure mode.
        """
        if name in cls._shadow_instances:
            return cls._shadow_instances[name]

        # Lazy import — backtesting depends on engine, but engine
        # imports the factory. Pulling sim_exchange in at module load
        # would create a hot import cycle.
        from backtesting.sim_exchange import SimulatedExchangeAdapter

        balance = float(
            kwargs.pop("initial_balance", None)
            or os.environ.get("QUANTAGENT_SHADOW_BALANCE")
            or _DEFAULT_SHADOW_BALANCE
        )

        data_adapter: ExchangeAdapter | None = None
        if name in cls._registry:
            try:
                data_adapter = cls._registry[name](**kwargs)
                _scrub_signing_keys(data_adapter)
            except Exception:
                # Credential / config error constructing the real adapter
                # in shadow mode is non-fatal: log and continue with a
                # data-less sim. The runner will surface the exact failure
                # when something tries to fetch data.
                logger.exception(
                    f"shadow factory: failed to construct data delegate "
                    f"for {name}; sim adapter will have no data source"
                )
                data_adapter = None

        instance = SimulatedExchangeAdapter(
            initial_balance=balance,
            data_adapter=data_adapter,
            name=f"shadow-{name}",
        )
        cls._shadow_instances[name] = instance
        return instance

    @classmethod
    def _build_paper_adapter(
        cls, name: str, **kwargs: Any
    ) -> ExchangeAdapter:
        """Construct (or fetch from cache) a paper-mode adapter.

        Paper mode is the simplest of the three: it returns the real
        registered adapter constructed with ``testnet=True``. The
        adapter signs real orders against the venue's testnet
        orderbook, so we get end-to-end execution validation against
        a real venue without risking mainnet funds.

        ``testnet=True`` is FORCED — if the caller passed an explicit
        ``testnet`` kwarg with any other value, it's overridden. The
        whole point of paper mode is that you can never accidentally
        hit mainnet through it. (For mainnet routing, ask for
        ``mode="live"``.)

        Unlike shadow mode there is NO key scrubbing (we need to sign
        real testnet orders) and NO simulated wrapper (we want real
        fills from the testnet orderbook).

        Raises:
            ValueError: when ``name`` isn't in the registry. Unlike
                shadow mode, paper has no usable fallback for an
                unregistered exchange — you need a real adapter
                implementation to talk to a real testnet endpoint.
        """
        if name in cls._paper_instances:
            return cls._paper_instances[name]
        if name not in cls._registry:
            raise ValueError(
                f"Unknown exchange: {name} (paper mode requires a "
                "registered adapter — testnet has no usable fallback)"
            )

        # Force testnet=True regardless of what the caller passed.
        # The strict override matches the safety contract documented
        # in get_adapter — paper mode can never hit mainnet.
        kwargs.pop("testnet", None)
        instance = cls._registry[name](testnet=True, **kwargs)
        cls._paper_instances[name] = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances and registrations."""
        cls._instances.clear()
        cls._shadow_instances.clear()
        cls._paper_instances.clear()
        cls._registry.clear()

    @classmethod
    def reset_shadow_cache(cls) -> None:
        """Clear only the shadow-instance cache. Used in tests when
        rebuilding shadow adapters within a single process."""
        cls._shadow_instances.clear()

    @classmethod
    def reset_paper_cache(cls) -> None:
        """Clear only the paper-instance cache. Used in tests when
        rebuilding paper adapters within a single process."""
        cls._paper_instances.clear()
