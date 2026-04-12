"""QuantAgent CLI entry point.

Commands:
    python -m quantagent run               — Start BotRunner + FastAPI (live mode)
    python -m quantagent run --shadow      — Load shadow bots, simulated execution
    python -m quantagent run --paper       — Load paper bots, real testnet orders, port 8001
    python -m quantagent migrate           — Run Alembic migrations (upgrade to head)
    python -m quantagent seed              — Seed dev database with test data
    python -m quantagent                   — Show help

``--shadow`` and ``--paper`` are MUTUALLY EXCLUSIVE — passing both
exits with an error. To run shadow and paper simultaneously, start
two separate processes in two tmux sessions on different ports
(shadow defaults to 8000, paper defaults to 8001).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Callable


def _setup_logging() -> None:
    """Configure structured logging.

    ``force=True`` is critical: without it, any module imported BEFORE
    ``_setup_logging`` runs (uvicorn, anthropic SDK, ccxt — all of
    which install handlers on the root logger at import time) would
    cause `basicConfig` to silently no-op, and the engine's startup
    log lines would never appear in stdout. Reinstalling the root
    handlers via ``force=True`` is the documented escape hatch.
    """
    level = logging.DEBUG if os.environ.get("VERBOSE") else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _make_bot_factory(
    repos,
    llm_provider,
    adapter_factory: Callable[..., object],
    event_bus,
    feature_flags=None,
    shadow_mode: bool = False,
    paper_mode: bool = False,
) -> Callable[[str, str], object]:
    """Build a `(symbol, bot_id) -> TraderBot` factory closure.

    The LLM provider, repos, event bus, and feature flags are all
    captured by reference so they're constructed ONCE at startup and
    shared across every spawned bot. The exchange adapter and bot-local
    state (TradingConfig, OHLCVFetcher, Executor, AnalysisPipeline)
    are constructed PER bot inside the closure because they're
    bot-symbol-scoped.

    Mirrors the wiring used by `scripts/run_trade.py` so that single-
    cycle dev runs and the production BotRunner exercise the same
    dependency graph. The only difference is that here the closure is
    invoked dynamically by `BotManager.spawn_bot` instead of being
    hand-built once at script start.

    Per CLAUDE.md Rule 3, the engine has zero knowledge of which
    exchange this is — it just calls the injected `adapter_factory`.
    Shadow mode is transparent: in shadow mode the factory returns a
    `SimulatedExchangeAdapter` with the real venue's adapter wired up
    as `data_adapter=`, so the same code path serves live data with
    fake fills.
    """
    # Lazy imports — keep CLI start-up cheap and avoid pulling the full
    # engine into argv parsing.
    from engine.config import FeatureFlags, TradingConfig
    from engine.conviction.agent import ConvictionAgent
    from engine.data.flow import FlowAgent, FlowSignalAgent
    from engine.data.flow.commodity import CommodityFlowProvider
    from engine.data.flow.crypto import CryptoFlowProvider
    from engine.data.flow.equity import EquityFlowProvider
    from engine.data.flow.options import OptionsEnrichment
    from engine.data.ohlcv import OHLCVFetcher
    from engine.execution.agent import DecisionAgent
    from engine.execution.executor import Executor
    from engine.execution.portfolio_risk_manager import (
        PortfolioRiskConfig,
        PortfolioRiskManager,
    )
    from engine.memory.cross_bot import CrossBotSignals
    from engine.memory.cycle_memory import CycleMemory
    from engine.memory.reflection_rules import ReflectionRules
    from engine.memory.regime_history import RegimeHistory
    from engine.pipeline import AnalysisPipeline
    from engine.signals.indicator_agent import IndicatorAgent
    from engine.signals.pattern_agent import PatternAgent
    from engine.signals.registry import SignalRegistry
    from engine.signals.trend_agent import TrendAgent
    from engine.trader_bot import TraderBot
    from sentinel.position_manager import PositionManager

    # Feature flags are shared across bots — load once.
    flags = feature_flags or FeatureFlags()
    # Process-wide adapter mode. main.py loads bots filtered by mode,
    # so every bot the factory builds belongs to this mode. The runner
    # also passes per-bot mode through to its sentinel adapter — these
    # two paths agree in the production startup flow because of the
    # mode-based bot filter. Shadow takes precedence over paper if
    # both flags are set (defensive — main.py's CLI dispatcher should
    # already reject the combination, but the factory honours one
    # canonical resolution rule regardless).
    if shadow_mode:
        factory_mode = "shadow"
    elif paper_mode:
        factory_mode = "paper"
    else:
        factory_mode = "live"
    # Mark every cycle this factory's pipelines persist as belonging
    # to the shadow data partition iff this is a shadow OR paper run.
    # Live cycles stay is_shadow=False so they show up in the live_*
    # views the QuantDataScientist mines.
    pipeline_is_shadow = shadow_mode or paper_mode

    # Single CryptoFlowProvider shared across every spawned bot. Its
    # rolling OI history buffer is keyed per-symbol internally, so one
    # instance correctly accumulates lookback windows for every symbol
    # the factory ever sees. Building a fresh provider inside
    # ``factory(...)`` would wipe the buffer on every hourly spawn —
    # the cold-start path would never end and FlowSignalAgent's
    # BUILDING / DROPPING rules would stay dormant.
    #
    # The provider is wired to ``repos.oi_snapshots`` so each fresh
    # snapshot is also persisted, and ``warmup_from_repo()`` (awaited
    # by the caller after construction) bulk-loads the recent window
    # so a process restart is effectively free after the first day of
    # uptime. ``getattr`` keeps the test stubs happy — MagicMock
    # repos and the SQLite fallback both supply the attribute.
    _oi_repo = getattr(repos, "oi_snapshots", None)
    _shared_flow_provider = CryptoFlowProvider(oi_repo=_oi_repo)
    # Default to a 1h-bot lookback (2h window). The shared provider
    # serves every spawned bot regardless of timeframe; if mixed
    # timeframes ship in the future the provider will need per-symbol
    # lookbacks. For now all production bots are 1h.
    _shared_flow_provider.set_lookback_for_timeframe("1h")

    # OptionsEnrichment is a second shared FlowProvider for BTC / ETH
    # only (Deribit does not list public options for altcoins — the
    # provider short-circuits for every other symbol). Shared across
    # spawns so its 15-minute Deribit cache persists instead of paying
    # fresh network round-trips every hourly TraderBot.
    _shared_options_provider = OptionsEnrichment()

    # CommodityFlowProvider pulls weekly CFTC COT positioning for our
    # HIP-3 commodity universe (gold / silver / WTI / brent) and
    # returns an empty dict for every other symbol. Wired to the
    # ``cot_cache`` repo so the 52-week rolling history survives
    # restarts; warmup_from_repo is awaited by _run_server before any
    # bot spawns.
    _cot_repo = getattr(repos, "cot_cache", None)
    _shared_commodity_provider = CommodityFlowProvider(cot_repo=_cot_repo)

    # EquityFlowProvider pulls daily FINRA RegSHO short-volume files
    # for our HIP-3 equity universe (TSLA / NVDA / GOOGL) and returns
    # an empty dict for every other symbol. Wired to the regsho_cache
    # table so the 20-day rolling Z-score window survives restarts.
    _regsho_repo = getattr(repos, "regsho_cache", None)
    _shared_equity_provider = EquityFlowProvider(regsho_repo=_regsho_repo)

    def factory(
        symbol: str,
        bot_id: str,
        adapter_override: object | None = None,
    ):
        # Look up bot config (timeframe, exchange, user_id) from the DB
        # if it's been stored. Falls back to defaults so the factory
        # remains usable for ad-hoc / unregistered spawns.
        bot_dict: dict | None = None
        try:
            # synchronous best-effort lookup — repos.bots.get_bot is
            # async, so we just inline the safe defaults if we can't
            # await here. The runner already has the dict in
            # _bot_configs, but the BotManager calls factory(...)
            # without that context. Use sane defaults.
            pass
        except Exception:
            bot_dict = None

        timeframe = (bot_dict or {}).get("timeframe", "1h")
        exchange = (bot_dict or {}).get("exchange", "hyperliquid")
        user_id = (bot_dict or {}).get("user_id", "system")

        config = TradingConfig(symbol=symbol, timeframe=timeframe)
        # Use the shared adapter from Sentinel if provided, so the
        # pipeline and Sentinel operate on the same instance. This is
        # critical for shadow mode: Sentinel feeds candles (triggering
        # SL/TP), while the pipeline opens positions — both must see
        # the same SimulatedExchangeAdapter state.
        adapter = adapter_override if adapter_override is not None else adapter_factory(exchange, mode=factory_mode)

        # Data layer
        fetcher = OHLCVFetcher(adapter, config)
        flow_agent = FlowAgent(
            [
                _shared_flow_provider,
                _shared_options_provider,
                _shared_commodity_provider,
                _shared_equity_provider,
            ]
        )

        # Signal layer — register every enabled agent. The four
        # currently-shipped agents are gated by their feature flags so
        # operators can disable any of them via env var or features.yaml
        # without code changes (CLAUDE.md Rule 9).
        registry = SignalRegistry()
        if flags.is_enabled("indicator_agent"):
            registry.register(IndicatorAgent(llm_provider, flags))
        if flags.is_enabled("pattern_agent"):
            registry.register(PatternAgent(llm_provider, flags))
        if flags.is_enabled("trend_agent"):
            registry.register(TrendAgent(llm_provider, flags))
        if flags.is_enabled("flow_signal_agent"):
            registry.register(FlowSignalAgent(flags))

        # Conviction + Decision (LLM-backed)
        conviction_agent = ConvictionAgent(llm_provider)
        decision_agent = DecisionAgent(llm_provider, config)

        # Sprint Portfolio-Risk-Manager Task 4: one PRM per bot.
        # PRM is stateful with respect to the drawdown hysteresis flag
        # (`_halted`) and the per-bot peak-equity tracker lives on the
        # AnalysisPipeline, so each bot needs its own instance —
        # sharing one PRM across bots would mix drawdown state
        # between unrelated portfolios. Default config is the spec
        # baseline (1% risk per trade, 15% per-asset cap, 30%
        # portfolio cap, 10/5/8% drawdown halt/reduce/resume) — when
        # we need per-bot tuning we'll thread a PortfolioRiskConfig
        # via bot config_json.
        portfolio_risk_manager = PortfolioRiskManager(PortfolioRiskConfig())

        # Execution + Sentinel position manager
        executor = Executor(adapter, event_bus, config)
        position_manager = PositionManager(adapter, event_bus)

        # Memory layer (shared repos, per-bot scoping happens via
        # bot_id / user_id passed into the pipeline)
        cycle_mem = CycleMemory(repos.cycles)
        rules = ReflectionRules(repos.rules)
        cross_bot = CrossBotSignals(repos.cross_bot)
        regime = RegimeHistory()

        pipeline = AnalysisPipeline(
            ohlcv_fetcher=fetcher,
            flow_agent=flow_agent,
            signal_registry=registry,
            conviction_agent=conviction_agent,
            decision_agent=decision_agent,
            event_bus=event_bus,
            cycle_memory=cycle_mem,
            reflection_rules=rules,
            cross_bot=cross_bot,
            regime_history=regime,
            cycle_repo=repos.cycles,
            config=config,
            bot_id=bot_id,
            user_id=user_id,
            is_shadow=pipeline_is_shadow,
            portfolio_risk_manager=portfolio_risk_manager,
            # Pure shadow mode (sim adapter, fake money) bypasses PRM
            # and uses a fixed $500 size so the data moat collects every
            # signal — exposure caps would otherwise starve the dataset
            # because shadow positions accumulate until the Sentinel
            # SL/TP monitor closes them. Paper mode (testnet) keeps PRM
            # active because validating live PRM is the whole point.
            shadow_fixed_size_usd=500.0 if shadow_mode else None,
            # Trade repo is wired in shadow mode so freshly opened
            # shadow positions get persisted to the trades table — the
            # Sentinel's SL/TP monitor reads from the same table to
            # close them when price breaches a level. Live + paper
            # don't need this path because their SL/TP orders live on
            # the exchange.
            trade_repo=repos.trades if shadow_mode else None,
            llm_provider=llm_provider,
            bot_repo=repos.bots,
            mode=factory_mode,
        )

        return TraderBot(
            bot_id=bot_id,
            pipeline=pipeline,
            executor=executor,
            position_manager=position_manager,
        )

    # Stash the shared CryptoFlowProvider on the factory function so
    # the async startup path (`_run_server`) can `await
    # bot_factory.flow_provider.warmup_from_repo()` and launch the
    # hourly cleanup loop without having to thread either the provider
    # or the repo through the BotRunner.
    factory.flow_provider = _shared_flow_provider  # type: ignore[attr-defined]
    factory.options_provider = _shared_options_provider  # type: ignore[attr-defined]
    factory.commodity_provider = _shared_commodity_provider  # type: ignore[attr-defined]
    factory.equity_provider = _shared_equity_provider  # type: ignore[attr-defined]
    return factory


# Dead symbols that should never run — removed from the exchange or
# were test entries that never had valid data.
_DEAD_SYMBOLS: frozenset[str] = frozenset({
    "SNDK-USDC", "USA500-USDC", "XYZ100-USDC",
})


def _deduplicate_bots(
    bots: list[dict],
    preferred_timeframe: str = "1h",
) -> list[dict]:
    """Keep one bot per symbol, preferring the configured timeframe.

    If the DB has multiple bot entries for the same symbol (e.g. 30m,
    1h, 4h from prior runs), the runner would register all of them.
    Only one sentinel is created per symbol, but every bot spawns its
    own scheduled loop — the extras hit BotManager's per-symbol
    concurrency limit and silently block the real bot.

    This guard deduplicates at startup so duplicates never reach the
    runner. It also filters out dead symbols.
    """
    _logger = logging.getLogger("quantagent")

    # Filter dead symbols first
    live: list[dict] = []
    for b in bots:
        sym = b.get("symbol", "")
        if sym in _DEAD_SYMBOLS:
            _logger.warning(
                f"Filtering dead symbol {sym} (bot {b.get('id', '?')})"
            )
        else:
            live.append(b)

    # Group by symbol
    by_symbol: dict[str, list[dict]] = {}
    for b in live:
        sym = b.get("symbol", "?")
        by_symbol.setdefault(sym, []).append(b)

    result: list[dict] = []
    for sym, group in by_symbol.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Multiple bots for the same symbol — pick the preferred TF,
        # then the most recently active if still ambiguous.
        preferred = [b for b in group if b.get("timeframe") == preferred_timeframe]
        candidates = preferred if preferred else group
        # Sort by last_cycle_at descending (None sorts last)
        candidates.sort(
            key=lambda b: b.get("last_cycle_at") or "",
            reverse=True,
        )
        keeper = candidates[0]
        result.append(keeper)

        dropped = [b for b in group if b is not keeper]
        dropped_ids = ", ".join(
            f"{b.get('id', '?')}({b.get('timeframe', '?')})" for b in dropped
        )
        _logger.warning(
            f"Duplicate bots for {sym}, keeping {keeper.get('id', '?')}"
            f"({keeper.get('timeframe', '?')}), dropping: {dropped_ids}"
        )

    if len(result) < len(bots):
        _logger.info(
            f"Bot dedup: {len(bots)} loaded → {len(result)} after "
            f"dedup ({len(bots) - len(result)} removed)"
        )

    return result


async def _run_server() -> None:
    """Start the BotRunner and FastAPI server together.

    Ctrl+C triggers graceful shutdown via signal handlers.
    """
    import uvicorn

    from api.app import create_app
    from engine.bot_manager import BotManager
    from engine.config import FeatureFlags
    from engine.events import create_event_bus
    from llm.claude import ClaudeProvider
    from storage.repositories import get_repositories

    logger = logging.getLogger("quantagent")

    from quantagent.version import ENGINE_VERSION
    logger.info(f"QuantAgent {ENGINE_VERSION} starting...")

    # ── Detect shadow / paper mode ──
    # The --shadow and --paper CLI flags (handled in main()) set
    # QUANTAGENT_SHADOW=1 / QUANTAGENT_PAPER=1 respectively. Both are
    # SIMPLE PROCESS FLAGS — no DATABASE_URL swap, no ensure_shadow_db
    # pre-create, no configure_shadow side-effects. The shared
    # `quantagent` database holds all three modes (live / shadow /
    # paper) via the per-row is_shadow / mode columns from Alembic 003
    # (mode column added there, paper added to the auto-derive set in
    # Paper Trading Task 2). The flags drive:
    #   1. which bots get loaded (filtered by exact `mode` match here)
    #   2. how the adapter_factory builds adapters (passes mode= through)
    #   3. logging banner so operators see the warning at boot
    #   4. is_shadow propagation into AnalysisPipeline cycle records
    #      (paper writes is_shadow=True so testnet fills are kept out
    #      of the live data moat — same as shadow)
    #
    # The mutual-exclusion check lives in main() — by the time we
    # reach this branch we know at most ONE of these flags is set.
    # The defensive `elif` chain below honours that contract.
    shadow_mode = os.environ.get("QUANTAGENT_SHADOW") == "1"
    paper_mode = os.environ.get("QUANTAGENT_PAPER") == "1"

    if shadow_mode:
        logger.warning(
            "⚠️ SHADOW MODE — loading shadow bots, simulated execution only"
        )
        bot_mode = "shadow"
    elif paper_mode:
        logger.warning(
            "⚠️ PAPER MODE — real orders on Hyperliquid testnet"
        )
        bot_mode = "paper"
    else:
        logger.info("LIVE MODE — real orders on mainnet")
        bot_mode = "live"

    # ── Initialize infrastructure ──
    logger.info("Initializing infrastructure...")
    repos = await get_repositories()
    event_bus = create_event_bus("memory")

    # ── LLM provider (constructed ONCE, shared by every spawned bot) ──
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — bot_factory will fail at first "
            "spawn. Set it in .env before running live."
        )
    llm_call_repo = getattr(repos, "llm_calls", None)
    llm_provider = ClaudeProvider(api_key=api_key, llm_call_repo=llm_call_repo) if api_key else None

    # ── Adapter factory (per-bot mode-aware) ──
    def adapter_factory(exchange: str, mode: str = "live"):
        # Side-effect import: registers HyperliquidAdapter with the
        # ExchangeFactory the first time something asks for an adapter.
        import exchanges.hyperliquid  # noqa: F401
        from exchanges.factory import ExchangeFactory
        return ExchangeFactory.get_adapter(exchange, mode=mode)

    # ── Bot factory: wires the full analysis pipeline per symbol ──
    feature_flags = FeatureFlags()
    bot_factory = _make_bot_factory(
        repos=repos,
        llm_provider=llm_provider,
        adapter_factory=adapter_factory,
        event_bus=event_bus,
        feature_flags=feature_flags,
        shadow_mode=shadow_mode,
        paper_mode=paper_mode,
    )

    # ── Warm the shared CryptoFlowProvider from the OI snapshot table
    # before any bot spawns. After this awaits the deques already hold
    # whatever the last process run wrote, so the very first analysis
    # cycle can compute oi_change_4h / oi_trend instead of blocking on
    # the cold-start window. Errors here are non-fatal — the provider
    # falls back to live cold start if the bulk-load fails.
    try:
        warmed = await bot_factory.flow_provider.warmup_from_repo()
        logger.info(
            f"CryptoFlowProvider warmup loaded {warmed} snapshots "
            f"({bot_factory.flow_provider.lookback_seconds}s lookback)"
        )
    except Exception:
        logger.exception("CryptoFlowProvider warmup failed; continuing")

    # Warm the shared CommodityFlowProvider from the cot_cache table
    # so the 52-week CFTC history is available on the first analysis
    # cycle after restart. Any failure is non-fatal — the provider
    # falls back to the next live cot_reports pull.
    try:
        cot_warmed = await bot_factory.commodity_provider.warmup_from_repo()
        logger.info(
            f"CommodityFlowProvider warmup loaded {cot_warmed} COT snapshots"
        )
    except Exception:
        logger.exception("CommodityFlowProvider warmup failed; continuing")

    # Warm the shared EquityFlowProvider from the regsho_cache table
    # so the 20-day Z-score window is live on cycle #1 after restart.
    try:
        regsho_warmed = await bot_factory.equity_provider.warmup_from_repo()
        logger.info(
            f"EquityFlowProvider warmup loaded {regsho_warmed} RegSHO snapshots"
        )
    except Exception:
        logger.exception("EquityFlowProvider warmup failed; continuing")

    # ── Hourly cleanup loop for the oi_snapshots table.
    # Keeps the table small by deleting anything older than 24h. Runs
    # in the background; cancelled on shutdown via shutdown_event.
    async def _oi_snapshots_cleanup_loop():
        while not shutdown_event.is_set():
            try:
                deleted = await repos.oi_snapshots.cleanup_older_than(86_400)
                if deleted:
                    logger.info(
                        f"oi_snapshots cleanup: deleted {deleted} rows older than 24h"
                    )
            except Exception:
                logger.exception("oi_snapshots cleanup failed")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=3600)
            except asyncio.TimeoutError:
                continue

    bot_manager = BotManager(
        event_bus=event_bus,
        bot_factory=bot_factory,
    )

    # ── PriceFeed initialization (Sprint Week 7 Task 7) ──
    # Gated by PRICE_FEED_ENABLED env var (feature flag). When active,
    # creates a HyperliquidPriceFeed (WebSocket + REST bootstrap),
    # wraps it in a RESTFallbackManager for resilience, and starts an
    # SLTPMonitor for tick-level shadow SL/TP. The PriceFeed reference
    # is passed through BotRunner → SentinelMonitor so every Sentinel
    # switches from REST-poll to event-driven mode (zero REST overhead).
    # When the flag is off, everything stays on the legacy REST path.
    price_feed_enabled = (
        os.environ.get("PRICE_FEED_ENABLED", "false").lower() == "true"
        and shadow_mode  # only shadow mode for now; live/paper use exchange SL/TP
    )
    price_feed = None
    fallback_manager = None
    sl_tp_monitor = None

    if price_feed_enabled:
        from engine.data.price_feed import HyperliquidPriceFeed, RESTFallbackManager
        from engine.sl_tp_monitor import SLTPMonitor

        testnet = paper_mode  # shadow runs against mainnet data
        # Bootstrap adapter for REST history seeding — public OHLCV, no signing.
        bootstrap_adapter = adapter_factory("hyperliquid", mode="live")

        price_feed = HyperliquidPriceFeed(
            event_bus=event_bus,
            testnet=testnet,
            candle_timeframe="1h",
            bootstrap_adapter=bootstrap_adapter,
        )

        fallback_manager = RESTFallbackManager(
            price_feed=price_feed,
            adapter=bootstrap_adapter,
            event_bus=event_bus,
        )

        sl_tp_monitor = SLTPMonitor(
            event_bus=event_bus,
            trade_repo=repos.trades,
            is_shadow=True,
        )

        logger.info("PriceFeed stack initialized (PRICE_FEED_ENABLED=true)")

    # ── Initialize BotRunner ──
    from quantagent.runner import BotRunner

    runner = BotRunner(
        repos=repos,
        adapter_factory=adapter_factory,
        llm_provider=llm_provider,
        event_bus=event_bus,
        bot_manager=bot_manager,
        shadow_mode=shadow_mode,
        price_feed=price_feed,
    )

    # Load active bots from DB filtered by mode so the runner restores
    # only the bots that belong to this process's mode. A shadow boot
    # never sees live or paper bots; a paper boot never sees live or
    # shadow bots; a live boot never sees shadow or paper bots. The
    # `bot_mode` was resolved by the shadow/paper detection block above.
    active_bots = await repos.bots.get_active_bots_by_mode(bot_mode)
    logger.info(f"Loaded {len(active_bots)} {bot_mode} bots from database")
    active_bots = _deduplicate_bots(active_bots)
    await runner.start_with_bots(active_bots)

    # ── Start PriceFeed stack (needs symbol list from active_bots) ──
    if price_feed_enabled and price_feed is not None:
        shadow_symbols = list({b.get("symbol", "BTC-USDC") for b in active_bots})
        try:
            await fallback_manager.start(symbols=shadow_symbols)
            for sym in shadow_symbols:
                sl_tp_monitor.register_symbol(sym)
            await sl_tp_monitor.start()
            # Wire PriceFeed into CryptoFlowProvider so it reads funding/OI
            # from WebSocket memory instead of REST, and fills its OI history
            # buffer from pushed events.
            bot_factory.flow_provider.set_price_feed(price_feed, event_bus)
            logger.info(
                f"PriceFeed active: {len(shadow_symbols)} symbols via WebSocket, "
                f"SLTPMonitor tracking shadow trades, Sentinels in event-driven mode"
            )
        except Exception:
            logger.exception(
                "PriceFeed startup failed; falling back to REST polling"
            )
            # Non-fatal — sentinels already run and will use REST path
            # because price_feed is set but connect failed; the sentinel's
            # _on_candle_close won't fire if no CandleClosed events arrive,
            # so the keepalive loop's day-boundary resets are all that runs.
            # The scheduled fallback loops in BotRunner still trigger
            # analysis cycles at the configured interval regardless.

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

    # Launch the OI snapshot cleanup loop now that shutdown_event
    # exists. The loop closes itself when the event is set.
    cleanup_task = asyncio.create_task(_oi_snapshots_cleanup_loop())

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
    cleanup_task.cancel()
    try:
        await cleanup_task
    except (asyncio.CancelledError, Exception):
        pass
    # Stop PriceFeed stack first — SLTPMonitor needs the bus to
    # unsubscribe, and the fallback manager needs the feed to disconnect.
    if sl_tp_monitor is not None:
        try:
            await sl_tp_monitor.stop()
        except Exception:
            logger.debug("SLTPMonitor stop() failed", exc_info=True)
    if fallback_manager is not None:
        try:
            await fallback_manager.stop()
        except Exception:
            logger.debug("RESTFallbackManager stop() failed", exc_info=True)
    try:
        await bot_factory.options_provider.close()
    except Exception:
        logger.debug("OptionsEnrichment close() failed", exc_info=True)
    try:
        await bot_factory.equity_provider.close()
    except Exception:
        logger.debug("EquityFlowProvider close() failed", exc_info=True)
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

    # ``--shadow`` and ``--paper`` are global mode flags: they apply to
    # any subcommand and are stripped from argv before subcommand
    # dispatch so the existing positional logic keeps working. Each
    # flag is a SIMPLE PROCESS MARKER — it sets the corresponding
    # `QUANTAGENT_*` env var and that's it. There is no DB URL swap,
    # no config mutation, no shadow-DB pre-create. The shared
    # `quantagent` database holds all three modes (live / shadow /
    # paper) via the per-row is_shadow / mode columns; _run_server()
    # reads the env vars, loads bots filtered by mode, and passes the
    # per-bot mode through to the adapter factory.
    #
    # Mode precedence: the two flags are MUTUALLY EXCLUSIVE because
    # one process is one mode. Trying both at once is almost certainly
    # an operator typo or misunderstanding — the safest response is to
    # exit loudly so they pick the one they actually meant. (For
    # genuinely running both modes simultaneously the operator must
    # start two separate processes in two tmux sessions on different
    # ports — that's the post-sprint deployment workflow.)
    shadow_requested = "--shadow" in args
    paper_requested = "--paper" in args

    if shadow_requested and paper_requested:
        print("ERROR: --shadow and --paper are mutually exclusive.")
        print("       Run them in separate processes (different ports).")
        sys.exit(1)

    if shadow_requested:
        args = [a for a in args if a != "--shadow"]
        os.environ["QUANTAGENT_SHADOW"] = "1"
        # Route LangSmith traces to a separate project so shadow-mode
        # cycles don't pollute the live observability dashboard. Plain
        # assignment (NOT setdefault) — the operator's `.env` almost
        # certainly points at `quantagent-live`, and we explicitly want
        # to override that for the shadow process. Mirrors the paper
        # branch below for symmetry. (Per-bot routing within a
        # mixed-mode process is a future enhancement — see Task 4
        # spec deviation #1 in SPRINT_WEEK7_paper_Update.md for the
        # full rationale on why we kept this process-level for now.)
        os.environ["LANGCHAIN_PROJECT"] = "quantagent-shadow"

    if paper_requested:
        args = [a for a in args if a != "--paper"]
        os.environ["QUANTAGENT_PAPER"] = "1"
        # Paper mode talks to the venue's testnet endpoint. The
        # HyperliquidAdapter constructor reads HYPERLIQUID_TESTNET first
        # (then branches credential lookup to the dedicated _TESTNET_*
        # env vars), so flipping this here is the cleanest way to make
        # sure every adapter built in this process uses testnet
        # regardless of how it was constructed (factory path, direct
        # ctor in scripts, etc.).
        os.environ["HYPERLIQUID_TESTNET"] = "true"
        # Route LangSmith traces to a separate project so paper-mode
        # cycles don't pollute live observability dashboards. The env
        # var is read by both `langsmith.Client` and `llm/claude.py`'s
        # tracing path (which honours `LANGCHAIN_PROJECT` per-process).
        os.environ["LANGCHAIN_PROJECT"] = "quantagent-paper"
        # Default port to 8001 so paper can run alongside a shadow or
        # live process on the same host without colliding. ``setdefault``
        # so an operator who explicitly sets ``PORT=9000`` keeps their
        # override — we only fill in the default when nothing is set.
        os.environ.setdefault("PORT", "8001")

    if not args or args[0] in ("--help", "-h"):
        print("QuantAgent v2 — AI-powered trading engine")
        print()
        print("Usage:")
        print("  python -m quantagent run            Start BotRunner + API server")
        print("  python -m quantagent run --shadow   Load shadow bots, simulated execution")
        print("  python -m quantagent run --paper    Load paper bots, testnet exchange + port 8001")
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
