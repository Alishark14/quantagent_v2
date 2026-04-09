#!/usr/bin/env python3
"""One-shot Hyperliquid testnet validation: full pipeline + real order.

Runs ONE complete analysis cycle (data → 4 signals → conviction → decision)
against Hyperliquid TESTNET and, if the pipeline outputs LONG or SHORT,
places a small real market order on the testnet orderbook.

This is a manual smoke test, NOT a service. Run once, see results, exit.

Usage:
    HYPERLIQUID_TESTNET=true python scripts/run_testnet_cycle.py \\
        --symbol BTC-USDC --timeframe 1h

Required environment:
    HYPERLIQUID_TESTNET=true                  (mandatory — script refuses without it)
    ANTHROPIC_API_KEY                          (Claude credentials)
    HYPERLIQUID_TESTNET_WALLET_ADDRESS         (testnet public wallet — preferred)
    HYPERLIQUID_TESTNET_PRIVATE_KEY            (testnet signing key — preferred)

The script falls back to the mainnet env vars
(``HYPERLIQUID_WALLET_ADDRESS`` / ``HYPERLIQUID_PRIVATE_KEY``) and then to
the spec aliases (``HYPERLIQUID_API_KEY`` / ``HYPERLIQUID_API_SECRET``)
for backward compatibility, but the dedicated testnet pair wins when set
so testnet runs never accidentally reuse mainnet keys.

Safety rails:
1. ``HYPERLIQUID_TESTNET=true`` env var is mandatory; missing → hard exit.
2. After constructing the adapter, ``adapter._testnet`` is read back and
   asserted to be True. If for any reason the testnet flag did not stick,
   the script exits BEFORE running the pipeline (no LLM cost burned, no
   order risk).
3. The factory cache is reset before adapter construction so a stale
   non-testnet ``hyperliquid`` instance from a previous import in the
   same process can never be returned.
4. Order size is computed as ``round(15.0 / current_price, 5)`` — small
   enough to be a smoke test, large enough to clear Hyperliquid's $10
   minimum notional with a $5 buffer.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# .env loader (matches scripts/run_trade.py byte-for-byte)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                if "#" in value:
                    value = value[:value.index("#")].strip()
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Cost-tracking LLM shim
# ---------------------------------------------------------------------------

class _CostTrackingLLM:
    """Delegating wrapper that aggregates total cost + latency.

    Wraps a real ``LLMProvider`` and forwards every method to the inner
    provider; on each call it accumulates ``response.cost`` and
    ``response.latency_ms`` so the script can print one-line totals at
    the end without scraping log output.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.total_cost: float = 0.0
        self.total_latency_ms: float = 0.0
        self.call_count: int = 0

    async def generate_text(self, *args, **kwargs):
        response = await self._inner.generate_text(*args, **kwargs)
        self._record(response)
        return response

    async def generate_vision(self, *args, **kwargs):
        response = await self._inner.generate_vision(*args, **kwargs)
        self._record(response)
        return response

    def _record(self, response) -> None:
        self.total_cost += float(getattr(response, "cost", 0.0) or 0.0)
        self.total_latency_ms += float(getattr(response, "latency_ms", 0.0) or 0.0)
        self.call_count += 1


# ---------------------------------------------------------------------------
# Pretty-printers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _print_action(action) -> None:
    print(f"\n  ACTION:        {action.action}")
    print(f"  Conviction:    {action.conviction_score:.3f}")
    if action.sl_price is not None:
        print(f"  SL price:      {action.sl_price}")
    if action.tp1_price is not None:
        print(f"  TP1 price:     {action.tp1_price}")
    if action.tp2_price is not None:
        print(f"  TP2 price:     {action.tp2_price}")
    if action.rr_ratio is not None:
        print(f"  R:R ratio:     {action.rr_ratio}")
    if action.position_size is not None:
        print(f"  Suggested $:   {action.position_size:.2f}")
    if action.reasoning:
        print(f"  Reasoning:     {action.reasoning[:300]}")


def _print_signal(signal) -> None:
    name = getattr(signal, "agent_name", "?")
    direction = getattr(signal, "direction", "?")
    confidence = getattr(signal, "confidence", 0.0)
    print(f"  - {name:<20} {direction:<8} (conf={confidence:.2f})")
    reasoning = getattr(signal, "reasoning", "") or ""
    if reasoning:
        print(f"      {reasoning[:160]}")


def _print_conviction(conviction) -> None:
    print(f"  Conviction score:  {conviction.conviction_score:.3f}")
    print(f"  Direction:         {conviction.direction}")
    print(f"  Regime:            {conviction.regime} "
          f"(conf={conviction.regime_confidence:.2f})")
    print(f"  Signal quality:    {conviction.signal_quality}")
    print(f"  Factual / Subj:    "
          f"{conviction.factual_weight:.1f} / {conviction.subjective_weight:.1f}")
    if conviction.contradictions:
        print(f"  Contradictions:    {conviction.contradictions}")
    if conviction.reasoning:
        print(f"  Reasoning:         {conviction.reasoning[:300]}")


# ---------------------------------------------------------------------------
# Endpoint inspection
# ---------------------------------------------------------------------------

def _adapter_endpoint(adapter) -> str:
    """Extract the live API URL from the inner ccxt object, if exposed."""
    inner = getattr(adapter, "_exchange", None)
    if inner is None:
        return "(adapter has no _exchange — cannot inspect)"
    urls = getattr(inner, "urls", None)
    if not isinstance(urls, dict):
        return "(no urls dict)"
    api = urls.get("api")
    if isinstance(api, dict):
        # ccxt sometimes nests by call type — return the first concrete URL
        for v in api.values():
            if isinstance(v, str) and v:
                return v
        return repr(api)
    if isinstance(api, str):
        return api
    return "(no api URL)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(symbol: str, timeframe: str) -> int:
    _load_env()

    # ── Step 1: Mandatory env-var safety checks ──
    if os.environ.get("HYPERLIQUID_TESTNET", "").lower() not in ("true", "1", "yes"):
        print("ERROR: HYPERLIQUID_TESTNET=true is required for this script.")
        print("       This script ONLY runs against testnet.")
        print("       Set it in .env or export it before running.")
        return 1

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        return 1

    # In testnet mode the HyperliquidAdapter prefers the dedicated
    # HYPERLIQUID_TESTNET_* env vars and falls back to the mainnet vars for
    # backward compatibility. This script ONLY runs on testnet, so prefer the
    # testnet pair, fall back to the mainnet pair, and accept the spec's
    # HYPERLIQUID_API_KEY / HYPERLIQUID_API_SECRET aliases as a final fallback.
    wallet = (
        os.environ.get("HYPERLIQUID_TESTNET_WALLET_ADDRESS")
        or os.environ.get("HYPERLIQUID_WALLET_ADDRESS")
        or os.environ.get("HYPERLIQUID_API_KEY")
    )
    private_key = (
        os.environ.get("HYPERLIQUID_TESTNET_PRIVATE_KEY")
        or os.environ.get("HYPERLIQUID_PRIVATE_KEY")
        or os.environ.get("HYPERLIQUID_API_SECRET")
    )

    if not wallet:
        print(
            "ERROR: HYPERLIQUID_TESTNET_WALLET_ADDRESS "
            "(or HYPERLIQUID_WALLET_ADDRESS / HYPERLIQUID_API_KEY) is not set."
        )
        return 1
    if not private_key:
        print(
            "ERROR: HYPERLIQUID_TESTNET_PRIVATE_KEY "
            "(or HYPERLIQUID_PRIVATE_KEY / HYPERLIQUID_API_SECRET) is not set."
        )
        return 1

    # Ensure the adapter's env-var fallback path sees what we resolved,
    # so the constructor reads from one canonical source.
    os.environ.setdefault("HYPERLIQUID_TESTNET_WALLET_ADDRESS", wallet)
    os.environ.setdefault("HYPERLIQUID_TESTNET_PRIVATE_KEY", private_key)

    # ── Logging setup ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "anthropic", "ccxt", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _banner(f"QuantAgent v2 — Testnet Validation Cycle")
    print(f"  Symbol:    {symbol}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Time UTC:  {datetime.now(timezone.utc).isoformat()}")

    # ── Step 2: Construct adapter on a clean factory cache ──
    from exchanges.factory import ExchangeFactory
    import exchanges.hyperliquid  # noqa: F401  — populates the registry

    # Reset the live cache so any stale, non-testnet adapter from a
    # previous import in this process can't be returned. The factory's
    # live path keys only by name (NOT by kwargs), so without the reset
    # `testnet=True` would be silently dropped if a cache hit existed.
    ExchangeFactory._instances.pop("hyperliquid", None)

    _banner("Adapter")
    adapter = ExchangeFactory.get_adapter("hyperliquid", testnet=True)

    # Defense in depth: read the testnet flag back off the adapter and
    # bail BEFORE we burn any LLM cost or risk an order if the flag did
    # not stick (e.g. someone wired a custom factory that ignored kwargs).
    is_testnet = bool(getattr(adapter, "_testnet", False))
    endpoint = _adapter_endpoint(adapter)
    print(f"  Endpoint:       {endpoint}")
    print(f"  testnet flag:   {is_testnet}")
    if not is_testnet:
        print("ERROR: adapter._testnet is False — testnet flag did not stick.")
        print("       Aborting before any LLM call or order placement.")
        return 1

    # ── Step 3: Wallet + balance ──
    print(f"  Wallet:         {wallet}")
    try:
        balance = await adapter.get_balance()
    except Exception as e:
        print(f"  Balance:        ERROR — {e}")
        return 1
    print(f"  Balance (USDC): {balance:.2f}")
    if balance <= 0:
        print("  WARNING: balance is 0 — order placement will fail. Continuing anyway "
              "for the pipeline-only smoke test.")

    # ── Step 4: Wire the full pipeline ──
    _banner("Wiring pipeline")

    from engine.config import FeatureFlags, TradingConfig
    from engine.conviction.agent import ConvictionAgent
    from engine.data.flow import FlowAgent, FlowSignalAgent
    from engine.data.flow.crypto import CryptoFlowProvider
    from engine.data.ohlcv import OHLCVFetcher
    from engine.events import (
        ConvictionScored,
        CycleCompleted,
        DataReady,
        InProcessBus,
        SignalsReady,
    )
    from engine.execution.agent import DecisionAgent
    from engine.memory.cross_bot import CrossBotSignals
    from engine.memory.cycle_memory import CycleMemory
    from engine.memory.reflection_rules import ReflectionRules
    from engine.memory.regime_history import RegimeHistory
    from engine.pipeline import AnalysisPipeline
    from engine.signals.indicator_agent import IndicatorAgent
    from engine.signals.pattern_agent import PatternAgent
    from engine.signals.registry import SignalRegistry
    from engine.signals.trend_agent import TrendAgent
    from llm.claude import ClaudeProvider
    from storage.repositories import get_repositories

    config = TradingConfig(symbol=symbol, timeframe=timeframe)
    bus = InProcessBus()

    # Capture the in-flight signals + conviction so we can print them
    # after the cycle completes (the bus dispatches sync but we don't
    # want to interleave noisy lines with the pipeline's own logging).
    captured: dict = {"signals": None, "conviction": None}
    bus.subscribe(SignalsReady, lambda e: captured.__setitem__("signals", e.signals))
    bus.subscribe(ConvictionScored, lambda e: captured.__setitem__("conviction", e.conviction))
    bus.subscribe(DataReady, lambda e: print(
        f"  [event] DataReady           — {e.market_data.num_candles} candles"
    ))
    bus.subscribe(SignalsReady, lambda e: print(
        f"  [event] SignalsReady        — {len(e.signals)} signals"
    ))
    bus.subscribe(ConvictionScored, lambda e: print(
        f"  [event] ConvictionScored    — {e.conviction.conviction_score:.2f} "
        f"{e.conviction.direction}"
    ))
    bus.subscribe(CycleCompleted, lambda e: print(
        f"  [event] CycleCompleted      — {e.action} (conviction={e.conviction:.2f})"
    ))

    inner_llm = ClaudeProvider(api_key=anthropic_key)
    llm = _CostTrackingLLM(inner_llm)

    repos = await get_repositories("sqlite")
    fetcher = OHLCVFetcher(adapter, config)
    flow_agent = FlowAgent([CryptoFlowProvider()])

    flags = FeatureFlags()
    registry = SignalRegistry()
    registry.register(IndicatorAgent(llm, flags))
    registry.register(PatternAgent(llm, flags))
    registry.register(TrendAgent(llm, flags))
    registry.register(FlowSignalAgent(flags))
    print(f"  Registered signal producers: "
          f"{[p.name() for p in registry.get_enabled()]}")

    conviction_agent = ConvictionAgent(llm)
    decision_agent = DecisionAgent(llm, config)

    pipeline = AnalysisPipeline(
        ohlcv_fetcher=fetcher,
        flow_agent=flow_agent,
        signal_registry=registry,
        conviction_agent=conviction_agent,
        decision_agent=decision_agent,
        event_bus=bus,
        cycle_memory=CycleMemory(repos.cycles),
        reflection_rules=ReflectionRules(repos.rules),
        cross_bot=CrossBotSignals(repos.cross_bot),
        regime_history=RegimeHistory(),
        cycle_repo=repos.cycles,
        config=config,
        bot_id="testnet-cycle",
        user_id="dev-testnet",
    )

    # ── Step 5: Run one cycle ──
    _banner("Running pipeline.run_cycle()")
    cycle_start = time.perf_counter()
    try:
        action = await pipeline.run_cycle()
    except Exception as e:
        print(f"  ERROR running cycle: {e}")
        return 1
    cycle_elapsed_ms = (time.perf_counter() - cycle_start) * 1000
    print(f"  Wall clock: {cycle_elapsed_ms:.0f}ms")

    # ── Step 6: Pretty-print the result ──
    _banner("Signals")
    signals = captured.get("signals") or []
    if not signals:
        print("  (no signals captured)")
    else:
        for sig in signals:
            _print_signal(sig)

    _banner("Conviction")
    conviction = captured.get("conviction")
    if conviction is None:
        print("  (no conviction captured)")
    else:
        _print_conviction(conviction)

    _banner("Decision")
    _print_action(action)

    # ── Step 7: Place an order if the pipeline says LONG/SHORT ──
    _banner("Order")
    placed_order = None
    if action.action in ("LONG", "SHORT"):
        side = "buy" if action.action == "LONG" else "sell"
        # Compute a small testnet order size — round to 5 decimals so
        # ccxt accepts the precision and the notional clears Hyperliquid's
        # $10 minimum with a $5 buffer.
        try:
            ticker = await adapter.get_ticker(symbol)
            current_price = float(ticker.get("last") or 0.0)
        except Exception as e:
            print(f"  ERROR fetching ticker: {e}")
            current_price = 0.0
        if current_price <= 0:
            print(f"  ERROR: ticker returned no price for {symbol}; skipping order.")
        else:
            order_size = round(15.0 / current_price, 5)
            print(f"  Current price: {current_price}")
            print(f"  Order side:    {side}")
            print(f"  Order size:    {order_size}  (~${order_size * current_price:.2f})")
            print("  PLACING ORDER on testnet...")
            try:
                placed_order = await adapter.place_market_order(
                    symbol=symbol, side=side, size=order_size
                )
            except Exception as e:
                print(f"  ERROR placing order: {e}")
                placed_order = None

            if placed_order is None:
                pass
            elif not placed_order.success:
                print(f"  Order FAILED: {placed_order.error}")
            else:
                print(f"  Order OK")
                print(f"    order_id:    {placed_order.order_id}")
                print(f"    fill_price:  {placed_order.fill_price}")
                print(f"    fill_size:   {placed_order.fill_size}")

                # Wait briefly then dump positions
                print("  Sleeping 2s for fill propagation...")
                await asyncio.sleep(2)
                try:
                    positions = await adapter.get_positions(symbol)
                except Exception as e:
                    print(f"  ERROR fetching positions: {e}")
                    positions = []
                if not positions:
                    print("  No open positions returned (may not have settled yet).")
                else:
                    for pos in positions:
                        print(
                            f"    {pos.symbol} {pos.direction} {pos.size} "
                            f"@ {pos.entry_price}  uPnL=${pos.unrealized_pnl:.2f}"
                        )
    elif action.action == "SKIP":
        print("  Pipeline decided SKIP — no order placed.")
    else:
        print(f"  Pipeline action is {action.action!r} — no entry order placed.")

    # ── Step 8: Cost + latency summary ──
    _banner("Run summary")
    print(f"  LLM calls:        {llm.call_count}")
    print(f"  LLM total cost:   ${llm.total_cost:.4f}")
    print(f"  LLM total ms:     {llm.total_latency_ms:.0f}")
    print(f"  Pipeline ms:      {cycle_elapsed_ms:.0f}")
    print(f"  Events emitted:   {bus.total_published}")
    print()

    return 0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one analysis cycle against Hyperliquid testnet "
                    "(real orders if pipeline says LONG/SHORT)."
    )
    parser.add_argument("--symbol", default="BTC-USDC", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    args = parser.parse_args()

    rc = asyncio.run(main(args.symbol, args.timeframe))
    sys.exit(rc)
