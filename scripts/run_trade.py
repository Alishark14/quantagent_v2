#!/usr/bin/env python3
"""Run a single TraderBot lifecycle: analyze + execute on exchange.

First real trade script. Defaults to testnet (fake money).

Usage:
    python scripts/run_trade.py --symbol BTC-USDC --timeframe 1h --testnet
    python scripts/run_trade.py --symbol BTC-USDC --timeframe 1h --dry-run
    python scripts/run_trade.py --symbol ETH-USDC --timeframe 4h --testnet --verbose

Flags:
    --testnet   Use Hyperliquid testnet (default: True)
    --live      Use mainnet (requires confirmation)
    --dry-run   Pipeline only, print action, no execution
    --verbose   Print agent signals, conviction breakdown, memory context
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


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


async def main(
    symbol: str,
    timeframe: str,
    verbose: bool,
    testnet: bool,
    dry_run: bool,
) -> None:
    _load_env()

    # Testnet env override
    if testnet:
        os.environ["HYPERLIQUID_TESTNET"] = "true"

    # Logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpx", "httpcore", "anthropic", "ccxt", "matplotlib", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Validate
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    # Safety: live trading confirmation
    if not testnet and not dry_run:
        print("\n" + "!" * 60)
        print("  WARNING: LIVE TRADING MODE — REAL MONEY AT RISK")
        print("!" * 60)
        confirm = input("  Type 'yes' to proceed with live trading: ")
        if confirm.strip().lower() != "yes":
            print("  Aborted.")
            sys.exit(0)

    mode = "DRY-RUN" if dry_run else ("TESTNET" if testnet else "LIVE")
    print(f"\n{'='*60}")
    print(f"  QuantAgent v2 — TraderBot [{mode}]")
    print(f"  Symbol: {symbol} | TF: {timeframe}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # ── Imports ──
    from engine.config import FeatureFlags, TradingConfig
    from engine.conviction.agent import ConvictionAgent
    from engine.data.flow import FlowAgent
    from engine.data.flow.crypto import CryptoFlowProvider
    from engine.data.ohlcv import OHLCVFetcher
    from engine.events import (
        ConvictionScored,
        CycleCompleted,
        DataReady,
        InProcessBus,
        SignalsReady,
        TradeOpened,
        TradeClosed,
    )
    from engine.execution.agent import DecisionAgent
    from engine.execution.executor import Executor
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
    from exchanges.factory import ExchangeFactory
    from llm.claude import ClaudeProvider
    from sentinel.position_manager import PositionManager
    from storage.repositories import get_repositories

    config = TradingConfig(symbol=symbol, timeframe=timeframe)
    bus = InProcessBus()

    # Event logging
    bus.subscribe(DataReady, lambda e: print(
        f"  [EVENT] DataReady — {e.market_data.num_candles} candles"
    ))

    if verbose:
        bus.subscribe(SignalsReady, lambda e: _print_signals(e.signals))
        bus.subscribe(ConvictionScored, lambda e: _print_conviction(e.conviction))
    else:
        bus.subscribe(SignalsReady, lambda e: print(
            f"  [EVENT] SignalsReady — {len(e.signals)} signals"
        ))
        bus.subscribe(ConvictionScored, lambda e: print(
            f"  [EVENT] ConvictionScored — {e.conviction.conviction_score:.2f} "
            f"{e.conviction.direction} ({e.conviction.regime})"
        ))

    bus.subscribe(CycleCompleted, lambda e: print(
        f"  [EVENT] CycleCompleted — {e.action} (conviction={e.conviction:.2f})"
    ))
    bus.subscribe(TradeOpened, lambda e: print(
        f"  [EVENT] TradeOpened — {e.trade_action.action} "
        f"fill={e.order_result.fill_price} id={e.order_result.order_id}"
    ))
    bus.subscribe(TradeClosed, lambda e: print(
        f"  [EVENT] TradeClosed — {e.symbol} P&L={e.pnl} ({e.exit_reason})"
    ))

    # ── Initialize stack ──
    print("Initializing...")
    import exchanges.hyperliquid  # noqa: F401
    adapter = ExchangeFactory.get_adapter("hyperliquid")
    llm = ClaudeProvider(api_key=api_key)
    repos = await get_repositories("sqlite")

    fetcher = OHLCVFetcher(adapter, config)
    flow_agent = FlowAgent([CryptoFlowProvider()])

    registry = SignalRegistry()
    flags = FeatureFlags()
    registry.register(IndicatorAgent(llm, flags))
    registry.register(PatternAgent(llm, flags))
    registry.register(TrendAgent(llm, flags))

    conviction_agent = ConvictionAgent(llm)
    decision_agent = DecisionAgent(llm, config)
    executor = Executor(adapter, bus, config)
    pm = PositionManager(adapter, bus)

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
        event_bus=bus,
        cycle_memory=cycle_mem,
        reflection_rules=rules,
        cross_bot=cross_bot,
        regime_history=regime,
        cycle_repo=repos.cycles,
        config=config,
        bot_id="trade-bot-001",
        user_id="dev-user",
    )

    # ── Run ──
    if dry_run:
        print("Running analysis (dry-run, no execution)...\n")
        action = await pipeline.run_cycle()
        _print_action(action)
        print(f"\n  [DRY-RUN] No orders placed.")
    else:
        print("Running TraderBot (analyze + execute)...\n")
        bot = TraderBot(
            bot_id="trade-bot-001",
            pipeline=pipeline,
            executor=executor,
            position_manager=pm,
        )
        result = await bot.run()
        _print_result(result)

        # Verify position on exchange
        if result.get("action") in ("LONG", "SHORT"):
            print("\n  Verifying position on exchange...")
            try:
                positions = await adapter.get_positions(symbol)
                if positions:
                    pos = positions[0]
                    print(f"    Position: {pos.direction} {pos.size} @ {pos.entry_price}")
                    print(f"    Unrealized P&L: ${pos.unrealized_pnl:.2f}")
                else:
                    print("    No position found (may not have filled yet)")
            except Exception as e:
                print(f"    Position check failed: {e}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Events: {bus.total_published}")
    for name, count in bus.per_type_counts.items():
        print(f"    {name}: {count}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_signals(signals) -> None:
    print(f"  [EVENT] SignalsReady — {len(signals)} signals:")
    for s in signals:
        print(
            f"    {s.agent_name}: {s.direction} "
            f"(confidence={s.confidence:.2f})"
        )
        if s.pattern_detected:
            print(f"      Pattern: {s.pattern_detected}")
        print(f"      {s.reasoning[:120]}")
        if s.contradictions and s.contradictions != "none":
            print(f"      Contradictions: {s.contradictions}")


def _print_conviction(conviction) -> None:
    print(f"  [EVENT] ConvictionScored:")
    print(f"    Score: {conviction.conviction_score:.2f}")
    print(f"    Direction: {conviction.direction}")
    print(f"    Regime: {conviction.regime} (conf={conviction.regime_confidence:.2f})")
    print(f"    Quality: {conviction.signal_quality}")
    print(f"    Factual/Subj: {conviction.factual_weight:.1f}/{conviction.subjective_weight:.1f}")
    if conviction.contradictions:
        print(f"    Contradictions: {conviction.contradictions}")
    print(f"    Reasoning: {conviction.reasoning[:200]}")


def _print_action(action) -> None:
    print(f"\n{'='*60}")
    print(f"  ACTION: {action.action}")
    print(f"  Conviction: {action.conviction_score:.2f}")
    if action.sl_price:
        print(f"  SL: {action.sl_price}")
        print(f"  TP1: {action.tp1_price}")
        print(f"  TP2: {action.tp2_price}")
        print(f"  RR: {action.rr_ratio}")
        if action.position_size:
            print(f"  Size: ${action.position_size:.2f}")
    print(f"  Reasoning: {action.reasoning}")
    print(f"{'='*60}")


def _print_result(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  STATUS: {result.get('status', '?')}")
    print(f"  ACTION: {result.get('action', '?')}")
    print(f"  Conviction: {result.get('conviction_score', 0):.2f}")
    print(f"  Duration: {result.get('duration_ms', 0):.0f}ms")

    order = result.get("order_result")
    if order:
        print(f"  Order ID: {order.get('order_id', 'N/A')}")
        print(f"  Fill Price: {order.get('fill_price', 'N/A')}")
        print(f"  Fill Size: {order.get('fill_size', 'N/A')}")
        if not order.get("success"):
            print(f"  Error: {order.get('error', '?')}")

    print(f"  Reasoning: {result.get('reasoning', '?')}")

    if result.get("error"):
        print(f"  ERROR: {result['error']}")

    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a TraderBot: analyze + execute")
    parser.add_argument("--symbol", default="BTC-USDC", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Pipeline only, no execution")
    parser.add_argument("--testnet", action="store_true", default=True,
                        help="Use testnet (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Use mainnet (requires confirmation)")
    args = parser.parse_args()

    # --live overrides --testnet
    use_testnet = not args.live

    asyncio.run(main(args.symbol, args.timeframe, args.verbose, use_testnet, args.dry_run))
