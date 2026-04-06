#!/usr/bin/env python3
"""Run a single analysis cycle end-to-end.

This is a manual verification script — real data, real LLM calls, real signals.
NOT a test. Requires API keys in .env.

Usage:
    python scripts/run_cycle.py --symbol BTC-USDC --timeframe 1h
    python scripts/run_cycle.py --symbol ETH-USDC --timeframe 4h
    python scripts/run_cycle.py --symbol SOL-USDC --timeframe 15m --verbose

Requirements:
    ANTHROPIC_API_KEY in .env
    HYPERLIQUID_WALLET_ADDRESS in .env (read-only is fine, no trading)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_env() -> None:
    """Load .env file from project root (simple parser, no dependency)."""
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
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                # Strip inline # comments (only for unquoted values)
                if "#" in value:
                    value = value[:value.index("#")].strip()
            if key and key not in os.environ:
                os.environ[key] = value


async def main(symbol: str, timeframe: str, verbose: bool, testnet: bool) -> None:
    # Load .env before any imports that read env vars
    _load_env()

    # Testnet override
    if testnet:
        os.environ["HYPERLIQUID_TESTNET"] = "true"

    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Validate required env vars
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    net_label = "TESTNET" if testnet else "MAINNET"
    print(f"\n{'='*60}")
    print(f"  QuantAgent v2 — Analysis Cycle [{net_label}]")
    print(f"  Symbol: {symbol} | TF: {timeframe}")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # ── Initialize components ──
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
    from exchanges.factory import ExchangeFactory
    from llm.claude import ClaudeProvider
    from storage.repositories import get_repositories

    config = TradingConfig(symbol=symbol, timeframe=timeframe)

    # Event bus with logging subscribers
    bus = InProcessBus()
    bus.subscribe(DataReady, lambda e: print(f"  [EVENT] DataReady — {e.market_data.num_candles} candles"))
    bus.subscribe(SignalsReady, lambda e: print(f"  [EVENT] SignalsReady — {len(e.signals)} signals"))
    bus.subscribe(ConvictionScored, lambda e: print(
        f"  [EVENT] ConvictionScored — {e.conviction.conviction_score:.2f} "
        f"{e.conviction.direction} ({e.conviction.regime})"
    ))
    bus.subscribe(CycleCompleted, lambda e: print(
        f"  [EVENT] CycleCompleted — {e.action} (conviction={e.conviction:.2f})"
    ))

    # Exchange adapter
    print("Initializing Hyperliquid adapter...")
    # Import to trigger factory registration
    import exchanges.hyperliquid  # noqa: F401
    adapter = ExchangeFactory.get_adapter("hyperliquid")

    # LLM provider
    print("Initializing Claude provider...")
    llm = ClaudeProvider(api_key=api_key)

    # Repositories (SQLite for dev)
    repos = await get_repositories("sqlite")

    # Data layer
    fetcher = OHLCVFetcher(adapter, config)
    flow_agent = FlowAgent([CryptoFlowProvider()])

    # Signal layer
    registry = SignalRegistry()
    flags = FeatureFlags()
    registry.register(IndicatorAgent(llm, flags))
    registry.register(PatternAgent(llm, flags))
    registry.register(TrendAgent(llm, flags))

    # Conviction + Decision
    conviction_agent = ConvictionAgent(llm)
    decision_agent = DecisionAgent(llm, config)

    # Memory
    cycle_mem = CycleMemory(repos.cycles)
    rules = ReflectionRules(repos.rules)
    cross_bot = CrossBotSignals(repos.cross_bot)
    regime = RegimeHistory()

    # Pipeline
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
        bot_id="test-bot-001",
        user_id="dev-user",
    )

    # ── Run cycle ──
    print("\nRunning analysis cycle...\n")
    action = await pipeline.run_cycle()

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"  RESULT: {action.action}")
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

    # Print cost summary from event bus metrics
    print(f"\n  Pipeline events: {bus.total_published}")
    for event_name, count in bus.per_type_counts.items():
        print(f"    {event_name}: {count}")
    if bus.handler_errors:
        print(f"    Handler errors: {bus.handler_errors}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single QuantAgent v2 analysis cycle")
    parser.add_argument("--symbol", default="BTC-USDC", help="Trading symbol (default: BTC-USDC)")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--testnet", action="store_true", help="Use Hyperliquid testnet")
    args = parser.parse_args()
    asyncio.run(main(args.symbol, args.timeframe, args.verbose, args.testnet))
