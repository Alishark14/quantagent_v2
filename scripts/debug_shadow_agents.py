#!/usr/bin/env python3
"""Diagnose why LLM signal agents return None in production shadow mode.

This script mirrors the EXACT wiring path that the production shadow
runner uses: _make_bot_factory → AnalysisPipeline → SignalRegistry.
It does NOT wire things manually like run_testnet_cycle.py or
debug_agents.py — the whole point is to exercise the SAME code path
that the scheduled loop takes, so any divergence in MarketData
construction, LLM provider threading, or feature flag resolution is
caught.

Usage:
    python scripts/debug_shadow_agents.py
    python scripts/debug_shadow_agents.py --symbol ETH-USDC --timeframe 4h

Required env:
    ANTHROPIC_API_KEY    — Claude credentials
    DATABASE_URL         — or falls back to sqlite

The script:
  1. Constructs infrastructure the same way _run_server() does
  2. Calls _make_bot_factory to get the same factory closure
  3. Invokes factory(symbol, bot_id) to build a TraderBot
  4. Extracts the pipeline's signal_registry and market_data fetcher
  5. Fetches MarketData through the pipeline's OHLCVFetcher
  6. Calls each registered agent individually with verbose output
  7. Reports which agents succeed and which return None, and why
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import sys
import traceback
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


def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


async def main(symbol: str, timeframe: str) -> int:
    _load_env()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpcore", "anthropic", "ccxt", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return 1

    _banner(f"Shadow Agent Diagnostic — {symbol} {timeframe}")

    # ── Step 1: Build infrastructure EXACTLY as _run_server() does ──
    _banner("Step 1: Build infrastructure (mirrors _run_server)")

    from engine.config import FeatureFlags
    from engine.events import create_event_bus
    from llm.claude import ClaudeProvider
    from storage.repositories import get_repositories

    repos = await get_repositories()
    event_bus = create_event_bus("memory")
    feature_flags = FeatureFlags()

    print(f"  Feature flags loaded:")
    for flag, val in feature_flags.all_flags().items():
        # Also check env override
        env_key = f"FEATURE_{flag.upper()}"
        env_val = os.environ.get(env_key)
        effective = feature_flags.is_enabled(flag)
        override = f" (env override: {env_key}={env_val})" if env_val is not None else ""
        print(f"    {flag}: yaml={val}, effective={effective}{override}")

    # ── Step 2: Construct LLM provider the same way main.py does ──
    _banner("Step 2: LLM provider (same as main.py)")
    llm_provider = ClaudeProvider(api_key=api_key) if api_key else None
    print(f"  llm_provider = {llm_provider}")
    print(f"  type = {type(llm_provider).__name__}")
    print(f"  model = {getattr(llm_provider, 'model', 'N/A')}")

    # Quick connectivity test
    if llm_provider:
        try:
            resp = await llm_provider.generate_text(
                system_prompt="Respond with exactly: ok",
                user_prompt="ping",
                agent_name="debug_ping",
                max_tokens=10,
                temperature=0.0,
            )
            print(f"  LLM ping: '{resp.content}' ({resp.latency_ms:.0f}ms)")
        except Exception as e:
            print(f"  LLM ping FAILED: {e}")
            return 1

    # ── Step 3: Build bot via _make_bot_factory (THE production path) ──
    _banner("Step 3: _make_bot_factory → factory(symbol, bot_id)")

    from quantagent.main import _make_bot_factory

    def adapter_factory(exchange: str, mode: str = "live"):
        import exchanges.hyperliquid  # noqa: F401
        from exchanges.factory import ExchangeFactory
        return ExchangeFactory.get_adapter(exchange, mode=mode)

    bot_factory = _make_bot_factory(
        repos=repos,
        llm_provider=llm_provider,
        adapter_factory=adapter_factory,
        event_bus=event_bus,
        feature_flags=feature_flags,
        shadow_mode=True,
        paper_mode=False,
    )

    bot_id = "debug-shadow-diag"
    bot = bot_factory(symbol, bot_id)
    pipeline = bot._pipeline

    print(f"  Bot created: {bot.bot_id}")
    print(f"  Pipeline config: {pipeline._config.symbol} / {pipeline._config.timeframe}")
    print(f"  Pipeline is_shadow: {pipeline._is_shadow}")
    print(f"  Pipeline PRM: {pipeline._prm}")

    # ── Step 4: Inspect what the factory registered in the SignalRegistry ──
    _banner("Step 4: SignalRegistry contents")
    registry = pipeline._signal_registry
    all_producers = registry._producers
    enabled_producers = registry.get_enabled()

    print(f"  Total registered producers: {len(all_producers)}")
    print(f"  Enabled producers: {len(enabled_producers)}")
    for p in all_producers:
        llm_attr = getattr(p, '_llm', 'N/A')
        print(
            f"    {p.name():<25} type={p.signal_type():<5} "
            f"enabled={p.is_enabled():<5} vision={p.requires_vision():<5} "
            f"_llm={type(llm_attr).__name__}"
        )
        # THE KEY CHECK: is the LLM provider actually set?
        if hasattr(p, '_llm'):
            if p._llm is None:
                print(f"      *** _llm IS NONE — THIS IS THE BUG ***")
            elif not hasattr(p._llm, 'generate_text'):
                print(f"      *** _llm has no generate_text method — wrong type ***")

    # ── Step 5: Fetch MarketData via the pipeline's OHLCVFetcher ──
    _banner("Step 5: Fetch MarketData (pipeline's OHLCVFetcher)")
    try:
        fetcher = pipeline._ohlcv
        market_data = await fetcher.fetch(symbol, timeframe)

        print(f"  symbol: {market_data.symbol}")
        print(f"  timeframe: {market_data.timeframe}")
        print(f"  candles: {len(market_data.candles) if market_data.candles else 0}")
        print(f"  indicators: {bool(market_data.indicators)} "
              f"(keys: {list(market_data.indicators.keys()) if market_data.indicators else []})")
        print(f"  swing_highs: {len(market_data.swing_highs)} values")
        print(f"  swing_lows: {len(market_data.swing_lows)} values")
        print(f"  parent_tf: {market_data.parent_tf is not None}")
        print(f"  flow: {market_data.flow}")
        print(f"  forecast_candles: {market_data.forecast_candles}")
        print(f"  forecast_description: '{market_data.forecast_description}'")
        print(f"  num_candles: {market_data.num_candles}")
        print(f"  lookback_description: '{market_data.lookback_description}'")

        if not market_data.candles:
            print("  *** NO CANDLES — all agents will return None ***")
        if not market_data.indicators:
            print("  *** NO INDICATORS — all LLM agents will return None ***")

        # Also fetch flow (as the pipeline does in run_cycle)
        try:
            from engine.data.flow import FlowAgent
            from engine.data.flow.crypto import CryptoFlowProvider
            flow_agent = FlowAgent([CryptoFlowProvider()])
            flow = await flow_agent.fetch_flow(symbol, fetcher._adapter)
            market_data.flow = flow
            print(f"  flow (after enrichment): funding={getattr(flow, 'funding_rate', None)}, "
                  f"oi={getattr(flow, 'open_interest', None)}")
        except Exception as e:
            print(f"  Flow enrichment failed (non-fatal): {e}")

    except Exception:
        print("  FAILED to fetch MarketData:")
        traceback.print_exc()
        return 1

    # ── Step 6: Test chart generation (vision agents need this) ──
    _banner("Step 6: Chart generation")
    try:
        from engine.data.charts import generate_candlestick_chart, generate_trendline_chart

        candlestick_png = generate_candlestick_chart(
            candles=market_data.candles,
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            swing_highs=market_data.swing_highs,
            swing_lows=market_data.swing_lows,
        )
        print(f"  Candlestick chart: {len(candlestick_png) if candlestick_png else 0} bytes")
        if not candlestick_png:
            print("  *** EMPTY — PatternAgent will return None ***")

        trendline_png = generate_trendline_chart(
            candles=market_data.candles,
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
        )
        print(f"  Trendline chart: {len(trendline_png) if trendline_png else 0} bytes")
        if not trendline_png:
            print("  *** EMPTY — TrendAgent will return None ***")

    except Exception:
        print("  Chart generation FAILED:")
        traceback.print_exc()

    # ── Step 7: Call each agent individually with the pipeline's MarketData ──
    _banner("Step 7: Call each agent individually")

    results: dict[str, str] = {}

    for producer in enabled_producers:
        name = producer.name()
        print(f"\n  --- {name} ---")
        try:
            result = await producer.analyze(market_data)
            if result is not None:
                results[name] = f"OK: {result.direction} (conf={result.confidence:.2f})"
                print(f"  RESULT: {result.direction} confidence={result.confidence:.2f}")
                print(f"  Reasoning: {result.reasoning[:200]}")
            else:
                results[name] = "RETURNED None (check DIAG logs above)"
                print(f"  RETURNED None — check DIAG log lines above for the exact return site")
        except Exception as exc:
            results[name] = f"EXCEPTION: {type(exc).__name__}: {exc}"
            print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    # ── Step 8: Also run via registry.run_all() to match production ──
    _banner("Step 8: registry.run_all() (production path)")
    try:
        signals = await registry.run_all(market_data)
        print(f"  run_all returned {len(signals)} signals out of {len(enabled_producers)} producers")
        for sig in signals:
            print(f"    {sig.agent_name}: {sig.direction} (conf={sig.confidence:.2f})")
    except Exception:
        print("  run_all FAILED:")
        traceback.print_exc()

    # ── Summary ──
    _banner("SUMMARY")
    for name, outcome in results.items():
        status = "PASS" if outcome.startswith("OK") else "FAIL"
        print(f"  [{status}] {name:<25} {outcome}")

    print()
    failing = [n for n, o in results.items() if not o.startswith("OK")]
    if not failing:
        print("  All agents produced signals. The bug may be in the scheduled")
        print("  loop wiring (BotRunner → BotManager → factory) rather than")
        print("  in the agents themselves. Compare the DIAG logs from this")
        print("  script with production logs.")
    else:
        print(f"  {len(failing)} agent(s) returned None: {failing}")
        print("  Check the DIAG log lines above — each return None site is tagged")
        print("  with a line number and reason.")

    print()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnose LLM signal agent failures in shadow mode. "
                    "Mirrors _make_bot_factory wiring exactly."
    )
    parser.add_argument("--symbol", default="BTC-USDC", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    args = parser.parse_args()
    rc = asyncio.run(main(args.symbol, args.timeframe))
    sys.exit(rc)
