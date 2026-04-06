#!/usr/bin/env python3
"""Debug signal agents individually to isolate failures.

Fetches real MarketData, then runs EACH agent one at a time with full
traceback on failure. This isolates whether the issue is in the data,
the LLM provider, the chart generation, or the JSON parsing.

Usage:
    python scripts/debug_agents.py
    python scripts/debug_agents.py --symbol ETH-USDC --timeframe 4h
"""

from __future__ import annotations

import asyncio
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
            # Remove surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            else:
                # Strip inline # comments (only for unquoted values)
                if "#" in value:
                    value = value[:value.index("#")].strip()
            if key and key not in os.environ:
                os.environ[key] = value


async def main(symbol: str, timeframe: str) -> None:
    _load_env()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Debug Signal Agents — {symbol} {timeframe}")
    print(f"{'='*60}\n")

    # ── Step 1: Fetch MarketData ──
    print("=" * 60)
    print("STEP 1: Fetching MarketData")
    print("=" * 60)
    try:
        from engine.config import TradingConfig
        from engine.data.ohlcv import OHLCVFetcher
        from exchanges.factory import ExchangeFactory
        import exchanges.hyperliquid  # noqa: F401 — registers adapter

        config = TradingConfig(symbol=symbol, timeframe=timeframe)
        adapter = ExchangeFactory.get_adapter("hyperliquid")
        fetcher = OHLCVFetcher(adapter, config)

        market_data = await fetcher.fetch(symbol, timeframe)

        print(f"  Candles: {market_data.num_candles}")
        print(f"  Last close: {market_data.candles[-1]['close'] if market_data.candles else 'N/A'}")
        print(f"  Indicators keys: {list(market_data.indicators.keys())}")
        print(f"  Swing highs: {market_data.swing_highs[:3]}")
        print(f"  Swing lows: {market_data.swing_lows[:3]}")
        print(f"  Parent TF: {market_data.parent_tf}")
        print(f"  Flow: {market_data.flow}")
        print(f"  forecast_candles: {market_data.forecast_candles}")
        print(f"  forecast_description: '{market_data.forecast_description}'")
        print()

        # Inspect indicator structure in detail
        print("  Indicator detail:")
        for k, v in market_data.indicators.items():
            val_str = str(v)
            if len(val_str) > 100:
                val_str = val_str[:100] + "..."
            print(f"    {k}: {val_str}")
        print()

    except Exception:
        print(f"  FAILED to fetch MarketData:")
        traceback.print_exc()
        return

    # ── Step 2: Test grounding header ──
    print("=" * 60)
    print("STEP 2: Generate grounding header")
    print("=" * 60)
    try:
        from engine.data.charts import generate_grounding_header

        grounding = generate_grounding_header(
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            indicators=market_data.indicators,
            flow=market_data.flow,
            parent_tf=market_data.parent_tf,
            swing_highs=market_data.swing_highs,
            swing_lows=market_data.swing_lows,
            forecast_candles=market_data.forecast_candles,
            forecast_description=market_data.forecast_description,
            num_candles=market_data.num_candles,
            lookback_description=market_data.lookback_description,
        )
        print(f"  Grounding header ({len(grounding)} chars):")
        for line in grounding.split("\n"):
            print(f"    {line}")
        print()
    except Exception:
        print(f"  FAILED to generate grounding header:")
        traceback.print_exc()
        return

    # ── Step 3: Test chart generation ──
    print("=" * 60)
    print("STEP 3: Generate charts")
    print("=" * 60)
    candlestick_png = None
    trendline_png = None

    try:
        from engine.data.charts import generate_candlestick_chart
        candlestick_png = generate_candlestick_chart(
            candles=market_data.candles,
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
            swing_highs=market_data.swing_highs,
            swing_lows=market_data.swing_lows,
        )
        print(f"  Candlestick chart: {len(candlestick_png)} bytes")
    except Exception:
        print(f"  FAILED to generate candlestick chart:")
        traceback.print_exc()

    try:
        from engine.data.charts import generate_trendline_chart
        trendline_png = generate_trendline_chart(
            candles=market_data.candles,
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
        )
        print(f"  Trendline chart: {len(trendline_png)} bytes")
    except Exception:
        print(f"  FAILED to generate trendline chart:")
        traceback.print_exc()
    print()

    # ── Step 4: Test LLM provider ──
    print("=" * 60)
    print("STEP 4: Test LLM provider (quick text call)")
    print("=" * 60)
    try:
        from llm.claude import ClaudeProvider
        llm = ClaudeProvider(api_key=api_key)

        from llm.base import LLMResponse
        response = await llm.generate_text(
            system_prompt="You are a test. Respond with exactly: {\"status\": \"ok\"}",
            user_prompt="Ping",
            agent_name="debug_ping",
            max_tokens=50,
            temperature=0.0,
        )
        print(f"  LLM response: '{response.content[:200]}'")
        print(f"  Tokens: {response.input_tokens}/{response.output_tokens}")
        print(f"  Cost: ${response.cost:.4f}")
        print(f"  Latency: {response.latency_ms:.0f}ms")
    except Exception:
        print(f"  FAILED LLM call:")
        traceback.print_exc()
        print("\n  Cannot proceed without LLM. Stopping.")
        return
    print()

    # ── Step 5: Run each agent individually ──
    from engine.config import FeatureFlags
    flags = FeatureFlags()

    # 5a: IndicatorAgent
    print("=" * 60)
    print("STEP 5a: IndicatorAgent (text-only)")
    print("=" * 60)
    try:
        from engine.signals.indicator_agent import IndicatorAgent

        agent = IndicatorAgent(llm, flags)
        print(f"  Enabled: {agent.is_enabled()}")
        print(f"  Requires vision: {agent.requires_vision()}")

        # Build prompt manually to inspect it
        from engine.signals.prompts.indicator_v1 import SYSTEM_PROMPT, USER_PROMPT
        system = SYSTEM_PROMPT.format(grounding_header=grounding)
        user = USER_PROMPT.format(
            forecast_candles=market_data.forecast_candles,
            forecast_description=market_data.forecast_description,
            symbol=market_data.symbol,
            timeframe=market_data.timeframe,
        )
        print(f"  System prompt length: {len(system)} chars")
        print(f"  User prompt: '{user[:200]}'")

        result = await agent.analyze(market_data)
        if result is not None:
            print(f"  SUCCESS: {result.direction} (confidence={result.confidence:.2f})")
            print(f"  Reasoning: {result.reasoning[:200]}")
            print(f"  Contradictions: {result.contradictions}")
            print(f"  Raw output ({len(result.raw_output)} chars): {result.raw_output[:300]}")
        else:
            print(f"  RETURNED None — agent handled error internally")
            print(f"  Running LLM call manually to see raw response...")
            try:
                raw_response = await llm.generate_text(
                    system_prompt=system,
                    user_prompt=user,
                    agent_name="indicator_agent_debug",
                    max_tokens=512,
                    temperature=0.3,
                )
                print(f"  Raw LLM response:\n    {raw_response.content[:500]}")
            except Exception:
                print(f"  Manual LLM call also failed:")
                traceback.print_exc()
    except Exception:
        print(f"  EXCEPTION in IndicatorAgent:")
        traceback.print_exc()
    print()

    # 5b: PatternAgent
    print("=" * 60)
    print("STEP 5b: PatternAgent (vision)")
    print("=" * 60)
    try:
        from engine.signals.pattern_agent import PatternAgent

        agent = PatternAgent(llm, flags)
        print(f"  Enabled: {agent.is_enabled()}")
        print(f"  Chart available: {candlestick_png is not None and len(candlestick_png) > 0}")

        result = await agent.analyze(market_data)
        if result is not None:
            print(f"  SUCCESS: {result.direction} (confidence={result.confidence:.2f})")
            print(f"  Pattern: {result.pattern_detected}")
            print(f"  Reasoning: {result.reasoning[:200]}")
            print(f"  Contradictions: {result.contradictions}")
            print(f"  Raw output ({len(result.raw_output)} chars): {result.raw_output[:300]}")
        else:
            print(f"  RETURNED None — agent handled error internally")
            if candlestick_png:
                print(f"  Running vision LLM call manually...")
                try:
                    from engine.signals.prompts.pattern_v1 import SYSTEM_PROMPT as PAT_SYS, USER_PROMPT as PAT_USR
                    sys_prompt = PAT_SYS.format(grounding_header=grounding)
                    usr_prompt = PAT_USR.format(
                        symbol=market_data.symbol,
                        timeframe=market_data.timeframe,
                        forecast_candles=market_data.forecast_candles,
                        forecast_description=market_data.forecast_description,
                    )
                    raw_response = await llm.generate_vision(
                        system_prompt=sys_prompt,
                        user_prompt=usr_prompt,
                        image_data=candlestick_png,
                        image_media_type="image/png",
                        agent_name="pattern_agent_debug",
                        max_tokens=512,
                        temperature=0.3,
                    )
                    print(f"  Raw LLM response:\n    {raw_response.content[:500]}")
                except Exception:
                    print(f"  Manual vision call also failed:")
                    traceback.print_exc()
    except Exception:
        print(f"  EXCEPTION in PatternAgent:")
        traceback.print_exc()
    print()

    # 5c: TrendAgent
    print("=" * 60)
    print("STEP 5c: TrendAgent (vision)")
    print("=" * 60)
    try:
        from engine.signals.trend_agent import TrendAgent

        agent = TrendAgent(llm, flags)
        print(f"  Enabled: {agent.is_enabled()}")
        print(f"  Chart available: {trendline_png is not None and len(trendline_png) > 0}")

        result = await agent.analyze(market_data)
        if result is not None:
            print(f"  SUCCESS: {result.direction} (confidence={result.confidence:.2f})")
            print(f"  Reasoning: {result.reasoning[:200]}")
            print(f"  Contradictions: {result.contradictions}")
            print(f"  Raw output ({len(result.raw_output)} chars): {result.raw_output[:300]}")
        else:
            print(f"  RETURNED None — agent handled error internally")
            if trendline_png:
                print(f"  Running vision LLM call manually...")
                try:
                    from engine.signals.prompts.trend_v1 import SYSTEM_PROMPT as TRD_SYS, USER_PROMPT as TRD_USR
                    sys_prompt = TRD_SYS.format(grounding_header=grounding)
                    usr_prompt = TRD_USR.format(
                        symbol=market_data.symbol,
                        timeframe=market_data.timeframe,
                        forecast_candles=market_data.forecast_candles,
                        forecast_description=market_data.forecast_description,
                    )
                    raw_response = await llm.generate_vision(
                        system_prompt=sys_prompt,
                        user_prompt=usr_prompt,
                        image_data=trendline_png,
                        image_media_type="image/png",
                        agent_name="trend_agent_debug",
                        max_tokens=512,
                        temperature=0.3,
                    )
                    print(f"  Raw LLM response:\n    {raw_response.content[:500]}")
                except Exception:
                    print(f"  Manual vision call also failed:")
                    traceback.print_exc()
    except Exception:
        print(f"  EXCEPTION in TrendAgent:")
        traceback.print_exc()
    print()

    print("=" * 60)
    print("  Debug complete.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Debug signal agents individually")
    parser.add_argument("--symbol", default="BTC-USDC")
    parser.add_argument("--timeframe", default="1h")
    args = parser.parse_args()
    asyncio.run(main(args.symbol, args.timeframe))
