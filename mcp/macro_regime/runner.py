"""Cron entry point for the Macro Regime Manager.

Usage::

    python -m mcp.macro_regime.runner --mode check
    python -m mcp.macro_regime.runner --mode deep
    python -m mcp.macro_regime.runner --mode emergency \\
            --trigger-symbols BTC-USDC,ETH-USDC

Suitable for cron::

    0 * * * * python -m mcp.macro_regime.runner --mode check    # hourly
    0 6 * * * python -m mcp.macro_regime.runner --mode deep     # daily

Modes (per ARCHITECTURE.md §13.2.1):

  * ``check``     — fetch + lightweight delta gate. If a trigger fires,
                    auto-escalates to ``deep`` for the same snapshot.
                    No LLM call when no trigger fires (~$0).
  * ``deep``      — fetch + full LLM assessment, writes macro_regime.json.
  * ``emergency`` — fetch + LLM assessment with urgency context, accepts
                    triggering symbols (--trigger-symbols).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from mcp.macro_regime.agent import MacroRegime, MacroRegimeManager
from mcp.macro_regime.data_fetcher import MacroDataFetcher, MacroSnapshot
from mcp.macro_regime.lightweight_check import CheckResult, LightweightCheck

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mcp.macro_regime.runner",
        description="Macro Regime Manager — fetch macro data, classify regime.",
    )
    parser.add_argument(
        "--mode",
        choices=("check", "deep", "emergency"),
        required=True,
        help="check (hourly delta gate), deep (daily LLM), emergency (swarm).",
    )
    parser.add_argument(
        "--trigger-symbols",
        default=None,
        help="Comma-separated triggering symbols for --mode emergency.",
    )
    parser.add_argument(
        "--output",
        default="macro_regime.json",
        help="Output JSON path (default macro_regime.json).",
    )
    parser.add_argument(
        "--snapshot-path",
        default="macro_regime_snapshot.json",
        help="Path used by LightweightCheck for the persisted prior snapshot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run analysis but DO NOT overwrite macro_regime.json.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


async def _run_deep_inner(
    args: argparse.Namespace,
    snapshot: MacroSnapshot,
    *,
    urgency: str,
    triggering_symbols: list[str] | None = None,
    reasons: list[str] | None = None,
) -> int:
    llm_provider = _build_llm_provider()
    if llm_provider is None:
        return 2
    agent = MacroRegimeManager(
        llm_provider=llm_provider,
        output_path=Path(args.output),
    )
    regime = await agent.run_deep(
        snapshot,
        urgency=urgency,
        triggering_symbols=triggering_symbols,
        reasons=reasons,
        dry_run=args.dry_run,
    )
    _print_deep_summary(regime, dry_run=args.dry_run)
    return 0 if regime.error is None else 1


# ---------------------------------------------------------------------------
# Lazy builders (monkeypatched in tests)
# ---------------------------------------------------------------------------


def _build_fetcher() -> MacroDataFetcher:
    """Construct the production MacroDataFetcher.

    Lazy so `--help` doesn't import httpx, and so tests can monkeypatch.
    """
    return MacroDataFetcher()


def _build_llm_provider():
    """Construct the production ClaudeProvider lazily."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set. Required for the "
            "Macro Regime Manager runner deep / emergency modes.",
            file=sys.stderr,
        )
        return None
    try:
        from llm.claude import ClaudeProvider
    except ImportError as e:
        print(f"ERROR: failed to import ClaudeProvider: {e}", file=sys.stderr)
        return None
    return ClaudeProvider(api_key=api_key)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_check_summary(result: CheckResult) -> None:
    print()
    print("=" * 72)
    print("MACRO REGIME MANAGER — Lightweight Check")
    print("=" * 72)
    if result.snapshot:
        srcs = sorted(result.snapshot.available_sources)
        print(f"Sources: {', '.join(srcs) if srcs else '(none)'}")
    if result.should_trigger_deep:
        print(f"Triggered: yes ({len(result.reasons)} reason(s))")
        for r in result.reasons:
            print(f"  - {r}")
    else:
        print("Triggered: no")
    print()


def _print_deep_summary(regime: MacroRegime, *, dry_run: bool) -> None:
    print()
    print("=" * 72)
    print("MACRO REGIME MANAGER — Deep Analysis")
    print("=" * 72)
    if regime.error:
        print(f"  ⚠️  RUN ERROR: {regime.error}")
    print(
        f"Regime: {regime.regime} (confidence: {regime.confidence:.2f}). "
        f"{len(regime.blackout_windows)} blackout window(s)."
    )
    if regime.reasoning:
        print(f"Reasoning: {regime.reasoning}")
    if dry_run:
        print(f"DRY RUN — would have written to {regime.output_path}")
    elif regime.error is None:
        print(f"Written to {regime.output_path}")
    print()


def _parse_symbols(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    """Top-level dispatcher — every mode shares the fetch step."""
    fetcher = _build_fetcher()
    snapshot = fetcher.fetch()

    if args.mode == "check":
        check = LightweightCheck(snapshot_path=Path(args.snapshot_path))
        result = check.run(snapshot)
        _print_check_summary(result)
        if not result.should_trigger_deep:
            return 0
        print("Lightweight check fired — escalating to Deep Analysis.")
        return await _run_deep_inner(
            args, snapshot, urgency="triggered", reasons=result.reasons
        )
    if args.mode == "deep":
        return await _run_deep_inner(args, snapshot, urgency="normal")
    if args.mode == "emergency":
        symbols = _parse_symbols(args.trigger_symbols)
        return await _run_deep_inner(
            args,
            snapshot,
            urgency="emergency",
            triggering_symbols=symbols,
        )
    print(f"ERROR: unknown mode {args.mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
