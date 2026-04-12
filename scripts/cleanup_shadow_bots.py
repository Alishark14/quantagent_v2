"""One-time cleanup: deduplicate shadow bots by symbol.

BTC, ETH, SOL have 3 bot entries each (30m, 1h, 4h) from prior shadow
runs. get_active_bots_by_mode("shadow") returns all of them, but the
system runs 1h only. The extra bots block the 1h bot via per-symbol
concurrency limits.

This script:
  1. For symbols with multiple active shadow bots: keeps ONLY the 1h
     bot, sets the rest to status='inactive'.
  2. Deactivates dead symbols: SNDK-USDC, USA500-USDC, XYZ100-USDC.
  3. Prints a before/after summary.

Idempotent — safe to run multiple times.

Usage:
    python scripts/cleanup_shadow_bots.py [--dry-run] [--backend sqlite]
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow running from repo root without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PREFERRED_TIMEFRAME = "1h"
DEAD_SYMBOLS = {"SNDK-USDC", "USA500-USDC", "XYZ100-USDC"}


async def run(dry_run: bool = False, backend: str = "postgresql") -> None:
    from storage.repositories import get_repositories

    os.environ.setdefault("DATABASE_BACKEND", backend)
    repos = await get_repositories()
    bot_repo = repos.bots

    # ── BEFORE snapshot ──
    all_shadow = await bot_repo.get_active_bots_by_mode("shadow")
    print(f"\n{'='*60}")
    print(f"BEFORE: {len(all_shadow)} active shadow bots")
    print(f"{'='*60}")

    by_symbol: dict[str, list[dict]] = {}
    for b in all_shadow:
        sym = b.get("symbol", "?")
        by_symbol.setdefault(sym, []).append(b)

    for sym in sorted(by_symbol):
        bots = by_symbol[sym]
        tfs = [b.get("timeframe", "?") for b in bots]
        marker = " ← DUPLICATE" if len(bots) > 1 else ""
        dead = " ← DEAD" if sym in DEAD_SYMBOLS else ""
        print(f"  {sym:20s}  {len(bots)} bot(s)  timeframes={tfs}{marker}{dead}")

    # ── Compute changes ──
    to_deactivate: list[dict] = []

    # 1. Dead symbols — deactivate all bots
    for sym in DEAD_SYMBOLS:
        for b in by_symbol.get(sym, []):
            to_deactivate.append(b)

    # 2. Duplicate symbols — keep only the preferred timeframe
    for sym, bots in by_symbol.items():
        if sym in DEAD_SYMBOLS:
            continue  # already handled
        if len(bots) <= 1:
            continue

        # Find the preferred bot (1h), keep it, deactivate the rest
        preferred = [b for b in bots if b.get("timeframe") == PREFERRED_TIMEFRAME]
        others = [b for b in bots if b.get("timeframe") != PREFERRED_TIMEFRAME]

        if not preferred:
            # No 1h bot — keep the first one, deactivate the rest
            preferred = [bots[0]]
            others = bots[1:]

        keeper = preferred[0]
        print(f"\n  {sym}: keeping {keeper['id']} ({keeper.get('timeframe')})")
        for b in others:
            print(f"  {sym}: deactivating {b['id']} ({b.get('timeframe')})")
            to_deactivate.append(b)
        # If multiple 1h bots somehow exist, keep only the first
        for b in preferred[1:]:
            print(f"  {sym}: deactivating extra 1h {b['id']}")
            to_deactivate.append(b)

    if not to_deactivate:
        print("\nNo changes needed — no duplicates or dead symbols found.")
        return

    # ── Show SQL ──
    print(f"\n{'='*60}")
    print(f"PLANNED: deactivate {len(to_deactivate)} bots")
    print(f"{'='*60}")
    for b in to_deactivate:
        print(f"  -- {b.get('symbol')} / {b.get('timeframe')}")
        print(f"  UPDATE bots SET is_active=false, deactivated_at=NOW() WHERE id='{b['id']}';")

    if dry_run:
        print("\n--dry-run: no changes applied.")
        return

    # ── Execute ──
    for b in to_deactivate:
        bot_id = b["id"]
        await bot_repo.deactivate_bot(bot_id)

    # ── AFTER snapshot ──
    after = await bot_repo.get_active_bots_by_mode("shadow")
    print(f"\n{'='*60}")
    print(f"AFTER: {len(after)} active shadow bots")
    print(f"{'='*60}")

    after_by_sym: dict[str, list[dict]] = {}
    for b in after:
        sym = b.get("symbol", "?")
        after_by_sym.setdefault(sym, []).append(b)

    for sym in sorted(after_by_sym):
        bots = after_by_sym[sym]
        tfs = [b.get("timeframe", "?") for b in bots]
        print(f"  {sym:20s}  {len(bots)} bot(s)  timeframes={tfs}")

    # Verify no duplicates remain
    dupes = [s for s, bs in after_by_sym.items() if len(bs) > 1]
    dead_remaining = [s for s in DEAD_SYMBOLS if s in after_by_sym]
    if dupes:
        print(f"\n⚠ DUPLICATES REMAIN: {dupes}")
    if dead_remaining:
        print(f"\n⚠ DEAD SYMBOLS REMAIN: {dead_remaining}")
    if not dupes and not dead_remaining:
        print(f"\n✓ Clean: {len(after)} unique symbols, no duplicates, no dead symbols.")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    be = "sqlite"
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            be = sys.argv[i + 1]
    asyncio.run(run(dry_run=dry, backend=be))
