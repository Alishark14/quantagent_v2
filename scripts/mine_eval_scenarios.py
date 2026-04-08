"""Auto-mine eval scenarios from recent live trades.

Scans the trade repository for two failure modes per
ARCHITECTURE.md §31.4.5:

* **Overconfident disasters** — conviction ≥ 0.85 AND PnL ≤ 0.
* **Missed opportunities** — conviction ≤ 0.5 (or skipped) AND
  forward_max_r ≥ 3.0.

Each match is written as a pre-filled scenario JSON skeleton to
``backtesting/evals/scenarios/auto_mined/pending_review/`` for human
review. Promotion to the permanent test suite (and to
``manifest.json``) is intentionally manual — the founder labels and
moves them.

Usage:
    python scripts/mine_eval_scenarios.py --days 7
    python scripts/mine_eval_scenarios.py --days 30 --bot-id dev-btc-1h
    python scripts/mine_eval_scenarios.py --days 7 --backend sqlite
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow `python scripts/mine_eval_scenarios.py` to find the project
# packages without needing PYTHONPATH or `python -m`. The script lives
# at <repo>/scripts/, so the repo root is one level up.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtesting.evals.auto_miner import AutoMiner, RepositoryTradeFetcher  # noqa: E402

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/mine_eval_scenarios.py",
        description="Mine eval scenarios from recent live trades.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="How many days of recent trades to scan (default 7).",
    )
    parser.add_argument(
        "--bot-id",
        action="append",
        default=None,
        help=(
            "Bot id to scan. Repeatable: --bot-id A --bot-id B. "
            "When omitted the script enumerates dev-user bots via the bot repo."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("sqlite", "postgresql"),
        default=None,
        help="Database backend (default: read DATABASE_BACKEND env var).",
    )
    parser.add_argument(
        "--per-bot-limit",
        type=int,
        default=200,
        help="Max trades per bot to fetch (default 200).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the pending_review directory (default: package default).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    from storage.repositories import get_repositories

    if args.days < 1:
        print(f"ERROR: --days must be >= 1, got {args.days}", file=sys.stderr)
        return 2

    repos = await get_repositories(args.backend)
    try:
        fetcher = RepositoryTradeFetcher(
            trade_repo=repos.trades,
            bot_repo=repos.bots if not args.bot_id else None,
            bot_ids=args.bot_id,
            per_bot_limit=args.per_bot_limit,
        )

        miner_kwargs: dict = {"trade_fetcher": fetcher}
        if args.output_dir:
            miner_kwargs["output_dir"] = Path(args.output_dir)
        miner = AutoMiner(**miner_kwargs)

        # We need separate counts for the summary line, so call the
        # internal classify steps directly instead of just `mine()`.
        trades = await miner.scan_recent_trades(days=args.days)
        disasters = miner.find_overconfident_disasters(trades)
        missed = miner.find_missed_opportunities(trades)
        written = await miner.mine(days=args.days)
    finally:
        if hasattr(repos, "close"):
            await repos.close()

    _print_summary(
        scanned=len(trades),
        days=args.days,
        disasters=len(disasters),
        missed=len(missed),
        output_dir=miner._output_dir,  # noqa: SLF001 — internal but stable
        written=written,
    )
    return 0


def _print_summary(
    *,
    scanned: int,
    days: int,
    disasters: int,
    missed: int,
    output_dir: Path,
    written: list[Path],
) -> None:
    print()
    print("=" * 72)
    print("AUTO-MINE EVAL SCENARIOS")
    print("=" * 72)
    print(
        f"Scanned {scanned} trade(s) from the last {days} day(s). "
        f"Found {disasters} overconfident disaster(s), {missed} missed opportunity(ies)."
    )
    print(f"Drafts saved to {output_dir}/")
    if not written:
        print("No scenarios mined.")
        print()
        return

    print()
    print("Drafts written:")
    for path in written:
        reason = _infer_reason(path.stem)
        print(f"  - {path.name}  ({reason})")
    print()


def _infer_reason(stem: str) -> str:
    if stem.startswith("overconfident_disaster"):
        return "overconfident_disaster"
    if stem.startswith("missed_opportunity"):
        return "missed_opportunity"
    return "auto_mined"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
