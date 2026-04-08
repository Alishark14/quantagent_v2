"""Auto-miner — turns live trading mistakes into pending eval scenarios.

Hand-labelling alone produces founder bias: you only test for setups
you remember to look for. The auto-miner closes the gap by scanning
recent live trades for two specific failure modes and packaging each
as a pre-filled scenario draft for human review.

Failure modes (per ARCHITECTURE.md §31.4.5):

1. **Overconfident disaster** — conviction > 0.85 but the trade lost
   money. The model was sure of a setup that didn't work. Freezing
   the input state lets us regression-test that the new prompt /
   model doesn't take the same bait again.

2. **Missed opportunity** — conviction < 0.5 (or skipped) but the
   forward price path shows a > 3R move in the would-be direction.
   The model was overly cautious on a setup it should have taken.

Drafts land in ``backtesting/evals/scenarios/auto_mined/pending_review/``
with the inputs frozen and the ``expected`` block left blank. The
founder reviews and either fills in expectations + promotes to
``promoted/`` (and wires up the manifest) or deletes.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from storage.repositories.base import BotRepository, TradeRepository


_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "scenarios" / "auto_mined" / "pending_review"
_DEFAULT_OVERCONFIDENCE_THRESHOLD = 0.85
_DEFAULT_LOW_CONVICTION_THRESHOLD = 0.50
_DEFAULT_MISSED_R_THRESHOLD = 3.0


# Trade fetchers can be sync or async; the auto-miner accepts both.
TradeFetcher = Callable[[], list[dict] | Awaitable[list[dict]]]


# ---------------------------------------------------------------------------
# Repository-backed fetcher
# ---------------------------------------------------------------------------


class RepositoryTradeFetcher:
    """Async fetcher that pulls closed trades from the trade repository.

    The auto-miner is duck-typed against any callable that returns a
    list of trade dicts; this class is the production implementation
    that talks to the real :class:`TradeRepository`. It iterates over
    every bot in the bot repository and concatenates each bot's recent
    trades, normalising the field names so the AutoMiner sees the same
    shape regardless of which storage backend produced them.

    The class is intentionally a callable (`__call__`) so it satisfies
    the ``TradeFetcher`` protocol — pass an instance directly to
    :class:`AutoMiner` like you would a lambda.
    """

    def __init__(
        self,
        trade_repo: "TradeRepository",
        bot_repo: "BotRepository | None" = None,
        bot_ids: list[str] | None = None,
        per_bot_limit: int = 200,
    ) -> None:
        """
        Args:
            trade_repo: Anything implementing TradeRepository.
            bot_repo: Optional BotRepository — when provided, the
                fetcher discovers bot ids dynamically. Either this or
                ``bot_ids`` must be set.
            bot_ids: Explicit list of bot ids to scan. Used when
                ``bot_repo`` is None.
            per_bot_limit: Max trades to pull per bot. Default 200 —
                covers ~30 days for an active bot at the standard
                cycle cadence.
        """
        if bot_repo is None and not bot_ids:
            raise ValueError(
                "RepositoryTradeFetcher requires either bot_repo or bot_ids"
            )
        self._trade_repo = trade_repo
        self._bot_repo = bot_repo
        self._bot_ids = list(bot_ids) if bot_ids else None
        self._per_bot_limit = int(per_bot_limit)

    async def __call__(self) -> list[dict]:
        """Return a flat list of normalised trade dicts across all bots."""
        bot_ids = await self._resolve_bot_ids()
        if not bot_ids:
            logger.info("RepositoryTradeFetcher: no bot ids to scan")
            return []

        all_trades: list[dict] = []
        for bot_id in bot_ids:
            try:
                trades = await self._trade_repo.get_trades_by_bot(
                    bot_id, limit=self._per_bot_limit
                )
            except Exception as e:
                logger.warning(
                    f"RepositoryTradeFetcher: get_trades_by_bot({bot_id}) "
                    f"failed: {e}"
                )
                continue
            for t in trades:
                all_trades.append(self._normalise(t))
        logger.info(
            f"RepositoryTradeFetcher: collected {len(all_trades)} trades "
            f"from {len(bot_ids)} bot(s)"
        )
        return all_trades

    async def _resolve_bot_ids(self) -> list[str]:
        if self._bot_ids:
            return self._bot_ids
        if self._bot_repo is None:
            return []
        # No `get_all_bots` on the ABC — but BotRepository has
        # `get_bots_by_user`. We can't enumerate users from the ABC, so
        # production callers should pass `bot_ids` explicitly when there
        # are multiple users. Single-tenant dev installs can override.
        try:
            bots = await self._bot_repo.get_bots_by_user("dev-user")
        except Exception as e:
            logger.warning(f"RepositoryTradeFetcher: bot discovery failed: {e}")
            return []
        return [b["id"] for b in bots if "id" in b]

    @staticmethod
    def _normalise(trade: dict) -> dict:
        """Map repo column names → the names the AutoMiner expects.

        The trades table uses ``conviction_score`` while the AutoMiner
        looks for ``conviction``; ``entry_time`` ISO string vs the
        miner's preferred ``entry_timestamp`` ms field; etc. Returns a
        new dict so the original record isn't mutated.
        """
        normalised = dict(trade)

        # conviction_score → conviction
        if "conviction" not in normalised and "conviction_score" in normalised:
            normalised["conviction"] = normalised["conviction_score"]

        # direction → action (the miner checks both)
        if "action" not in normalised and "direction" in normalised:
            normalised["action"] = normalised["direction"]

        # trade id under either name
        if "trade_id" not in normalised and "id" in normalised:
            normalised["trade_id"] = normalised["id"]

        # entry_timestamp (ms) — derive from entry_time ISO if missing
        if "entry_timestamp" not in normalised:
            entry_time = normalised.get("entry_time")
            if entry_time is not None:
                ts_ms = _iso_to_ms(entry_time)
                if ts_ms is not None:
                    normalised["entry_timestamp"] = ts_ms
        return normalised


def _iso_to_ms(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        cleaned = str(value).replace("Z", "+00:00")
        return int(datetime.fromisoformat(cleaned).timestamp() * 1000)
    except (TypeError, ValueError):
        return None


class AutoMiner:
    """Scans recent trades and writes pending-review scenarios to disk."""

    def __init__(
        self,
        trade_fetcher: TradeFetcher,
        output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
        overconfidence_threshold: float = _DEFAULT_OVERCONFIDENCE_THRESHOLD,
        low_conviction_threshold: float = _DEFAULT_LOW_CONVICTION_THRESHOLD,
        missed_r_threshold: float = _DEFAULT_MISSED_R_THRESHOLD,
    ) -> None:
        """
        Args:
            trade_fetcher: Sync or async callable returning a list of
                trade dicts. Loose coupling — keeps the auto-miner
                independent of the actual repository class.
            output_dir: Where to drop the JSON drafts.
            overconfidence_threshold: Conviction above which a losing
                trade is flagged as an overconfident disaster.
            low_conviction_threshold: Conviction at or below which a
                trade (or skipped setup) is a candidate for the
                missed-opportunity check.
            missed_r_threshold: Minimum R-multiple a forward path must
                hit for a low-conviction skip to count as "missed".
        """
        self._fetch = trade_fetcher
        self._output_dir = Path(output_dir)
        self._overconfidence_threshold = float(overconfidence_threshold)
        self._low_conviction_threshold = float(low_conviction_threshold)
        self._missed_r_threshold = float(missed_r_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_recent_trades(self, days: int = 7) -> list[dict]:
        """Return trade dicts from the last ``days`` days.

        The auto-miner doesn't filter — it just hands the full list
        back. Callers can filter further before scoring. The fetcher
        is responsible for the time window; ``days`` is informational
        and used by the cutoff helper.
        """
        result = self._fetch()
        if hasattr(result, "__await__"):
            trades: list[dict] = await result  # type: ignore[assignment]
        else:
            trades = list(result)  # type: ignore[arg-type]

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        cutoff_ms = int(cutoff.timestamp() * 1000)
        # If the trade has no timestamp, keep it — better than dropping data.
        return [
            t
            for t in trades
            if int(t.get("entry_timestamp") or t.get("timestamp") or 0) >= cutoff_ms
            or "entry_timestamp" not in t and "timestamp" not in t
        ]

    def find_overconfident_disasters(self, trades: list[dict]) -> list[dict]:
        """Conviction above threshold AND PnL ≤ 0."""
        out: list[dict] = []
        for t in trades:
            conviction = self._safe_float(t.get("conviction"))
            pnl = self._safe_float(t.get("pnl"))
            if conviction is None or pnl is None:
                continue
            if conviction >= self._overconfidence_threshold and pnl <= 0:
                out.append(t)
        return out

    def find_missed_opportunities(self, trades: list[dict]) -> list[dict]:
        """Low conviction (or skip) trades whose forward path showed ≥ N R.

        A trade dict qualifies if BOTH:
        - ``conviction`` ≤ ``low_conviction_threshold`` (or the trade
          was outright skipped, indicated by ``action == 'SKIP'``)
        - ``forward_max_r`` ≥ ``missed_r_threshold``

        ``forward_max_r`` is expected to be pre-computed by whatever
        component records the trade (e.g. tracking module). Trades
        without it are skipped — we don't compute forward paths here
        because the auto-miner is supposed to be cheap and synchronous.
        """
        out: list[dict] = []
        for t in trades:
            conviction = self._safe_float(t.get("conviction"))
            forward_max_r = self._safe_float(t.get("forward_max_r"))
            action = t.get("action") or t.get("direction")

            is_skip = isinstance(action, str) and action.upper() == "SKIP"
            is_low_conviction = (
                conviction is not None
                and conviction <= self._low_conviction_threshold
            )
            if not (is_skip or is_low_conviction):
                continue
            if forward_max_r is None or forward_max_r < self._missed_r_threshold:
                continue
            out.append(t)
        return out

    async def mine(self, days: int = 7) -> list[Path]:
        """Top-level: scan, classify, and write JSON drafts to disk.

        Returns the list of file paths written.
        """
        trades = await self.scan_recent_trades(days=days)
        disasters = self.find_overconfident_disasters(trades)
        missed = self.find_missed_opportunities(trades)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for trade in disasters:
            written.append(
                self._write_draft(trade, "overconfident_disaster", "trap_setups")
            )
        for trade in missed:
            written.append(
                self._write_draft(trade, "missed_opportunity", "clear_setups")
            )
        return written

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_draft(
        self, trade: dict, mining_reason: str, default_category: str
    ) -> Path:
        """Persist one trade as a Scenario JSON skeleton."""
        scenario_id = self._scenario_id(trade, mining_reason)
        path = self._output_dir / f"{scenario_id}.json"

        now = datetime.now(tz=timezone.utc).isoformat()
        draft = {
            "id": scenario_id,
            "name": f"[auto-mined] {mining_reason} on {trade.get('symbol', 'UNKNOWN')}",
            "category": default_category,
            "version": 1,
            "created_at": now,
            "last_validated": now,
            "inputs": {
                "symbol": trade.get("symbol", "UNKNOWN"),
                "timeframe": trade.get("timeframe", "1h"),
                "ohlcv": trade.get("ohlcv_at_entry") or [],
                "indicators": trade.get("indicators_at_entry") or {},
                "flow_data": trade.get("flow_data_at_entry"),
                "regime_context": trade.get("regime_at_entry"),
                "timestamp": (
                    self._isoformat_ms(trade.get("entry_timestamp"))
                    or self._isoformat_ms(trade.get("timestamp"))
                    or now
                ),
            },
            # PENDING REVIEW — labeller fills this in.
            "expected": {
                "expected_action": "SKIP",  # placeholder
                "key_features_to_mention": [],
                "notes": (
                    "AUTO-MINED — needs human labelling. Mining reason: "
                    f"{mining_reason}."
                ),
            },
            "metadata": {
                "auto_mined": True,
                "mining_reason": mining_reason,
                "source_trade_id": trade.get("trade_id") or trade.get("id"),
                "trade_pnl": trade.get("pnl"),
                "trade_conviction": trade.get("conviction"),
                "trade_forward_max_r": trade.get("forward_max_r"),
            },
        }
        path.write_text(json.dumps(draft, indent=2))
        logger.info(f"AutoMiner: wrote draft {path.name}")
        return path

    @staticmethod
    def _scenario_id(trade: dict, reason: str) -> str:
        """Build a deterministic, filesystem-safe scenario id."""
        symbol = str(trade.get("symbol", "unknown")).lower().replace("/", "_")
        tf = str(trade.get("timeframe", "tf")).lower()
        ts_raw = trade.get("entry_timestamp") or trade.get("timestamp") or 0
        try:
            ts_ms = int(ts_raw)
        except (TypeError, ValueError):
            ts_ms = 0
        slug = f"{reason}_{symbol}_{tf}_{ts_ms}"
        # Strip anything that isn't alnum, dash, or underscore.
        return re.sub(r"[^A-Za-z0-9_\-]+", "_", slug)

    @staticmethod
    def _safe_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _isoformat_ms(ts_ms) -> str | None:
        if ts_ms is None:
            return None
        try:
            return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return None
