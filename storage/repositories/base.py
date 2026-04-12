"""Abstract repository interfaces for all database access.

Engine code NEVER touches SQL directly — all DB access goes through these interfaces.
PostgreSQL is the standard backend. SQLite exists only for local dev fallback.
"""

from abc import ABC, abstractmethod


class TradeRepository(ABC):
    """Repository for trade lifecycle records.

    Shadow-mode contract: every read method that returns a list of
    trades takes an optional ``include_shadow: bool = False`` keyword.
    Production callers leave it at the default and never see shadow
    rows. The shadow analytics paths opt in explicitly. Writes accept
    an ``is_shadow`` key in the input dict (or via the
    ``is_shadow`` kwarg on ``save_trade``) and persist it. The flag is
    NOT inferred from the bot record at the repo layer — the caller is
    responsible for passing the bot's mode through, because looking it
    up here would couple TradeRepository to BotRepository and force a
    second query on every write.
    """

    @abstractmethod
    async def save_trade(self, trade: dict) -> str:
        """Save a trade record. Returns the trade ID.

        Reads ``trade.get("is_shadow", False)`` and persists it on the
        new ``is_shadow`` column.
        """
        ...

    @abstractmethod
    async def get_trade(self, trade_id: str) -> dict | None:
        """Get a trade by ID. Returns None if not found.

        Single-row lookup by primary key — does NOT filter by shadow
        flag because the caller already knows the trade's identity.
        """
        ...

    @abstractmethod
    async def get_open_positions(
        self, user_id: str, bot_id: str, *, include_shadow: bool = False
    ) -> list[dict]:
        """Get open positions filtered by user_id AND bot_id.

        Defaults to live-only (``is_shadow = false``) so production
        position-sync never sees shadow virtual fills. Pass
        ``include_shadow=True`` from shadow-aware code paths only.
        """
        ...

    @abstractmethod
    async def get_trades_by_bot(
        self, bot_id: str, limit: int = 50, *, include_shadow: bool = False
    ) -> list[dict]:
        """Get recent trades for a bot, ordered by entry_time descending.

        Defaults to live-only.
        """
        ...

    @abstractmethod
    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        """Update trade fields. Returns True if trade was found and updated."""
        ...

    @abstractmethod
    async def get_open_shadow_trades(self, symbol: str) -> list[dict]:
        """Return all open shadow trades for ``symbol``.

        Used by the Sentinel SL/TP monitor to find positions that need
        their high/low watched on each scan. Filters by
        ``status='open' AND is_shadow=true AND symbol=?``. The
        complementary live-trade scan is owned by the exchange (native
        SL/TP orders), so this method intentionally returns shadow rows
        only — there is no live equivalent.
        """
        ...

    @abstractmethod
    async def close_trade(
        self,
        trade_id: str,
        *,
        exit_price: float,
        exit_reason: str,
        exit_time,
        pnl: float,
    ) -> bool:
        """Close an open trade with exit details + computed PnL.

        Sets ``status='closed'`` and stamps the four exit fields. The
        caller computes PnL because the formula depends on direction
        (long vs short) and the repo doesn't fetch the row first to
        avoid an extra round-trip.
        """
        ...


class OISnapshotRepository(ABC):
    """Repository for the rolling per-symbol open-interest history.

    The CryptoFlowProvider keeps an in-memory deque of recent OI
    snapshots per symbol so it can compute ``oi_change_*`` deltas
    locally. The deque resets on every process restart, which means
    every fresh boot pays a multi-hour cold-start penalty before
    FlowSignalAgent's BUILDING / DROPPING / divergence rules can fire.
    Persisting snapshots to a small dedicated table eliminates that
    penalty: on startup the provider bulk-loads the recent window from
    the table into its deques, and on every fetch it writes the new
    snapshot back. A periodic cleanup keeps the table small.
    """

    @abstractmethod
    async def insert_snapshot(
        self, symbol: str, timestamp, oi_value: float
    ) -> None:
        """Insert a single (symbol, timestamp, oi_value) row.

        Implementations must be idempotent on the (symbol, timestamp)
        primary key — `ON CONFLICT DO NOTHING` semantics so concurrent
        writers don't trip duplicate-key errors.
        """
        ...

    @abstractmethod
    async def get_recent_snapshots(
        self, lookback_seconds: int
    ) -> list[dict]:
        """Return rows newer than ``now - lookback_seconds``.

        Each row is a dict with keys ``symbol``, ``timestamp`` (epoch
        seconds, float), ``oi_value`` (float). Ordered by symbol then
        timestamp ascending so the caller can rebuild per-symbol deques
        in chronological order with a single linear scan.
        """
        ...

    @abstractmethod
    async def cleanup_older_than(self, seconds: int) -> int:
        """Delete snapshots older than ``now - seconds``. Returns row count."""
        ...


class COTCacheRepository(ABC):
    """Persistent cache for weekly CFTC Commitment of Traders snapshots.

    CFTC publishes COT data once a week (Friday evening for data as of
    the prior Tuesday). The cot_reports library can be flaky — the
    CFTC ZIP server sometimes 503s and a transient failure on a
    restart would leave ``CommodityFlowProvider`` blind until the next
    successful fetch. Persisting the weekly snapshots means:

    * Warmup on process boot rebuilds the 52-week history without a
      single network call.
    * A live fetch failure falls back transparently to the stored
      history — the provider still serves the most recent snapshot.
    * The percentile / divergence math runs off a rolling window that
      survives restarts.
    """

    @abstractmethod
    async def upsert(
        self,
        symbol: str,
        report_date,
        *,
        managed_money_net: float | None,
        commercial_net: float | None,
        total_oi: float | None,
        raw_json: str | None = None,
    ) -> None:
        """Insert or update a single (symbol, report_date) row.

        Implementations must be idempotent on the composite primary
        key (ON CONFLICT DO UPDATE) so re-running a weekly fetch
        against a row we already have refreshes the stored values
        instead of failing.
        """
        ...

    @abstractmethod
    async def get_recent(
        self, symbol: str, limit: int = 52
    ) -> list[dict]:
        """Return up to ``limit`` most recent rows for ``symbol``.

        Rows ascending by report_date so the caller can feed the list
        straight into a rolling percentile/divergence computation.
        Each row is a dict with keys ``symbol``, ``report_date``
        (datetime.date), ``managed_money_net``, ``commercial_net``,
        ``total_oi``, ``raw_json``.
        """
        ...

    @abstractmethod
    async def cleanup_older_than_weeks(self, weeks: int) -> int:
        """Delete rows older than ``weeks`` weeks. Returns rows deleted."""
        ...


class RegSHOCacheRepository(ABC):
    """Persistent cache for daily FINRA RegSHO short-volume snapshots.

    FINRA publishes a daily off-exchange short-volume file for every
    US equity; ``EquityFlowProvider`` pulls that file once per trading
    day and computes a 20-day Z-score / trend off the rolling history.
    Persisting the per-(symbol, trade_date) rows means:

    * Warmup on process boot rebuilds the 20-day window without a
      single network call, so the first analysis cycle after restart
      already has Z-scores for TSLA / NVDA / GOOGL.
    * A live FINRA fetch failure (holiday, server outage, network
      blip) falls back transparently to the stored history.
    * The same table can be queried by the analytics layer to
      backtest SVR signals without touching FINRA again.
    """

    @abstractmethod
    async def upsert(
        self,
        symbol: str,
        trade_date,
        *,
        short_volume: int | None,
        total_volume: int | None,
        short_volume_ratio: float | None,
    ) -> None:
        """Insert or update one (symbol, trade_date) row.

        Implementations must be idempotent on the composite PK so the
        same day's fetch can be replayed without duplicate-key errors.
        """
        ...

    @abstractmethod
    async def get_recent(
        self, symbol: str, limit: int = 20
    ) -> list[dict]:
        """Return up to ``limit`` most-recent rows for ``symbol``.

        Rows come back ASCENDING by trade_date so the caller can feed
        them straight into a rolling Z-score / trend computation.
        Each dict: ``symbol``, ``trade_date`` (datetime.date),
        ``short_volume``, ``total_volume``, ``short_volume_ratio``.
        """
        ...

    @abstractmethod
    async def cleanup_older_than_days(self, days: int) -> int:
        """Delete rows older than ``days`` days. Returns rows deleted."""
        ...


class CycleRepository(ABC):
    """Repository for analysis cycle records.

    Shadow-mode contract mirrors :class:`TradeRepository`: read methods
    take ``include_shadow: bool = False`` and default to live-only;
    writes accept ``is_shadow`` in the input dict.
    """

    @abstractmethod
    async def save_cycle(self, cycle: dict) -> str:
        """Save a cycle record. Returns the cycle ID.

        Reads ``cycle.get("is_shadow", False)`` and persists it.
        """
        ...

    @abstractmethod
    async def get_recent_cycles(
        self, bot_id: str, limit: int = 5, *, include_shadow: bool = False
    ) -> list[dict]:
        """Get recent cycles for a bot, ordered by timestamp descending.

        Defaults to live-only.
        """
        ...


class RuleRepository(ABC):
    """Repository for reflection rules (per-asset, per-timeframe)."""

    @abstractmethod
    async def save_rule(self, rule: dict) -> str:
        """Save a reflection rule. Returns the rule ID."""
        ...

    @abstractmethod
    async def get_rules(self, symbol: str, timeframe: str) -> list[dict]:
        """Get active rules for a symbol+timeframe combination."""
        ...

    @abstractmethod
    async def update_rule_score(self, rule_id: str, delta: int) -> bool:
        """Adjust rule score by delta. Deactivates if score drops below -2.
        Returns True if rule was found and updated."""
        ...

    @abstractmethod
    async def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate a rule. Returns True if rule was found."""
        ...


class BotRepository(ABC):
    """Repository for bot configuration and health."""

    @abstractmethod
    async def save_bot(self, bot: dict) -> str:
        """Save a bot record. Returns the bot ID."""
        ...

    @abstractmethod
    async def get_bot(self, bot_id: str) -> dict | None:
        """Get a bot by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def get_bots_by_user(self, user_id: str) -> list[dict]:
        """Get all bots for a user."""
        ...

    @abstractmethod
    async def get_active_bots(self, *, include_shadow: bool = False) -> list[dict]:
        """Get all bots with status='active' across all users.

        Used by BotRunner on startup to restore state without
        requiring re-registration via API. Defaults to live-only —
        shadow bots are excluded unless ``include_shadow=True`` is
        passed. Production startup leaves the default; shadow-mode
        startup uses :meth:`get_active_bots_by_mode` instead.
        """
        ...

    @abstractmethod
    async def get_active_bots_by_mode(self, mode: str) -> list[dict]:
        """Get all active bots whose ``mode`` column equals ``mode``.

        Used by the shadow-mode redesign so a single BotRunner can be
        booted in either mode (``"live"`` or ``"shadow"``) and load
        only the bots that belong to its mode without inverse
        filtering. Returns rows with ``status='active' AND mode=?``.
        """
        ...

    @abstractmethod
    async def update_bot_health(self, bot_id: str, health: dict) -> bool:
        """Update bot health snapshot. Returns True if bot was found."""
        ...

    @abstractmethod
    async def deactivate_bot(self, bot_id: str) -> bool:
        """Set ``is_active=false`` and ``deactivated_at=now()``.

        Returns True if the bot was found and deactivated. Idempotent —
        calling on an already-deactivated bot is a no-op that returns True.
        """
        ...

    @abstractmethod
    async def update_last_cycle(self, bot_id: str) -> bool:
        """Stamp ``last_cycle_at=now()`` on the bot row.

        Called after every analysis cycle so the dedup logic can prefer
        the most-recently-active bot when duplicates exist.
        """
        ...


class SentinelEventRepository(ABC):
    """Repository for sentinel setup/skip event analytics."""

    @abstractmethod
    async def insert_event(self, event: dict) -> None:
        """Insert a sentinel event. Fire-and-forget — callers swallow errors."""
        ...


class LLMCallRepository(ABC):
    """Repository for per-call LLM analytics."""

    @abstractmethod
    async def insert_call(self, call: dict) -> None:
        """Insert an LLM call record. Fire-and-forget — callers swallow errors."""
        ...

    @abstractmethod
    async def get_calls_by_agent(
        self, agent_name: str, limit: int = 100
    ) -> list[dict]:
        """Return recent calls for an agent, newest first."""
        ...


class CrossBotRepository(ABC):
    """Repository for cross-bot signal sharing (user_id scoped)."""

    @abstractmethod
    async def save_signal(self, signal: dict) -> None:
        """Save a cross-bot signal."""
        ...

    @abstractmethod
    async def get_recent_signals(
        self, symbol: str, user_id: str, limit: int = 10
    ) -> list[dict]:
        """Get recent signals for a symbol, filtered by user_id (multi-tenant isolation)."""
        ...
