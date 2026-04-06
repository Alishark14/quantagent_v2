"""Abstract repository interfaces for all database access.

Engine code NEVER touches SQL directly — all DB access goes through these interfaces.
PostgreSQL is the standard backend. SQLite exists only for local dev fallback.
"""

from abc import ABC, abstractmethod


class TradeRepository(ABC):
    """Repository for trade lifecycle records."""

    @abstractmethod
    async def save_trade(self, trade: dict) -> str:
        """Save a trade record. Returns the trade ID."""
        ...

    @abstractmethod
    async def get_trade(self, trade_id: str) -> dict | None:
        """Get a trade by ID. Returns None if not found."""
        ...

    @abstractmethod
    async def get_open_positions(self, user_id: str, bot_id: str) -> list[dict]:
        """Get open positions filtered by user_id AND bot_id."""
        ...

    @abstractmethod
    async def get_trades_by_bot(self, bot_id: str, limit: int = 50) -> list[dict]:
        """Get recent trades for a bot, ordered by entry_time descending."""
        ...

    @abstractmethod
    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        """Update trade fields. Returns True if trade was found and updated."""
        ...


class CycleRepository(ABC):
    """Repository for analysis cycle records."""

    @abstractmethod
    async def save_cycle(self, cycle: dict) -> str:
        """Save a cycle record. Returns the cycle ID."""
        ...

    @abstractmethod
    async def get_recent_cycles(self, bot_id: str, limit: int = 5) -> list[dict]:
        """Get recent cycles for a bot, ordered by timestamp descending."""
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
    async def update_bot_health(self, bot_id: str, health: dict) -> bool:
        """Update bot health snapshot. Returns True if bot was found."""
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
