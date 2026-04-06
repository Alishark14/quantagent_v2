"""Tests for FastAPI web layer endpoints.

Uses in-memory mock repos so tests run without any database.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Set API_KEYS before importing auth module
os.environ["API_KEYS"] = "test-key-123,another-key-456"

from api.app import create_app
from api.auth import get_current_user
from api.dependencies import (
    get_bot_repo,
    get_cycle_repo,
    get_health_tracker,
    get_rule_repo,
    get_trade_repo,
)
from tracking.health import HealthTracker


# ---------------------------------------------------------------------------
# In-memory mock repositories
# ---------------------------------------------------------------------------


class MockBotRepository:
    """In-memory bot repository for testing."""

    def __init__(self) -> None:
        self._bots: dict[str, dict] = {}

    async def save_bot(self, bot: dict) -> str:
        bot_id = bot.get("id") or str(uuid4())
        self._bots[bot_id] = {**bot, "id": bot_id}
        return bot_id

    async def get_bot(self, bot_id: str) -> dict | None:
        return self._bots.get(bot_id)

    async def get_bots_by_user(self, user_id: str) -> list[dict]:
        return [b for b in self._bots.values() if b.get("user_id") == user_id]

    async def update_bot_health(self, bot_id: str, health: dict) -> bool:
        if bot_id in self._bots:
            self._bots[bot_id]["last_health"] = health
            if "status" in health:
                self._bots[bot_id]["status"] = health["status"]
            return True
        return False


class MockTradeRepository:
    """In-memory trade repository for testing."""

    def __init__(self) -> None:
        self._trades: dict[str, dict] = {}

    async def save_trade(self, trade: dict) -> str:
        trade_id = trade.get("id") or str(uuid4())
        self._trades[trade_id] = {**trade, "id": trade_id}
        return trade_id

    async def get_trade(self, trade_id: str) -> dict | None:
        return self._trades.get(trade_id)

    async def get_open_positions(self, user_id: str, bot_id: str) -> list[dict]:
        return [
            t for t in self._trades.values()
            if t.get("user_id") == user_id
            and t.get("bot_id") == bot_id
            and t.get("status") == "open"
        ]

    async def get_trades_by_bot(self, bot_id: str, limit: int = 50) -> list[dict]:
        if bot_id:
            trades = [t for t in self._trades.values() if t.get("bot_id") == bot_id]
        else:
            trades = list(self._trades.values())
        return trades[:limit]

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        if trade_id in self._trades:
            self._trades[trade_id].update(updates)
            return True
        return False


class MockCycleRepository:
    """In-memory cycle repository for testing."""

    def __init__(self) -> None:
        self._cycles: list[dict] = []

    async def save_cycle(self, cycle: dict) -> str:
        cycle_id = cycle.get("id") or str(uuid4())
        self._cycles.append({**cycle, "id": cycle_id})
        return cycle_id

    async def get_recent_cycles(self, bot_id: str, limit: int = 5) -> list[dict]:
        matching = [c for c in self._cycles if c.get("bot_id") == bot_id]
        return matching[-limit:]


class MockRuleRepository:
    """In-memory rule repository for testing."""

    def __init__(self) -> None:
        self._rules: list[dict] = []

    async def save_rule(self, rule: dict) -> str:
        rule_id = rule.get("id") or str(uuid4())
        self._rules.append({**rule, "id": rule_id})
        return rule_id

    async def get_rules(self, symbol: str, timeframe: str) -> list[dict]:
        return [
            r for r in self._rules
            if r.get("symbol") == symbol and r.get("timeframe") == timeframe
            and r.get("active", True)
        ]

    async def update_rule_score(self, rule_id: str, delta: int) -> bool:
        for r in self._rules:
            if r.get("id") == rule_id:
                r["score"] = r.get("score", 0) + delta
                return True
        return False

    async def deactivate_rule(self, rule_id: str) -> bool:
        for r in self._rules:
            if r.get("id") == rule_id:
                r["active"] = False
                return True
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_USER_ID = "625faa3f"  # SHA-256("test-key-123")[:8] — matches auth.py logic


@pytest.fixture
def mock_repos():
    """Create fresh mock repos for each test."""
    return {
        "bots": MockBotRepository(),
        "trades": MockTradeRepository(),
        "cycles": MockCycleRepository(),
        "rules": MockRuleRepository(),
    }


@pytest.fixture
def health_tracker():
    """Fresh HealthTracker."""
    return HealthTracker()


@pytest.fixture
def app(mock_repos, health_tracker):
    """Create a FastAPI app with mock dependencies."""
    application = create_app()

    # Override dependencies with mocks
    application.dependency_overrides[get_bot_repo] = lambda: mock_repos["bots"]
    application.dependency_overrides[get_trade_repo] = lambda: mock_repos["trades"]
    application.dependency_overrides[get_cycle_repo] = lambda: mock_repos["cycles"]
    application.dependency_overrides[get_rule_repo] = lambda: mock_repos["rules"]
    application.dependency_overrides[get_health_tracker] = lambda: health_tracker

    return application


@pytest.fixture
def client(app):
    """TestClient with mocked dependencies. Uses context manager to trigger lifespan."""
    # Skip lifespan for unit tests (repos are mocked via overrides)
    app.router.lifespan_context = _noop_lifespan
    return TestClient(app)


from contextlib import asynccontextmanager


@asynccontextmanager
async def _noop_lifespan(app):
    yield


HEADERS = {"X-API-Key": "test-key-123"}
HEADERS_INVALID = {"X-API-Key": "bad-key"}


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Test API key authentication."""

    def test_missing_api_key_returns_401(self, client):
        resp = client.get("/v1/bots")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["detail"]

    def test_invalid_api_key_returns_401(self, client):
        resp = client.get("/v1/bots", headers=HEADERS_INVALID)
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["detail"]

    def test_valid_api_key_passes(self, client):
        resp = client.get("/v1/bots", headers=HEADERS)
        assert resp.status_code == 200

    def test_second_valid_key_also_works(self, client):
        resp = client.get("/v1/bots", headers={"X-API-Key": "another-key-456"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Bot CRUD tests
# ---------------------------------------------------------------------------


class TestBotCRUD:
    """Test bot management endpoints."""

    def test_create_bot(self, client):
        resp = client.post(
            "/v1/bots",
            json={
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "exchange": "hyperliquid",
            },
            headers=HEADERS,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["symbol"] == "BTC-USDC"
        assert data["timeframe"] == "1h"
        assert data["exchange"] == "hyperliquid"
        assert data["status"] == "created"
        assert data["id"]  # non-empty UUID
        assert data["user_id"] == TEST_USER_ID

    def test_create_bot_with_custom_config(self, client):
        resp = client.post(
            "/v1/bots",
            json={
                "symbol": "ETH-USDC",
                "timeframe": "4h",
                "account_balance": 5000.0,
                "conviction_threshold": 0.7,
                "max_position_pct": 0.5,
            },
            headers=HEADERS,
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["symbol"] == "ETH-USDC"
        assert data["config"]["account_balance"] == 5000.0
        assert data["config"]["conviction_threshold"] == 0.7

    def test_list_bots_empty(self, client):
        resp = client.get("/v1/bots", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["bots"] == []
        assert data["count"] == 0

    def test_list_bots_after_create(self, client):
        # Create 2 bots
        client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        client.post(
            "/v1/bots",
            json={"symbol": "ETH-USDC", "timeframe": "4h"},
            headers=HEADERS,
        )
        resp = client.get("/v1/bots", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        symbols = {b["symbol"] for b in data["bots"]}
        assert symbols == {"BTC-USDC", "ETH-USDC"}

    def test_get_bot_by_id(self, client):
        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        resp = client.get(f"/v1/bots/{bot_id}", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == bot_id
        assert data["symbol"] == "BTC-USDC"

    def test_get_nonexistent_bot_returns_404(self, client):
        resp = client.get("/v1/bots/nonexistent-id", headers=HEADERS)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_delete_bot_marks_stopped(self, client):
        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        resp = client.delete(f"/v1/bots/{bot_id}", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_delete_nonexistent_bot_returns_404(self, client):
        resp = client.delete("/v1/bots/nonexistent-id", headers=HEADERS)
        assert resp.status_code == 404

    def test_bots_isolated_by_user(self, client):
        """Bots created by one user are not visible to another."""
        # Create bot with first key
        client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )

        # List bots with second key (different user)
        resp = client.get(
            "/v1/bots",
            headers={"X-API-Key": "another-key-456"},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# Analyze endpoint tests
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Test the manual analysis trigger endpoint."""

    def test_analyze_no_cycles(self, client):
        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        resp = client.post(f"/v1/bots/{bot_id}/analyze", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "NO_DATA"
        assert data["action"] == "SKIP"

    def test_analyze_with_cycle_data(self, client, mock_repos):
        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        # Seed a cycle record
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mock_repos["cycles"].save_cycle({
                "bot_id": bot_id,
                "action": "LONG",
                "conviction_score": 0.85,
                "signals_json": json.dumps([{"agent": "indicator", "direction": "BULLISH"}]),
            })
        )

        resp = client.post(f"/v1/bots/{bot_id}/analyze", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "OK"
        assert data["action"] == "LONG"
        assert data["conviction_score"] == 0.85

    def test_analyze_nonexistent_bot_returns_404(self, client):
        resp = client.post("/v1/bots/nonexistent-id/analyze", headers=HEADERS)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Trade endpoint tests
# ---------------------------------------------------------------------------


class TestTrades:
    """Test trade listing endpoints."""

    def _seed_trade(self, mock_repos, bot_id: str, **overrides) -> str:
        """Helper to seed a trade record."""
        import asyncio
        trade_id = str(uuid4())
        trade = {
            "id": trade_id,
            "user_id": TEST_USER_ID,
            "bot_id": bot_id,
            "symbol": "BTC-USDC",
            "timeframe": "1h",
            "direction": "LONG",
            "entry_price": 50000.0,
            "size": 0.1,
            "status": "closed",
            "pnl": 100.0,
            **overrides,
        }
        asyncio.get_event_loop().run_until_complete(
            mock_repos["trades"].save_trade(trade)
        )
        return trade_id

    def test_list_trades_empty(self, client):
        resp = client.get("/v1/trades", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["trades"] == []
        assert data["count"] == 0

    def test_list_trades_with_data(self, client, mock_repos):
        self._seed_trade(mock_repos, "bot-1")
        self._seed_trade(mock_repos, "bot-1")

        resp = client.get("/v1/trades?bot_id=bot-1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_list_trades_filter_by_symbol(self, client, mock_repos):
        self._seed_trade(mock_repos, "bot-1", symbol="BTC-USDC")
        self._seed_trade(mock_repos, "bot-1", symbol="ETH-USDC")

        resp = client.get(
            "/v1/trades?bot_id=bot-1&symbol=ETH-USDC",
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["trades"][0]["symbol"] == "ETH-USDC"

    def test_list_trades_limit(self, client, mock_repos):
        for _ in range(5):
            self._seed_trade(mock_repos, "bot-1")

        resp = client.get("/v1/trades?bot_id=bot-1&limit=2", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_get_trade_by_id(self, client, mock_repos):
        trade_id = self._seed_trade(mock_repos, "bot-1")

        resp = client.get(f"/v1/trades/{trade_id}", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == trade_id
        assert data["symbol"] == "BTC-USDC"
        assert data["direction"] == "LONG"

    def test_get_nonexistent_trade_returns_404(self, client):
        resp = client.get("/v1/trades/nonexistent-id", headers=HEADERS)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_trades_isolated_by_user(self, client, mock_repos):
        """Trades from another user are not visible."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            mock_repos["trades"].save_trade({
                "id": "other-user-trade",
                "user_id": "other-user",
                "bot_id": "bot-x",
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "direction": "SHORT",
                "status": "open",
            })
        )

        # This user should not see the other user's trades
        resp = client.get("/v1/trades?bot_id=bot-x", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealth:
    """Test health endpoint."""

    def test_health_returns_valid_snapshot(self, client):
        """Health endpoint does not require auth."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], float)
        assert data["total_events"] == 0
        assert data["error_count"] == 0
        assert data["db_status"] == "ok"

    def test_health_degraded_on_errors(self, client, health_tracker):
        health_tracker.record_error("test", "something broke")

        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["error_count"] == 1
        assert len(data["recent_errors"]) == 1

    def test_health_no_auth_required(self, client):
        """Health check works without API key for monitoring systems."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Positions endpoint tests
# ---------------------------------------------------------------------------


class TestPositions:
    """Test open positions endpoint."""

    def test_positions_empty(self, client):
        resp = client.get("/v1/positions", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["positions"] == []
        assert data["count"] == 0

    def test_positions_with_open_trades(self, client, mock_repos):
        import asyncio

        # Create a bot first
        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        # Seed an open trade
        asyncio.get_event_loop().run_until_complete(
            mock_repos["trades"].save_trade({
                "id": "open-trade-1",
                "user_id": TEST_USER_ID,
                "bot_id": bot_id,
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "direction": "LONG",
                "entry_price": 50000.0,
                "size": 0.1,
                "pnl": 200.0,
                "status": "open",
            })
        )

        resp = client.get("/v1/positions", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        pos = data["positions"][0]
        assert pos["symbol"] == "BTC-USDC"
        assert pos["direction"] == "LONG"
        assert pos["entry_price"] == 50000.0
        assert pos["bot_id"] == bot_id

    def test_positions_excludes_closed_trades(self, client, mock_repos):
        import asyncio

        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        # Seed a closed trade (should not appear in positions)
        asyncio.get_event_loop().run_until_complete(
            mock_repos["trades"].save_trade({
                "id": "closed-trade-1",
                "user_id": TEST_USER_ID,
                "bot_id": bot_id,
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "direction": "LONG",
                "status": "closed",
            })
        )

        resp = client.get("/v1/positions", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# Rules endpoint tests
# ---------------------------------------------------------------------------


class TestRules:
    """Test reflection rules endpoint."""

    def test_rules_empty(self, client):
        resp = client.get("/v1/rules?symbol=BTC-USDC&timeframe=1h", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["rules"] == []
        assert data["count"] == 0

    def test_rules_with_data(self, client, mock_repos):
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_repos["rules"].save_rule({
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "rule_text": "Avoid LONG when RSI > 80",
                "score": 3,
                "active": True,
                "created_at": "2026-04-06T00:00:00Z",
            })
        )
        asyncio.get_event_loop().run_until_complete(
            mock_repos["rules"].save_rule({
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "rule_text": "Favor SHORT in ranging regime",
                "score": 1,
                "active": True,
                "created_at": "2026-04-06T01:00:00Z",
            })
        )

        resp = client.get("/v1/rules?symbol=BTC-USDC&timeframe=1h", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_rules_filtered_by_symbol(self, client, mock_repos):
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_repos["rules"].save_rule({
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "rule_text": "BTC rule",
                "score": 1,
                "active": True,
                "created_at": "2026-04-06T00:00:00Z",
            })
        )
        asyncio.get_event_loop().run_until_complete(
            mock_repos["rules"].save_rule({
                "symbol": "ETH-USDC",
                "timeframe": "1h",
                "rule_text": "ETH rule",
                "score": 1,
                "active": True,
                "created_at": "2026-04-06T00:00:00Z",
            })
        )

        resp = client.get("/v1/rules?symbol=ETH-USDC&timeframe=1h", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["rules"][0]["rule_text"] == "ETH rule"

    def test_rules_requires_auth(self, client):
        resp = client.get("/v1/rules?symbol=BTC-USDC&timeframe=1h")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Bot detail with last cycle tests
# ---------------------------------------------------------------------------


class TestBotWithCycle:
    """Test bot detail includes last cycle info."""

    def test_bot_detail_includes_last_cycle(self, client, mock_repos):
        import asyncio

        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        asyncio.get_event_loop().run_until_complete(
            mock_repos["cycles"].save_cycle({
                "bot_id": bot_id,
                "action": "SKIP",
                "conviction_score": 0.3,
                "timestamp": "2026-04-06T12:00:00Z",
            })
        )

        resp = client.get(f"/v1/bots/{bot_id}", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["last_cycle"] is not None
        assert data["last_cycle"]["action"] == "SKIP"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_create_bot_missing_required_fields(self, client):
        resp = client.post("/v1/bots", json={}, headers=HEADERS)
        assert resp.status_code == 422  # Validation error

    def test_create_bot_invalid_conviction_threshold(self, client):
        resp = client.post(
            "/v1/bots",
            json={
                "symbol": "BTC-USDC",
                "timeframe": "1h",
                "conviction_threshold": 5.0,  # max is 1.0
            },
            headers=HEADERS,
        )
        assert resp.status_code == 422

    def test_delete_then_get_shows_stopped(self, client):
        create_resp = client.post(
            "/v1/bots",
            json={"symbol": "BTC-USDC", "timeframe": "1h"},
            headers=HEADERS,
        )
        bot_id = create_resp.json()["id"]

        client.delete(f"/v1/bots/{bot_id}", headers=HEADERS)

        resp = client.get(f"/v1/bots/{bot_id}", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_health_event_counts_structure(self, client, health_tracker):
        # Simulate some events
        class FakeEvent:
            pass

        health_tracker.on_any_event(FakeEvent())
        health_tracker.on_any_event(FakeEvent())

        resp = client.get("/v1/health")
        data = resp.json()
        assert data["total_events"] == 2
        assert "FakeEvent" in data["event_counts"]
        assert data["event_counts"]["FakeEvent"] == 2
