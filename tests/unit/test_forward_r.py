"""Tests for tracking.forward_r — ForwardMaxRStamper + compute helper.

Covers:

* compute_forward_max_r pure helper: LONG max favourable excursion,
  SHORT max favourable excursion, no excursion (price ran adversely)
  → 0.0, polars DataFrame input, list-of-dicts input, invalid inputs
  return None.
* ForwardMaxRStamper.stamp_trade looks up the trade, computes the
  metric, and persists it via repo.update_trade.
* Stamper handles missing entry_price, missing direction, missing
  forward path data (FileNotFoundError) gracefully → returns None,
  doesn't raise.
* Risk derivation order: explicit `risk` > `sl_price` distance >
  default 1% of entry.
* Stamper persists nothing when trade dict has no id.
* on_trade_closed event handler: trade_id present → stamps;
  trade_id missing → no-op.
* TrackingModule + InProcessBus integration: stamper registered as
  optional dep, async event handler awaited correctly via the
  async-aware _safe wrapper.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from tracking.forward_r import ForwardMaxRStamper, compute_forward_max_r


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeTradeRepo:
    def __init__(self, trades: dict[str, dict] | None = None) -> None:
        self.trades = dict(trades or {})
        self.updates: list[tuple[str, dict]] = []

    async def get_trade(self, trade_id: str) -> dict | None:
        record = self.trades.get(trade_id)
        return dict(record) if record is not None else None

    async def update_trade(self, trade_id: str, updates: dict) -> bool:
        self.updates.append((trade_id, dict(updates)))
        if trade_id in self.trades:
            self.trades[trade_id].update(updates)
            return True
        return False


class _FakeForwardPathLoader:
    """Returns a list of {high, low} dicts for any (symbol, timestamp) pair.

    The forward_r stamper accepts list-of-dicts input transparently
    via :func:`_extract_highs_lows`, so we can avoid pulling polars
    into these tests.
    """

    def __init__(
        self,
        path: list[dict] | None = None,
        raise_for: set[tuple[str, int]] | None = None,
    ) -> None:
        self._path = path or []
        self._raise_for = raise_for or set()
        self.calls: list[dict] = []

    def recommended_resolution(self, timeframe: str) -> str:
        return "5m" if timeframe in ("4h", "1d") else "1m"

    def load(
        self,
        symbol: str,
        entry_timestamp: int,
        duration_candles: int = 60,
        resolution: str = "1m",
    ):
        self.calls.append(
            {
                "symbol": symbol,
                "entry_timestamp": entry_timestamp,
                "duration_candles": duration_candles,
                "resolution": resolution,
            }
        )
        if (symbol, entry_timestamp) in self._raise_for:
            raise FileNotFoundError(
                f"no parquet for {symbol} at {entry_timestamp}"
            )
        return list(self._path)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _trade(
    trade_id: str = "t1",
    *,
    direction: str = "LONG",
    entry_price: float = 100.0,
    sl_price: float | None = 99.0,
    timeframe: str = "1h",
    symbol: str = "BTC-USDC",
    entry_timestamp_ms: int | None = None,
) -> dict:
    return {
        "id": trade_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "entry_time": _now_iso(),
        "entry_timestamp_ms": entry_timestamp_ms or 1_700_000_000_000,
    }


# ---------------------------------------------------------------------------
# compute_forward_max_r — pure helper
# ---------------------------------------------------------------------------


def test_compute_forward_max_r_long_basic():
    path = [
        {"high": 101, "low": 99},
        {"high": 105, "low": 100},  # 5 dollars above 100 entry
        {"high": 103, "low": 98},
    ]
    r = compute_forward_max_r(direction="LONG", entry_price=100, risk=1.0, forward_path=path)
    assert r == pytest.approx(5.0)


def test_compute_forward_max_r_short_basic():
    path = [
        {"high": 100, "low": 95},  # 5 dollars below 100 entry
        {"high": 99, "low": 97},
    ]
    r = compute_forward_max_r(direction="SHORT", entry_price=100, risk=2.0, forward_path=path)
    assert r == pytest.approx(2.5)  # 5 / 2 risk


def test_compute_forward_max_r_long_no_favourable_move_returns_zero():
    """Price only moved adversely (down) — LONG max-favourable should be 0."""
    path = [
        {"high": 99.5, "low": 95},
        {"high": 99.0, "low": 94},
    ]
    r = compute_forward_max_r(direction="LONG", entry_price=100, risk=1.0, forward_path=path)
    assert r == pytest.approx(0.0)


def test_compute_forward_max_r_short_no_favourable_move_returns_zero():
    path = [
        {"high": 105, "low": 100.5},
        {"high": 110, "low": 101},
    ]
    r = compute_forward_max_r(direction="SHORT", entry_price=100, risk=1.0, forward_path=path)
    assert r == pytest.approx(0.0)


def test_compute_forward_max_r_empty_path_returns_zero():
    r = compute_forward_max_r(direction="LONG", entry_price=100, risk=1.0, forward_path=[])
    assert r == pytest.approx(0.0)


def test_compute_forward_max_r_invalid_entry_returns_none():
    assert compute_forward_max_r("LONG", 0.0, 1.0, [{"high": 1, "low": 1}]) is None
    assert compute_forward_max_r("LONG", -10, 1.0, [{"high": 1, "low": 1}]) is None
    assert compute_forward_max_r("LONG", None, 1.0, [{"high": 1, "low": 1}]) is None


def test_compute_forward_max_r_invalid_risk_returns_none():
    assert compute_forward_max_r("LONG", 100, 0.0, [{"high": 1, "low": 1}]) is None
    assert compute_forward_max_r("LONG", 100, -1.0, [{"high": 1, "low": 1}]) is None


def test_compute_forward_max_r_unknown_direction_returns_none():
    assert compute_forward_max_r("SIDEWAYS", 100, 1.0, [{"high": 1, "low": 1}]) is None


def test_compute_forward_max_r_case_insensitive():
    path = [{"high": 102, "low": 99}]
    r = compute_forward_max_r("long", 100, 1.0, path)
    assert r == pytest.approx(2.0)


def test_compute_forward_max_r_with_polars_dataframe():
    polars = pytest.importorskip("polars")
    df = polars.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [100, 100, 100],
            "high": [101, 104, 102],
            "low": [99, 100, 100],
            "close": [100, 102, 101],
            "volume": [10, 10, 10],
        }
    )
    r = compute_forward_max_r("LONG", 100, 1.0, df)
    assert r == pytest.approx(4.0)


def test_compute_forward_max_r_with_empty_polars_dataframe():
    polars = pytest.importorskip("polars")
    df = polars.DataFrame(
        schema={
            "timestamp": polars.Int64,
            "high": polars.Float64,
            "low": polars.Float64,
        }
    )
    r = compute_forward_max_r("LONG", 100, 1.0, df)
    assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ForwardMaxRStamper.stamp_trade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stamp_trade_persists_computed_value():
    repo = _FakeTradeRepo({"t1": _trade("t1")})
    loader = _FakeForwardPathLoader(
        path=[{"high": 102, "low": 99}, {"high": 105, "low": 100}]
    )
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)

    value = await stamper.stamp_trade("t1")
    # entry=100, sl=99 → risk=1; max high=105 → 5R
    assert value == pytest.approx(5.0)
    assert repo.updates == [("t1", {"forward_max_r": 5.0})]
    assert loader.calls[0]["resolution"] == "1m"


@pytest.mark.asyncio
async def test_stamp_trade_uses_5m_for_4h_timeframe():
    repo = _FakeTradeRepo({"t1": _trade("t1", timeframe="4h")})
    loader = _FakeForwardPathLoader(path=[{"high": 102, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    await stamper.stamp_trade("t1")
    assert loader.calls[0]["resolution"] == "5m"


@pytest.mark.asyncio
async def test_stamp_trade_unknown_id_returns_none():
    repo = _FakeTradeRepo({})
    loader = _FakeForwardPathLoader()
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    result = await stamper.stamp_trade("ghost")
    assert result is None
    assert repo.updates == []


@pytest.mark.asyncio
async def test_stamp_trade_missing_parquet_returns_none(caplog):
    repo = _FakeTradeRepo(
        {"t1": _trade("t1", entry_timestamp_ms=1_700_000_000_000)}
    )
    loader = _FakeForwardPathLoader(
        raise_for={("BTC-USDC", 1_700_000_000_000)}
    )
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    with caplog.at_level("WARNING"):
        result = await stamper.stamp_trade("t1")
    assert result is None
    assert repo.updates == []
    assert any("no forward path data" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_stamp_trade_missing_entry_price_returns_none():
    bad_trade = _trade("t1")
    bad_trade["entry_price"] = None
    repo = _FakeTradeRepo({"t1": bad_trade})
    loader = _FakeForwardPathLoader(path=[{"high": 102, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    assert await stamper.stamp_trade("t1") is None
    assert repo.updates == []


@pytest.mark.asyncio
async def test_stamp_trade_falls_back_to_default_risk_when_no_sl():
    """No sl_price → 1% of entry as risk."""
    bad_trade = _trade("t1")
    bad_trade["sl_price"] = None
    repo = _FakeTradeRepo({"t1": bad_trade})
    # entry=100, default risk=1.0 (1% of 100), max excursion 5 → 5R
    loader = _FakeForwardPathLoader(path=[{"high": 105, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    value = await stamper.stamp_trade("t1")
    assert value == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_stamp_trade_explicit_risk_wins_over_sl():
    trade = _trade("t1", sl_price=99.0)
    trade["risk"] = 2.0  # Should be preferred over sl-derived risk of 1.0
    repo = _FakeTradeRepo({"t1": trade})
    loader = _FakeForwardPathLoader(path=[{"high": 104, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    value = await stamper.stamp_trade("t1")
    # max excursion=4, risk=2 → 2R
    assert value == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_stamp_trade_dict_without_id_does_not_persist():
    repo = _FakeTradeRepo({})
    loader = _FakeForwardPathLoader(path=[{"high": 105, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)
    trade_no_id = _trade("ignored")
    trade_no_id.pop("id")
    value = await stamper.stamp_trade_dict(trade_no_id)
    # Compute still works (returns 5.0) but persistence is skipped.
    assert value == pytest.approx(5.0)
    assert repo.updates == []


# ---------------------------------------------------------------------------
# Event handler — on_trade_closed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_trade_closed_with_trade_id_stamps():
    from engine.events import TradeClosed

    repo = _FakeTradeRepo({"t1": _trade("t1")})
    loader = _FakeForwardPathLoader(path=[{"high": 105, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)

    event = TradeClosed(
        source="test",
        timestamp=datetime.now(tz=timezone.utc),
        symbol="BTC-USDC",
        pnl=-50.0,
        exit_reason="sl",
        trade_id="t1",
    )
    value = await stamper.on_trade_closed(event)
    assert value == pytest.approx(5.0)
    assert repo.updates == [("t1", {"forward_max_r": 5.0})]


@pytest.mark.asyncio
async def test_on_trade_closed_without_trade_id_skips(caplog):
    from engine.events import TradeClosed

    repo = _FakeTradeRepo({})
    loader = _FakeForwardPathLoader(path=[{"high": 105, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)

    event = TradeClosed(
        source="test",
        timestamp=datetime.now(tz=timezone.utc),
        symbol="BTC-USDC",
        pnl=0.0,
        exit_reason="CLOSE_ALL",
        # trade_id NOT set — older emission site
    )
    with caplog.at_level("DEBUG", logger="tracking.forward_r"):
        value = await stamper.on_trade_closed(event)
    assert value is None
    assert repo.updates == []


# ---------------------------------------------------------------------------
# TrackingModule integration with the async-aware _safe wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracking_module_dispatches_async_handler():
    """Stamper.on_trade_closed runs and persists when wired through TrackingModule."""
    from engine.events import InProcessBus, TradeClosed
    from tracking import TrackingModule

    repo = _FakeTradeRepo({"t1": _trade("t1")})
    loader = _FakeForwardPathLoader(path=[{"high": 110, "low": 99}])
    stamper = ForwardMaxRStamper(repo=repo, forward_path_loader=loader)

    bus = InProcessBus()
    tracking = TrackingModule(forward_max_r_stamper=stamper)
    tracking.subscribe_all(bus)

    await bus.publish(
        TradeClosed(
            source="test",
            timestamp=datetime.now(tz=timezone.utc),
            symbol="BTC-USDC",
            pnl=-50.0,
            exit_reason="sl",
            trade_id="t1",
        )
    )

    # The async handler should have run to completion before publish() returns.
    assert repo.updates == [("t1", {"forward_max_r": 10.0})]


@pytest.mark.asyncio
async def test_tracking_module_async_handler_failure_does_not_propagate(caplog):
    """If the stamper raises, the bus must still complete normally."""
    from engine.events import InProcessBus, TradeClosed
    from tracking import TrackingModule

    class _ExplodingStamper:
        async def on_trade_closed(self, event):
            raise RuntimeError("kaboom")

    bus = InProcessBus()
    tracking = TrackingModule(forward_max_r_stamper=_ExplodingStamper())
    tracking.subscribe_all(bus)

    with caplog.at_level("ERROR"):
        await bus.publish(
            TradeClosed(
                source="test",
                timestamp=datetime.now(tz=timezone.utc),
                symbol="BTC-USDC",
                pnl=0.0,
                exit_reason="sl",
                trade_id="t1",
            )
        )
    # Failure was logged but didn't crash publish()
    assert any("kaboom" in (r.message or "") or "kaboom" in (r.exc_text or "")
               for r in caplog.records)


def test_tracking_module_default_stamper_is_none():
    """Stamper is opt-in; default TrackingModule has no forward_r wiring."""
    from tracking import TrackingModule

    tracking = TrackingModule()
    assert tracking.forward_max_r_stamper is None
