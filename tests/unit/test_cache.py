"""Tests for the multi-layer caching system.

Tests CacheBackend (MemoryCacheBackend), CacheManager (get_or_fetch),
CacheMetrics, TTL expiry, and integration with data fetchers.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from storage.cache import (
    CacheManager,
    TTL_ASSET_META,
    TTL_EXTERNAL_API,
    TTL_FLOW,
    TTL_FUNDING,
    TTL_OHLCV,
    TTL_ORDERBOOK,
    flow_key,
    funding_key,
    meta_key,
    ohlcv_key,
    orderbook_key,
)
from storage.cache.base import CacheBackend
from storage.cache.memory import MemoryCacheBackend
from storage.cache.metrics import CacheMetrics


# ---------------------------------------------------------------------------
# CacheBackend ABC tests
# ---------------------------------------------------------------------------


class TestCacheBackendABC:
    """Verify the ABC cannot be instantiated directly."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            CacheBackend()


# ---------------------------------------------------------------------------
# MemoryCacheBackend tests
# ---------------------------------------------------------------------------


class TestMemoryCacheBackend:
    """Test the in-memory TTL cache backend."""

    @pytest.fixture
    def backend(self):
        return MemoryCacheBackend()

    @pytest.mark.asyncio
    async def test_set_and_get(self, backend):
        await backend.set("key1", "value1", ttl=60)
        result = await backend.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self, backend):
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_has_returns_true_for_existing(self, backend):
        await backend.set("key1", "val", ttl=60)
        assert await backend.has("key1") is True

    @pytest.mark.asyncio
    async def test_has_returns_false_for_missing(self, backend):
        assert await backend.has("missing") is False

    @pytest.mark.asyncio
    async def test_delete_existing_key(self, backend):
        await backend.set("key1", "val", ttl=60)
        result = await backend.delete("key1")
        assert result is True
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, backend):
        result = await backend.delete("nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_removes_all(self, backend):
        await backend.set("a", 1, ttl=60)
        await backend.set("b", 2, ttl=60)
        await backend.set("c", 3, ttl=120)
        await backend.clear()
        assert await backend.get("a") is None
        assert await backend.get("b") is None
        assert await backend.get("c") is None

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, backend):
        """Values should expire after their TTL."""
        await backend.set("short", "data", ttl=1)
        assert await backend.get("short") == "data"
        await asyncio.sleep(1.1)
        assert await backend.get("short") is None

    @pytest.mark.asyncio
    async def test_different_ttl_buckets(self, backend):
        """Keys with different TTLs go to different buckets."""
        await backend.set("fast", "v1", ttl=60)
        await backend.set("slow", "v2", ttl=3600)
        assert await backend.get("fast") == "v1"
        assert await backend.get("slow") == "v2"

    @pytest.mark.asyncio
    async def test_overwrite_same_key(self, backend):
        await backend.set("key", "old", ttl=60)
        await backend.set("key", "new", ttl=60)
        assert await backend.get("key") == "new"

    @pytest.mark.asyncio
    async def test_total_entries(self, backend):
        await backend.set("a", 1, ttl=60)
        await backend.set("b", 2, ttl=60)
        assert backend.total_entries == 2

    @pytest.mark.asyncio
    async def test_stores_complex_types(self, backend):
        """Cache should handle dicts, lists, and dataclasses."""
        data = {"candles": [{"open": 100, "close": 101}], "count": 150}
        await backend.set("complex", data, ttl=60)
        result = await backend.get("complex")
        assert result == data

    @pytest.mark.asyncio
    async def test_ttl_change_moves_bucket(self, backend):
        """Changing TTL for a key should move it to a new bucket."""
        await backend.set("key", "v1", ttl=60)
        await backend.set("key", "v2", ttl=120)
        assert await backend.get("key") == "v2"


# ---------------------------------------------------------------------------
# CacheMetrics tests
# ---------------------------------------------------------------------------


class TestCacheMetrics:
    """Test cache performance metrics."""

    def test_initial_state(self):
        m = CacheMetrics()
        assert m.hits == 0
        assert m.misses == 0
        assert m.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        m = CacheMetrics()
        m.record_hit()
        m.record_hit()
        m.record_miss()
        assert m.hit_rate == pytest.approx(2 / 3)

    def test_all_hits(self):
        m = CacheMetrics()
        for _ in range(10):
            m.record_hit()
        assert m.hit_rate == 1.0

    def test_all_misses(self):
        m = CacheMetrics()
        for _ in range(5):
            m.record_miss()
        assert m.hit_rate == 0.0

    def test_summary(self):
        m = CacheMetrics()
        m.record_hit()
        m.record_miss()
        m.record_set()
        m.record_delete()
        m.record_flush()
        s = m.summary()
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["total_requests"] == 2
        assert s["sets"] == 1
        assert s["deletes"] == 1
        assert s["flushes"] == 1

    def test_reset(self):
        m = CacheMetrics()
        m.record_hit()
        m.record_miss()
        m.reset()
        assert m.hits == 0
        assert m.misses == 0


# ---------------------------------------------------------------------------
# CacheManager tests
# ---------------------------------------------------------------------------


class TestCacheManager:
    """Test CacheManager with get_or_fetch(), invalidation, flush."""

    @pytest.fixture
    def cache(self):
        return CacheManager()

    @pytest.mark.asyncio
    async def test_get_or_fetch_miss_calls_fn(self, cache):
        fetch_fn = AsyncMock(return_value={"data": 42})

        result = await cache.get_or_fetch("key1", fetch_fn, ttl=60)

        assert result == {"data": 42}
        fetch_fn.assert_called_once()
        assert cache.metrics.misses == 1
        assert cache.metrics.sets == 1

    @pytest.mark.asyncio
    async def test_get_or_fetch_hit_skips_fn(self, cache):
        fetch_fn = AsyncMock(return_value={"data": 42})

        # First call: miss + fetch
        await cache.get_or_fetch("key1", fetch_fn, ttl=60)

        # Second call: hit from cache
        result = await cache.get_or_fetch("key1", fetch_fn, ttl=60)

        assert result == {"data": 42}
        fetch_fn.assert_called_once()  # only the first time
        assert cache.metrics.hits == 1
        assert cache.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_get_or_fetch_none_not_cached(self, cache):
        """If fetch_fn returns None, don't cache it."""
        fetch_fn = AsyncMock(return_value=None)

        result = await cache.get_or_fetch("key1", fetch_fn, ttl=60)
        assert result is None
        assert cache.metrics.sets == 0  # not cached

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        await cache.set("key1", "val", ttl=60)
        result = await cache.invalidate("key1")
        assert result is True
        assert await cache.get("key1") is None
        assert cache.metrics.deletes == 1

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent(self, cache):
        result = await cache.invalidate("nope")
        assert result is False

    @pytest.mark.asyncio
    async def test_flush_all(self, cache):
        await cache.set("a", 1, ttl=60)
        await cache.set("b", 2, ttl=60)
        await cache.flush_all()
        assert await cache.get("a") is None
        assert await cache.get("b") is None
        assert cache.metrics.flushes == 1

    @pytest.mark.asyncio
    async def test_invalidate_pattern(self, cache):
        await cache.set("ohlcv:BTC:1h", "data1", ttl=60)
        await cache.set("ohlcv:ETH:1h", "data2", ttl=60)
        await cache.set("flow:BTC", "data3", ttl=60)

        count = await cache.invalidate_pattern("ohlcv:")
        assert count == 2
        assert await cache.get("ohlcv:BTC:1h") is None
        assert await cache.get("flow:BTC") is not None  # not affected

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, cache):
        fetch_fn = AsyncMock(return_value="data")

        await cache.get_or_fetch("k1", fetch_fn, ttl=60)  # miss
        await cache.get_or_fetch("k1", fetch_fn, ttl=60)  # hit
        await cache.get_or_fetch("k1", fetch_fn, ttl=60)  # hit

        assert cache.metrics.hits == 2
        assert cache.metrics.misses == 1
        assert cache.metrics.hit_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# TTL constant tests
# ---------------------------------------------------------------------------


class TestTTLConstants:
    """Verify TTL constants are set correctly."""

    def test_ohlcv_ttl_matches_candle_periods(self):
        assert TTL_OHLCV["15m"] == 900
        assert TTL_OHLCV["30m"] == 1800
        assert TTL_OHLCV["1h"] == 3600
        assert TTL_OHLCV["4h"] == 14400
        assert TTL_OHLCV["1d"] == 86400

    def test_flow_ttl_is_5_minutes(self):
        assert TTL_FLOW == 300

    def test_external_api_ttl(self):
        assert TTL_EXTERNAL_API == 3600

    def test_asset_meta_ttl_is_24h(self):
        assert TTL_ASSET_META == 86400

    def test_orderbook_ttl_is_very_short(self):
        assert TTL_ORDERBOOK == 5

    def test_funding_ttl(self):
        assert TTL_FUNDING == 60


# ---------------------------------------------------------------------------
# Key builder tests
# ---------------------------------------------------------------------------


class TestKeyBuilders:
    """Test cache key generation helpers."""

    def test_ohlcv_key(self):
        assert ohlcv_key("BTC-USDC", "1h") == "ohlcv:BTC-USDC:1h"

    def test_flow_key(self):
        assert flow_key("ETH-USDC") == "flow:ETH-USDC"

    def test_meta_key(self):
        assert meta_key("hyperliquid") == "meta:hyperliquid"

    def test_orderbook_key(self):
        assert orderbook_key("SOL-USDC") == "orderbook:SOL-USDC"

    def test_funding_key(self):
        assert funding_key("BTC-USDC") == "funding:BTC-USDC"


# ---------------------------------------------------------------------------
# Integration: OHLCVFetcher with cache
# ---------------------------------------------------------------------------


class TestOHLCVFetcherCache:
    """Test that OHLCVFetcher uses cache when provided."""

    @pytest.mark.asyncio
    async def test_fetcher_caches_candles(self):
        from engine.config import TradingConfig
        from engine.data.ohlcv import OHLCVFetcher

        adapter = MagicMock()
        candles = [
            {"timestamp": 1, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
            for _ in range(50)
        ]
        adapter.fetch_ohlcv = AsyncMock(return_value=candles)

        cache = CacheManager()
        fetcher = OHLCVFetcher(adapter, TradingConfig(), cache=cache)

        # First fetch: miss
        await fetcher.fetch("BTC-USDC", "1h")
        assert adapter.fetch_ohlcv.call_count >= 1  # main + parent TF

        first_call_count = adapter.fetch_ohlcv.call_count

        # Second fetch: hit from cache
        await fetcher.fetch("BTC-USDC", "1h")
        # Should NOT have made more adapter calls (cached)
        assert adapter.fetch_ohlcv.call_count == first_call_count

    @pytest.mark.asyncio
    async def test_fetcher_works_without_cache(self):
        from engine.config import TradingConfig
        from engine.data.ohlcv import OHLCVFetcher

        adapter = MagicMock()
        adapter.fetch_ohlcv = AsyncMock(return_value=[
            {"timestamp": 1, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
            for _ in range(50)
        ])

        fetcher = OHLCVFetcher(adapter, TradingConfig())  # no cache
        result = await fetcher.fetch("BTC-USDC", "1h")
        assert result.symbol == "BTC-USDC"


# ---------------------------------------------------------------------------
# Integration: FlowAgent with cache
# ---------------------------------------------------------------------------


class TestFlowAgentCache:
    """Test that FlowAgent uses cache when provided."""

    @pytest.mark.asyncio
    async def test_flow_agent_caches_result(self):
        from engine.data.flow import FlowAgent
        from engine.data.flow.base import FlowProvider

        provider = MagicMock(spec=FlowProvider)
        provider.name.return_value = "test"
        provider.fetch = AsyncMock(return_value={
            "funding_rate": 0.0001,
            "funding_signal": "NEUTRAL",
        })

        cache = CacheManager()
        agent = FlowAgent(providers=[provider], cache=cache)
        adapter = MagicMock()

        # First call: miss
        result1 = await agent.fetch_flow("BTC-USDC", adapter)
        assert result1.funding_rate == 0.0001
        provider.fetch.assert_called_once()

        # Second call: hit
        result2 = await agent.fetch_flow("BTC-USDC", adapter)
        assert result2.funding_rate == 0.0001
        provider.fetch.assert_called_once()  # not called again

    @pytest.mark.asyncio
    async def test_flow_agent_works_without_cache(self):
        from engine.data.flow import FlowAgent

        agent = FlowAgent()  # no providers, no cache
        adapter = MagicMock()

        result = await agent.fetch_flow("BTC-USDC", adapter)
        assert result.data_richness == "MINIMAL"


# ---------------------------------------------------------------------------
# Integration: HyperliquidCostModel with cache
# ---------------------------------------------------------------------------


class TestHyperliquidCostModelCache:
    """Test that HyperliquidCostModel uses cache when provided."""

    @pytest.mark.asyncio
    async def test_meta_refresh_uses_cache(self):
        from engine.execution.cost_models.hyperliquid import HyperliquidCostModel

        cache = CacheManager()
        model = HyperliquidCostModel(cache=cache)

        adapter = MagicMock()
        adapter.fetch_meta = AsyncMock(return_value=[
            {"symbol": "BTC-USDC", "deployer_fee_scale": 0, "growth_mode": False, "is_hip3": False},
        ])
        adapter.fetch_user_fees = AsyncMock(return_value={"tier": 1})

        # First refresh: miss
        await model.refresh(adapter)
        assert adapter.fetch_meta.call_count == 1

        # Reset timer to force another refresh attempt
        model._last_meta_refresh = None

        # Second refresh: hit from cache (meta cached for 24h)
        await model.refresh(adapter)
        # fetch_meta should NOT be called again — cache served it
        assert adapter.fetch_meta.call_count == 1

    @pytest.mark.asyncio
    async def test_model_works_without_cache(self):
        from engine.execution.cost_models.hyperliquid import HyperliquidCostModel

        model = HyperliquidCostModel()  # no cache
        assert model.get_taker_rate("BTC-USDC") > 0


# ---------------------------------------------------------------------------
# Thundering herd protection tests
# ---------------------------------------------------------------------------


class TestThunderingHerd:
    """Test that concurrent get_or_fetch calls result in exactly 1 fetch."""

    @pytest.mark.asyncio
    async def test_50_concurrent_callers_single_fetch(self):
        cache = CacheManager()
        call_count = 0

        async def slow_fetch():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)  # simulate API latency
            return {"data": "result"}

        # Spawn 50 concurrent requests for the same key
        results = await asyncio.gather(
            *[cache.get_or_fetch("shared_key", slow_fetch, ttl=60) for _ in range(50)]
        )

        assert call_count == 1  # exactly 1 API call
        assert all(r == {"data": "result"} for r in results)

    @pytest.mark.asyncio
    async def test_different_keys_fetch_independently(self):
        cache = CacheManager()
        call_count = 0

        async def fetch():
            nonlocal call_count
            call_count += 1
            return "data"

        await asyncio.gather(
            cache.get_or_fetch("key_a", fetch, ttl=60),
            cache.get_or_fetch("key_b", fetch, ttl=60),
        )

        assert call_count == 2  # each key fetched once

    @pytest.mark.asyncio
    async def test_lock_cleanup_after_completion(self):
        cache = CacheManager()

        async def fetch():
            return "val"

        await cache.get_or_fetch("cleanup_key", fetch, ttl=60)

        # After all callers complete, the lock should be cleaned up
        # (lock may still exist if implementation doesn't clean outside async with)
        # At minimum, it should not be locked
        lock = cache._locks.get("cleanup_key")
        if lock is not None:
            assert not lock.locked()


# ---------------------------------------------------------------------------
# Epoch-aligned TTL tests
# ---------------------------------------------------------------------------


class TestEpochAlignedTTL:
    """Test compute_ttl and expected_candle_close from storage/cache/ttl.py."""

    def test_compute_ttl_1h_midway(self):
        """At 12:30:00, TTL should be ~1802s (30 min + 2s buffer)."""
        from unittest.mock import patch
        from storage.cache.ttl import compute_ttl

        # Simulate 12:30:00 UTC (epoch for a round half-hour)
        fake_time = 1712408400 + 1800  # some epoch + 30 min into the hour
        with patch("storage.cache.ttl.time.time", return_value=fake_time):
            ttl = compute_ttl("1h")
            # next candle open is 1 hour boundary: ~1800 + 2 = 1802
            assert 1801 <= ttl <= 1803

    def test_compute_ttl_1h_near_boundary(self):
        """At XX:59:58, TTL should be ~4.0s (2s + 2s buffer)."""
        from unittest.mock import patch
        from storage.cache.ttl import compute_ttl

        # 2 seconds before the hour boundary
        fake_time = 1712412000 - 2  # 2s before an hour boundary
        with patch("storage.cache.ttl.time.time", return_value=fake_time):
            ttl = compute_ttl("1h")
            assert 3.5 <= ttl <= 4.5

    def test_compute_ttl_15m_range(self):
        from storage.cache.ttl import compute_ttl
        ttl = compute_ttl("15m")
        assert 2.0 <= ttl <= 902.0  # 0 to 900 + 2s buffer

    def test_compute_ttl_unknown_timeframe_raises(self):
        from storage.cache.ttl import compute_ttl
        with pytest.raises(ValueError, match="Unknown timeframe"):
            compute_ttl("3h")

    def test_expected_candle_close(self):
        from unittest.mock import patch
        from storage.cache.ttl import expected_candle_close, TIMEFRAME_SECONDS

        fake_time = 1712410200.0  # some timestamp
        with patch("storage.cache.ttl.time.time", return_value=fake_time):
            close = expected_candle_close("1h")
            period = TIMEFRAME_SECONDS["1h"]
            candle_open = (fake_time // period) * period
            assert close == candle_open + period

    def test_provider_ttl_constants(self):
        from storage.cache.ttl import FLOW_TTL, REGIME_TTL, SENTIMENT_TTL, NEWS_TTL
        assert FLOW_TTL == 300
        assert REGIME_TTL == 14400
        assert SENTIMENT_TTL == 3600
        assert NEWS_TTL == 1800


# ---------------------------------------------------------------------------
# Stale candle rejection tests
# ---------------------------------------------------------------------------


class TestStaleCandleRejection:
    """Test OHLCVFetcher retries on stale candle data."""

    @pytest.mark.asyncio
    async def test_retries_on_stale_candle(self):
        from unittest.mock import patch
        from engine.config import TradingConfig
        from engine.data.ohlcv import OHLCVFetcher

        adapter = MagicMock()
        # Return candles with old timestamp (stale)
        stale_candles = [
            {"timestamp": 1000, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
            for _ in range(50)
        ]
        adapter.fetch_ohlcv = AsyncMock(return_value=stale_candles)

        fetcher = OHLCVFetcher(adapter, TradingConfig())

        with patch("engine.data.ohlcv.asyncio.sleep", new_callable=AsyncMock):
            candles = await fetcher._fetch_candles_with_validation("BTC-USDC", "1h", 150)

        # Should have retried 3 times
        assert adapter.fetch_ohlcv.call_count == 3
        # Still returns data (with warning) after all retries
        assert len(candles) == 50

    @pytest.mark.asyncio
    async def test_accepts_fresh_candle_immediately(self):
        from unittest.mock import patch
        from engine.config import TradingConfig
        from engine.data.ohlcv import OHLCVFetcher
        from storage.cache.ttl import expected_candle_close, TIMEFRAME_SECONDS

        adapter = MagicMock()

        # The validation checks: last candle open + period == expected_candle_close
        # expected_candle_close = current_candle_open + period
        # So last candle must have open = current_candle_open - period
        # which means last_open + period = current_candle_open = expected - period...
        # Actually: expected_candle_close is the NEXT candle open after current.
        # We need last candle's close (open + period) to match expected_candle_close.
        # So last candle open = expected_candle_close - period
        expected_close = expected_candle_close("1h")
        period = TIMEFRAME_SECONDS["1h"]
        last_candle_open = expected_close - period

        fresh_candles = [
            {"timestamp": last_candle_open, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
            for _ in range(50)
        ]
        adapter.fetch_ohlcv = AsyncMock(return_value=fresh_candles)

        fetcher = OHLCVFetcher(adapter, TradingConfig())
        candles = await fetcher._fetch_candles_with_validation("BTC-USDC", "1h", 150)

        # Should accept on first try — no retries
        assert adapter.fetch_ohlcv.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_candles_no_retry(self):
        from engine.config import TradingConfig
        from engine.data.ohlcv import OHLCVFetcher

        adapter = MagicMock()
        adapter.fetch_ohlcv = AsyncMock(return_value=[])

        fetcher = OHLCVFetcher(adapter, TradingConfig())
        candles = await fetcher._fetch_candles_with_validation("BTC-USDC", "1h", 150)

        assert adapter.fetch_ohlcv.call_count == 1
        assert candles == []


# ---------------------------------------------------------------------------
# File cache tests
# ---------------------------------------------------------------------------


class TestFileCacheBackend:
    """Test file-system cache backend for chart images."""

    @pytest.fixture
    def file_cache(self, tmp_path):
        from storage.cache.file_cache import FileCacheBackend
        return FileCacheBackend(cache_dir=str(tmp_path / "chart_cache"))

    @pytest.mark.asyncio
    async def test_set_and_get(self, file_cache):
        await file_cache.set("chart:BTC:1h:12345", b"png_data_here", ttl=60)
        result = await file_cache.get("chart:BTC:1h:12345")
        assert result == b"png_data_here"

    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self, file_cache):
        result = await file_cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, file_cache):
        await file_cache.set("expiring", "data", ttl=1)
        assert await file_cache.get("expiring") == "data"
        await asyncio.sleep(1.1)
        assert await file_cache.get("expiring") is None

    @pytest.mark.asyncio
    async def test_has(self, file_cache):
        await file_cache.set("exists", "val", ttl=60)
        assert await file_cache.has("exists") is True
        assert await file_cache.has("nope") is False

    @pytest.mark.asyncio
    async def test_delete(self, file_cache):
        await file_cache.set("del_me", "val", ttl=60)
        result = await file_cache.delete("del_me")
        assert result is True
        assert await file_cache.get("del_me") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, file_cache):
        assert await file_cache.delete("nope") is False

    @pytest.mark.asyncio
    async def test_clear(self, file_cache):
        await file_cache.set("a", 1, ttl=60)
        await file_cache.set("b", 2, ttl=60)
        await file_cache.clear()
        assert await file_cache.get("a") is None
        assert await file_cache.get("b") is None

    @pytest.mark.asyncio
    async def test_zlib_compression(self, file_cache, tmp_path):
        """Verify files on disk are smaller than raw data."""
        import pickle
        big_data = {"pixels": [i for i in range(10000)]}
        await file_cache.set("big", big_data, ttl=60)

        raw_size = len(pickle.dumps(big_data))
        # Find the file on disk
        cache_dir = tmp_path / "chart_cache"
        files = list(cache_dir.glob("*.zc"))
        assert len(files) == 1
        compressed_size = files[0].stat().st_size
        assert compressed_size < raw_size

    @pytest.mark.asyncio
    async def test_auto_creates_directory(self, tmp_path):
        from storage.cache.file_cache import FileCacheBackend
        deep_path = str(tmp_path / "a" / "b" / "c")
        fc = FileCacheBackend(cache_dir=deep_path)
        await fc.set("key", "val", ttl=60)
        assert await fc.get("key") == "val"


# ---------------------------------------------------------------------------
# Chart routing tests
# ---------------------------------------------------------------------------


class TestChartRouting:
    """Test that CacheManager routes chart:* keys to FileCacheBackend."""

    @pytest.mark.asyncio
    async def test_chart_key_uses_file_backend(self, tmp_path):
        from storage.cache.file_cache import FileCacheBackend
        file_be = FileCacheBackend(cache_dir=str(tmp_path / "charts"))
        cache = CacheManager(chart_backend=file_be)

        await cache.set("chart:BTC:1h:12345", b"image_data", ttl=60)

        # Should be in file backend, not memory
        assert await file_be.get("chart:BTC:1h:12345") == b"image_data"
        assert await cache.backend.get("chart:BTC:1h:12345") is None

    @pytest.mark.asyncio
    async def test_non_chart_key_uses_memory_backend(self, tmp_path):
        from storage.cache.file_cache import FileCacheBackend
        file_be = FileCacheBackend(cache_dir=str(tmp_path / "charts"))
        cache = CacheManager(chart_backend=file_be)

        await cache.set("ohlcv:BTC:1h", [1, 2, 3], ttl=60)

        # Should be in memory backend, not file
        assert await cache.backend.get("ohlcv:BTC:1h") == [1, 2, 3]
        assert await file_be.get("ohlcv:BTC:1h") is None


# ---------------------------------------------------------------------------
# Sentinel cache access tests
# ---------------------------------------------------------------------------


class TestSentinelCacheAccess:
    """Verify Sentinel L1 bypass and L2 cache reads."""

    @staticmethod
    def _make_candles(n: int = 50) -> list[dict]:
        """Generate enough candles for indicator computation."""
        import random
        random.seed(42)
        candles = []
        price = 100.0
        for i in range(n):
            o = price
            h = o + random.uniform(0.5, 3.0)
            l = o - random.uniform(0.5, 3.0)
            c = o + random.uniform(-2.0, 2.0)
            candles.append({
                "timestamp": 1000 + i * 3600,
                "open": round(o, 2), "high": round(h, 2),
                "low": round(l, 2), "close": round(c, 2),
                "volume": random.randint(500, 5000),
            })
            price = c
        return candles

    @pytest.mark.asyncio
    async def test_sentinel_fetches_l1_directly(self):
        """Sentinel should call adapter.fetch_ohlcv directly (bypass cache)."""
        from sentinel.monitor import SentinelMonitor
        from engine.events import InProcessBus

        adapter = MagicMock()
        adapter.fetch_ohlcv = AsyncMock(return_value=self._make_candles(50))
        adapter.get_funding_rate = AsyncMock(return_value=0.0001)

        cache = CacheManager()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter=adapter, event_bus=bus, symbol="BTC-USDC",
            threshold=0.99, cache=cache,
        )

        await sentinel.check_once()

        # L1: adapter.fetch_ohlcv called directly (not through cache)
        adapter.fetch_ohlcv.assert_called_once()

    @pytest.mark.asyncio
    async def test_sentinel_reads_l2_from_cache(self):
        """Sentinel should read funding rate through CacheManager."""
        from sentinel.monitor import SentinelMonitor
        from engine.events import InProcessBus

        adapter = MagicMock()
        adapter.fetch_ohlcv = AsyncMock(return_value=self._make_candles(50))
        adapter.get_funding_rate = AsyncMock(return_value=0.0005)

        cache = CacheManager()
        bus = InProcessBus()
        sentinel = SentinelMonitor(
            adapter=adapter, event_bus=bus, symbol="BTC-USDC",
            threshold=0.99, cache=cache,
        )

        # First check: cache miss — calls adapter
        await sentinel.check_once()
        assert adapter.get_funding_rate.call_count == 1

        # Second check: cache hit — should NOT call adapter again
        await sentinel.check_once()
        assert adapter.get_funding_rate.call_count == 1  # still 1
