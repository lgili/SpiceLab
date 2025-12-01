"""Tests for the distributed cache module."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from spicelab.cache.distributed import (
    DistributedCacheStats,
    MockDistributedCache,
    is_redis_available,
)


class TestDistributedCacheStats:
    """Tests for DistributedCacheStats dataclass."""

    def test_initial_stats(self) -> None:
        """Test initial statistics are zero."""
        stats = DistributedCacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = DistributedCacheStats(hits=80, misses=20)
        assert stats.hit_rate == pytest.approx(0.8)

    def test_reset(self) -> None:
        """Test resetting stats."""
        stats = DistributedCacheStats(hits=10, misses=5, sets=15)
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = DistributedCacheStats(hits=10, misses=5)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert "hit_rate" in d


class TestIsRedisAvailable:
    """Tests for is_redis_available function."""

    def test_returns_bool(self) -> None:
        """Test that function returns boolean."""
        result = is_redis_available()
        assert isinstance(result, bool)


class TestMockDistributedCache:
    """Tests for MockDistributedCache class (in-memory mock)."""

    @pytest.fixture
    def cache(self) -> MockDistributedCache:
        """Create mock cache instance."""
        return MockDistributedCache(prefix="test:")

    def test_create_cache(self) -> None:
        """Test creating mock cache."""
        cache = MockDistributedCache()
        assert cache.ping() is True

    def test_set_and_get(self, cache: MockDistributedCache) -> None:
        """Test basic set and get operations."""
        cache.set("key1", {"data": "value"})
        result = cache.get("key1")
        assert result == {"data": "value"}

    def test_get_nonexistent(self, cache: MockDistributedCache) -> None:
        """Test getting nonexistent key."""
        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats.misses == 1

    def test_set_with_ttl(self, cache: MockDistributedCache) -> None:
        """Test set with TTL."""
        cache.set("expiring", "value", ttl=1)
        assert cache.get("expiring") == "value"

        # Wait for expiry
        time.sleep(1.1)
        assert cache.get("expiring") is None

    def test_exists(self, cache: MockDistributedCache) -> None:
        """Test exists check."""
        assert cache.exists("key1") is False
        cache.set("key1", "value")
        assert cache.exists("key1") is True

    def test_delete(self, cache: MockDistributedCache) -> None:
        """Test delete operation."""
        cache.set("to_delete", "value")
        assert cache.exists("to_delete")

        result = cache.delete("to_delete")
        assert result is True
        assert cache.exists("to_delete") is False

    def test_delete_nonexistent(self, cache: MockDistributedCache) -> None:
        """Test deleting nonexistent key."""
        result = cache.delete("nonexistent")
        assert result is False

    def test_get_ttl(self, cache: MockDistributedCache) -> None:
        """Test getting TTL."""
        # Nonexistent key
        assert cache.get_ttl("nonexistent") == -2

        # Key without TTL
        cache.set("no_ttl", "value")
        assert cache.get_ttl("no_ttl") == -1

        # Key with TTL
        cache.set("with_ttl", "value", ttl=60)
        ttl = cache.get_ttl("with_ttl")
        assert 58 <= ttl <= 60

    def test_expire(self, cache: MockDistributedCache) -> None:
        """Test setting expiry on existing key."""
        cache.set("key", "value")
        assert cache.get_ttl("key") == -1

        cache.expire("key", 30)
        ttl = cache.get_ttl("key")
        assert 28 <= ttl <= 30

    def test_keys_pattern(self, cache: MockDistributedCache) -> None:
        """Test listing keys with pattern."""
        cache.set("user:1", "a")
        cache.set("user:2", "b")
        cache.set("session:1", "c")

        user_keys = cache.keys("user:*")
        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

    def test_clear(self, cache: MockDistributedCache) -> None:
        """Test clearing cache."""
        cache.set("key1", "a")
        cache.set("key2", "b")

        count = cache.clear()
        assert count == 2
        assert cache.exists("key1") is False

    def test_clear_pattern(self, cache: MockDistributedCache) -> None:
        """Test clearing with pattern."""
        cache.set("user:1", "a")
        cache.set("user:2", "b")
        cache.set("session:1", "c")

        count = cache.clear("user:*")
        assert count == 2
        assert cache.exists("session:1") is True

    def test_mget(self, cache: MockDistributedCache) -> None:
        """Test getting multiple values."""
        cache.set("k1", "v1")
        cache.set("k2", "v2")

        result = cache.mget(["k1", "k2", "k3"])
        assert result["k1"] == "v1"
        assert result["k2"] == "v2"
        assert result["k3"] is None

    def test_mset(self, cache: MockDistributedCache) -> None:
        """Test setting multiple values."""
        result = cache.mset({"a": 1, "b": 2, "c": 3})
        assert result is True

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_mset_with_ttl(self, cache: MockDistributedCache) -> None:
        """Test setting multiple values with TTL."""
        cache.mset({"x": 1, "y": 2}, ttl=60)

        ttl_x = cache.get_ttl("x")
        ttl_y = cache.get_ttl("y")
        assert 58 <= ttl_x <= 60
        assert 58 <= ttl_y <= 60

    def test_metadata(self, cache: MockDistributedCache) -> None:
        """Test storing and retrieving metadata."""
        cache.set("key", "value", metadata={"source": "test", "version": 1})

        meta = cache.get_metadata("key")
        assert meta is not None
        assert meta["source"] == "test"
        assert meta["version"] == 1

    def test_metadata_nonexistent(self, cache: MockDistributedCache) -> None:
        """Test getting metadata for nonexistent key."""
        meta = cache.get_metadata("nonexistent")
        assert meta is None

    def test_info(self, cache: MockDistributedCache) -> None:
        """Test getting cache info."""
        cache.set("k1", "v1")
        cache.set("k2", "v2")

        info = cache.info()
        assert info["type"] == "mock"
        assert info["total_keys"] == 2

    def test_close(self, cache: MockDistributedCache) -> None:
        """Test closing (no-op for mock)."""
        cache.close()  # Should not raise

    def test_stats_tracking(self, cache: MockDistributedCache) -> None:
        """Test that stats are tracked correctly."""
        cache.set("key", "value")  # 1 set
        cache.get("key")  # 1 hit
        cache.get("missing")  # 1 miss
        cache.delete("key")  # 1 delete

        assert cache.stats.sets == 1
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.deletes == 1

    def test_default_ttl(self) -> None:
        """Test default TTL."""
        cache = MockDistributedCache(default_ttl=30)
        cache.set("key", "value")

        ttl = cache.get_ttl("key")
        assert 28 <= ttl <= 30


class TestMockCacheDataTypes:
    """Tests for different data types in mock cache."""

    @pytest.fixture
    def cache(self) -> MockDistributedCache:
        """Create mock cache instance."""
        return MockDistributedCache()

    def test_string_value(self, cache: MockDistributedCache) -> None:
        """Test caching string values."""
        cache.set("string", "hello world")
        assert cache.get("string") == "hello world"

    def test_int_value(self, cache: MockDistributedCache) -> None:
        """Test caching integer values."""
        cache.set("int", 42)
        assert cache.get("int") == 42

    def test_float_value(self, cache: MockDistributedCache) -> None:
        """Test caching float values."""
        cache.set("float", 3.14159)
        assert cache.get("float") == pytest.approx(3.14159)

    def test_list_value(self, cache: MockDistributedCache) -> None:
        """Test caching list values."""
        cache.set("list", [1, 2, 3, 4, 5])
        assert cache.get("list") == [1, 2, 3, 4, 5]

    def test_dict_value(self, cache: MockDistributedCache) -> None:
        """Test caching dictionary values."""
        data = {"name": "test", "values": [1, 2, 3]}
        cache.set("dict", data)
        assert cache.get("dict") == data

    def test_nested_dict(self, cache: MockDistributedCache) -> None:
        """Test caching nested dictionary."""
        data = {
            "level1": {
                "level2": {
                    "level3": "deep"
                }
            }
        }
        cache.set("nested", data)
        assert cache.get("nested") == data

    def test_numpy_array(self, cache: MockDistributedCache) -> None:
        """Test caching numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cache.set("numpy", arr)

        result = cache.get("numpy")
        np.testing.assert_array_equal(result, arr)

    def test_numpy_2d_array(self, cache: MockDistributedCache) -> None:
        """Test caching 2D numpy arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        cache.set("numpy2d", arr)

        result = cache.get("numpy2d")
        np.testing.assert_array_equal(result, arr)

    def test_none_value(self, cache: MockDistributedCache) -> None:
        """Test caching None value."""
        cache.set("none", None)
        # Note: get returns None for both missing and None value
        # Use exists() to distinguish
        assert cache.exists("none")

    def test_bool_value(self, cache: MockDistributedCache) -> None:
        """Test caching boolean values."""
        cache.set("true", True)
        cache.set("false", False)

        assert cache.get("true") is True
        assert cache.get("false") is False


class TestMockCacheConcurrency:
    """Tests for cache behavior with multiple operations."""

    def test_overwrite(self) -> None:
        """Test overwriting existing key."""
        cache = MockDistributedCache()

        cache.set("key", "value1")
        cache.set("key", "value2")

        assert cache.get("key") == "value2"

    def test_many_keys(self) -> None:
        """Test with many keys."""
        cache = MockDistributedCache()

        for i in range(100):
            cache.set(f"key_{i}", f"value_{i}")

        for i in range(100):
            assert cache.get(f"key_{i}") == f"value_{i}"

    def test_large_value(self) -> None:
        """Test with large value."""
        cache = MockDistributedCache()

        # 1MB of data
        large_data = np.random.randn(1000000)
        cache.set("large", large_data)

        result = cache.get("large")
        np.testing.assert_array_equal(result, large_data)


class TestMockCacheEdgeCases:
    """Tests for edge cases."""

    def test_empty_key(self) -> None:
        """Test empty key."""
        cache = MockDistributedCache()
        cache.set("", "value")
        assert cache.get("") == "value"

    def test_special_characters_key(self) -> None:
        """Test keys with special characters."""
        cache = MockDistributedCache()

        cache.set("key:with:colons", "v1")
        cache.set("key/with/slashes", "v2")
        cache.set("key with spaces", "v3")

        assert cache.get("key:with:colons") == "v1"
        assert cache.get("key/with/slashes") == "v2"
        assert cache.get("key with spaces") == "v3"

    def test_unicode_key(self) -> None:
        """Test Unicode key."""
        cache = MockDistributedCache()

        cache.set("日本語キー", "value")
        assert cache.get("日本語キー") == "value"

    def test_unicode_value(self) -> None:
        """Test Unicode value."""
        cache = MockDistributedCache()

        cache.set("key", "中文值")
        assert cache.get("key") == "中文值"

    def test_very_long_key(self) -> None:
        """Test very long key."""
        cache = MockDistributedCache()

        long_key = "a" * 1000
        cache.set(long_key, "value")
        assert cache.get(long_key) == "value"

    def test_empty_mget(self) -> None:
        """Test mget with empty list."""
        cache = MockDistributedCache()
        result = cache.mget([])
        assert result == {}

    def test_empty_mset(self) -> None:
        """Test mset with empty dict."""
        cache = MockDistributedCache()
        result = cache.mset({})
        assert result is True
