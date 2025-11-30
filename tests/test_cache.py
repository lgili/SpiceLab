"""Tests for the compressed cache module."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spicelab.cache import (
    CacheEntry,
    CacheStats,
    CompressedCache,
    generate_cache_key,
    hash_circuit,
)


def _check_zstd_available() -> bool:
    """Check if zstandard is available."""
    try:
        import zstandard  # noqa: F401

        return True
    except ImportError:
        return False


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_initial_stats(self) -> None:
        """Test initial statistics are zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.evictions == 0

    def test_hit_rate_no_operations(self) -> None:
        """Test hit rate with no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=80, misses=20)
        assert stats.hit_rate == pytest.approx(0.8)

    def test_compression_ratio_no_data(self) -> None:
        """Test compression ratio with no data."""
        stats = CacheStats()
        assert stats.avg_compression_ratio == 0.0

    def test_compression_ratio_calculation(self) -> None:
        """Test compression ratio calculation."""
        stats = CacheStats(
            total_compressed_bytes=1000,
            total_original_bytes=10000,
        )
        assert stats.avg_compression_ratio == pytest.approx(0.1)

    def test_space_saved_bytes(self) -> None:
        """Test space saved calculation."""
        stats = CacheStats(
            total_compressed_bytes=1000,
            total_original_bytes=10000,
        )
        assert stats.space_saved_bytes == 9000

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = CacheStats(hits=10, misses=5)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert "hit_rate" in d


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self) -> None:
        """Test creating a cache entry."""
        entry = CacheEntry(
            key="test_key",
            compressed_size=100,
            original_size=1000,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        assert entry.key == "test_key"
        assert entry.compressed_size == 100
        assert entry.original_size == 1000

    def test_compression_ratio(self) -> None:
        """Test compression ratio property."""
        entry = CacheEntry(
            key="test",
            compressed_size=100,
            original_size=1000,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        assert entry.compression_ratio == pytest.approx(0.1)

    def test_compression_ratio_zero_original(self) -> None:
        """Test compression ratio with zero original size."""
        entry = CacheEntry(
            key="test",
            compressed_size=0,
            original_size=0,
            created_at=time.time(),
            last_accessed=time.time(),
        )
        assert entry.compression_ratio == 0.0


class TestCompressedCache:
    """Tests for CompressedCache class."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        return tmp_path / "cache"

    @pytest.fixture
    def cache(self, cache_dir: Path) -> CompressedCache:
        """Create a cache instance."""
        return CompressedCache(cache_dir=cache_dir, max_size_mb=10)

    def test_init_creates_directory(self, cache_dir: Path) -> None:
        """Test that init creates cache directory."""
        CompressedCache(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_set_and_get_simple(self, cache: CompressedCache) -> None:
        """Test basic set and get operations."""
        data = {"key": "value", "number": 42}
        cache.set("test_key", data)

        result = cache.get("test_key")
        assert result == data

    def test_set_and_get_numpy_array(self, cache: CompressedCache) -> None:
        """Test caching numpy arrays."""
        data = np.random.randn(1000, 10)
        cache.set("numpy_data", data)

        result = cache.get("numpy_data")
        np.testing.assert_array_equal(result, data)

    def test_set_and_get_large_data(self, cache: CompressedCache) -> None:
        """Test caching large data."""
        # Create a large array that should compress well
        data = np.zeros((10000, 100))
        cache.set("large_data", data)

        result = cache.get("large_data")
        np.testing.assert_array_equal(result, data)

        # Check entry exists
        entry = cache.get_entry("large_data")
        assert entry is not None
        # If zstd is available, compression should happen
        # If not, sizes will be equal (fallback mode)
        assert entry.compressed_size <= entry.original_size

    def test_get_nonexistent_key(self, cache: CompressedCache) -> None:
        """Test getting a key that doesn't exist."""
        result = cache.get("nonexistent")
        assert result is None
        assert cache.stats.misses == 1

    def test_contains(self, cache: CompressedCache) -> None:
        """Test contains check."""
        assert not cache.contains("test")
        cache.set("test", "value")
        assert cache.contains("test")

    def test_delete(self, cache: CompressedCache) -> None:
        """Test deleting entries."""
        cache.set("test", "value")
        assert cache.contains("test")

        result = cache.delete("test")
        assert result is True
        assert not cache.contains("test")

    def test_delete_nonexistent(self, cache: CompressedCache) -> None:
        """Test deleting nonexistent entry."""
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear(self, cache: CompressedCache) -> None:
        """Test clearing all entries."""
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")

        assert cache.entry_count == 5

        cleared = cache.clear()
        assert cleared == 5
        assert cache.entry_count == 0

    def test_stats_tracking(self, cache: CompressedCache) -> None:
        """Test that statistics are tracked correctly."""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key2")  # Hit
        cache.get("nonexistent")  # Miss

        assert cache.stats.sets == 2
        assert cache.stats.hits == 2
        assert cache.stats.misses == 1

    def test_access_count_tracking(self, cache: CompressedCache) -> None:
        """Test that access count is tracked."""
        cache.set("test", "value")

        for _ in range(5):
            cache.get("test")

        entry = cache.get_entry("test")
        assert entry is not None
        # Access count is 5 (from the loop) + 1 from get_entry internal access
        # Actually get_entry doesn't call get, so it should be 5
        # But the entry starts at 1, so 5 gets = 6 total accesses
        assert entry.access_count == 6  # 1 initial + 5 gets

    def test_lru_eviction(self, cache_dir: Path) -> None:
        """Test LRU eviction when cache is full."""
        # Create small cache
        cache = CompressedCache(cache_dir=cache_dir, max_size_mb=1)

        # Fill cache with entries
        for i in range(100):
            data = np.random.randn(1000)  # ~8KB each
            cache.set(f"key_{i}", data)

        # Some entries should have been evicted
        assert cache.size_mb <= 1.0

    def test_manual_eviction(self, cache: CompressedCache) -> None:
        """Test manual LRU eviction."""
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
            time.sleep(0.01)  # Ensure different timestamps

        # Keep only 5 entries
        evicted = cache.evict_lru(keep_count=5)
        assert evicted == 5
        assert cache.entry_count == 5

    def test_get_entries(self, cache: CompressedCache) -> None:
        """Test getting all entries."""
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")

        entries = cache.get_entries()
        assert len(entries) == 5
        assert all(isinstance(e, CacheEntry) for e in entries)

    def test_disabled_cache(self, cache_dir: Path) -> None:
        """Test that disabled cache doesn't store anything."""
        cache = CompressedCache(cache_dir=cache_dir, enabled=False)

        cache.set("test", "value")
        result = cache.get("test")

        assert result is None
        assert cache.stats.misses == 1

    def test_size_properties(self, cache: CompressedCache) -> None:
        """Test size tracking properties."""
        assert cache.size_bytes == 0
        assert cache.size_mb == 0.0

        cache.set("test", np.random.randn(1000))

        assert cache.size_bytes > 0
        assert cache.size_mb > 0.0

    def test_metadata_storage(self, cache: CompressedCache) -> None:
        """Test storing metadata with entries."""
        cache.set("test", "value", metadata={"source": "test", "version": 1})

        entry = cache.get_entry("test")
        assert entry is not None
        assert entry.metadata["source"] == "test"
        assert entry.metadata["version"] == 1

    def test_persistence(self, cache_dir: Path) -> None:
        """Test cache persistence across instances."""
        # Create and populate cache
        cache1 = CompressedCache(cache_dir=cache_dir)
        cache1.set("persistent_key", {"data": "important"})
        cache1.save_stats()

        # Create new instance with same directory
        cache2 = CompressedCache(cache_dir=cache_dir)

        # Should find the cached data
        assert cache2.contains("persistent_key")
        result = cache2.get("persistent_key")
        assert result == {"data": "important"}


class TestHashCircuit:
    """Tests for hash_circuit function."""

    def test_hash_string_netlist(self) -> None:
        """Test hashing a string netlist."""
        netlist = ".title Test\nR1 1 0 1k\n.end"
        hash1 = hash_circuit(netlist)

        assert len(hash1) == 64  # SHA-256 hex length
        assert hash1 == hash_circuit(netlist)  # Deterministic

    def test_hash_different_netlists(self) -> None:
        """Test that different netlists produce different hashes."""
        netlist1 = "R1 1 0 1k"
        netlist2 = "R1 1 0 2k"

        hash1 = hash_circuit(netlist1)
        hash2 = hash_circuit(netlist2)

        assert hash1 != hash2

    def test_hash_circuit_object(self) -> None:
        """Test hashing an object with build_netlist method."""

        class MockCircuit:
            def build_netlist(self) -> str:
                return "R1 1 0 1k"

        circuit = MockCircuit()
        hash_val = hash_circuit(circuit)

        assert len(hash_val) == 64

    def test_hash_circuit_with_to_netlist(self) -> None:
        """Test hashing an object with to_netlist method."""

        class MockCircuit:
            def to_netlist(self) -> str:
                return "R1 1 0 1k"

        circuit = MockCircuit()
        hash_val = hash_circuit(circuit)

        assert len(hash_val) == 64


class TestGenerateCacheKey:
    """Tests for generate_cache_key function."""

    def test_key_from_circuit_string(self) -> None:
        """Test generating key from circuit string."""
        key = generate_cache_key("R1 1 0 1k")
        assert len(key) == 64

    def test_key_with_analyses(self) -> None:
        """Test key includes analysis specifications."""

        class MockAnalysis:
            def to_dict(self) -> dict[str, Any]:
                return {"type": "tran", "stop": "1m"}

        key1 = generate_cache_key("R1 1 0 1k", analyses=[MockAnalysis()])
        key2 = generate_cache_key("R1 1 0 1k", analyses=None)

        assert key1 != key2

    def test_key_with_parameters(self) -> None:
        """Test key includes parameters."""
        key1 = generate_cache_key("R1 1 0 1k", parameters={"temp": 25})
        key2 = generate_cache_key("R1 1 0 1k", parameters={"temp": 85})

        assert key1 != key2

    def test_key_deterministic(self) -> None:
        """Test that key generation is deterministic."""
        key1 = generate_cache_key(
            "R1 1 0 1k",
            parameters={"a": 1, "b": 2},
        )
        key2 = generate_cache_key(
            "R1 1 0 1k",
            parameters={"b": 2, "a": 1},  # Different order
        )

        assert key1 == key2  # Should be same due to sorting


class TestCompressionEfficiency:
    """Tests for compression efficiency."""

    @pytest.fixture
    def cache(self, tmp_path: Path) -> CompressedCache:
        """Create a cache instance."""
        return CompressedCache(cache_dir=tmp_path / "cache")

    @pytest.mark.skipif(
        not _check_zstd_available(),
        reason="zstandard not available",
    )
    def test_compressible_data(self, cache: CompressedCache) -> None:
        """Test that compressible data is compressed well."""
        # Highly compressible: repeating pattern
        data = np.zeros((10000, 100))
        entry = cache.set("zeros", data)

        # Should achieve good compression (10:1 or better)
        assert entry.compression_ratio < 0.2

    @pytest.mark.skipif(
        not _check_zstd_available(),
        reason="zstandard not available",
    )
    def test_random_data_compression(self, cache: CompressedCache) -> None:
        """Test compression of random data."""
        # Random data compresses poorly
        data = np.random.randn(1000, 100)
        entry = cache.set("random", data)

        # Random float data typically doesn't compress well
        # But zstd should still achieve some compression
        assert entry.compression_ratio < 1.0

    @pytest.mark.skipif(
        not _check_zstd_available(),
        reason="zstandard not available",
    )
    def test_text_data_compression(self, cache: CompressedCache) -> None:
        """Test compression of text data."""
        # Text typically compresses well
        data = "Hello World! " * 10000
        entry = cache.set("text", data)

        assert entry.compression_ratio < 0.1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_value(self, tmp_path: Path) -> None:
        """Test caching empty values."""
        cache = CompressedCache(cache_dir=tmp_path / "cache")

        cache.set("empty_dict", {})
        cache.set("empty_list", [])
        cache.set("empty_string", "")

        assert cache.get("empty_dict") == {}
        assert cache.get("empty_list") == []
        assert cache.get("empty_string") == ""

    def test_none_value(self, tmp_path: Path) -> None:
        """Test caching None value."""
        cache = CompressedCache(cache_dir=tmp_path / "cache")

        cache.set("none_value", None)
        result = cache.get("none_value")

        # Need to distinguish between "not found" and "cached None"
        assert cache.contains("none_value")

    def test_special_characters_in_key(self, tmp_path: Path) -> None:
        """Test keys with special characters."""
        cache = CompressedCache(cache_dir=tmp_path / "cache")

        # Keys are hashed, so special chars should be fine
        cache.set("key/with/slashes", "value1")
        cache.set("key with spaces", "value2")
        cache.set("key:with:colons", "value3")

        assert cache.get("key/with/slashes") == "value1"
        assert cache.get("key with spaces") == "value2"
        assert cache.get("key:with:colons") == "value3"

    def test_unicode_key_and_value(self, tmp_path: Path) -> None:
        """Test Unicode in keys and values."""
        cache = CompressedCache(cache_dir=tmp_path / "cache")

        cache.set("키_key_键", {"データ": "значение"})
        result = cache.get("키_key_键")

        assert result == {"データ": "значение"}

    def test_very_long_key(self, tmp_path: Path) -> None:
        """Test very long keys."""
        cache = CompressedCache(cache_dir=tmp_path / "cache")

        long_key = "a" * 10000
        cache.set(long_key, "value")

        assert cache.get(long_key) == "value"
