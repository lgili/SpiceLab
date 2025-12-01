"""Compressed cache for simulation results.

This module provides zstd-compressed caching for simulation results,
reducing disk usage by 80-95% compared to uncompressed storage.

Key features:
- zstd compression with configurable levels
- LRU eviction when cache exceeds size limits
- Cache statistics tracking (hits, misses, compression ratios)
- Thread-safe operations
- Automatic cache directory management

Example::

    from spicelab.cache import CompressedCache

    cache = CompressedCache(max_size_mb=1000, compression_level=3)

    # Store result
    cache.set("simulation_key", result_dataset)

    # Retrieve result
    cached = cache.get("simulation_key")

    # Check statistics
    print(f"Hit rate: {cache.stats.hit_rate:.1%}")
    print(f"Compression: {cache.stats.avg_compression_ratio:.1%}")

"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Optional zstandard import
try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    zstd = None  # type: ignore[assignment]
    ZSTD_AVAILABLE = False


@dataclass
class CacheStats:
    """Statistics for cache operations.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        sets: Number of cache sets
        evictions: Number of cache evictions
        total_compressed_bytes: Total bytes after compression
        total_original_bytes: Total bytes before compression
        compression_time_ms: Total time spent compressing
        decompression_time_ms: Total time spent decompressing

    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    total_compressed_bytes: int = 0
    total_original_bytes: int = 0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0 to 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def avg_compression_ratio(self) -> float:
        """Average compression ratio (compressed/original)."""
        if self.total_original_bytes == 0:
            return 0.0
        return self.total_compressed_bytes / self.total_original_bytes

    @property
    def space_saved_bytes(self) -> int:
        """Total bytes saved by compression."""
        return self.total_original_bytes - self.total_compressed_bytes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "avg_compression_ratio": self.avg_compression_ratio,
            "space_saved_bytes": self.space_saved_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "total_original_bytes": self.total_original_bytes,
            "compression_time_ms": self.compression_time_ms,
            "decompression_time_ms": self.decompression_time_ms,
        }


@dataclass
class CacheEntry:
    """Metadata for a single cache entry.

    Attributes:
        key: Cache key
        compressed_size: Size after compression in bytes
        original_size: Size before compression in bytes
        created_at: Unix timestamp when entry was created
        last_accessed: Unix timestamp of last access
        access_count: Number of times entry was accessed

    """

    key: str
    compressed_size: int
    original_size: int
    created_at: float
    last_accessed: float
    access_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio for this entry."""
        if self.original_size == 0:
            return 0.0
        return self.compressed_size / self.original_size


class CompressedCache:
    """zstd-compressed cache for simulation results.

    This cache stores serialized Python objects (datasets, results) with
    zstd compression, significantly reducing disk usage.

    Args:
        cache_dir: Directory for cache storage. Defaults to ~/.spicelab/cache
        max_size_mb: Maximum cache size in MB. When exceeded, LRU eviction occurs.
        compression_level: zstd compression level (1-22). Higher = smaller files.
        enabled: Whether caching is enabled. Useful for testing.

    Example::

        cache = CompressedCache(max_size_mb=500, compression_level=3)

        # Store a result
        cache.set("my_simulation", result_dataset)

        # Check if exists
        if cache.contains("my_simulation"):
            result = cache.get("my_simulation")

        # Clear old entries
        cache.evict_lru(keep_count=100)

        # Clear everything
        cache.clear()

    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        max_size_mb: int = 1000,
        compression_level: int = 3,
        enabled: bool = True,
    ) -> None:
        """Initialize the compressed cache."""
        if cache_dir is None:
            cache_dir = Path.home() / ".spicelab" / "cache"
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compression_level = compression_level
        self.enabled = enabled

        self._stats = CacheStats()
        self._entries: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()

        # Initialize zstd compressor/decompressor if available
        if ZSTD_AVAILABLE and zstd is not None:
            self._compressor = zstd.ZstdCompressor(level=compression_level)
            self._decompressor = zstd.ZstdDecompressor()
        else:
            self._compressor = None
            self._decompressor = None

        # Create cache directory
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def size_bytes(self) -> int:
        """Current cache size in bytes."""
        with self._lock:
            return sum(e.compressed_size for e in self._entries.values())

    @property
    def size_mb(self) -> float:
        """Current cache size in MB."""
        return self.size_bytes / (1024 * 1024)

    @property
    def entry_count(self) -> int:
        """Number of entries in cache."""
        with self._lock:
            return len(self._entries)

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Use first 2 chars of hash as subdirectory for better filesystem performance
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        subdir = key_hash[:2]
        filename = f"{key_hash}.zst"
        return self.cache_dir / subdir / filename

    def _get_index_path(self) -> Path:
        """Get path to the cache index file."""
        return self.cache_dir / "index.json"

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_path = self._get_index_path()
        if not index_path.exists():
            return

        try:
            with open(index_path) as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = CacheEntry(
                    key=entry_data["key"],
                    compressed_size=entry_data["compressed_size"],
                    original_size=entry_data["original_size"],
                    created_at=entry_data["created_at"],
                    last_accessed=entry_data["last_accessed"],
                    access_count=entry_data.get("access_count", 1),
                    metadata=entry_data.get("metadata", {}),
                )
                # Verify file exists
                if self._get_cache_path(entry.key).exists():
                    self._entries[entry.key] = entry
        except (json.JSONDecodeError, KeyError, OSError):
            # Index corrupted, rebuild from files
            self._rebuild_index()

    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_path = self._get_index_path()
        data = {
            "version": 1,
            "entries": [
                {
                    "key": e.key,
                    "compressed_size": e.compressed_size,
                    "original_size": e.original_size,
                    "created_at": e.created_at,
                    "last_accessed": e.last_accessed,
                    "access_count": e.access_count,
                    "metadata": e.metadata,
                }
                for e in self._entries.values()
            ],
        }

        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _rebuild_index(self) -> None:
        """Rebuild index by scanning cache directory."""
        self._entries.clear()

        for subdir in self.cache_dir.iterdir():
            if not subdir.is_dir() or subdir.name == "__pycache__":
                continue
            for file in subdir.glob("*.zst"):
                # We can't recover the original key from hash, so use hash as key
                key_hash = file.stem
                stat = file.stat()
                self._entries[key_hash] = CacheEntry(
                    key=key_hash,
                    compressed_size=stat.st_size,
                    original_size=stat.st_size,  # Unknown, assume same
                    created_at=stat.st_ctime,
                    last_accessed=stat.st_mtime,
                )

        self._save_index()

    def contains(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enabled:
            return False

        with self._lock:
            if key not in self._entries:
                return False
            # Verify file exists
            return self._get_cache_path(key).exists()

    def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found

        """
        if not self.enabled:
            self._stats.misses += 1
            return None

        with self._lock:
            if key not in self._entries:
                self._stats.misses += 1
                return None

            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                # Entry in index but file missing
                del self._entries[key]
                self._stats.misses += 1
                return None

            try:
                # Read and decompress
                t0 = time.perf_counter()
                compressed_data = cache_path.read_bytes()

                if self._decompressor is not None:
                    data = self._decompressor.decompress(compressed_data)
                else:
                    # Fallback: no compression
                    data = compressed_data

                decompress_time = (time.perf_counter() - t0) * 1000
                self._stats.decompression_time_ms += decompress_time

                # Deserialize
                result = pickle.loads(data)

                # Update access tracking
                entry = self._entries[key]
                entry.last_accessed = time.time()
                entry.access_count += 1

                self._stats.hits += 1
                return result

            except Exception:
                # Corrupted entry
                self._stats.misses += 1
                return None

    def set(
        self, key: str, value: Any, metadata: dict[str, Any] | None = None
    ) -> CacheEntry:
        """Store a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be pickleable)
            metadata: Optional metadata to store with entry

        Returns:
            CacheEntry with compression statistics

        """
        if not self.enabled:
            return CacheEntry(
                key=key,
                compressed_size=0,
                original_size=0,
                created_at=time.time(),
                last_accessed=time.time(),
            )

        with self._lock:
            # Serialize
            data = pickle.dumps(value)
            original_size = len(data)

            # Compress
            t0 = time.perf_counter()
            if self._compressor is not None:
                compressed_data = self._compressor.compress(data)
            else:
                # Fallback: no compression
                compressed_data = data
            compress_time = (time.perf_counter() - t0) * 1000
            compressed_size = len(compressed_data)

            # Update stats
            self._stats.compression_time_ms += compress_time
            self._stats.total_original_bytes += original_size
            self._stats.total_compressed_bytes += compressed_size
            self._stats.sets += 1

            # Write to disk
            cache_path = self._get_cache_path(key)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(compressed_data)

            # Create entry
            now = time.time()
            entry = CacheEntry(
                key=key,
                compressed_size=compressed_size,
                original_size=original_size,
                created_at=now,
                last_accessed=now,
                metadata=metadata or {},
            )
            self._entries[key] = entry

            # Check if eviction needed
            if self.size_bytes > self.max_size_bytes:
                self._evict_lru()

            # Save index periodically
            if self._stats.sets % 10 == 0:
                self._save_index()

            return entry

    def delete(self, key: str) -> bool:
        """Delete an entry from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found

        """
        with self._lock:
            if key not in self._entries:
                return False

            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

            del self._entries[key]
            return True

    def _evict_lru(self) -> int:
        """Evict least recently used entries until under size limit.

        Returns:
            Number of entries evicted

        """
        evicted = 0
        target_size = int(self.max_size_bytes * 0.9)  # Evict to 90% of max

        # Sort by last_accessed
        sorted_entries = sorted(
            self._entries.values(), key=lambda e: e.last_accessed
        )

        current_size = self.size_bytes
        for entry in sorted_entries:
            if current_size <= target_size:
                break

            if self.delete(entry.key):
                current_size -= entry.compressed_size
                evicted += 1
                self._stats.evictions += 1

        if evicted > 0:
            self._save_index()

        return evicted

    def evict_lru(self, keep_count: int | None = None, keep_bytes: int | None = None) -> int:
        """Manually evict LRU entries.

        Args:
            keep_count: Keep only this many entries
            keep_bytes: Keep only this many bytes

        Returns:
            Number of entries evicted

        """
        with self._lock:
            evicted = 0

            # Sort by last_accessed
            sorted_entries = sorted(
                self._entries.values(), key=lambda e: e.last_accessed
            )

            current_size = self.size_bytes
            current_count = len(self._entries)

            for entry in sorted_entries:
                should_evict = False

                if keep_count is not None and current_count > keep_count:
                    should_evict = True
                if keep_bytes is not None and current_size > keep_bytes:
                    should_evict = True

                if not should_evict:
                    break

                if self.delete(entry.key):
                    current_size -= entry.compressed_size
                    current_count -= 1
                    evicted += 1
                    self._stats.evictions += 1

            if evicted > 0:
                self._save_index()

            return evicted

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared

        """
        with self._lock:
            count = len(self._entries)

            # Remove all cache files
            for key in list(self._entries.keys()):
                self.delete(key)

            # Remove subdirectories
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir() and subdir.name != "__pycache__":
                    shutil.rmtree(subdir, ignore_errors=True)

            self._entries.clear()
            self._save_index()

            return count

    def get_entries(self) -> list[CacheEntry]:
        """Get all cache entries.

        Returns:
            List of CacheEntry objects

        """
        with self._lock:
            return list(self._entries.values())

    def get_entry(self, key: str) -> CacheEntry | None:
        """Get metadata for a cache entry.

        Args:
            key: Cache key

        Returns:
            CacheEntry or None if not found

        """
        with self._lock:
            return self._entries.get(key)

    def save_stats(self) -> None:
        """Save cache statistics and index to disk."""
        with self._lock:
            self._save_index()

            stats_path = self.cache_dir / "stats.json"
            with open(stats_path, "w") as f:
                json.dump(self._stats.to_dict(), f, indent=2)


def hash_circuit(circuit: Any) -> str:
    """Generate a hash from a circuit object.

    This function creates a unique identifier for a circuit based on its
    netlist representation. Useful for cache key generation.

    Args:
        circuit: A Circuit object with a build_netlist() or to_netlist() method

    Returns:
        SHA-256 hash of the circuit netlist

    """
    # Try different methods to get netlist
    netlist = None

    if hasattr(circuit, "build_netlist"):
        netlist = circuit.build_netlist()
    elif hasattr(circuit, "to_netlist"):
        netlist = circuit.to_netlist()
    elif hasattr(circuit, "netlist"):
        netlist = circuit.netlist
    elif isinstance(circuit, str):
        netlist = circuit
    else:
        # Fallback: use pickle hash
        netlist = pickle.dumps(circuit)
        return hashlib.sha256(netlist).hexdigest()

    if isinstance(netlist, str):
        netlist = netlist.encode("utf-8")

    return hashlib.sha256(netlist).hexdigest()


def generate_cache_key(
    circuit: Any,
    analyses: list[Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """Generate a cache key for a simulation.

    This creates a unique key based on the circuit, analysis types,
    and any parameters that affect the simulation result.

    Args:
        circuit: Circuit object or netlist string
        analyses: List of analysis specifications
        parameters: Additional parameters affecting simulation

    Returns:
        Cache key string

    """
    hasher = hashlib.sha256()

    # Hash circuit
    circuit_hash = hash_circuit(circuit)
    hasher.update(circuit_hash.encode())

    # Hash analyses
    if analyses:
        for analysis in analyses:
            if hasattr(analysis, "to_dict"):
                analysis_str = json.dumps(analysis.to_dict(), sort_keys=True)
            elif hasattr(analysis, "__dict__"):
                analysis_str = json.dumps(
                    {k: str(v) for k, v in analysis.__dict__.items()},
                    sort_keys=True,
                )
            else:
                analysis_str = str(analysis)
            hasher.update(analysis_str.encode())

    # Hash parameters
    if parameters:
        param_str = json.dumps(
            {k: str(v) for k, v in sorted(parameters.items())},
            sort_keys=True,
        )
        hasher.update(param_str.encode())

    return hasher.hexdigest()
