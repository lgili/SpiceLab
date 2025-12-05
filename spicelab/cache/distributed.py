"""Distributed cache using Redis for team collaboration.

This module provides a distributed cache implementation using Redis,
allowing simulation results to be shared across team members and machines.

Example::

    from spicelab.cache import DistributedCache

    # Connect to Redis
    cache = DistributedCache(host="localhost", port=6379)

    # Store simulation result
    cache.set("sim_001", result_data, ttl=3600)

    # Retrieve from any machine
    result = cache.get("sim_001")

    # Check cache statistics
    print(cache.stats)

"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Try to import redis
_REDIS_AVAILABLE = False
try:
    import redis

    _REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]


def is_redis_available() -> bool:
    """Check if Redis client is available."""
    return _REDIS_AVAILABLE


@dataclass
class DistributedCacheStats:
    """Statistics for distributed cache.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        sets: Number of set operations
        deletes: Number of delete operations
        errors: Number of errors encountered
        bytes_sent: Total bytes sent to Redis
        bytes_received: Total bytes received from Redis

    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.bytes_sent = 0
        self.bytes_received = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
        }


@dataclass
class CacheEntry:
    """Entry stored in distributed cache.

    Attributes:
        key: Cache key
        value: Cached value (serialized)
        created_at: Unix timestamp when created
        ttl: Time to live in seconds
        metadata: Additional metadata

    """

    key: str
    value: bytes
    created_at: float = field(default_factory=time.time)
    ttl: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DistributedCache:
    """Distributed cache using Redis.

    This cache allows simulation results to be shared across
    multiple machines and team members.

    Args:
        host: Redis server host
        port: Redis server port
        db: Redis database number
        password: Redis password (optional)
        prefix: Key prefix for namespacing
        default_ttl: Default TTL in seconds (None for no expiry)
        serializer: Serialization method ('pickle' or 'json')

    Example::

        cache = DistributedCache(host="redis.example.com")

        # Store with TTL
        cache.set("simulation_result", data, ttl=3600)

        # Retrieve
        data = cache.get("simulation_result")

        # Store numpy arrays
        cache.set("waveform", np.array([1, 2, 3]))

    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        prefix: str = "spicelab:",
        default_ttl: int | None = None,
        serializer: str = "pickle",
        connect_timeout: float = 5.0,
    ) -> None:
        """Initialize distributed cache."""
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "redis is required for distributed cache. "
                "Install with: pip install redis"
            )

        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.serializer = serializer

        # Statistics
        self.stats = DistributedCacheStats()

        # Create Redis client
        self._client: redis.Redis[bytes] = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            socket_timeout=connect_timeout,
            socket_connect_timeout=connect_timeout,
            decode_responses=False,
        )

        # Connection pool for better performance
        self._pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=10,
        )

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        if self.serializer == "json":
            # Handle numpy arrays
            if isinstance(value, np.ndarray):
                return json.dumps({
                    "__numpy__": True,
                    "data": value.tolist(),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }).encode()
            return json.dumps(value, default=str).encode()
        else:
            return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        if self.serializer == "json":
            obj = json.loads(data.decode())
            # Handle numpy arrays
            if isinstance(obj, dict) and obj.get("__numpy__"):
                return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
            return obj
        else:
            return pickle.loads(data)

    def ping(self) -> bool:
        """Check if Redis server is reachable.

        Returns:
            True if server responds, False otherwise

        """
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found

        """
        full_key = self._make_key(key)
        try:
            data = self._client.get(full_key)
            if data is None:
                self.stats.misses += 1
                return None

            self.stats.hits += 1
            self.stats.bytes_received += len(data)
            return self._deserialize(data)
        except Exception:
            self.stats.errors += 1
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None uses default)
            metadata: Optional metadata (stored separately)

        Returns:
            True if successful, False otherwise

        """
        full_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl

        try:
            data = self._serialize(value)
            self.stats.bytes_sent += len(data)

            if ttl is not None:
                result = self._client.setex(full_key, ttl, data)
            else:
                result = self._client.set(full_key, data)

            # Store metadata if provided
            if metadata:
                meta_key = f"{full_key}:meta"
                meta_data = json.dumps(metadata).encode()
                if ttl is not None:
                    self._client.setex(meta_key, ttl, meta_data)
                else:
                    self._client.set(meta_key, meta_data)

            self.stats.sets += 1
            return bool(result)
        except Exception:
            self.stats.errors += 1
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found

        """
        full_key = self._make_key(key)
        try:
            result = self._client.delete(full_key)
            # Also delete metadata
            self._client.delete(f"{full_key}:meta")
            self.stats.deletes += 1
            return bool(result > 0)
        except Exception:
            self.stats.errors += 1
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists, False otherwise

        """
        full_key = self._make_key(key)
        try:
            return bool(self._client.exists(full_key))
        except Exception:
            return False

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for a cached entry.

        Args:
            key: Cache key

        Returns:
            Metadata dictionary or None

        """
        meta_key = f"{self._make_key(key)}:meta"
        try:
            data = self._client.get(meta_key)
            if data:
                result = json.loads(data.decode())
                return dict(result) if isinstance(result, dict) else None
            return None
        except Exception:
            return None

    def get_ttl(self, key: str) -> int:
        """Get remaining TTL for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no expiry, -2 if not found

        """
        full_key = self._make_key(key)
        try:
            return int(self._client.ttl(full_key))
        except Exception:
            return -2

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on existing key.

        Args:
            key: Cache key
            ttl: New TTL in seconds

        Returns:
            True if successful, False otherwise

        """
        full_key = self._make_key(key)
        try:
            return bool(self._client.expire(full_key, ttl))
        except Exception:
            return False

    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Args:
            pattern: Glob-style pattern

        Returns:
            List of matching keys (without prefix)

        """
        full_pattern = self._make_key(pattern)
        try:
            keys = self._client.keys(full_pattern)
            prefix_len = len(self.prefix)
            return [k.decode()[prefix_len:] for k in keys if not k.decode().endswith(":meta")]
        except Exception:
            return []

    def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern.

        Args:
            pattern: Glob-style pattern

        Returns:
            Number of keys deleted

        """
        full_pattern = self._make_key(pattern)
        try:
            keys = self._client.keys(full_pattern)
            if keys:
                return int(self._client.delete(*keys))
            return 0
        except Exception:
            return 0

    def mget(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values at once.

        Args:
            keys: List of cache keys

        Returns:
            Dictionary of key -> value (None for missing)

        """
        if not keys:
            return {}

        full_keys = [self._make_key(k) for k in keys]
        try:
            values = self._client.mget(full_keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
                    self.stats.hits += 1
                    self.stats.bytes_received += len(value)
                else:
                    result[key] = None
                    self.stats.misses += 1
            return result
        except Exception:
            self.stats.errors += 1
            return {k: None for k in keys}

    def mset(
        self,
        items: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Set multiple values at once.

        Args:
            items: Dictionary of key -> value
            ttl: TTL for all items (None uses default)

        Returns:
            True if all successful, False otherwise

        """
        if not items:
            return True

        ttl = ttl if ttl is not None else self.default_ttl

        try:
            pipe = self._client.pipeline()
            for key, value in items.items():
                full_key = self._make_key(key)
                data = self._serialize(value)
                self.stats.bytes_sent += len(data)

                if ttl is not None:
                    pipe.setex(full_key, ttl, data)
                else:
                    pipe.set(full_key, data)

            pipe.execute()
            self.stats.sets += len(items)
            return True
        except Exception:
            self.stats.errors += 1
            return False

    def info(self) -> dict[str, Any]:
        """Get Redis server info.

        Returns:
            Server info dictionary

        """
        try:
            info = self._client.info()
            return {
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
            }
        except Exception:
            return {}

    def close(self) -> None:
        """Close connection to Redis."""
        try:
            self._client.close()
        except Exception:
            pass


class MockDistributedCache:
    """In-memory mock for distributed cache (for testing without Redis).

    This provides the same interface as DistributedCache but uses
    an in-memory dictionary instead of Redis.

    """

    def __init__(
        self,
        prefix: str = "spicelab:",
        default_ttl: int | None = None,
    ) -> None:
        """Initialize mock cache."""
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.stats = DistributedCacheStats()
        self._data: dict[str, tuple[Any, float | None]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def _is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        if key not in self._data:
            return True
        _, expires_at = self._data[key]
        if expires_at is None:
            return False
        return time.time() > expires_at

    def ping(self) -> bool:
        """Always returns True for mock."""
        return True

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        full_key = self._make_key(key)
        if self._is_expired(full_key):
            self._data.pop(full_key, None)
            self.stats.misses += 1
            return None

        self.stats.hits += 1
        return self._data[full_key][0]

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Set value in cache."""
        full_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl

        expires_at = None
        if ttl is not None:
            expires_at = time.time() + ttl

        self._data[full_key] = (value, expires_at)
        if metadata:
            self._metadata[full_key] = metadata

        self.stats.sets += 1
        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        full_key = self._make_key(key)
        existed = full_key in self._data
        self._data.pop(full_key, None)
        self._metadata.pop(full_key, None)
        self.stats.deletes += 1
        return existed

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._make_key(key)
        if self._is_expired(full_key):
            self._data.pop(full_key, None)
            return False
        return full_key in self._data

    def get_metadata(self, key: str) -> dict[str, Any] | None:
        """Get metadata for key."""
        full_key = self._make_key(key)
        return self._metadata.get(full_key)

    def get_ttl(self, key: str) -> int:
        """Get remaining TTL."""
        full_key = self._make_key(key)
        if full_key not in self._data:
            return -2
        _, expires_at = self._data[full_key]
        if expires_at is None:
            return -1
        remaining = int(expires_at - time.time())
        return max(0, remaining)

    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL on key."""
        full_key = self._make_key(key)
        if full_key not in self._data:
            return False
        value, _ = self._data[full_key]
        self._data[full_key] = (value, time.time() + ttl)
        return True

    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern."""
        import fnmatch

        full_pattern = self._make_key(pattern)
        prefix_len = len(self.prefix)
        result = []
        for key in list(self._data.keys()):
            if self._is_expired(key):
                self._data.pop(key, None)
                continue
            if fnmatch.fnmatch(key, full_pattern):
                result.append(key[prefix_len:])
        return result

    def clear(self, pattern: str = "*") -> int:
        """Clear keys matching pattern."""
        keys = self.keys(pattern)
        for key in keys:
            self.delete(key)
        return len(keys)

    def mget(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values."""
        return {k: self.get(k) for k in keys}

    def mset(self, items: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values."""
        for key, value in items.items():
            self.set(key, value, ttl=ttl)
        return True

    def info(self) -> dict[str, Any]:
        """Get mock info."""
        return {
            "type": "mock",
            "total_keys": len(self._data),
        }

    def close(self) -> None:
        """No-op for mock."""
        pass
