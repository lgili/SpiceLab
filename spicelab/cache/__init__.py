"""Cache module for simulation result caching.

This module provides:
- CompressedCache: zstd-compressed local cache for simulation results
- CacheStats: Statistics tracking for cache hits/misses
- CacheKey: Utilities for generating cache keys from circuits

Example::

    from spicelab.cache import CompressedCache, generate_cache_key

    cache = CompressedCache()

    # Generate key from circuit
    key = generate_cache_key(circuit, analyses)

    # Check cache
    result = cache.get(key)
    if result is None:
        result = run_simulation(circuit, analyses)
        cache.set(key, result)

    # Get statistics
    print(cache.stats)

"""

from .compressed import (
    CacheEntry,
    CacheStats,
    CompressedCache,
    generate_cache_key,
    hash_circuit,
)
from .distributed import (
    DistributedCache,
    DistributedCacheStats,
    MockDistributedCache,
    is_redis_available,
)

__all__ = [
    "CacheEntry",
    "CacheStats",
    "CompressedCache",
    "DistributedCache",
    "DistributedCacheStats",
    "MockDistributedCache",
    "generate_cache_key",
    "hash_circuit",
    "is_redis_available",
]
