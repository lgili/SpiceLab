"""Lazy ResultHandle implementation for deferred dataset loading (Phase 3.4).

This module provides LazyResultHandle which delays loading raw simulation data
until .dataset() is actually called. Benefits:

1. Faster sweep/Monte Carlo - only load results you need
2. Memory-efficient - avoid loading large datasets into RAM
3. Memory-mapped reading - for very large RAW files

Example:
    >>> handle = LazyResultHandle.from_raw_file(
    ...     path="results.raw",
    ...     reader=read_ngspice_raw,
    ...     meta=meta
    ... )
    >>> # No data loaded yet
    >>>
    >>> ds = handle.dataset()  # Now loads
    >>> # Data is cached for subsequent calls
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.types import ResultHandle, ResultMeta
from .result import DatasetResultHandle

if TYPE_CHECKING:
    pass

__all__ = ["LazyResultHandle", "LazyDatasetResultHandle"]


class LazyDatasetResultHandle(ResultHandle):
    """ResultHandle that defers dataset loading until accessed.

    Phase 3.4: Lazy loading optimization for large simulation results.

    Benefits:
    - Faster orchestrator/sweep execution (skip unused results)
    - Lower memory footprint (only load what you need)
    - Supports memory-mapped RAW files (future)

    Example:
        >>> handle = LazyDatasetResultHandle(
        ...     loader=lambda: read_ngspice_raw("results.raw"),
        ...     meta=meta
        ... )
        >>> # No data loaded yet
        >>>
        >>> ds = handle.dataset()  # Triggers loading
        >>> ds2 = handle.dataset()  # Returns cached result
    """

    def __init__(
        self,
        loader: Callable[[], Any],
        meta: ResultMeta,
        raw_path: Path | str | None = None,
    ) -> None:
        """Initialize lazy result handle.

        Args:
            loader: Function that loads and returns xarray.Dataset
            meta: Result metadata
            raw_path: Optional path to RAW file (for debugging/repr)
        """
        self._loader = loader
        self._meta = meta
        self._raw_path = Path(raw_path) if raw_path else None
        self._loaded = False

    @cached_property
    def _dataset(self) -> Any:
        """Load dataset lazily and cache result."""
        self._loaded = True
        return self._loader()

    def dataset(self) -> Any:
        """Return dataset, loading if necessary.

        First call triggers loading via loader function.
        Subsequent calls return cached result.
        """
        return self._dataset

    def to_polars(self) -> Any:
        """Convert to polars DataFrame (loads dataset if needed)."""
        # Delegate to DatasetResultHandle implementation
        eager_handle = DatasetResultHandle(self.dataset(), self._meta)
        return eager_handle.to_polars()

    def attrs(self) -> dict[str, Any]:
        """Return metadata attributes."""
        return self._meta.attrs

    def is_loaded(self) -> bool:
        """Check if dataset has been loaded."""
        return self._loaded

    @classmethod
    def from_raw_file(
        cls,
        path: Path | str,
        reader: Callable[[Path | str], Any],
        meta: ResultMeta,
    ) -> LazyDatasetResultHandle:
        """Create lazy handle from RAW file path.

        Args:
            path: Path to RAW file
            reader: Function that reads RAW file (e.g., read_ngspice_raw)
            meta: Result metadata

        Returns:
            LazyDatasetResultHandle that will load when .dataset() called.

        Example:
            >>> from spicelab.io.readers import read_ngspice_raw
            >>> handle = LazyDatasetResultHandle.from_raw_file(
            ...     path="simulation.raw",
            ...     reader=read_ngspice_raw,
            ...     meta=meta
            ... )
        """
        return cls(
            loader=lambda: reader(path),
            meta=meta,
            raw_path=path,
        )

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        path = f" from {self._raw_path}" if self._raw_path else ""
        return f"<LazyDatasetResultHandle {status}{path}>"


# Alias for backward compatibility
LazyResultHandle = LazyDatasetResultHandle
