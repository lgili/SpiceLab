"""HDF5-based storage for simulation results.

This module provides efficient storage and retrieval of simulation results
using HDF5 format with optional compression.

Example::

    from spicelab.storage import HDF5ResultStorage

    # Create storage
    storage = HDF5ResultStorage("simulations.h5")

    # Save a single result
    storage.save_result(
        name="transient_001",
        time=time_array,
        data={"vout": vout, "iload": iload},
        metadata={"temp": 25, "vdd": 3.3}
    )

    # Load result
    result = storage.load_result("transient_001")
    print(result["time"], result["vout"])

    # Save parameter sweep
    storage.save_sweep(
        name="temp_sweep",
        parameter="temperature",
        values=[-40, 25, 85],
        results=[result1, result2, result3]
    )

"""

from __future__ import annotations

import json
import time as time_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Try to import h5py
_H5PY_AVAILABLE = False
try:
    import h5py

    _H5PY_AVAILABLE = True
except ImportError:
    h5py = None  # type: ignore


def is_hdf5_available() -> bool:
    """Check if HDF5 support is available."""
    return _H5PY_AVAILABLE


@dataclass
class StorageMetadata:
    """Metadata for stored simulation results.

    Attributes:
        name: Name/identifier of the result
        created_at: Unix timestamp when created
        modified_at: Unix timestamp when last modified
        simulation_type: Type of simulation (tran, ac, dc, etc.)
        parameters: Simulation parameters
        description: Optional description
        tags: Optional tags for organization

    """

    name: str
    created_at: float = field(default_factory=time_module.time)
    modified_at: float = field(default_factory=time_module.time)
    simulation_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "simulation_type": self.simulation_type,
            "parameters": json.dumps(self.parameters),
            "description": self.description,
            "tags": json.dumps(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageMetadata:
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            created_at=data.get("created_at", 0.0),
            modified_at=data.get("modified_at", 0.0),
            simulation_type=data.get("simulation_type", ""),
            parameters=json.loads(data.get("parameters", "{}")),
            description=data.get("description", ""),
            tags=json.loads(data.get("tags", "[]")),
        )


@dataclass
class DatasetInfo:
    """Information about a stored dataset.

    Attributes:
        name: Dataset name/path
        shape: Array shape
        dtype: Data type
        size_bytes: Size in bytes
        compression: Compression method used
        chunks: Chunk size if chunked

    """

    name: str
    shape: tuple[int, ...]
    dtype: str
    size_bytes: int
    compression: str | None = None
    chunks: tuple[int, ...] | None = None

    def __str__(self) -> str:
        """Return string representation."""
        size_str = _format_size(self.size_bytes)
        comp_str = f", {self.compression}" if self.compression else ""
        return f"{self.name}: {self.shape} {self.dtype} ({size_str}{comp_str})"


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore
    return f"{size_bytes:.1f} TB"


class HDF5ResultStorage:
    """Store and retrieve simulation results in HDF5 format.

    This class provides a high-level interface for storing simulation
    results with support for compression, metadata, and batch operations.

    Args:
        path: Path to HDF5 file
        mode: File mode ('r', 'r+', 'w', 'w-', 'a')
        compression: Compression method ('gzip', 'lzf', None)
        compression_level: Compression level (1-9 for gzip)

    Example::

        storage = HDF5ResultStorage("results.h5")

        # Save transient simulation
        storage.save_result(
            name="tran_001",
            time=time_array,
            data={"vout": vout, "vin": vin},
            metadata={"temp": 25}
        )

        # Load and use
        result = storage.load_result("tran_001")
        plt.plot(result["time"], result["vout"])

    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "a",
        compression: str | None = "gzip",
        compression_level: int = 4,
    ) -> None:
        """Initialize HDF5 storage."""
        if not _H5PY_AVAILABLE:
            raise ImportError(
                "h5py is required for HDF5 storage. "
                "Install with: pip install h5py"
            )

        self.path = Path(path)
        self.mode = mode
        self.compression = compression
        self.compression_level = compression_level
        self._file: h5py.File | None = None

    def __enter__(self) -> HDF5ResultStorage:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def open(self) -> None:
        """Open the HDF5 file."""
        if self._file is None:
            self._file = h5py.File(self.path, self.mode)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def file(self) -> h5py.File:
        """Get the HDF5 file object, opening if necessary."""
        if self._file is None:
            self.open()
        return self._file  # type: ignore

    def save_result(
        self,
        name: str,
        time: NDArray[Any] | None = None,
        data: dict[str, NDArray[Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> None:
        """Save a simulation result.

        Args:
            name: Unique name/identifier for this result
            time: Time array (optional, stored as 'time' dataset)
            data: Dictionary of signal name -> array
            metadata: Optional metadata dictionary
            overwrite: Whether to overwrite existing result

        """
        # Create or get group
        if name in self.file:
            if overwrite:
                del self.file[name]
            else:
                raise ValueError(f"Result '{name}' already exists")

        group = self.file.create_group(name)

        # Save time array if provided
        if time is not None:
            self._create_dataset(group, "time", time)

        # Save data arrays
        if data is not None:
            data_group = group.create_group("data")
            for signal_name, signal_data in data.items():
                self._create_dataset(data_group, signal_name, signal_data)

        # Save metadata
        meta = StorageMetadata(
            name=name,
            parameters=metadata or {},
        )
        group.attrs["metadata"] = json.dumps(meta.to_dict())

    def _create_dataset(
        self,
        group: h5py.Group,
        name: str,
        data: NDArray[Any],
    ) -> h5py.Dataset:
        """Create a dataset with compression."""
        kwargs: dict[str, Any] = {}

        if self.compression is not None and data.size > 1000:
            kwargs["compression"] = self.compression
            if self.compression == "gzip":
                kwargs["compression_opts"] = self.compression_level
            # Enable chunking for compression
            kwargs["chunks"] = True

        return group.create_dataset(name, data=data, **kwargs)

    def load_result(self, name: str) -> dict[str, Any]:
        """Load a simulation result.

        Args:
            name: Name of the result to load

        Returns:
            Dictionary with 'time', signal data, and 'metadata'

        """
        if name not in self.file:
            raise KeyError(f"Result '{name}' not found")

        group = self.file[name]
        result: dict[str, Any] = {}

        # Load time if present
        if "time" in group:
            result["time"] = group["time"][:]

        # Load data signals
        if "data" in group:
            data_group = group["data"]
            for signal_name in data_group:
                result[signal_name] = data_group[signal_name][:]

        # Load metadata
        if "metadata" in group.attrs:
            meta_dict = json.loads(group.attrs["metadata"])
            result["metadata"] = StorageMetadata.from_dict(meta_dict)

        return result

    def save_sweep(
        self,
        name: str,
        parameter: str,
        values: list[Any],
        results: list[dict[str, NDArray[Any]]],
        time: NDArray[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> None:
        """Save a parameter sweep.

        Args:
            name: Name for the sweep
            parameter: Name of swept parameter
            values: Parameter values for each result
            results: List of result dictionaries
            time: Shared time array (if same for all)
            metadata: Optional metadata
            overwrite: Whether to overwrite existing

        """
        if name in self.file:
            if overwrite:
                del self.file[name]
            else:
                raise ValueError(f"Sweep '{name}' already exists")

        group = self.file.create_group(name)
        group.attrs["type"] = "sweep"
        group.attrs["parameter"] = parameter
        group.attrs["values"] = json.dumps(values)

        # Save shared time if provided
        if time is not None:
            self._create_dataset(group, "time", time)

        # Save each result
        for i, (value, result) in enumerate(zip(values, results)):
            result_group = group.create_group(f"result_{i:04d}")
            result_group.attrs["value"] = json.dumps(value)

            for signal_name, signal_data in result.items():
                self._create_dataset(result_group, signal_name, signal_data)

        # Save metadata
        if metadata:
            group.attrs["metadata"] = json.dumps(metadata)

    def load_sweep(self, name: str) -> dict[str, Any]:
        """Load a parameter sweep.

        Args:
            name: Name of the sweep

        Returns:
            Dictionary with parameter info, values, and results

        """
        if name not in self.file:
            raise KeyError(f"Sweep '{name}' not found")

        group = self.file[name]

        result: dict[str, Any] = {
            "parameter": group.attrs.get("parameter", ""),
            "values": json.loads(group.attrs.get("values", "[]")),
            "results": [],
        }

        # Load shared time if present
        if "time" in group:
            result["time"] = group["time"][:]

        # Load results
        i = 0
        while f"result_{i:04d}" in group:
            result_group = group[f"result_{i:04d}"]
            signals: dict[str, NDArray[Any]] = {}

            for signal_name in result_group:
                if isinstance(result_group[signal_name], h5py.Dataset):
                    signals[signal_name] = result_group[signal_name][:]

            result["results"].append(signals)
            i += 1

        # Load metadata
        if "metadata" in group.attrs:
            result["metadata"] = json.loads(group.attrs["metadata"])

        return result

    def save_batch(
        self,
        results: dict[str, dict[str, NDArray[Any]]],
        group_name: str = "batch",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save multiple results as a batch.

        Args:
            results: Dictionary mapping result names to data dictionaries
            group_name: Name for the batch group
            metadata: Optional batch metadata

        """
        if group_name in self.file:
            del self.file[group_name]

        batch_group = self.file.create_group(group_name)
        batch_group.attrs["type"] = "batch"
        batch_group.attrs["count"] = len(results)

        for name, data in results.items():
            result_group = batch_group.create_group(name)
            for signal_name, signal_data in data.items():
                self._create_dataset(result_group, signal_name, signal_data)

        if metadata:
            batch_group.attrs["metadata"] = json.dumps(metadata)

    def load_batch(self, group_name: str = "batch") -> dict[str, dict[str, NDArray[Any]]]:
        """Load a batch of results.

        Args:
            group_name: Name of the batch group

        Returns:
            Dictionary mapping result names to data dictionaries

        """
        if group_name not in self.file:
            raise KeyError(f"Batch '{group_name}' not found")

        batch_group = self.file[group_name]
        results: dict[str, dict[str, NDArray[Any]]] = {}

        for name in batch_group:
            if isinstance(batch_group[name], h5py.Group):
                item_group = batch_group[name]
                signals: dict[str, NDArray[Any]] = {}

                for signal_name in item_group:
                    if isinstance(item_group[signal_name], h5py.Dataset):
                        signals[signal_name] = item_group[signal_name][:]

                results[name] = signals

        return results

    def list_results(self) -> list[str]:
        """List all stored results.

        Returns:
            List of result names

        """
        return list(self.file.keys())

    def delete_result(self, name: str) -> bool:
        """Delete a result.

        Args:
            name: Name of result to delete

        Returns:
            True if deleted, False if not found

        """
        if name in self.file:
            del self.file[name]
            return True
        return False

    def get_info(self, name: str) -> list[DatasetInfo]:
        """Get information about datasets in a result.

        Args:
            name: Name of result

        Returns:
            List of DatasetInfo objects

        """
        if name not in self.file:
            raise KeyError(f"Result '{name}' not found")

        infos: list[DatasetInfo] = []
        self._collect_dataset_info(self.file[name], name, infos)
        return infos

    def _collect_dataset_info(
        self,
        group: h5py.Group,
        prefix: str,
        infos: list[DatasetInfo],
    ) -> None:
        """Recursively collect dataset information."""
        for key in group:
            item = group[key]
            path = f"{prefix}/{key}"

            if isinstance(item, h5py.Dataset):
                infos.append(
                    DatasetInfo(
                        name=path,
                        shape=item.shape,
                        dtype=str(item.dtype),
                        size_bytes=item.nbytes,
                        compression=item.compression,
                        chunks=item.chunks,
                    )
                )
            elif isinstance(item, h5py.Group):
                self._collect_dataset_info(item, path, infos)

    def get_metadata(self, name: str) -> StorageMetadata | None:
        """Get metadata for a result.

        Args:
            name: Name of result

        Returns:
            StorageMetadata or None if not found

        """
        if name not in self.file:
            return None

        group = self.file[name]
        if "metadata" in group.attrs:
            meta_dict = json.loads(group.attrs["metadata"])
            return StorageMetadata.from_dict(meta_dict)
        return None

    @property
    def size_bytes(self) -> int:
        """Get file size in bytes."""
        if self.path.exists():
            return self.path.stat().st_size
        return 0

    @property
    def size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.size_bytes / (1024 * 1024)

    def flush(self) -> None:
        """Flush pending writes to disk."""
        if self._file is not None:
            self._file.flush()

    def compact(self) -> None:
        """Compact the file to reclaim space from deleted data.

        Note: This requires closing and reopening the file.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

        # Create temporary file
        temp_path = self.path.with_suffix(".h5.tmp")

        with h5py.File(self.path, "r") as src:
            with h5py.File(temp_path, "w") as dst:
                for name in src:
                    src.copy(name, dst)

        # Replace original with compacted
        temp_path.replace(self.path)

        # Reopen
        self.open()


def save_to_hdf5(
    path: str | Path,
    name: str,
    time: NDArray[Any] | None = None,
    data: dict[str, NDArray[Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    compression: str | None = "gzip",
) -> None:
    """Convenience function to save a single result.

    Args:
        path: Path to HDF5 file
        name: Result name
        time: Time array
        data: Signal data dictionary
        metadata: Optional metadata
        compression: Compression method

    """
    with HDF5ResultStorage(path, compression=compression) as storage:
        storage.save_result(name, time=time, data=data, metadata=metadata)


def load_from_hdf5(
    path: str | Path,
    name: str,
) -> dict[str, Any]:
    """Convenience function to load a single result.

    Args:
        path: Path to HDF5 file
        name: Result name

    Returns:
        Result dictionary

    """
    with HDF5ResultStorage(path, mode="r") as storage:
        return storage.load_result(name)
