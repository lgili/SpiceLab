"""Storage module for efficient simulation result persistence.

This module provides:
- HDF5ResultStorage: Store and retrieve simulation results in HDF5 format
- Batch storage for parameter sweeps
- Compression options for disk space optimization
- xarray integration for labeled multi-dimensional data

Example::

    from spicelab.storage import HDF5ResultStorage

    # Store simulation results
    storage = HDF5ResultStorage("results.h5")
    storage.save_result("sim_001", time, voltage, current)

    # Load results
    time, voltage, current = storage.load_result("sim_001")

    # Batch operations
    storage.save_sweep("temp_sweep", results_dict)

"""

from .hdf5 import (
    HDF5ResultStorage,
    StorageMetadata,
    DatasetInfo,
)

__all__ = [
    "HDF5ResultStorage",
    "StorageMetadata",
    "DatasetInfo",
]
