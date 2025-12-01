"""Parallel execution module for batch simulations.

This module provides:
- ParallelExecutor: Execute simulations in parallel using ProcessPoolExecutor
- BatchSimulation: High-level interface for parameter sweeps
- Progress tracking with optional tqdm integration

Example::

    from spicelab.parallel import ParallelExecutor, BatchSimulation

    # Low-level parallel execution
    executor = ParallelExecutor(max_workers=8)
    results = executor.map(simulate_func, parameter_sets)

    # High-level batch simulation
    batch = BatchSimulation(circuit, analyses)
    batch.add_sweep("temp", [-40, 25, 85])
    batch.add_sweep("vdd", [3.0, 3.3, 3.6])
    results = batch.run(parallel=True)

"""

from .executor import (
    BatchResult,
    JobResult,
    JobStatus,
    ParallelExecutor,
    ProgressCallback,
)

__all__ = [
    "BatchResult",
    "JobResult",
    "JobStatus",
    "ParallelExecutor",
    "ProgressCallback",
]
