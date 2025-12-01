"""Incremental simulation module for efficient re-simulation.

This module provides:
- IncrementalSimulator: Skip simulations when inputs haven't changed
- Circuit hashing for change detection
- Dirty checking for subcircuits and parameters
- Cache integration for result persistence

Example::

    from spicelab.incremental import IncrementalSimulator

    # Create incremental simulator with caching
    sim = IncrementalSimulator(cache_dir="./sim_cache")

    # First run - full simulation
    result1 = sim.run(circuit, analysis)

    # Second run - returns cached result (no re-simulation)
    result2 = sim.run(circuit, analysis)

    # Modify circuit - triggers re-simulation
    circuit.R1.value = "2k"
    result3 = sim.run(circuit, analysis)

"""

from .simulator import (
    IncrementalSimulator,
    SimulationState,
    ChangeType,
    ChangeInfo,
    hash_circuit,
    hash_analysis,
)

__all__ = [
    "IncrementalSimulator",
    "SimulationState",
    "ChangeType",
    "ChangeInfo",
    "hash_circuit",
    "hash_analysis",
]
