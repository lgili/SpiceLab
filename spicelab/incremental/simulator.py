"""Incremental simulation with change detection and caching.

This module provides efficient re-simulation by detecting changes to
circuits and analyses, and returning cached results when possible.

Example::

    from spicelab.incremental import IncrementalSimulator

    sim = IncrementalSimulator()

    # First simulation
    result = sim.run(circuit, analysis)
    print(sim.stats)  # Shows cache miss

    # Same circuit - returns cached
    result = sim.run(circuit, analysis)
    print(sim.stats)  # Shows cache hit

"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ChangeType(Enum):
    """Type of change detected in circuit or analysis."""

    NONE = auto()
    COMPONENT_VALUE = auto()
    COMPONENT_ADDED = auto()
    COMPONENT_REMOVED = auto()
    TOPOLOGY = auto()
    PARAMETER = auto()
    ANALYSIS = auto()
    SUBCIRCUIT = auto()


@dataclass
class ChangeInfo:
    """Information about a detected change.

    Attributes:
        change_type: Type of change
        component: Component name if applicable
        old_value: Previous value
        new_value: New value
        description: Human-readable description

    """

    change_type: ChangeType
    component: str | None = None
    old_value: Any = None
    new_value: Any = None
    description: str = ""

    def __str__(self) -> str:
        """Return string representation."""
        if self.component:
            return f"{self.change_type.name}: {self.component} ({self.description})"
        return f"{self.change_type.name}: {self.description}"


@dataclass
class SimulationState:
    """State of a simulation for change detection.

    Attributes:
        circuit_hash: Hash of the circuit
        analysis_hash: Hash of the analysis configuration
        parameter_hash: Hash of simulation parameters
        timestamp: When this state was created
        result_key: Key to retrieve cached result

    """

    circuit_hash: str
    analysis_hash: str
    parameter_hash: str = ""
    timestamp: float = field(default_factory=time.time)
    result_key: str = ""

    def __post_init__(self) -> None:
        """Generate result key from hashes."""
        if not self.result_key:
            combined = f"{self.circuit_hash}:{self.analysis_hash}:{self.parameter_hash}"
            self.result_key = hashlib.sha256(combined.encode()).hexdigest()[:16]

    def matches(self, other: SimulationState) -> bool:
        """Check if this state matches another."""
        return (
            self.circuit_hash == other.circuit_hash
            and self.analysis_hash == other.analysis_hash
            and self.parameter_hash == other.parameter_hash
        )


@runtime_checkable
class CircuitProtocol(Protocol):
    """Protocol for circuit objects."""

    def build_netlist(self) -> str:
        """Build netlist string."""
        ...


@runtime_checkable
class AnalysisProtocol(Protocol):
    """Protocol for analysis objects."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        ...


def hash_circuit(circuit: Any) -> str:
    """Generate a hash for a circuit.

    Args:
        circuit: Circuit object or netlist string

    Returns:
        SHA-256 hash of the circuit

    """
    if isinstance(circuit, str):
        netlist = circuit
    elif hasattr(circuit, "build_netlist"):
        netlist = circuit.build_netlist()
    elif hasattr(circuit, "to_netlist"):
        netlist = circuit.to_netlist()
    else:
        # Try to convert to string
        netlist = str(circuit)

    # Normalize whitespace for consistent hashing
    # Remove empty lines and strip each line
    lines = [line.strip() for line in netlist.strip().split("\n") if line.strip()]
    normalized = "\n".join(lines)
    return hashlib.sha256(normalized.encode()).hexdigest()


def hash_analysis(analysis: Any) -> str:
    """Generate a hash for an analysis configuration.

    Args:
        analysis: Analysis object or dictionary

    Returns:
        SHA-256 hash of the analysis

    """
    if isinstance(analysis, dict):
        data = analysis
    elif hasattr(analysis, "to_dict"):
        data = analysis.to_dict()
    elif hasattr(analysis, "__dict__"):
        data = {k: v for k, v in analysis.__dict__.items() if not k.startswith("_")}
    else:
        data = {"value": str(analysis)}

    # Sort keys for consistent ordering
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


def hash_parameters(parameters: dict[str, Any] | None) -> str:
    """Generate a hash for simulation parameters.

    Args:
        parameters: Parameter dictionary

    Returns:
        SHA-256 hash of parameters

    """
    if not parameters:
        return ""
    json_str = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class SimulationStats:
    """Statistics for incremental simulation.

    Attributes:
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        simulations_skipped: Number of simulations skipped
        simulations_run: Number of simulations actually run
        total_time_saved_ms: Estimated time saved from caching

    """

    cache_hits: int = 0
    cache_misses: int = 0
    simulations_skipped: int = 0
    simulations_run: int = 0
    total_time_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def reset(self) -> None:
        """Reset all statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.simulations_skipped = 0
        self.simulations_run = 0
        self.total_time_saved_ms = 0.0


class IncrementalSimulator:
    """Simulator with incremental change detection and caching.

    This simulator tracks circuit and analysis states to avoid
    re-running simulations when nothing has changed.

    Args:
        cache_dir: Directory for cached results (None for memory-only)
        max_cache_entries: Maximum number of cached results
        auto_save: Automatically save cache to disk

    Example::

        sim = IncrementalSimulator(cache_dir="./cache")

        # First run
        result1 = sim.run(circuit, analysis)

        # Same inputs - returns cached
        result2 = sim.run(circuit, analysis)
        assert sim.stats.cache_hits == 1

        # Modified circuit - re-runs simulation
        circuit.R1.value = "2k"
        result3 = sim.run(circuit, analysis)
        assert sim.stats.cache_misses == 2

    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        max_cache_entries: int = 100,
        auto_save: bool = True,
    ) -> None:
        """Initialize incremental simulator."""
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_entries = max_cache_entries
        self.auto_save = auto_save

        # In-memory cache
        self._cache: dict[str, Any] = {}
        self._states: dict[str, SimulationState] = {}
        self._access_order: list[str] = []

        # Statistics
        self.stats = SimulationStats()

        # Last state for comparison
        self._last_state: SimulationState | None = None

        # Load cache from disk if available
        if self.cache_dir:
            self._load_cache()

    def run(
        self,
        circuit: Any,
        analysis: Any,
        parameters: dict[str, Any] | None = None,
        runner: Any | None = None,
    ) -> Any:
        """Run simulation with caching.

        Args:
            circuit: Circuit to simulate
            analysis: Analysis configuration
            parameters: Additional parameters
            runner: Simulation runner (required on first run)

        Returns:
            Simulation result (from cache or fresh run)

        """
        # Create current state
        current_state = SimulationState(
            circuit_hash=hash_circuit(circuit),
            analysis_hash=hash_analysis(analysis),
            parameter_hash=hash_parameters(parameters),
        )

        # Check if we have a cached result
        if current_state.result_key in self._cache:
            self.stats.cache_hits += 1
            self.stats.simulations_skipped += 1
            self._update_access(current_state.result_key)
            self._last_state = current_state
            return self._cache[current_state.result_key]

        # Cache miss - need to run simulation
        self.stats.cache_misses += 1

        if runner is None:
            raise ValueError(
                "No cached result found and no runner provided. "
                "Provide a runner to execute the simulation."
            )

        # Run the simulation
        start_time = time.time()
        result = self._run_simulation(runner, circuit, analysis, parameters)
        duration_ms = (time.time() - start_time) * 1000

        self.stats.simulations_run += 1

        # Store in cache
        self._cache[current_state.result_key] = result
        self._states[current_state.result_key] = current_state
        self._update_access(current_state.result_key)

        # Evict if necessary
        self._evict_if_needed()

        # Auto-save
        if self.auto_save and self.cache_dir:
            self._save_cache()

        self._last_state = current_state

        return result

    def _run_simulation(
        self,
        runner: Any,
        circuit: Any,
        analysis: Any,
        parameters: dict[str, Any] | None,
    ) -> Any:
        """Execute the actual simulation."""
        if hasattr(runner, "run"):
            if parameters:
                return runner.run(circuit, analysis, **parameters)
            return runner.run(circuit, analysis)
        elif callable(runner):
            if parameters:
                return runner(circuit, analysis, **parameters)
            return runner(circuit, analysis)
        else:
            raise TypeError(f"Runner must be callable or have run() method: {type(runner)}")

    def _update_access(self, key: str) -> None:
        """Update access order for LRU eviction."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is too large."""
        while len(self._cache) > self.max_cache_entries and self._access_order:
            oldest_key = self._access_order.pop(0)
            self._cache.pop(oldest_key, None)
            self._states.pop(oldest_key, None)

    def check_changes(
        self,
        circuit: Any,
        analysis: Any,
        parameters: dict[str, Any] | None = None,
    ) -> list[ChangeInfo]:
        """Check what has changed since last simulation.

        Args:
            circuit: Circuit to check
            analysis: Analysis to check
            parameters: Parameters to check

        Returns:
            List of detected changes

        """
        if self._last_state is None:
            return [ChangeInfo(ChangeType.NONE, description="No previous state")]

        changes: list[ChangeInfo] = []

        current_circuit_hash = hash_circuit(circuit)
        current_analysis_hash = hash_analysis(analysis)
        current_param_hash = hash_parameters(parameters)

        if current_circuit_hash != self._last_state.circuit_hash:
            changes.append(
                ChangeInfo(
                    ChangeType.TOPOLOGY,
                    description="Circuit has changed",
                    old_value=self._last_state.circuit_hash[:8],
                    new_value=current_circuit_hash[:8],
                )
            )

        if current_analysis_hash != self._last_state.analysis_hash:
            changes.append(
                ChangeInfo(
                    ChangeType.ANALYSIS,
                    description="Analysis configuration has changed",
                    old_value=self._last_state.analysis_hash[:8],
                    new_value=current_analysis_hash[:8],
                )
            )

        if current_param_hash != self._last_state.parameter_hash:
            changes.append(
                ChangeInfo(
                    ChangeType.PARAMETER,
                    description="Parameters have changed",
                    old_value=self._last_state.parameter_hash[:8] or "(none)",
                    new_value=current_param_hash[:8] or "(none)",
                )
            )

        if not changes:
            changes.append(ChangeInfo(ChangeType.NONE, description="No changes detected"))

        return changes

    def is_dirty(
        self,
        circuit: Any,
        analysis: Any,
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        """Check if simulation needs to be re-run.

        Args:
            circuit: Circuit to check
            analysis: Analysis to check
            parameters: Parameters to check

        Returns:
            True if simulation needs to run, False if cached

        """
        current_state = SimulationState(
            circuit_hash=hash_circuit(circuit),
            analysis_hash=hash_analysis(analysis),
            parameter_hash=hash_parameters(parameters),
        )
        return current_state.result_key not in self._cache

    def invalidate(
        self,
        circuit: Any | None = None,
        analysis: Any | None = None,
    ) -> int:
        """Invalidate cached results.

        Args:
            circuit: Invalidate results for this circuit (None = all)
            analysis: Invalidate results for this analysis (None = all)

        Returns:
            Number of entries invalidated

        """
        if circuit is None and analysis is None:
            count = len(self._cache)
            self._cache.clear()
            self._states.clear()
            self._access_order.clear()
            return count

        circuit_hash = hash_circuit(circuit) if circuit else None
        analysis_hash = hash_analysis(analysis) if analysis else None

        to_remove = []
        for key, state in self._states.items():
            if circuit_hash and state.circuit_hash != circuit_hash:
                continue
            if analysis_hash and state.analysis_hash != analysis_hash:
                continue
            to_remove.append(key)

        for key in to_remove:
            self._cache.pop(key, None)
            self._states.pop(key, None)
            if key in self._access_order:
                self._access_order.remove(key)

        return len(to_remove)

    def get_cached_result(
        self,
        circuit: Any,
        analysis: Any,
        parameters: dict[str, Any] | None = None,
    ) -> Any | None:
        """Get cached result without running simulation.

        Args:
            circuit: Circuit
            analysis: Analysis
            parameters: Parameters

        Returns:
            Cached result or None if not found

        """
        state = SimulationState(
            circuit_hash=hash_circuit(circuit),
            analysis_hash=hash_analysis(analysis),
            parameter_hash=hash_parameters(parameters),
        )
        return self._cache.get(state.result_key)

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / "incremental_cache.pkl"
        if not cache_file.exists():
            return

        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                self._cache = data.get("cache", {})
                self._states = data.get("states", {})
                self._access_order = data.get("access_order", [])
        except Exception:
            # If loading fails, start fresh
            pass

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "incremental_cache.pkl"

        try:
            data = {
                "cache": self._cache,
                "states": self._states,
                "access_order": self._access_order,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            # Silently fail if can't save
            pass

    @property
    def cache_size(self) -> int:
        """Number of cached results."""
        return len(self._cache)

    def clear_stats(self) -> None:
        """Clear statistics."""
        self.stats.reset()


class DependencyTracker:
    """Track dependencies between circuit components.

    This class helps identify which simulations need to be re-run
    when specific components change.

    Example::

        tracker = DependencyTracker()
        tracker.add_dependency("output", ["R1", "R2", "C1"])

        # Check what's affected by R1 change
        affected = tracker.get_affected("R1")
        # Returns {"output"}

    """

    def __init__(self) -> None:
        """Initialize dependency tracker."""
        self._dependencies: dict[str, set[str]] = {}
        self._reverse_deps: dict[str, set[str]] = {}

    def add_dependency(self, target: str, sources: list[str]) -> None:
        """Add a dependency relationship.

        Args:
            target: The dependent item (e.g., simulation name)
            sources: Items that target depends on (e.g., components)

        """
        if target not in self._dependencies:
            self._dependencies[target] = set()

        for source in sources:
            self._dependencies[target].add(source)

            if source not in self._reverse_deps:
                self._reverse_deps[source] = set()
            self._reverse_deps[source].add(target)

    def get_dependencies(self, target: str) -> set[str]:
        """Get all dependencies of a target."""
        return self._dependencies.get(target, set())

    def get_affected(self, source: str) -> set[str]:
        """Get all targets affected by a source change."""
        return self._reverse_deps.get(source, set())

    def clear(self) -> None:
        """Clear all dependencies."""
        self._dependencies.clear()
        self._reverse_deps.clear()


def compare_circuits(circuit1: Any, circuit2: Any) -> list[ChangeInfo]:
    """Compare two circuits and identify differences.

    Args:
        circuit1: First circuit
        circuit2: Second circuit

    Returns:
        List of changes between circuits

    """
    hash1 = hash_circuit(circuit1)
    hash2 = hash_circuit(circuit2)

    if hash1 == hash2:
        return [ChangeInfo(ChangeType.NONE, description="Circuits are identical")]

    # Get netlists for detailed comparison
    if hasattr(circuit1, "build_netlist"):
        netlist1 = circuit1.build_netlist()
    else:
        netlist1 = str(circuit1)

    if hasattr(circuit2, "build_netlist"):
        netlist2 = circuit2.build_netlist()
    else:
        netlist2 = str(circuit2)

    lines1 = set(netlist1.strip().split("\n"))
    lines2 = set(netlist2.strip().split("\n"))

    changes: list[ChangeInfo] = []

    # Lines removed
    for line in lines1 - lines2:
        if line.strip():
            changes.append(
                ChangeInfo(
                    ChangeType.COMPONENT_REMOVED,
                    description=f"Removed: {line[:50]}...",
                    old_value=line,
                )
            )

    # Lines added
    for line in lines2 - lines1:
        if line.strip():
            changes.append(
                ChangeInfo(
                    ChangeType.COMPONENT_ADDED,
                    description=f"Added: {line[:50]}...",
                    new_value=line,
                )
            )

    if not changes:
        changes.append(
            ChangeInfo(
                ChangeType.TOPOLOGY,
                description="Circuits differ (whitespace or ordering)",
            )
        )

    return changes
