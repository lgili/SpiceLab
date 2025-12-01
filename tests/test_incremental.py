"""Tests for the incremental simulation module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spicelab.incremental import (
    IncrementalSimulator,
    SimulationState,
    ChangeType,
    ChangeInfo,
    hash_circuit,
    hash_analysis,
)
from spicelab.incremental.simulator import (
    SimulationStats,
    DependencyTracker,
    compare_circuits,
    hash_parameters,
)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types_exist(self) -> None:
        """Test that all change types exist."""
        assert ChangeType.NONE
        assert ChangeType.COMPONENT_VALUE
        assert ChangeType.COMPONENT_ADDED
        assert ChangeType.COMPONENT_REMOVED
        assert ChangeType.TOPOLOGY
        assert ChangeType.PARAMETER
        assert ChangeType.ANALYSIS
        assert ChangeType.SUBCIRCUIT


class TestChangeInfo:
    """Tests for ChangeInfo dataclass."""

    def test_create_change_info(self) -> None:
        """Test creating change info."""
        info = ChangeInfo(
            change_type=ChangeType.COMPONENT_VALUE,
            component="R1",
            old_value="1k",
            new_value="2k",
            description="Resistor value changed",
        )
        assert info.change_type == ChangeType.COMPONENT_VALUE
        assert info.component == "R1"
        assert info.old_value == "1k"
        assert info.new_value == "2k"

    def test_str_with_component(self) -> None:
        """Test string representation with component."""
        info = ChangeInfo(
            change_type=ChangeType.COMPONENT_VALUE,
            component="R1",
            description="value changed",
        )
        s = str(info)
        assert "R1" in s
        assert "COMPONENT_VALUE" in s

    def test_str_without_component(self) -> None:
        """Test string representation without component."""
        info = ChangeInfo(
            change_type=ChangeType.TOPOLOGY,
            description="circuit changed",
        )
        s = str(info)
        assert "TOPOLOGY" in s


class TestSimulationState:
    """Tests for SimulationState dataclass."""

    def test_create_state(self) -> None:
        """Test creating simulation state."""
        state = SimulationState(
            circuit_hash="abc123",
            analysis_hash="def456",
            parameter_hash="ghi789",
        )
        assert state.circuit_hash == "abc123"
        assert state.analysis_hash == "def456"
        assert state.result_key  # Auto-generated

    def test_result_key_generated(self) -> None:
        """Test that result key is auto-generated."""
        state = SimulationState(
            circuit_hash="abc",
            analysis_hash="def",
        )
        assert len(state.result_key) == 16

    def test_matches(self) -> None:
        """Test state matching."""
        state1 = SimulationState("abc", "def", "ghi")
        state2 = SimulationState("abc", "def", "ghi")
        state3 = SimulationState("abc", "def", "xyz")

        assert state1.matches(state2)
        assert not state1.matches(state3)


class TestSimulationStats:
    """Tests for SimulationStats dataclass."""

    def test_initial_stats(self) -> None:
        """Test initial statistics."""
        stats = SimulationStats()
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        stats = SimulationStats(cache_hits=80, cache_misses=20)
        assert stats.hit_rate == pytest.approx(0.8)

    def test_reset(self) -> None:
        """Test resetting stats."""
        stats = SimulationStats(cache_hits=10, cache_misses=5)
        stats.reset()
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0


class TestHashCircuit:
    """Tests for hash_circuit function."""

    def test_hash_string(self) -> None:
        """Test hashing a string netlist."""
        netlist = "R1 1 0 1k\n.end"
        h = hash_circuit(netlist)
        assert len(h) == 64  # SHA-256 hex length

    def test_hash_deterministic(self) -> None:
        """Test that hashing is deterministic."""
        netlist = "R1 1 0 1k\n.end"
        h1 = hash_circuit(netlist)
        h2 = hash_circuit(netlist)
        assert h1 == h2

    def test_hash_different_netlists(self) -> None:
        """Test that different netlists have different hashes."""
        h1 = hash_circuit("R1 1 0 1k")
        h2 = hash_circuit("R1 1 0 2k")
        assert h1 != h2

    def test_hash_whitespace_normalized(self) -> None:
        """Test that whitespace is normalized."""
        h1 = hash_circuit("R1 1 0 1k\n.end")
        h2 = hash_circuit("R1 1 0 1k\n\n.end\n")
        assert h1 == h2

    def test_hash_object_with_build_netlist(self) -> None:
        """Test hashing object with build_netlist method."""

        class MockCircuit:
            def build_netlist(self) -> str:
                return "R1 1 0 1k"

        h = hash_circuit(MockCircuit())
        assert len(h) == 64

    def test_hash_object_with_to_netlist(self) -> None:
        """Test hashing object with to_netlist method."""

        class MockCircuit:
            def to_netlist(self) -> str:
                return "R1 1 0 1k"

        h = hash_circuit(MockCircuit())
        assert len(h) == 64


class TestHashAnalysis:
    """Tests for hash_analysis function."""

    def test_hash_dict(self) -> None:
        """Test hashing a dictionary."""
        analysis = {"type": "tran", "stop": "1m"}
        h = hash_analysis(analysis)
        assert len(h) == 64

    def test_hash_deterministic(self) -> None:
        """Test deterministic hashing."""
        analysis = {"type": "tran", "stop": "1m"}
        h1 = hash_analysis(analysis)
        h2 = hash_analysis(analysis)
        assert h1 == h2

    def test_hash_order_independent(self) -> None:
        """Test that dict order doesn't affect hash."""
        h1 = hash_analysis({"a": 1, "b": 2})
        h2 = hash_analysis({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_object_with_to_dict(self) -> None:
        """Test hashing object with to_dict method."""

        class MockAnalysis:
            def to_dict(self) -> dict[str, Any]:
                return {"type": "ac"}

        h = hash_analysis(MockAnalysis())
        assert len(h) == 64


class TestHashParameters:
    """Tests for hash_parameters function."""

    def test_hash_none(self) -> None:
        """Test hashing None parameters."""
        h = hash_parameters(None)
        assert h == ""

    def test_hash_empty(self) -> None:
        """Test hashing empty dict."""
        h = hash_parameters({})
        assert h == ""

    def test_hash_dict(self) -> None:
        """Test hashing parameter dict."""
        h = hash_parameters({"temp": 25, "vdd": 3.3})
        assert len(h) == 64


class TestIncrementalSimulator:
    """Tests for IncrementalSimulator class."""

    @pytest.fixture
    def sim(self) -> IncrementalSimulator:
        """Create simulator instance."""
        return IncrementalSimulator(max_cache_entries=10)

    @pytest.fixture
    def mock_runner(self) -> Any:
        """Create a mock runner."""
        call_count = [0]

        def runner(circuit: Any, analysis: Any, **params: Any) -> dict[str, Any]:
            call_count[0] += 1
            return {"result": "simulated", "call": call_count[0]}

        runner.call_count = call_count  # type: ignore
        return runner

    def test_create_simulator(self) -> None:
        """Test creating simulator."""
        sim = IncrementalSimulator()
        assert sim.cache_size == 0
        assert sim.stats.cache_hits == 0

    def test_first_run_cache_miss(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test that first run is a cache miss."""
        result = sim.run("R1 1 0 1k", {"type": "tran"}, runner=mock_runner)

        assert sim.stats.cache_misses == 1
        assert sim.stats.cache_hits == 0
        assert sim.stats.simulations_run == 1

    def test_second_run_cache_hit(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test that second identical run is cache hit."""
        circuit = "R1 1 0 1k"
        analysis = {"type": "tran"}

        result1 = sim.run(circuit, analysis, runner=mock_runner)
        result2 = sim.run(circuit, analysis, runner=mock_runner)

        assert sim.stats.cache_misses == 1
        assert sim.stats.cache_hits == 1
        assert result1 == result2

    def test_changed_circuit_cache_miss(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test that changed circuit causes cache miss."""
        analysis = {"type": "tran"}

        sim.run("R1 1 0 1k", analysis, runner=mock_runner)
        sim.run("R1 1 0 2k", analysis, runner=mock_runner)

        assert sim.stats.cache_misses == 2

    def test_changed_analysis_cache_miss(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test that changed analysis causes cache miss."""
        circuit = "R1 1 0 1k"

        sim.run(circuit, {"type": "tran", "stop": "1m"}, runner=mock_runner)
        sim.run(circuit, {"type": "tran", "stop": "2m"}, runner=mock_runner)

        assert sim.stats.cache_misses == 2

    def test_changed_parameters_cache_miss(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test that changed parameters cause cache miss."""
        circuit = "R1 1 0 1k"
        analysis = {"type": "tran"}

        sim.run(circuit, analysis, parameters={"temp": 25}, runner=mock_runner)
        sim.run(circuit, analysis, parameters={"temp": 85}, runner=mock_runner)

        assert sim.stats.cache_misses == 2

    def test_no_runner_cached_returns_result(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test that cached result is returned without runner."""
        circuit = "R1 1 0 1k"
        analysis = {"type": "tran"}

        # First run with runner
        result1 = sim.run(circuit, analysis, runner=mock_runner)

        # Second run without runner - should use cache
        result2 = sim.run(circuit, analysis)

        assert result1 == result2

    def test_no_runner_no_cache_raises(self, sim: IncrementalSimulator) -> None:
        """Test that missing runner with no cache raises error."""
        with pytest.raises(ValueError, match="No cached result"):
            sim.run("R1 1 0 1k", {"type": "tran"})

    def test_is_dirty_no_cache(self, sim: IncrementalSimulator) -> None:
        """Test is_dirty when no cache exists."""
        assert sim.is_dirty("R1 1 0 1k", {"type": "tran"})

    def test_is_dirty_with_cache(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test is_dirty with cached result."""
        circuit = "R1 1 0 1k"
        analysis = {"type": "tran"}

        sim.run(circuit, analysis, runner=mock_runner)

        assert not sim.is_dirty(circuit, analysis)
        assert sim.is_dirty("R1 1 0 2k", analysis)

    def test_check_changes_no_previous(self, sim: IncrementalSimulator) -> None:
        """Test check_changes with no previous state."""
        changes = sim.check_changes("R1 1 0 1k", {"type": "tran"})
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.NONE

    def test_check_changes_no_change(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test check_changes when nothing changed."""
        circuit = "R1 1 0 1k"
        analysis = {"type": "tran"}

        sim.run(circuit, analysis, runner=mock_runner)
        changes = sim.check_changes(circuit, analysis)

        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.NONE

    def test_check_changes_circuit_changed(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test check_changes when circuit changed."""
        analysis = {"type": "tran"}

        sim.run("R1 1 0 1k", analysis, runner=mock_runner)
        changes = sim.check_changes("R1 1 0 2k", analysis)

        assert any(c.change_type == ChangeType.TOPOLOGY for c in changes)

    def test_invalidate_all(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test invalidating all cache entries."""
        sim.run("R1 1 0 1k", {"type": "tran"}, runner=mock_runner)
        sim.run("R1 1 0 2k", {"type": "tran"}, runner=mock_runner)

        count = sim.invalidate()
        assert count == 2
        assert sim.cache_size == 0

    def test_invalidate_specific_circuit(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test invalidating specific circuit."""
        sim.run("R1 1 0 1k", {"type": "tran"}, runner=mock_runner)
        sim.run("R1 1 0 2k", {"type": "tran"}, runner=mock_runner)

        count = sim.invalidate(circuit="R1 1 0 1k")
        assert count == 1
        assert sim.cache_size == 1

    def test_get_cached_result(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test getting cached result directly."""
        circuit = "R1 1 0 1k"
        analysis = {"type": "tran"}

        sim.run(circuit, analysis, runner=mock_runner)

        cached = sim.get_cached_result(circuit, analysis)
        assert cached is not None
        assert cached["result"] == "simulated"

    def test_get_cached_result_miss(self, sim: IncrementalSimulator) -> None:
        """Test getting non-existent cached result."""
        cached = sim.get_cached_result("R1 1 0 1k", {"type": "tran"})
        assert cached is None

    def test_lru_eviction(self, mock_runner: Any) -> None:
        """Test LRU eviction when cache is full."""
        sim = IncrementalSimulator(max_cache_entries=3)

        # Add 4 entries to trigger eviction
        for i in range(4):
            sim.run(f"R1 1 0 {i}k", {"type": "tran"}, runner=mock_runner)

        assert sim.cache_size == 3

    def test_clear_stats(
        self, sim: IncrementalSimulator, mock_runner: Any
    ) -> None:
        """Test clearing statistics."""
        sim.run("R1 1 0 1k", {"type": "tran"}, runner=mock_runner)
        sim.run("R1 1 0 1k", {"type": "tran"}, runner=mock_runner)

        assert sim.stats.cache_hits == 1
        sim.clear_stats()
        assert sim.stats.cache_hits == 0

    def test_runner_with_run_method(self, sim: IncrementalSimulator) -> None:
        """Test runner with run() method."""

        class MockRunner:
            def run(self, circuit: Any, analysis: Any) -> dict[str, Any]:
                return {"result": "from_run_method"}

        result = sim.run("R1 1 0 1k", {"type": "tran"}, runner=MockRunner())
        assert result["result"] == "from_run_method"


class TestIncrementalSimulatorPersistence:
    """Tests for cache persistence."""

    def test_save_and_load_cache(self, tmp_path: Path) -> None:
        """Test saving and loading cache from disk."""
        cache_dir = tmp_path / "cache"

        # Create simulator and add entries
        sim1 = IncrementalSimulator(cache_dir=cache_dir)

        def runner(c: Any, a: Any) -> dict[str, Any]:
            return {"data": "test"}

        sim1.run("R1 1 0 1k", {"type": "tran"}, runner=runner)
        sim1._save_cache()

        # Create new simulator with same cache dir
        sim2 = IncrementalSimulator(cache_dir=cache_dir)

        # Should have cached result
        result = sim2.get_cached_result("R1 1 0 1k", {"type": "tran"})
        assert result is not None
        assert result["data"] == "test"


class TestDependencyTracker:
    """Tests for DependencyTracker class."""

    @pytest.fixture
    def tracker(self) -> DependencyTracker:
        """Create tracker instance."""
        return DependencyTracker()

    def test_add_dependency(self, tracker: DependencyTracker) -> None:
        """Test adding dependencies."""
        tracker.add_dependency("sim1", ["R1", "R2", "C1"])

        deps = tracker.get_dependencies("sim1")
        assert deps == {"R1", "R2", "C1"}

    def test_get_affected(self, tracker: DependencyTracker) -> None:
        """Test getting affected targets."""
        tracker.add_dependency("sim1", ["R1", "R2"])
        tracker.add_dependency("sim2", ["R1", "C1"])

        affected = tracker.get_affected("R1")
        assert affected == {"sim1", "sim2"}

        affected = tracker.get_affected("C1")
        assert affected == {"sim2"}

    def test_get_affected_no_deps(self, tracker: DependencyTracker) -> None:
        """Test getting affected when no dependencies."""
        affected = tracker.get_affected("unknown")
        assert affected == set()

    def test_clear(self, tracker: DependencyTracker) -> None:
        """Test clearing dependencies."""
        tracker.add_dependency("sim1", ["R1"])
        tracker.clear()

        assert tracker.get_dependencies("sim1") == set()
        assert tracker.get_affected("R1") == set()


class TestCompareCircuits:
    """Tests for compare_circuits function."""

    def test_identical_circuits(self) -> None:
        """Test comparing identical circuits."""
        changes = compare_circuits("R1 1 0 1k", "R1 1 0 1k")
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.NONE

    def test_different_value(self) -> None:
        """Test comparing circuits with different values."""
        changes = compare_circuits("R1 1 0 1k", "R1 1 0 2k")
        assert len(changes) >= 1
        # Should detect removed and added lines
        types = {c.change_type for c in changes}
        assert ChangeType.COMPONENT_ADDED in types or ChangeType.COMPONENT_REMOVED in types

    def test_added_component(self) -> None:
        """Test detecting added component."""
        changes = compare_circuits("R1 1 0 1k", "R1 1 0 1k\nR2 2 0 2k")
        assert any(c.change_type == ChangeType.COMPONENT_ADDED for c in changes)

    def test_removed_component(self) -> None:
        """Test detecting removed component."""
        changes = compare_circuits("R1 1 0 1k\nR2 2 0 2k", "R1 1 0 1k")
        assert any(c.change_type == ChangeType.COMPONENT_REMOVED for c in changes)
