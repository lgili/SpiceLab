"""Memory profiling tests for Monte Carlo and other intensive operations.

These tests use tracemalloc to monitor memory usage and ensure operations
don't leak memory or use excessive amounts.
"""

import gc
import tracemalloc

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net


def _build_rc_circuit() -> tuple[Circuit, Resistor]:
    """Build a simple RC circuit for testing."""
    circuit = Circuit("rc_test")

    vin = Vdc("Vin", 1.0)
    r1 = Resistor("R1", 1000.0)
    c1 = Capacitor("C1", 1e-6)

    circuit.add(vin, r1, c1)

    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], Net("out"))
    circuit.connect(c1.ports[0], Net("out"))
    circuit.connect(c1.ports[1], GND)

    return circuit, r1


# ==============================================================================
# Circuit Creation Memory Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_creation_memory():
    """Circuit creation should have bounded memory usage."""
    gc.collect()
    tracemalloc.start()

    # Create many circuits
    circuits = []
    for i in range(100):
        circuit, _ = _build_rc_circuit()
        circuit._name = f"circuit_{i}"
        circuits.append(circuit)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 100 simple circuits should use < 50MB
    assert peak < 50 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB (expected < 50MB)"


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_creation_no_leak():
    """Creating and discarding circuits should not leak memory."""
    gc.collect()
    tracemalloc.start()
    initial_snapshot = tracemalloc.take_snapshot()

    # Create and discard many circuits
    for _ in range(100):
        circuit, _ = _build_rc_circuit()
        _ = circuit.build_netlist()
        del circuit

    gc.collect()
    final_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Compare snapshots - should not grow significantly
    top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")

    # Sum of memory differences
    total_diff = sum(stat.size_diff for stat in top_stats[:10])

    # Should not grow by more than 5MB after GC
    assert total_diff < 5 * 1024 * 1024, f"Memory grew by {total_diff / 1e6:.1f}MB"


# ==============================================================================
# Large Circuit Memory Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_large_circuit_memory_bounded():
    """Large circuits should have predictable memory usage."""
    gc.collect()
    tracemalloc.start()

    # Build a 1000-component circuit
    circuit = Circuit("large")
    prev_net = GND

    for i in range(1000):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    # Generate netlist
    netlist = circuit.build_netlist()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 1000 component circuit should use < 100MB
    assert peak < 100 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB (expected < 100MB)"

    # Netlist should be reasonable size
    assert len(netlist) < 200_000, f"Netlist too large: {len(netlist)} chars"


@pytest.mark.stress
@pytest.mark.slow
def test_netlist_generation_memory():
    """Netlist generation should not use excessive temporary memory."""
    # Build circuit first
    circuit = Circuit("netlist_mem")
    prev_net = GND

    for i in range(500):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    gc.collect()
    tracemalloc.start()

    # Now measure just netlist generation
    _ = circuit.build_netlist()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Netlist generation for 500 components should use < 20MB
    assert peak < 20 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB (expected < 20MB)"


# ==============================================================================
# Hash Generation Memory Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_hash_generation_memory():
    """Hash generation should have bounded memory usage."""
    # Build circuit first
    circuit = Circuit("hash_mem")
    prev_net = GND

    for i in range(500):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    gc.collect()
    tracemalloc.start()

    # Generate hash multiple times
    for _ in range(10):
        _ = circuit.hash()

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Hash generation should use < 10MB
    assert peak < 10 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB (expected < 10MB)"


# ==============================================================================
# Validation Memory Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_validation_memory():
    """Circuit validation should have bounded memory usage."""
    from spicelab.validators import validate_circuit

    # Build circuit first
    circuit = Circuit("validate_mem")
    prev_net = GND

    v = Vdc("Vin", 1.0)
    circuit.add(v)
    circuit.connect(v.ports[0], Net("in"))
    circuit.connect(v.ports[1], GND)

    for i in range(500):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        if i == 0:
            circuit.connect(r.ports[0], Net("in"))
        else:
            circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    gc.collect()
    tracemalloc.start()

    # Validate multiple times
    for _ in range(5):
        _ = validate_circuit(circuit)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Validation should use < 20MB
    assert peak < 20 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB (expected < 20MB)"


# ==============================================================================
# Repeated Operations Memory Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_repeated_operations_no_leak():
    """Repeated circuit operations should not leak memory."""
    gc.collect()
    tracemalloc.start()
    initial = tracemalloc.take_snapshot()

    # Perform many operations
    for iteration in range(50):
        circuit = Circuit(f"iter_{iteration}")

        # Build circuit
        for i in range(50):
            r = Resistor(f"R{i}", 1000.0)
            c = Capacitor(f"C{i}", 1e-9)
            circuit.add(r, c)
            circuit.connect(r.ports[0], GND)
            circuit.connect(r.ports[1], Net(f"n{i}"))
            circuit.connect(c.ports[0], Net(f"n{i}"))
            circuit.connect(c.ports[1], GND)

        # Generate outputs
        netlist = circuit.build_netlist()
        h = circuit.hash()
        summary = circuit.summary()

        # Discard
        del circuit, netlist, h, summary

        # Periodic GC
        if iteration % 10 == 0:
            gc.collect()

    gc.collect()
    final = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Check for memory growth
    top_stats = final.compare_to(initial, "lineno")
    total_diff = sum(stat.size_diff for stat in top_stats[:20])

    # Should not grow by more than 10MB after full GC
    assert total_diff < 10 * 1024 * 1024, f"Memory grew by {total_diff / 1e6:.1f}MB"


# ==============================================================================
# Monte Carlo Simulation Memory (Mock)
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_monte_carlo_like_memory():
    """Simulating Monte Carlo-like repeated circuit creation should be memory bounded.

    This tests the pattern used in Monte Carlo without requiring actual simulation.
    """
    gc.collect()
    tracemalloc.start()

    circuits = []
    netlists = []

    # Simulate 100 "Monte Carlo runs" - create circuits with varied parameters
    for run in range(100):
        circuit = Circuit(f"mc_run_{run}")

        # Vary resistance values like Monte Carlo would
        r_value = 1000.0 * (1 + run * 0.01)  # 0-100% variation

        v = Vdc("Vin", 1.0)
        r = Resistor("R1", r_value)
        c = Capacitor("C1", 1e-6)

        circuit.add(v, r, c)
        circuit.connect(v.ports[0], Net("in"))
        circuit.connect(v.ports[1], GND)
        circuit.connect(r.ports[0], Net("in"))
        circuit.connect(r.ports[1], Net("out"))
        circuit.connect(c.ports[0], Net("out"))
        circuit.connect(c.ports[1], GND)

        # Generate netlist for each run
        netlist = circuit.build_netlist()

        circuits.append(circuit)
        netlists.append(netlist)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 100 simple circuits should use < 50MB
    assert peak < 50 * 1024 * 1024, f"Peak memory: {peak / 1e6:.1f}MB (expected < 50MB)"

    # All circuits should be valid
    assert len(circuits) == 100
    assert all(len(nl) > 50 for nl in netlists)  # Each netlist should have content
