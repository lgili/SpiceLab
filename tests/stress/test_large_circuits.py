"""Stress tests for large circuits.

These tests verify that SpiceLab can handle circuits with many components,
complex topologies, and large netlists without performance degradation
or memory issues.
"""

import time
from collections.abc import Callable

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net


def measure_time(func: Callable[[], None]) -> float:
    """Measure execution time of a function in seconds."""
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


# ==============================================================================
# Large Component Count Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_100_components():
    """Circuit should handle 100 components efficiently."""
    circuit = Circuit("stress_100")

    prev_net = GND
    for i in range(100):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    # Should generate netlist quickly
    netlist = circuit.build_netlist()

    assert len(circuit._components) == 100
    assert "R0" in netlist
    assert "R99" in netlist
    assert ".end" in netlist.lower()


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_500_components():
    """Circuit should handle 500 components efficiently."""
    circuit = Circuit("stress_500")

    prev_net = GND
    for i in range(500):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    netlist = circuit.build_netlist()

    assert len(circuit._components) == 500
    assert "R0" in netlist
    assert "R499" in netlist


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_1000_components():
    """Circuit should handle 1000 components efficiently."""
    circuit = Circuit("stress_1k")

    prev_net = GND
    for i in range(1000):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    start = time.perf_counter()
    netlist = circuit.build_netlist()
    elapsed = time.perf_counter() - start

    assert len(circuit._components) == 1000
    # Netlist generation should complete in reasonable time
    assert elapsed < 5.0, f"Netlist generation took {elapsed:.2f}s, expected < 5s"


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_2000_components():
    """Circuit should handle 2000 components."""
    circuit = Circuit("stress_2k")

    prev_net = GND
    for i in range(2000):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    start = time.perf_counter()
    netlist = circuit.build_netlist()
    elapsed = time.perf_counter() - start

    assert len(circuit._components) == 2000
    assert elapsed < 10.0, f"Netlist generation took {elapsed:.2f}s, expected < 10s"


# ==============================================================================
# Mixed Component Type Stress Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_mixed_components_500():
    """Circuit should handle 500 mixed component types."""
    circuit = Circuit("stress_mixed_500")

    prev_net = GND
    for i in range(500):
        # Cycle through component types
        comp_type = i % 4
        if comp_type == 0:
            comp = Resistor(f"R{i // 4}", 1000.0)
        elif comp_type == 1:
            comp = Capacitor(f"C{i // 4}", 100e-9)
        elif comp_type == 2:
            comp = Inductor(f"L{i // 4}", 1e-3)
        else:
            comp = Vdc(f"V{i // 4}", 5.0)

        circuit.add(comp)
        circuit.connect(comp.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(comp.ports[1], next_net)
        prev_net = next_net

    netlist = circuit.build_netlist()

    assert len(circuit._components) == 500
    assert "R" in netlist
    assert "C" in netlist
    assert "L" in netlist
    assert "V" in netlist


# ==============================================================================
# Complex Topology Stress Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_ladder_network_100_stages():
    """Circuit should handle a 100-stage RC ladder network."""
    circuit = Circuit("ladder_100")

    # Create ladder network: R-C-R-C-...
    # Input connected to GND
    input_net = Net("in")
    vin = Vdc("Vin", 1.0)
    circuit.add(vin)
    circuit.connect(vin.ports[0], input_net)
    circuit.connect(vin.ports[1], GND)

    prev_net = input_net
    for i in range(100):
        # Series resistor
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)

        # Node between R and C
        mid_net = Net(f"n{i}")
        circuit.connect(r.ports[1], mid_net)

        # Shunt capacitor
        c = Capacitor(f"C{i}", 100e-12)
        circuit.add(c)
        circuit.connect(c.ports[0], mid_net)
        circuit.connect(c.ports[1], GND)

        prev_net = mid_net

    netlist = circuit.build_netlist()

    # 1 Vdc + 100 R + 100 C = 201 components
    assert len(circuit._components) == 201
    assert "R0" in netlist
    assert "R99" in netlist
    assert "C0" in netlist
    assert "C99" in netlist


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_star_topology_100_branches():
    """Circuit should handle a star topology with 100 branches from center."""
    circuit = Circuit("star_100")

    center = Net("center")

    # Input source
    vin = Vdc("Vin", 1.0)
    circuit.add(vin)
    circuit.connect(vin.ports[0], center)
    circuit.connect(vin.ports[1], GND)

    # 100 branches from center
    for i in range(100):
        r = Resistor(f"R{i}", 1000.0)
        c = Capacitor(f"C{i}", 100e-12)

        circuit.add(r, c)
        circuit.connect(r.ports[0], center)

        branch_net = Net(f"branch{i}")
        circuit.connect(r.ports[1], branch_net)
        circuit.connect(c.ports[0], branch_net)
        circuit.connect(c.ports[1], GND)

    netlist = circuit.build_netlist()

    # 1 Vdc + 100 R + 100 C = 201 components
    assert len(circuit._components) == 201


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_mesh_topology_10x10():
    """Circuit should handle a 10x10 mesh/grid topology."""
    circuit = Circuit("mesh_10x10")

    # Create 10x10 grid of nodes
    nodes: list[list[Net]] = []
    for row in range(10):
        row_nodes = []
        for col in range(10):
            if row == 0 and col == 0:
                node = GND
            else:
                node = Net(f"n{row}_{col}")
            row_nodes.append(node)
        nodes.append(row_nodes)

    # Connect horizontal resistors
    r_count = 0
    for row in range(10):
        for col in range(9):
            r = Resistor(f"RH{r_count}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], nodes[row][col])
            circuit.connect(r.ports[1], nodes[row][col + 1])
            r_count += 1

    # Connect vertical resistors
    for row in range(9):
        for col in range(10):
            r = Resistor(f"RV{r_count}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], nodes[row][col])
            circuit.connect(r.ports[1], nodes[row + 1][col])
            r_count += 1

    netlist = circuit.build_netlist()

    # 10*9 horizontal + 9*10 vertical = 180 resistors
    assert len(circuit._components) == 180


# ==============================================================================
# Performance Scaling Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_netlist_generation_scales_linearly():
    """Netlist generation time should scale roughly linearly with component count."""
    sizes = [100, 200, 400]
    times = []

    for size in sizes:
        circuit = Circuit(f"scale_{size}")
        prev_net = GND

        for i in range(size):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], prev_net)
            next_net = Net(f"n{i}")
            circuit.connect(r.ports[1], next_net)
            prev_net = next_net

        elapsed = measure_time(lambda: circuit.build_netlist())
        times.append(elapsed)

    # Check that doubling size doesn't more than triple time
    # (allowing for overhead and variance)
    if times[0] > 0.001:  # Only check if measurable
        ratio_1_to_2 = times[1] / times[0]
        ratio_2_to_3 = times[2] / times[1]

        # Should be roughly 2x, allow up to 4x for overhead
        assert ratio_1_to_2 < 4.0, f"Time didn't scale linearly: {ratio_1_to_2:.2f}x"
        assert ratio_2_to_3 < 4.0, f"Time didn't scale linearly: {ratio_2_to_3:.2f}x"


@pytest.mark.stress
@pytest.mark.slow
def test_hash_generation_scales_linearly():
    """Hash generation time should scale roughly linearly with component count."""
    sizes = [100, 200, 400]
    times = []

    for size in sizes:
        circuit = Circuit(f"hash_scale_{size}")
        prev_net = GND

        for i in range(size):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], prev_net)
            next_net = Net(f"n{i}")
            circuit.connect(r.ports[1], next_net)
            prev_net = next_net

        elapsed = measure_time(lambda: circuit.hash())
        times.append(elapsed)

    # Check scaling (same logic as netlist test)
    if times[0] > 0.001:
        ratio_1_to_2 = times[1] / times[0]
        ratio_2_to_3 = times[2] / times[1]

        assert ratio_1_to_2 < 4.0, f"Hash time didn't scale linearly: {ratio_1_to_2:.2f}x"
        assert ratio_2_to_3 < 4.0, f"Hash time didn't scale linearly: {ratio_2_to_3:.2f}x"


# ==============================================================================
# Memory Stress Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_memory_not_exploding():
    """Circuit memory usage should be reasonable for large circuits."""
    import sys

    circuit = Circuit("memory_test")

    # Add 1000 components
    prev_net = GND
    for i in range(1000):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    # Generate netlist (creates strings)
    netlist = circuit.build_netlist()

    # Basic size check - should be reasonable
    # Each resistor line ~30 chars, 1000 resistors = ~30KB + overhead
    assert len(netlist) < 100_000, f"Netlist unexpectedly large: {len(netlist)} chars"

    # Check circuit object size is reasonable
    # This is a rough heuristic
    circuit_size = sys.getsizeof(circuit)
    # Note: getsizeof doesn't count referenced objects, but gives baseline
    assert circuit_size < 1_000_000, f"Circuit object too large: {circuit_size} bytes"


@pytest.mark.stress
@pytest.mark.slow
def test_multiple_large_circuits_can_coexist():
    """Multiple large circuits should be able to coexist in memory."""
    circuits = []

    for circuit_idx in range(10):
        circuit = Circuit(f"multi_{circuit_idx}")
        prev_net = GND

        for i in range(200):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], prev_net)
            next_net = Net(f"n{i}")
            circuit.connect(r.ports[1], next_net)
            prev_net = next_net

        circuits.append(circuit)

    # All circuits should be valid and independent
    assert len(circuits) == 10

    for idx, circuit in enumerate(circuits):
        assert len(circuit._components) == 200
        assert circuit.name == f"multi_{idx}"

    # Each should generate independent netlist
    netlists = [c.build_netlist() for c in circuits]
    assert len(set(netlists)) == 10  # All different (different names)


# ==============================================================================
# Directive Stress Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_many_directives():
    """Circuit should handle many directives efficiently."""
    circuit = Circuit("many_directives")

    # Add a component
    r = Resistor("R1", 1000.0)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    # Add 100 directives
    for i in range(100):
        circuit.add_directive(f".param VAL{i}={i}")

    netlist = circuit.build_netlist()

    assert len(circuit._directives) == 100
    assert ".param VAL0=0" in netlist
    assert ".param VAL99=99" in netlist


@pytest.mark.stress
@pytest.mark.slow
def test_circuit_with_large_model_directives():
    """Circuit should handle large model directives."""
    circuit = Circuit("large_models")

    # Create a large model string (simulating vendor model)
    large_model = ".model LARGE_DEVICE NPN(\n"
    for i in range(50):
        large_model += f"+  PARAM{i}={i * 0.001}\n"
    large_model += "+)"

    circuit.add_directive(large_model)

    r = Resistor("R1", 1000.0)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    netlist = circuit.build_netlist()

    assert "LARGE_DEVICE" in netlist
    assert "PARAM0" in netlist
    assert "PARAM49" in netlist


# ==============================================================================
# Summary/Introspection Stress Tests
# ==============================================================================


@pytest.mark.stress
@pytest.mark.slow
def test_summary_with_500_components():
    """Circuit summary should handle 500 components."""
    circuit = Circuit("summary_500")

    prev_net = GND
    for i in range(500):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    start = time.perf_counter()
    summary = circuit.summary()
    elapsed = time.perf_counter() - start

    assert "Components (500)" in summary
    assert elapsed < 2.0, f"Summary generation took {elapsed:.2f}s, expected < 2s"


@pytest.mark.stress
@pytest.mark.slow
def test_to_dot_with_200_components():
    """Circuit DOT generation should handle 200 components."""
    circuit = Circuit("dot_200")

    prev_net = GND
    for i in range(200):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    start = time.perf_counter()
    dot = circuit.to_dot()
    elapsed = time.perf_counter() - start

    assert "graph circuit" in dot
    assert "R0" in dot
    assert "R199" in dot
    assert elapsed < 2.0, f"DOT generation took {elapsed:.2f}s, expected < 2s"
