"""Performance benchmarks for circuit operations.

These benchmarks measure the performance of critical operations to
track performance over time and catch regressions.

Run with: pytest tests/benchmarks/ --benchmark-only
"""

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec, Probe, SweepSpec, stable_hash

# ==============================================================================
# Circuit Creation Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_circuit_creation(benchmark):
    """Benchmark creating an empty circuit."""
    result = benchmark(lambda: Circuit("benchmark_circuit"))
    assert result.name == "benchmark_circuit"


@pytest.mark.benchmark
def test_benchmark_component_creation(benchmark):
    """Benchmark creating a resistor component."""
    result = benchmark(lambda: Resistor("R1", 1000.0))
    assert result.ref == "R1"


@pytest.mark.benchmark
def test_benchmark_add_single_component(benchmark):
    """Benchmark adding a single component to circuit."""

    def add_component():
        circuit = Circuit("test")
        r = Resistor("R1", 1000.0)
        circuit.add(r)
        return circuit

    result = benchmark(add_component)
    assert len(result._components) == 1


@pytest.mark.benchmark
def test_benchmark_add_10_components(benchmark):
    """Benchmark adding 10 components to circuit."""

    def add_components():
        circuit = Circuit("test")
        for i in range(10):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
        return circuit

    result = benchmark(add_components)
    assert len(result._components) == 10


@pytest.mark.benchmark
def test_benchmark_add_100_components(benchmark):
    """Benchmark adding 100 components to circuit."""

    def add_components():
        circuit = Circuit("test")
        for i in range(100):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
        return circuit

    result = benchmark(add_components)
    assert len(result._components) == 100


# ==============================================================================
# Connection Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_connect_to_gnd(benchmark):
    """Benchmark connecting a port to GND."""
    circuit = Circuit("test")
    r = Resistor("R1", 1000.0)
    circuit.add(r)

    def connect():
        circuit.connect(r.ports[0], GND)

    benchmark(connect)
    assert circuit._port_to_net.get(r.ports[0]) is GND


@pytest.mark.benchmark
def test_benchmark_connect_to_named_net(benchmark):
    """Benchmark connecting a port to a named net."""
    circuit = Circuit("test")
    r = Resistor("R1", 1000.0)
    circuit.add(r)
    net = Net("out")

    def connect():
        circuit.connect(r.ports[1], net)

    benchmark(connect)


@pytest.mark.benchmark
def test_benchmark_connect_ports_together(benchmark):
    """Benchmark connecting two ports together."""
    circuit = Circuit("test")
    r1 = Resistor("R1", 1000.0)
    r2 = Resistor("R2", 1000.0)
    circuit.add(r1, r2)

    def connect():
        circuit.connect(r1.ports[1], r2.ports[0])

    benchmark(connect)


@pytest.mark.benchmark
def test_benchmark_build_and_connect_10_series(benchmark):
    """Benchmark building and connecting 10 series resistors."""

    def build_series():
        circuit = Circuit("series")
        prev_net = GND
        for i in range(10):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], prev_net)
            next_net = Net(f"n{i}")
            circuit.connect(r.ports[1], next_net)
            prev_net = next_net
        return circuit

    result = benchmark(build_series)
    assert len(result._components) == 10


@pytest.mark.benchmark
def test_benchmark_build_and_connect_100_series(benchmark):
    """Benchmark building and connecting 100 series resistors."""

    def build_series():
        circuit = Circuit("series")
        prev_net = GND
        for i in range(100):
            r = Resistor(f"R{i}", 1000.0)
            circuit.add(r)
            circuit.connect(r.ports[0], prev_net)
            next_net = Net(f"n{i}")
            circuit.connect(r.ports[1], next_net)
            prev_net = next_net
        return circuit

    result = benchmark(build_series)
    assert len(result._components) == 100


# ==============================================================================
# Netlist Generation Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_netlist_10_components(benchmark):
    """Benchmark netlist generation for 10 components."""
    circuit = Circuit("netlist_10")
    prev_net = GND
    for i in range(10):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.build_netlist)
    assert "R0" in result
    assert "R9" in result


@pytest.mark.benchmark
def test_benchmark_netlist_100_components(benchmark):
    """Benchmark netlist generation for 100 components."""
    circuit = Circuit("netlist_100")
    prev_net = GND
    for i in range(100):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.build_netlist)
    assert "R0" in result
    assert "R99" in result


@pytest.mark.benchmark
def test_benchmark_netlist_500_components(benchmark):
    """Benchmark netlist generation for 500 components."""
    circuit = Circuit("netlist_500")
    prev_net = GND
    for i in range(500):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.build_netlist)
    assert "R0" in result
    assert "R499" in result


# ==============================================================================
# Hash Generation Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_hash_10_components(benchmark):
    """Benchmark hash generation for 10 components."""
    circuit = Circuit("hash_10")
    prev_net = GND
    for i in range(10):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.hash)
    assert len(result) == 12


@pytest.mark.benchmark
def test_benchmark_hash_100_components(benchmark):
    """Benchmark hash generation for 100 components."""
    circuit = Circuit("hash_100")
    prev_net = GND
    for i in range(100):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.hash)
    assert len(result) == 12


@pytest.mark.benchmark
def test_benchmark_stable_hash_dict(benchmark):
    """Benchmark stable_hash on a dictionary."""
    data = {f"key_{i}": i * 0.5 for i in range(100)}

    result = benchmark(lambda: stable_hash(data))
    assert len(result) == 12


@pytest.mark.benchmark
def test_benchmark_stable_hash_nested(benchmark):
    """Benchmark stable_hash on nested structures."""
    data = {
        "level1": {
            "level2": {
                "values": list(range(50)),
                "nested": {"a": 1, "b": 2, "c": 3},
            }
        }
    }

    result = benchmark(lambda: stable_hash(data))
    assert len(result) == 12


# ==============================================================================
# Summary/Introspection Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_summary_10_components(benchmark):
    """Benchmark summary generation for 10 components."""
    circuit = Circuit("summary_10")
    prev_net = GND
    for i in range(10):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.summary)
    assert "Components (10)" in result


@pytest.mark.benchmark
def test_benchmark_summary_100_components(benchmark):
    """Benchmark summary generation for 100 components."""
    circuit = Circuit("summary_100")
    prev_net = GND
    for i in range(100):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.summary)
    assert "Components (100)" in result


@pytest.mark.benchmark
def test_benchmark_to_dot_50_components(benchmark):
    """Benchmark DOT generation for 50 components."""
    circuit = Circuit("dot_50")
    prev_net = GND
    for i in range(50):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    result = benchmark(circuit.to_dot)
    assert "graph circuit" in result


# ==============================================================================
# Type Creation Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_analysis_spec_creation(benchmark):
    """Benchmark AnalysisSpec creation."""
    result = benchmark(lambda: AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"}))
    assert result.mode == "tran"


@pytest.mark.benchmark
def test_benchmark_sweep_spec_creation(benchmark):
    """Benchmark SweepSpec creation with unit normalization."""
    result = benchmark(lambda: SweepSpec({"R": ["1k", "2k", "5k", "10k"]}))
    assert "R" in result.variables


@pytest.mark.benchmark
def test_benchmark_probe_creation(benchmark):
    """Benchmark Probe creation."""
    result = benchmark(lambda: Probe.v("out"))
    assert result.kind == "voltage"


@pytest.mark.benchmark
def test_benchmark_probe_list_creation(benchmark):
    """Benchmark creating list of probes."""

    def create_probes():
        return [Probe.v(f"n{i}") for i in range(10)]

    result = benchmark(create_probes)
    assert len(result) == 10


# ==============================================================================
# Mixed Component Type Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_mixed_components_50(benchmark):
    """Benchmark circuit with 50 mixed component types."""

    def build_mixed():
        circuit = Circuit("mixed")
        prev_net = GND

        for i in range(50):
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

        return circuit.build_netlist()

    result = benchmark(build_mixed)
    assert ".end" in result.lower()


# ==============================================================================
# Directive Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_add_directive(benchmark):
    """Benchmark adding a directive."""
    circuit = Circuit("directive_test")

    def add_dir():
        circuit.add_directive(".option RELTOL=1e-6")

    benchmark(add_dir)


@pytest.mark.benchmark
def test_benchmark_add_directive_once_existing(benchmark):
    """Benchmark add_directive_once with existing directive."""
    circuit = Circuit("directive_test")
    circuit.add_directive(".option RELTOL=1e-6")

    def add_once():
        circuit.add_directive_once(".option RELTOL=1e-6")

    benchmark(add_once)
    # Should not add duplicate
    assert len(circuit._directives) == 1


@pytest.mark.benchmark
def test_benchmark_netlist_with_directives(benchmark):
    """Benchmark netlist generation with many directives."""
    circuit = Circuit("with_directives")
    r = Resistor("R1", 1000.0)
    circuit.add(r)
    circuit.connect(r.ports[0], GND)
    circuit.connect(r.ports[1], Net("out"))

    for i in range(50):
        circuit.add_directive(f".param VAL{i}={i}")

    result = benchmark(circuit.build_netlist)
    assert ".param VAL0=0" in result
    assert ".param VAL49=49" in result
