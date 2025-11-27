"""Performance benchmarks for analysis and simulation setup.

These benchmarks measure the performance of simulation preparation
without requiring an actual SPICE engine. They test:
- AnalysisSpec creation and validation
- Sweep expansion (combinatorics)
- Job hashing for caching
- Circuit preparation for simulation
- Netlist generation with analysis directives

Run with: pytest tests/benchmarks/test_analysis_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec, Probe, SweepSpec


def _build_rc_circuit(name: str = "rc_filter") -> Circuit:
    """Build a simple RC low-pass filter circuit."""
    circuit = Circuit(name)

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

    return circuit


def _build_large_circuit(n_stages: int = 100) -> Circuit:
    """Build a larger RC ladder circuit for stress testing."""
    circuit = Circuit(f"rc_ladder_{n_stages}")

    vin = Vdc("Vin", 1.0)
    circuit.add(vin)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)

    prev_net = Net("in")
    for i in range(n_stages):
        r = Resistor(f"R{i}", 1000.0)
        c = Capacitor(f"C{i}", 1e-9)
        circuit.add(r, c)
        circuit.connect(r.ports[0], prev_net)
        mid_net = Net(f"n{i}")
        circuit.connect(r.ports[1], mid_net)
        circuit.connect(c.ports[0], mid_net)
        circuit.connect(c.ports[1], GND)
        prev_net = mid_net

    return circuit


# ==============================================================================
# AnalysisSpec Creation Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_op_analysis_creation(benchmark) -> None:
    """Benchmark creating an operating point analysis."""
    result = benchmark(lambda: AnalysisSpec("op", {}))
    assert result.mode == "op"


@pytest.mark.benchmark
def test_benchmark_tran_analysis_creation(benchmark) -> None:
    """Benchmark creating a transient analysis."""
    result = benchmark(
        lambda: AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"})
    )
    assert result.mode == "tran"


@pytest.mark.benchmark
def test_benchmark_ac_analysis_creation(benchmark) -> None:
    """Benchmark creating an AC analysis."""
    result = benchmark(
        lambda: AnalysisSpec("ac", {"variation": "dec", "npoints": 100, "fstart": "1", "fstop": "1G"})
    )
    assert result.mode == "ac"


@pytest.mark.benchmark
def test_benchmark_dc_analysis_creation(benchmark) -> None:
    """Benchmark creating a DC sweep analysis."""
    result = benchmark(
        lambda: AnalysisSpec("dc", {"srcnam": "Vin", "vstart": 0, "vstop": 5, "vincr": 0.1})
    )
    assert result.mode == "dc"


@pytest.mark.benchmark
def test_benchmark_multiple_analyses_creation(benchmark) -> None:
    """Benchmark creating multiple analyses at once."""

    def create_analyses():
        return [
            AnalysisSpec("op", {}),
            AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"}),
            AnalysisSpec("ac", {"variation": "dec", "npoints": 100, "fstart": "1", "fstop": "1G"}),
        ]

    result = benchmark(create_analyses)
    assert len(result) == 3


# ==============================================================================
# SweepSpec Creation and Expansion Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_sweep_creation_simple(benchmark) -> None:
    """Benchmark creating a simple sweep spec."""
    result = benchmark(lambda: SweepSpec({"R1": ["1k", "2k", "5k"]}))
    assert "R1" in result.variables


@pytest.mark.benchmark
def test_benchmark_sweep_creation_multi_variable(benchmark) -> None:
    """Benchmark creating a multi-variable sweep spec."""
    result = benchmark(
        lambda: SweepSpec({
            "R1": ["1k", "2k", "5k", "10k"],
            "C1": ["100n", "1u", "10u"],
        })
    )
    assert len(result.variables) == 2


@pytest.mark.benchmark
def test_benchmark_sweep_expansion_small(benchmark) -> None:
    """Benchmark expanding a small sweep (12 combinations)."""
    from spicelab.orchestrator import _expand_sweep

    sweep = SweepSpec({
        "R1": ["1k", "2k", "5k", "10k"],
        "C1": ["100n", "1u", "10u"],
    })

    result = benchmark(_expand_sweep, sweep)
    assert len(result) == 12  # 4 x 3


@pytest.mark.benchmark
def test_benchmark_sweep_expansion_medium(benchmark) -> None:
    """Benchmark expanding a medium sweep (100 combinations)."""
    from spicelab.orchestrator import _expand_sweep

    sweep = SweepSpec({
        "R1": [f"{i}k" for i in range(1, 11)],  # 10 values
        "C1": [f"{i}n" for i in range(1, 11)],  # 10 values
    })

    result = benchmark(_expand_sweep, sweep)
    assert len(result) == 100  # 10 x 10


@pytest.mark.benchmark
def test_benchmark_sweep_expansion_large(benchmark) -> None:
    """Benchmark expanding a large sweep (1000 combinations)."""
    from spicelab.orchestrator import _expand_sweep

    sweep = SweepSpec({
        "R1": [f"{i}k" for i in range(1, 11)],   # 10 values
        "C1": [f"{i}n" for i in range(1, 11)],   # 10 values
        "R2": [f"{i}k" for i in range(1, 11)],   # 10 values
    })

    result = benchmark(_expand_sweep, sweep)
    assert len(result) == 1000  # 10 x 10 x 10


# ==============================================================================
# Job Hashing Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_job_hash_simple(benchmark) -> None:
    """Benchmark hashing a simple job."""
    from spicelab.orchestrator import Job, _job_hash

    circuit = _build_rc_circuit()
    job = Job(
        circuit=circuit,
        analyses=[AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"})],
        engine="ngspice",
    )
    combos = [{}]

    result = benchmark(_job_hash, job, combos)
    assert len(result) == 12  # hash length


@pytest.mark.benchmark
def test_benchmark_job_hash_with_sweep(benchmark) -> None:
    """Benchmark hashing a job with sweep."""
    from spicelab.orchestrator import Job, _expand_sweep, _job_hash

    circuit = _build_rc_circuit()
    sweep = SweepSpec({"R1": ["1k", "2k", "5k"]})
    combos = _expand_sweep(sweep)

    job = Job(
        circuit=circuit,
        analyses=[AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"})],
        sweep=sweep,
        engine="ngspice",
    )

    result = benchmark(_job_hash, job, combos)
    assert len(result) == 12


@pytest.mark.benchmark
def test_benchmark_job_hash_large_circuit(benchmark) -> None:
    """Benchmark hashing a job with large circuit."""
    from spicelab.orchestrator import Job, _job_hash

    circuit = _build_large_circuit(100)
    job = Job(
        circuit=circuit,
        analyses=[
            AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"}),
            AnalysisSpec("ac", {"variation": "dec", "npoints": 100, "fstart": "1", "fstop": "1G"}),
        ],
        engine="ngspice",
    )
    combos = [{}]

    result = benchmark(_job_hash, job, combos)
    assert len(result) == 12


# ==============================================================================
# Probe Creation Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_probe_v_creation(benchmark) -> None:
    """Benchmark creating voltage probes."""
    result = benchmark(lambda: Probe.v("out"))
    assert result.kind == "voltage"


@pytest.mark.benchmark
def test_benchmark_probe_i_creation(benchmark) -> None:
    """Benchmark creating current probes."""
    result = benchmark(lambda: Probe.i("Vin"))
    assert result.kind == "current"


@pytest.mark.benchmark
def test_benchmark_probe_list_10(benchmark) -> None:
    """Benchmark creating 10 probes."""

    def create_probes():
        return [Probe.v(f"n{i}") for i in range(10)]

    result = benchmark(create_probes)
    assert len(result) == 10


@pytest.mark.benchmark
def test_benchmark_probe_list_100(benchmark) -> None:
    """Benchmark creating 100 probes."""

    def create_probes():
        return [Probe.v(f"n{i}") for i in range(100)]

    result = benchmark(create_probes)
    assert len(result) == 100


# ==============================================================================
# Netlist with Analysis Directives Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_netlist_with_tran_directive(benchmark) -> None:
    """Benchmark generating netlist with transient analysis directive."""
    circuit = _build_rc_circuit()
    circuit.add_directive(".tran 1u 10m")

    result = benchmark(circuit.build_netlist)
    assert ".tran" in result


@pytest.mark.benchmark
def test_benchmark_netlist_with_ac_directive(benchmark) -> None:
    """Benchmark generating netlist with AC analysis directive."""
    circuit = _build_rc_circuit()
    circuit.add_directive(".ac dec 100 1 1G")

    result = benchmark(circuit.build_netlist)
    assert ".ac" in result


@pytest.mark.benchmark
def test_benchmark_netlist_with_multiple_directives(benchmark) -> None:
    """Benchmark generating netlist with multiple analysis directives."""
    circuit = _build_rc_circuit()
    circuit.add_directive(".tran 1u 10m")
    circuit.add_directive(".ac dec 100 1 1G")
    circuit.add_directive(".op")
    circuit.add_directive(".probe v(out)")
    circuit.add_directive(".option RELTOL=1e-6")

    result = benchmark(circuit.build_netlist)
    assert ".tran" in result
    assert ".ac" in result


@pytest.mark.benchmark
def test_benchmark_large_circuit_with_analysis(benchmark) -> None:
    """Benchmark large circuit netlist with analysis directives."""
    circuit = _build_large_circuit(100)
    circuit.add_directive(".tran 1u 10m")
    circuit.add_directive(".ac dec 100 1 1G")

    result = benchmark(circuit.build_netlist)
    assert ".tran" in result


# ==============================================================================
# Full Simulation Preparation Benchmarks (No Engine)
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_full_sim_prep_simple(benchmark) -> None:
    """Benchmark full simulation preparation for simple circuit."""
    from spicelab.orchestrator import Job, _expand_sweep, _job_hash

    def prepare_simulation():
        circuit = _build_rc_circuit()
        analyses = [
            AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"}),
        ]
        job = Job(circuit=circuit, analyses=analyses, engine="ngspice")
        combos = _expand_sweep(job.sweep)
        job_h = _job_hash(job, combos)
        netlist = circuit.build_netlist()
        return job_h, netlist

    result = benchmark(prepare_simulation)
    assert len(result[0]) == 12
    assert "R1" in result[1]


@pytest.mark.benchmark
def test_benchmark_full_sim_prep_with_sweep(benchmark) -> None:
    """Benchmark full simulation preparation with parameter sweep."""
    from spicelab.orchestrator import Job, _expand_sweep, _job_hash

    def prepare_simulation():
        circuit = _build_rc_circuit()
        analyses = [
            AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"}),
            AnalysisSpec("ac", {"variation": "dec", "npoints": 100, "fstart": "1", "fstop": "1G"}),
        ]
        sweep = SweepSpec({"R1": ["1k", "2k", "5k", "10k"]})
        job = Job(circuit=circuit, analyses=analyses, sweep=sweep, engine="ngspice")
        combos = _expand_sweep(job.sweep)
        job_h = _job_hash(job, combos)
        netlist = circuit.build_netlist()
        return job_h, netlist, combos

    result = benchmark(prepare_simulation)
    assert len(result[0]) == 12
    assert len(result[2]) == 4


@pytest.mark.benchmark
def test_benchmark_full_sim_prep_large_sweep(benchmark) -> None:
    """Benchmark full simulation preparation with large sweep (100 combos)."""
    from spicelab.orchestrator import Job, _expand_sweep, _job_hash

    def prepare_simulation():
        circuit = _build_rc_circuit()
        analyses = [AnalysisSpec("tran", {"tstep": "1u", "tstop": "10m"})]
        sweep = SweepSpec({
            "R1": [f"{i}k" for i in range(1, 11)],
            "C1": [f"{i}n" for i in range(1, 11)],
        })
        job = Job(circuit=circuit, analyses=analyses, sweep=sweep, engine="ngspice")
        combos = _expand_sweep(job.sweep)
        job_h = _job_hash(job, combos)
        return job_h, combos

    result = benchmark(prepare_simulation)
    assert len(result[0]) == 12
    assert len(result[1]) == 100


@pytest.mark.benchmark
def test_benchmark_monte_carlo_prep_100_runs(benchmark) -> None:
    """Benchmark Monte Carlo-like preparation (100 circuit variants)."""
    from spicelab.orchestrator import Job, _job_hash

    def prepare_monte_carlo():
        jobs = []
        for i in range(100):
            circuit = Circuit(f"mc_{i}")
            r = Resistor("R1", 1000.0 * (1 + i * 0.01))
            c = Capacitor("C1", 1e-6)
            vin = Vdc("Vin", 1.0)
            circuit.add(vin, r, c)
            circuit.connect(vin.ports[0], Net("in"))
            circuit.connect(vin.ports[1], GND)
            circuit.connect(r.ports[0], Net("in"))
            circuit.connect(r.ports[1], Net("out"))
            circuit.connect(c.ports[0], Net("out"))
            circuit.connect(c.ports[1], GND)

            analyses = [AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})]
            job = Job(circuit=circuit, analyses=analyses, engine="ngspice")
            job_h = _job_hash(job, [{}])
            jobs.append((job, job_h))

        return jobs

    result = benchmark(prepare_monte_carlo)
    assert len(result) == 100
