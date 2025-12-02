"""Benchmark comparing performance with real NGSpice simulations.

This module provides comprehensive benchmarks using actual NGSpice
simulations to measure the impact of:
- Compressed caching (CompressedCache)
- Parallel execution (ParallelExecutor)
- Incremental simulation (IncrementalSimulator)
- Combined optimizations

Run with: python -m pytest tests/benchmarks/test_real_ngspice_benchmark.py -v -s
"""

from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from spicelab.cache.compressed import ZSTD_AVAILABLE, CompressedCache
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Inductor, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines.orchestrator import run_simulation


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_ms:.2f} ms (\u00b1{self.std_ms:.2f})\n"
            f"  Min/Max: {self.min_ms:.2f} / {self.max_ms:.2f} ms"
        )


def run_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 3,
    warmup: int = 1,
) -> BenchmarkResult:
    """Run a benchmark function multiple times."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    return BenchmarkResult(name=name, iterations=iterations, times_ms=times)


def create_rc_lowpass(
    name: str = "rc_lowpass", r_value: str = "1k", c_value: str = "1u"
) -> Circuit:
    """Create an RC low-pass filter circuit."""
    circuit = Circuit(name)

    # AC voltage source with DC offset
    vin = Vdc("Vin", 1.0)
    r = Resistor("R1", r_value)
    c = Capacitor("C1", c_value)

    circuit.add(vin, r, c)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r.ports[0], Net("in"))
    circuit.connect(r.ports[1], Net("out"))
    circuit.connect(c.ports[0], Net("out"))
    circuit.connect(c.ports[1], GND)

    return circuit


def create_rlc_series(
    name: str = "rlc_series",
    r_value: str = "100",
    l_value: str = "10m",
    c_value: str = "1u",
) -> Circuit:
    """Create an RLC series circuit."""
    circuit = Circuit(name)

    vin = Vdc("Vin", 1.0)
    r = Resistor("R1", r_value)
    ind = Inductor("L1", l_value)
    c = Capacitor("C1", c_value)

    circuit.add(vin, r, ind, c)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r.ports[0], Net("in"))
    circuit.connect(r.ports[1], Net("node1"))
    circuit.connect(ind.ports[0], Net("node1"))
    circuit.connect(ind.ports[1], Net("node2"))
    circuit.connect(c.ports[0], Net("node2"))
    circuit.connect(c.ports[1], GND)

    return circuit


def create_voltage_divider(name: str = "vdiv", r1: str = "10k", r2: str = "10k") -> Circuit:
    """Create a voltage divider circuit."""
    circuit = Circuit(name)

    vin = Vdc("Vin", 10.0)
    r1_comp = Resistor("R1", r1)
    r2_comp = Resistor("R2", r2)

    circuit.add(vin, r1_comp, r2_comp)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1_comp.ports[0], Net("in"))
    circuit.connect(r1_comp.ports[1], Net("out"))
    circuit.connect(r2_comp.ports[0], Net("out"))
    circuit.connect(r2_comp.ports[1], GND)

    return circuit


@pytest.mark.benchmark
class TestRealNGSpiceBenchmark:
    """Benchmark tests using real NGSpice simulations."""

    @pytest.fixture(autouse=True)
    def check_ngspice(self) -> None:
        """Skip tests if ngspice is not available."""
        try:
            from spicelab.spice.registry import get_active_adapter

            get_active_adapter()
        except Exception:
            pytest.skip("ngspice adapter not available")

    def test_cache_vs_no_cache_real(self) -> None:
        """Compare performance with and without caching using real simulations."""
        if not ZSTD_AVAILABLE:
            print("\n\u26a0\ufe0f  zstandard not installed, using fallback compression")

        # Create circuit
        circuit = create_rc_lowpass()
        analyses = [AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})]

        n_runs = 5  # Run same simulation multiple times

        print("\n" + "=" * 60)
        print("REAL NGSPICE: CACHE OPTIMIZATION BENCHMARK")
        print("=" * 60)
        print("\nCircuit: RC Low-pass filter")
        print("Analysis: Transient 1ms with 1us step")
        print(f"Runs: {n_runs}")

        # Without cache - run each simulation fresh
        def run_without_cache() -> list[Any]:
            results = []
            for _ in range(n_runs):
                result = run_simulation(circuit, analyses, engine="ngspice")
                results.append(result)
            return results

        result_no_cache = run_benchmark("Without Cache", run_without_cache, iterations=3, warmup=1)

        # With cache
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompressedCache(cache_dir=Path(tmpdir), max_size_mb=100)

            def run_with_cache() -> list[Any]:
                results = []
                for i in range(n_runs):
                    key = f"rc_tran_{i % 2}"  # Only 2 unique keys to simulate cache hits
                    cached = cache.get(key)
                    if cached is not None:
                        results.append(cached)
                    else:
                        result = run_simulation(circuit, analyses, engine="ngspice")
                        cache.set(key, result)
                        results.append(result)
                return results

            # Populate cache first
            run_with_cache()
            cache.stats.hits = 0
            cache.stats.misses = 0

            result_with_cache = run_benchmark("With Cache", run_with_cache, iterations=3, warmup=0)

            print(f"\n{result_no_cache}")
            print(f"\n{result_with_cache}")

            speedup = (
                result_no_cache.mean_ms / result_with_cache.mean_ms
                if result_with_cache.mean_ms > 0
                else 0
            )
            print(f"\n\U0001f4ca Speedup: {speedup:.2f}x")
            print(f"\U0001f4ca Cache hit rate: {cache.stats.hit_rate:.1%}")

    def test_parallel_vs_sequential_real(self) -> None:
        """Compare parallel vs sequential execution with real simulations."""
        from spicelab.parallel.executor import ParallelExecutor

        # Create different circuits to simulate
        circuits = [
            (create_rc_lowpass(f"rc_{i}", r_value=f"{(i+1)*1}k"), [AnalysisSpec("op")])
            for i in range(8)
        ]

        print("\n" + "=" * 60)
        print("REAL NGSPICE: PARALLEL EXECUTION BENCHMARK")
        print("=" * 60)
        print(f"\nCircuits: {len(circuits)} RC filters with different R values")
        print("Analysis: Operating Point (OP)")

        # Sequential execution
        def run_sequential() -> list[Any]:
            results = []
            for circuit, analyses in circuits:
                result = run_simulation(circuit, analyses, engine="ngspice")
                results.append(result)
            return results

        result_sequential = run_benchmark("Sequential", run_sequential, iterations=3, warmup=1)

        # Parallel execution
        parallel_results = {}
        for n_workers in [2, 4]:

            def run_parallel(workers: int = n_workers) -> list[Any]:
                executor = ParallelExecutor(max_workers=workers, use_processes=True)

                def sim_task(item: tuple[Circuit, list[AnalysisSpec]]) -> Any:
                    circuit, analyses = item
                    return run_simulation(circuit, analyses, engine="ngspice")

                batch_result = executor.map(sim_task, circuits)
                return [r.value for r in batch_result.results if r.success]

            result = run_benchmark(
                f"Parallel ({n_workers} workers)",
                run_parallel,
                iterations=3,
                warmup=1,
            )
            parallel_results[n_workers] = result

        print(f"\n{result_sequential}")
        for n_workers, result in parallel_results.items():
            print(f"\n{result}")
            speedup = result_sequential.mean_ms / result.mean_ms if result.mean_ms > 0 else 0
            efficiency = speedup / n_workers * 100
            print(f"  \U0001f4ca Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")

    def test_incremental_simulation_real(self) -> None:
        """Compare incremental vs full re-simulation with real circuits."""
        from spicelab.incremental.simulator import IncrementalSimulator

        circuit = create_voltage_divider()
        analyses = [AnalysisSpec("op")]
        n_iterations = 5

        print("\n" + "=" * 60)
        print("REAL NGSPICE: INCREMENTAL SIMULATION BENCHMARK")
        print("=" * 60)
        print("\nCircuit: Voltage divider")
        print("Analysis: Operating Point (OP)")
        print(f"Iterations: {n_iterations}")

        # Full simulation - always fresh
        def run_full() -> list[Any]:
            results = []
            for i in range(n_iterations):
                # Create slightly different circuit each time
                c = create_voltage_divider(f"vdiv_{i}", r1=f"{10+i}k", r2="10k")
                result = run_simulation(c, analyses, engine="ngspice")
                results.append(result)
            return results

        result_full = run_benchmark("Full Simulation", run_full, iterations=3, warmup=1)

        # Incremental simulation
        with tempfile.TemporaryDirectory() as tmpdir:
            sim = IncrementalSimulator(cache_dir=Path(tmpdir))

            def runner(circ: Any, anlys: Any) -> Any:
                return run_simulation(circ, [anlys], engine="ngspice")

            def run_incremental() -> list[Any]:
                results = []
                for _ in range(n_iterations):
                    # Same circuit - should hit cache after first
                    result = sim.run(circuit, analyses[0], runner=runner)
                    results.append(result)
                return results

            result_incremental = run_benchmark(
                "Incremental", run_incremental, iterations=3, warmup=1
            )

            print(f"\n{result_full}")
            print(f"\n{result_incremental}")

            speedup = (
                result_full.mean_ms / result_incremental.mean_ms
                if result_incremental.mean_ms > 0
                else 0
            )
            print(f"\n\U0001f4ca Speedup: {speedup:.2f}x")
            print(
                f"\U0001f4ca Cache hits: {sim.stats.cache_hits}, misses: {sim.stats.cache_misses}"
            )


@pytest.mark.benchmark
class TestMonteCarloRealBenchmark:
    """Monte Carlo benchmark with real NGSpice simulations."""

    @pytest.fixture(autouse=True)
    def check_ngspice(self) -> None:
        """Skip tests if ngspice is not available."""
        try:
            from spicelab.spice.registry import get_active_adapter

            get_active_adapter()
        except Exception:
            pytest.skip("ngspice adapter not available")

    def test_monte_carlo_real(self) -> None:
        """Monte Carlo simulation with real NGSpice."""
        from spicelab.parallel.executor import ParallelExecutor

        n_samples = 10  # Fewer samples for real simulation (takes time)

        # Generate random parameter variations
        np.random.seed(42)
        r_variations = np.random.uniform(0.95, 1.05, n_samples)  # +/- 5% tolerance

        print("\n" + "=" * 60)
        print("REAL NGSPICE: MONTE CARLO BENCHMARK")
        print("=" * 60)
        print("\nCircuit: RC Low-pass filter with R tolerance")
        print(f"Samples: {n_samples}")
        print("R variation: +/- 5%")

        # Sequential Monte Carlo
        def run_mc_sequential() -> list[Any]:
            results = []
            for r_mult in r_variations:
                r_val = f"{1000 * r_mult:.0f}"  # Base 1k ohm
                circuit = create_rc_lowpass("mc_rc", r_value=r_val)
                result = run_simulation(circuit, [AnalysisSpec("op")], engine="ngspice")
                results.append(result)
            return results

        result_seq = run_benchmark("Sequential MC", run_mc_sequential, iterations=2, warmup=1)

        # Parallel Monte Carlo
        def run_mc_parallel() -> list[Any]:
            executor = ParallelExecutor(max_workers=4, use_processes=True)

            def mc_task(r_mult: float) -> Any:
                r_val = f"{1000 * r_mult:.0f}"
                circuit = create_rc_lowpass("mc_rc", r_value=r_val)
                return run_simulation(circuit, [AnalysisSpec("op")], engine="ngspice")

            batch_result = executor.map(mc_task, list(r_variations))
            return [r.value for r in batch_result.results if r.success]

        result_par = run_benchmark("Parallel MC (4w)", run_mc_parallel, iterations=2, warmup=1)

        # With cache (for repeated values)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompressedCache(cache_dir=Path(tmpdir), max_size_mb=100)

            def run_mc_cached() -> list[Any]:
                results = []
                for r_mult in r_variations:
                    # Round to reduce unique values (simulate binning)
                    r_rounded = round(r_mult, 2)
                    key = f"mc_r_{r_rounded}"

                    cached = cache.get(key)
                    if cached is not None:
                        results.append(cached)
                    else:
                        r_val = f"{1000 * r_mult:.0f}"
                        circuit = create_rc_lowpass("mc_rc", r_value=r_val)
                        result = run_simulation(circuit, [AnalysisSpec("op")], engine="ngspice")
                        cache.set(key, result)
                        results.append(result)
                return results

            # Populate cache
            run_mc_cached()
            cache.stats.hits = 0
            cache.stats.misses = 0

            result_cached = run_benchmark("Cached MC", run_mc_cached, iterations=2, warmup=0)

            print(f"\n{result_seq}")
            print(f"\n{result_par}")
            print(f"\n{result_cached}")

            print("\n" + "-" * 60)
            print("SUMMARY")
            print("-" * 60)
            baseline = result_seq.mean_ms
            print(f"\nBaseline (sequential): {baseline:.2f} ms")
            par_speedup = baseline / result_par.mean_ms
            print(f"Parallel (4 workers):  {result_par.mean_ms:.2f} ms ({par_speedup:.2f}x)")
            cache_speedup = baseline / result_cached.mean_ms
            print(f"Cached:                {result_cached.mean_ms:.2f} ms ({cache_speedup:.2f}x)")
            print(f"\nCache hit rate: {cache.stats.hit_rate:.1%}")


@pytest.mark.benchmark
class TestTransientAnalysisBenchmark:
    """Benchmark transient analysis with different step sizes."""

    @pytest.fixture(autouse=True)
    def check_ngspice(self) -> None:
        """Skip tests if ngspice is not available."""
        try:
            from spicelab.spice.registry import get_active_adapter

            get_active_adapter()
        except Exception:
            pytest.skip("ngspice adapter not available")

    def test_transient_step_comparison(self) -> None:
        """Compare transient analysis with different step sizes."""
        circuit = create_rlc_series()

        step_configs = [
            ("10u", "1m", "100 points"),
            ("1u", "1m", "1000 points"),
            ("100n", "100u", "1000 points"),
        ]

        print("\n" + "=" * 60)
        print("REAL NGSPICE: TRANSIENT ANALYSIS STEP SIZE BENCHMARK")
        print("=" * 60)
        print("\nCircuit: RLC Series (R=100, L=10mH, C=1uF)")

        results = {}
        for step, stop, desc in step_configs:
            analyses = [AnalysisSpec("tran", {"tstep": step, "tstop": stop})]

            def run_tran(a: list[AnalysisSpec] = analyses) -> Any:
                return run_simulation(circuit, a, engine="ngspice")

            result = run_benchmark(f"Tran {step}/{stop} ({desc})", run_tran, iterations=3, warmup=1)
            results[(step, stop)] = result
            print(f"\n{result}")

        print("\n" + "-" * 60)
        print("Performance scaling with simulation points")
        print("-" * 60)
        base_time = list(results.values())[0].mean_ms
        for (step, stop), result in results.items():
            ratio = result.mean_ms / base_time
            print(f"  {step}/{stop}: {result.mean_ms:.2f} ms ({ratio:.2f}x vs baseline)")
