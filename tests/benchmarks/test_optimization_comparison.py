"""Benchmark comparing performance with and without optimizations.

This module provides comprehensive benchmarks to measure the impact of:
- Compressed caching (CompressedCache)
- Parallel execution (ParallelExecutor)
- Incremental simulation (IncrementalSimulator)
- Combined optimizations

Run with: python -m pytest tests/benchmarks/test_optimization_comparison.py -v -s
"""

from __future__ import annotations

import gc
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from spicelab.cache.compressed import ZSTD_AVAILABLE


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    times_ms: list[float] = field(default_factory=list)
    memory_mb: float = 0.0
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        """Mean execution time in milliseconds."""
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def std_ms(self) -> float:
        """Standard deviation of execution time."""
        return statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        """Minimum execution time."""
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        """Maximum execution time."""
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def total_ms(self) -> float:
        """Total execution time."""
        return sum(self.times_ms)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_ms:.2f} ms (±{self.std_ms:.2f})\n"
            f"  Min/Max: {self.min_ms:.2f} / {self.max_ms:.2f} ms\n"
            f"  Total: {self.total_ms:.2f} ms"
        )


def run_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Run a benchmark function multiple times.

    Args:
        name: Benchmark name
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations (not counted)

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func()
        gc.collect()

    # Actual benchmark
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return BenchmarkResult(name=name, iterations=iterations, times_ms=times)


class MockSimulationResult:
    """Mock simulation result for benchmarking."""

    def __init__(self, size: int = 10000) -> None:
        self.time = np.linspace(0, 1e-3, size)
        self.voltage = np.sin(2 * np.pi * 1000 * self.time) + np.random.randn(size) * 0.01
        self.current = np.cos(2 * np.pi * 1000 * self.time) + np.random.randn(size) * 0.001
        self.metadata = {"analysis": "tran", "circuit": "test", "params": {"temp": 25}}


def mock_simulation(params: dict[str, Any], delay_ms: float = 10.0) -> MockSimulationResult:
    """Mock simulation function that takes some time.

    Args:
        params: Simulation parameters
        delay_ms: Simulated processing time in milliseconds

    Returns:
        Mock simulation result
    """
    # Simulate computation time
    time.sleep(delay_ms / 1000)

    # Create result with some parameter-dependent variation
    result = MockSimulationResult()
    if "temp" in params:
        result.voltage = result.voltage * (1 + params["temp"] / 1000)
    return result


class TestCacheOptimization:
    """Benchmark tests for cache optimization."""

    def test_cache_vs_no_cache(self) -> None:
        """Compare performance with and without caching."""
        from spicelab.cache.compressed import CompressedCache

        if not ZSTD_AVAILABLE:
            print("\n⚠️  zstandard not installed, using fallback (less compression)")

        n_simulations = 20
        n_unique = 5  # Only 5 unique simulations, rest are repeats

        # Generate parameter sets (with repeats)
        param_sets = [{"temp": i % n_unique * 10} for i in range(n_simulations)]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompressedCache(cache_dir=Path(tmpdir), max_size_mb=100)

            # Benchmark WITHOUT cache (all simulations run)
            def run_without_cache() -> list[MockSimulationResult]:
                results = []
                for params in param_sets:
                    results.append(mock_simulation(params, delay_ms=5))
                return results

            result_no_cache = run_benchmark(
                "Without Cache", run_without_cache, iterations=3, warmup=1
            )

            # Benchmark WITH cache
            def run_with_cache() -> list[MockSimulationResult]:
                results = []
                for params in param_sets:
                    key = f"sim_{params['temp']}"
                    cached = cache.get(key)
                    if cached is not None:
                        results.append(cached)
                    else:
                        result = mock_simulation(params, delay_ms=5)
                        cache.set(key, result)
                        results.append(result)
                return results

            # First run populates cache
            run_with_cache()
            cache.stats.hits = 0
            cache.stats.misses = 0

            result_with_cache = run_benchmark("With Cache", run_with_cache, iterations=3, warmup=0)

            # Print results
            print("\n" + "=" * 60)
            print("CACHE OPTIMIZATION BENCHMARK")
            print("=" * 60)
            print(f"\nSimulations: {n_simulations} total, {n_unique} unique")
            print(f"\n{result_no_cache}")
            print(f"\n{result_with_cache}")

            speedup = (
                result_no_cache.mean_ms / result_with_cache.mean_ms
                if result_with_cache.mean_ms > 0
                else 0
            )
            print(f"\n📊 Speedup: {speedup:.2f}x")
            print(f"📊 Cache hit rate: {cache.stats.hit_rate:.1%}")

            if ZSTD_AVAILABLE and cache.stats.avg_compression_ratio > 0:
                print(f"📊 Compression ratio: {cache.stats.avg_compression_ratio:.1%}")

            # Verify cache provides benefit
            assert result_with_cache.mean_ms < result_no_cache.mean_ms, (
                f"Cache should be faster! No cache: {result_no_cache.mean_ms:.2f}ms, "
                f"With cache: {result_with_cache.mean_ms:.2f}ms"
            )


class TestParallelOptimization:
    """Benchmark tests for parallel execution optimization."""

    def test_parallel_vs_sequential(self) -> None:
        """Compare parallel vs sequential execution."""
        from spicelab.parallel.executor import ParallelExecutor

        n_simulations = 16
        delay_ms = 20  # Each simulation takes 20ms

        param_sets = [{"temp": -40 + i * 10} for i in range(n_simulations)]

        # Sequential execution
        def run_sequential() -> list[MockSimulationResult]:
            return [mock_simulation(p, delay_ms=delay_ms) for p in param_sets]

        result_sequential = run_benchmark("Sequential", run_sequential, iterations=3, warmup=1)

        # Parallel execution with different worker counts
        parallel_results = {}
        for n_workers in [2, 4, 8]:

            def run_parallel(workers: int = n_workers) -> list[Any]:
                executor = ParallelExecutor(max_workers=workers, use_processes=False)
                batch_result = executor.map(
                    lambda p: mock_simulation(p, delay_ms=delay_ms),
                    param_sets,
                )
                return [r.value for r in batch_result.results if r.success]

            result = run_benchmark(
                f"Parallel ({n_workers} workers)",
                run_parallel,
                iterations=3,
                warmup=1,
            )
            parallel_results[n_workers] = result

        # Print results
        print("\n" + "=" * 60)
        print("PARALLEL EXECUTION BENCHMARK")
        print("=" * 60)
        print(f"\nSimulations: {n_simulations}, delay per sim: {delay_ms}ms")
        print(f"\n{result_sequential}")

        for n_workers, result in parallel_results.items():
            print(f"\n{result}")
            speedup = result_sequential.mean_ms / result.mean_ms if result.mean_ms > 0 else 0
            efficiency = speedup / n_workers * 100
            print(f"  📊 Speedup: {speedup:.2f}x (efficiency: {efficiency:.1f}%)")

        # Verify parallel provides benefit
        best_parallel = min(parallel_results.values(), key=lambda r: r.mean_ms)
        assert (
            best_parallel.mean_ms < result_sequential.mean_ms * 0.8
        ), "Parallel execution should be significantly faster!"


class TestIncrementalOptimization:
    """Benchmark tests for incremental simulation."""

    def test_incremental_vs_full(self) -> None:
        """Compare incremental vs full re-simulation."""
        from spicelab.incremental.simulator import IncrementalSimulator

        n_iterations = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            sim = IncrementalSimulator(cache_dir=Path(tmpdir))

            # Mock circuit and analysis as objects with hashable content
            @dataclass
            class MockCircuit:
                netlist: str

                def __hash__(self) -> int:
                    return hash(self.netlist)

            @dataclass
            class MockAnalysis:
                cmd: str

                def __hash__(self) -> int:
                    return hash(self.cmd)

            circuit = MockCircuit("V1 in 0 DC 5\nR1 in out 1k\nC1 out 0 1u")
            analysis = MockAnalysis(".tran 1u 1m")

            # Runner function that simulates work
            def mock_runner(circ: Any, anlys: Any) -> MockSimulationResult:
                return mock_simulation({"circuit": str(circ)}, delay_ms=10)

            # Full simulation every time (simulate no caching - always different circuit)
            def run_full() -> list[MockSimulationResult]:
                results = []
                for i in range(n_iterations):
                    # Each iteration has a unique circuit - no cache hits
                    c = MockCircuit(f"V1 in 0 DC 5\nR1 in out 1k\nC1 out 0 1u\n* iter {i}")
                    results.append(mock_simulation({"circuit": c.netlist}, delay_ms=10))
                return results

            result_full = run_benchmark("Full Simulation", run_full, iterations=3, warmup=1)

            # Incremental - same circuit should be cached after first run
            def run_incremental() -> list[Any]:
                results = []
                for _ in range(n_iterations):
                    # Same circuit every time - should hit cache after first
                    result = sim.run(circuit, analysis, runner=mock_runner)
                    results.append(result)
                return results

            result_incremental = run_benchmark(
                "Incremental", run_incremental, iterations=3, warmup=1
            )

            # Print results
            print("\n" + "=" * 60)
            print("INCREMENTAL SIMULATION BENCHMARK")
            print("=" * 60)
            print(f"\nIterations per run: {n_iterations}")
            print(f"\n{result_full}")
            print(f"\n{result_incremental}")

            speedup = (
                result_full.mean_ms / result_incremental.mean_ms
                if result_incremental.mean_ms > 0
                else 0
            )
            print(f"\n📊 Speedup: {speedup:.2f}x")
            print(f"📊 Cache stats: hits={sim.stats.cache_hits}, misses={sim.stats.cache_misses}")


class TestMonteCarloOptimization:
    """Benchmark Monte Carlo simulation with optimizations."""

    def test_monte_carlo_comparison(self) -> None:
        """Compare Monte Carlo with and without optimizations."""
        from spicelab.cache.compressed import CompressedCache
        from spicelab.parallel.executor import ParallelExecutor

        n_samples = 50
        delay_ms = 5  # Simulation time per sample

        # Generate random parameter samples
        np.random.seed(42)
        samples = [
            {
                "temp": np.random.uniform(-40, 85),
                "vdd": np.random.uniform(4.5, 5.5),
                "r_tol": np.random.uniform(-0.05, 0.05),
            }
            for _ in range(n_samples)
        ]

        print("\n" + "=" * 60)
        print("MONTE CARLO OPTIMIZATION BENCHMARK")
        print("=" * 60)
        print(f"\nSamples: {n_samples}, delay per sample: {delay_ms}ms")

        # 1. Baseline: Sequential, no cache
        def run_baseline() -> list[MockSimulationResult]:
            return [mock_simulation(s, delay_ms=delay_ms) for s in samples]

        result_baseline = run_benchmark(
            "Baseline (sequential, no cache)", run_baseline, iterations=3, warmup=1
        )
        print(f"\n{result_baseline}")

        # 2. With cache only
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompressedCache(cache_dir=Path(tmpdir), max_size_mb=100)

            def run_with_cache() -> list[MockSimulationResult]:
                results = []
                for s in samples:
                    # Round parameters for cache key (simulate tolerance grouping)
                    key = f"mc_{int(s['temp']/10)*10}_{int(s['vdd']*10)}"
                    cached = cache.get(key)
                    if cached is not None:
                        results.append(cached)
                    else:
                        result = mock_simulation(s, delay_ms=delay_ms)
                        cache.set(key, result)
                        results.append(result)
                return results

            # Populate cache first
            run_with_cache()
            cache.stats.hits = 0
            cache.stats.misses = 0

            result_cache = run_benchmark("With Cache", run_with_cache, iterations=3, warmup=0)
            print(f"\n{result_cache}")
            print(f"  Cache hit rate: {cache.stats.hit_rate:.1%}")

        # 3. With parallel only
        def run_parallel() -> list[Any]:
            executor = ParallelExecutor(max_workers=4, use_processes=False)
            batch_result = executor.map(
                lambda s: mock_simulation(s, delay_ms=delay_ms),
                samples,
            )
            return [r.value for r in batch_result.results if r.success]

        result_parallel = run_benchmark(
            "With Parallel (4 workers)", run_parallel, iterations=3, warmup=1
        )
        print(f"\n{result_parallel}")

        # 4. Combined: Cache + Parallel
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = CompressedCache(cache_dir=Path(tmpdir), max_size_mb=100)

            def process_sample(sample: dict[str, Any]) -> MockSimulationResult:
                key = f"mc_{int(sample['temp']/10)*10}_{int(sample['vdd']*10)}"
                cached = cache.get(key)
                if cached is not None:
                    return cached
                result = mock_simulation(sample, delay_ms=delay_ms)
                cache.set(key, result)
                return result

            def run_combined() -> list[Any]:
                executor = ParallelExecutor(max_workers=4, use_processes=False)
                batch_result = executor.map(process_sample, samples)
                return [r.value for r in batch_result.results if r.success]

            # Populate cache first
            run_combined()

            result_combined = run_benchmark(
                "Combined (Cache + Parallel)", run_combined, iterations=3, warmup=0
            )
            print(f"\n{result_combined}")

        # Summary
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)

        baseline_ms = result_baseline.mean_ms
        print(f"\nBaseline: {baseline_ms:.2f} ms")
        cache_speedup = baseline_ms / result_cache.mean_ms
        print(f"Cache:    {result_cache.mean_ms:.2f} ms ({cache_speedup:.2f}x speedup)")
        par_speedup = baseline_ms / result_parallel.mean_ms
        print(f"Parallel: {result_parallel.mean_ms:.2f} ms ({par_speedup:.2f}x speedup)")
        comb_speedup = baseline_ms / result_combined.mean_ms
        print(f"Combined: {result_combined.mean_ms:.2f} ms ({comb_speedup:.2f}x speedup)")

        # Verify optimizations help
        assert result_parallel.mean_ms < baseline_ms * 0.5, "Parallel should be >2x faster"


class TestLargeScaleBenchmark:
    """Large-scale benchmark for comprehensive comparison."""

    def test_large_scale_monte_carlo(self) -> None:
        """Large-scale Monte Carlo benchmark (100+ samples)."""
        from spicelab.parallel.executor import ParallelExecutor

        n_samples = 100
        delay_ms = 2  # Fast simulation for many samples

        np.random.seed(123)
        samples = [
            {"temp": np.random.uniform(-40, 85), "process": np.random.choice(["ss", "tt", "ff"])}
            for _ in range(n_samples)
        ]

        print("\n" + "=" * 60)
        print("LARGE-SCALE MONTE CARLO BENCHMARK")
        print("=" * 60)
        print(f"\nSamples: {n_samples}, delay per sample: {delay_ms}ms")
        print(f"Expected baseline time: {n_samples * delay_ms}ms")

        # Sequential baseline
        def run_sequential() -> list[MockSimulationResult]:
            return [mock_simulation(s, delay_ms=delay_ms) for s in samples]

        result_seq = run_benchmark("Sequential", run_sequential, iterations=2, warmup=1)

        # Parallel with different worker counts
        results_parallel = {}
        for workers in [2, 4, 8]:

            def run_par(w: int = workers) -> list[Any]:
                executor = ParallelExecutor(max_workers=w, use_processes=False)
                batch_result = executor.map(
                    lambda s: mock_simulation(s, delay_ms=delay_ms), samples
                )
                return [r.value for r in batch_result.results if r.success]

            results_parallel[workers] = run_benchmark(
                f"Parallel ({workers}w)", run_par, iterations=2, warmup=1
            )

        # Print results table
        print("\n" + "-" * 60)
        print(f"{'Configuration':<25} {'Mean (ms)':<12} {'Speedup':<10}")
        print("-" * 60)
        print(f"{'Sequential':<25} {result_seq.mean_ms:<12.2f} {'1.00x':<10}")

        for workers, result in results_parallel.items():
            speedup = result_seq.mean_ms / result.mean_ms if result.mean_ms > 0 else 0
            label = f"Parallel ({workers} workers)"
            print(f"{label:<25} {result.mean_ms:<12.2f} {speedup:.2f}x")

        print("-" * 60)

        # Theoretical maximum speedup
        print(f"\nTheoretical max (8 workers): {n_samples * delay_ms / 8:.2f}ms")
        print(f"Actual best: {min(r.mean_ms for r in results_parallel.values()):.2f}ms")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "-s"])
