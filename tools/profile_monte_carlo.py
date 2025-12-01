#!/usr/bin/env python3
"""Profile Monte Carlo simulation memory usage.

This script profiles memory consumption during Monte Carlo-like circuit operations
to identify memory leaks and establish memory baselines.

Usage:
    python tools/profile_monte_carlo.py [--runs 10,50,100] [--circuits 10,50,100]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND, Net


def build_test_circuit(name: str, variation: float = 0.0) -> Circuit:
    """Build a simple RC circuit for Monte Carlo testing.

    Args:
        name: Circuit name
        variation: Percentage variation to apply to resistance (simulates MC)
    """
    circuit = Circuit(name)

    vin = Vdc("Vin", 1.0)
    r1 = Resistor("R1", 1000.0 * (1 + variation))
    c1 = Capacitor("C1", 1e-6)

    circuit.add(vin, r1, c1)

    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)
    circuit.connect(r1.ports[0], Net("in"))
    circuit.connect(r1.ports[1], Net("out"))
    circuit.connect(c1.ports[0], Net("out"))
    circuit.connect(c1.ports[1], GND)

    return circuit


def build_large_circuit(name: str, n_components: int) -> Circuit:
    """Build a larger circuit for memory testing."""
    circuit = Circuit(name)

    vin = Vdc("Vin", 1.0)
    circuit.add(vin)
    circuit.connect(vin.ports[0], Net("in"))
    circuit.connect(vin.ports[1], GND)

    prev_net = Net("in")
    for i in range(n_components):
        r = Resistor(f"R{i}", 1000.0)
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    return circuit


def measure_monte_carlo_memory(n_runs: int, circuit_size: int = 3) -> dict:
    """Measure memory usage during Monte Carlo-like operations.

    Args:
        n_runs: Number of Monte Carlo iterations
        circuit_size: Number of components per circuit

    Returns:
        dict with memory metrics
    """
    results: dict[str, float | int] = {}

    gc.collect()
    tracemalloc.start()
    initial_snapshot = tracemalloc.take_snapshot()

    start_time = time.perf_counter()

    # Simulate Monte Carlo: create circuits with varied parameters
    circuits = []
    netlists = []
    hashes = []

    for run in range(n_runs):
        variation = run * 0.01  # 0-100% variation
        if circuit_size <= 10:
            circuit = build_test_circuit(f"mc_run_{run}", variation)
        else:
            circuit = build_large_circuit(f"mc_run_{run}", circuit_size)

        netlist = circuit.build_netlist()
        h = circuit.hash()

        circuits.append(circuit)
        netlists.append(netlist)
        hashes.append(h)

    results["build_time"] = time.perf_counter() - start_time

    current, peak = tracemalloc.get_traced_memory()
    results["peak_memory"] = peak
    results["current_memory"] = current

    # Memory per run
    results["memory_per_run"] = peak / n_runs if n_runs > 0 else 0

    # Check for memory growth pattern
    mid_snapshot = tracemalloc.take_snapshot()
    mid_stats = mid_snapshot.compare_to(initial_snapshot, "lineno")
    results["top_allocations"] = sum(stat.size_diff for stat in mid_stats[:20])

    # Cleanup and measure leak
    del circuits, netlists, hashes
    gc.collect()

    final_snapshot = tracemalloc.take_snapshot()
    final_stats = final_snapshot.compare_to(initial_snapshot, "lineno")
    results["retained_after_gc"] = sum(stat.size_diff for stat in final_stats[:20])

    tracemalloc.stop()

    results["n_runs"] = n_runs
    results["circuit_size"] = circuit_size

    return results


def measure_memory_over_batches(n_batches: int, runs_per_batch: int) -> list[dict]:
    """Measure memory growth over multiple batches to detect leaks.

    Returns list of memory measurements per batch.
    """
    results = []

    gc.collect()
    tracemalloc.start()

    for batch in range(n_batches):
        batch_start = time.perf_counter()

        # Create and discard circuits
        for run in range(runs_per_batch):
            circuit = build_test_circuit(f"batch_{batch}_run_{run}", run * 0.01)
            _ = circuit.build_netlist()
            _ = circuit.hash()
            del circuit

        gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        batch_time = time.perf_counter() - batch_start

        results.append(
            {
                "batch": batch,
                "current_memory": current,
                "peak_memory": peak,
                "batch_time": batch_time,
            }
        )

    tracemalloc.stop()
    return results


def print_results(results: list[dict]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("MONTE CARLO MEMORY PROFILING RESULTS")
    print("=" * 100)

    # Header
    print(
        f"{'Runs':>8} | "
        f"{'Circuit':>8} | "
        f"{'Peak (MB)':>12} | "
        f"{'Per Run (KB)':>12} | "
        f"{'Retained (KB)':>12} | "
        f"{'Time (s)':>10}"
    )
    print("-" * 100)

    for r in results:
        peak_mb = r["peak_memory"] / (1024 * 1024)
        per_run_kb = r["memory_per_run"] / 1024
        retained_kb = r["retained_after_gc"] / 1024
        time_s = r["build_time"]

        print(
            f"{r['n_runs']:>8} | "
            f"{r['circuit_size']:>8} | "
            f"{peak_mb:>12.2f} | "
            f"{per_run_kb:>12.2f} | "
            f"{retained_kb:>12.2f} | "
            f"{time_s:>10.2f}"
        )

    print("=" * 100)


def print_batch_results(results: list[dict]) -> None:
    """Print batch memory results."""
    print("\n" + "=" * 80)
    print("MEMORY GROWTH OVER BATCHES (leak detection)")
    print("=" * 80)

    print(f"{'Batch':>8} | {'Current (MB)':>12} | {'Peak (MB)':>12} | {'Time (s)':>10}")
    print("-" * 80)

    for r in results:
        current_mb = r["current_memory"] / (1024 * 1024)
        peak_mb = r["peak_memory"] / (1024 * 1024)
        time_s = r["batch_time"]

        print(
            f"{r['batch']:>8} | " f"{current_mb:>12.2f} | " f"{peak_mb:>12.2f} | " f"{time_s:>10.3f}"
        )

    print("=" * 80)

    # Analyze growth
    if len(results) >= 2:
        first = results[0]["current_memory"]
        last = results[-1]["current_memory"]
        growth = (last - first) / (1024 * 1024)

        # Use absolute growth threshold (1MB) for small values
        print(f"\nMemory growth: {growth:.2f} MB")
        if abs(growth) < 1.0:
            print("No significant memory leak detected")
        elif growth < 5.0:
            print("Moderate memory growth - investigate")
        else:
            print("Significant memory leak detected!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Monte Carlo memory usage")
    parser.add_argument(
        "--runs",
        type=str,
        default="10,50,100,500",
        help="Comma-separated list of run counts to test",
    )
    parser.add_argument(
        "--circuits",
        type=str,
        default="3,10,50",
        help="Comma-separated list of circuit sizes (components) to test",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=10,
        help="Number of batches for leak detection test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Runs per batch for leak detection",
    )
    args = parser.parse_args()

    run_counts = [int(s.strip()) for s in args.runs.split(",")]
    circuit_sizes = [int(s.strip()) for s in args.circuits.split(",")]

    print(f"Run counts: {run_counts}")
    print(f"Circuit sizes: {circuit_sizes}")

    # Run measurements
    results = []
    for circuit_size in circuit_sizes:
        for n_runs in run_counts:
            print(
                f"\nMeasuring {n_runs} runs x {circuit_size} components...",
                end=" ",
                flush=True,
            )
            try:
                r = measure_monte_carlo_memory(n_runs, circuit_size)
                results.append(r)
                print(f"done ({r['peak_memory'] / (1024 * 1024):.2f} MB peak)")
            except Exception as e:
                print(f"FAILED: {e}")

    if results:
        print_results(results)

    # Leak detection test
    print(f"\n\nRunning leak detection ({args.batches} batches x {args.batch_size} runs)...")
    batch_results = measure_memory_over_batches(args.batches, args.batch_size)
    print_batch_results(batch_results)


if __name__ == "__main__":
    main()
