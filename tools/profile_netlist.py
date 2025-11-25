#!/usr/bin/env python3
"""Profile netlist generation performance.

This script profiles the performance of circuit building and netlist generation
for various circuit sizes to identify bottlenecks.

Usage:
    python tools/profile_netlist.py [--sizes 10,100,1000] [--profile]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor
from spicelab.core.net import GND, Net


def build_rc_ladder(n_stages: int, name: str = "rc_ladder") -> Circuit:
    """Build an RC ladder network with n stages.

    This is a common circuit topology that stresses connectivity tracking.
    """
    circuit = Circuit(name)

    prev_net = GND
    for i in range(n_stages):
        # Series resistor
        r = Resistor(f"R{i}", "1k")
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)

        # Node between R and C
        mid_net = Net(f"n{i}")
        circuit.connect(r.ports[1], mid_net)

        # Shunt capacitor
        c = Capacitor(f"C{i}", "100n")
        circuit.add(c)
        circuit.connect(c.ports[0], mid_net)
        circuit.connect(c.ports[1], GND)

        prev_net = mid_net

    return circuit


def build_resistor_chain(n_resistors: int, name: str = "resistor_chain") -> Circuit:
    """Build a simple resistor chain: R-R-R-...-R."""
    circuit = Circuit(name)

    prev_net = GND
    for i in range(n_resistors):
        r = Resistor(f"R{i}", "1k")
        circuit.add(r)
        circuit.connect(r.ports[0], prev_net)
        next_net = Net(f"n{i}")
        circuit.connect(r.ports[1], next_net)
        prev_net = next_net

    return circuit


def measure_circuit_operations(n_components: int, circuit_type: str = "chain") -> dict:
    """Measure time for various circuit operations.

    Returns dict with timing data for each operation.
    """
    results: dict[str, float] = {}

    # 1. Measure circuit building time
    start = time.perf_counter()
    if circuit_type == "ladder":
        circuit = build_rc_ladder(n_components // 2, f"measure_{n_components}")
    else:
        circuit = build_resistor_chain(n_components, f"measure_{n_components}")
    results["build_time"] = time.perf_counter() - start

    # 2. Measure netlist generation time
    start = time.perf_counter()
    netlist = circuit.build_netlist()
    results["netlist_time"] = time.perf_counter() - start

    # 3. Measure hash generation time
    start = time.perf_counter()
    circuit_hash = circuit.hash()
    results["hash_time"] = time.perf_counter() - start

    # 4. Measure summary generation time
    start = time.perf_counter()
    summary = circuit.summary()
    results["summary_time"] = time.perf_counter() - start

    # Metadata
    results["n_components"] = len(circuit._components)
    results["netlist_size"] = len(netlist)

    return results


def profile_netlist_generation(n_components: int) -> pstats.Stats:
    """Profile netlist generation with cProfile."""
    profiler = cProfile.Profile()
    profiler.enable()

    # Build and generate netlist
    circuit = build_resistor_chain(n_components, f"profile_{n_components}")
    _ = circuit.build_netlist()

    profiler.disable()

    # Create stats object
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    return stats


def print_results(results: list[dict]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Components':>12} | "
        f"{'Build (ms)':>12} | "
        f"{'Netlist (ms)':>12} | "
        f"{'Hash (ms)':>12} | "
        f"{'Summary (ms)':>12} | "
        f"{'Total (ms)':>12}"
    )
    print("-" * 80)

    for r in results:
        build_ms = r["build_time"] * 1000
        netlist_ms = r["netlist_time"] * 1000
        hash_ms = r["hash_time"] * 1000
        summary_ms = r["summary_time"] * 1000
        total_ms = build_ms + netlist_ms + hash_ms + summary_ms

        print(
            f"{r['n_components']:>12} | "
            f"{build_ms:>12.2f} | "
            f"{netlist_ms:>12.2f} | "
            f"{hash_ms:>12.2f} | "
            f"{summary_ms:>12.2f} | "
            f"{total_ms:>12.2f}"
        )

    print("=" * 80)

    # Scaling analysis
    if len(results) >= 2:
        print("\nSCALING ANALYSIS:")
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]
            n_ratio = curr["n_components"] / prev["n_components"]

            for metric in ["build_time", "netlist_time", "hash_time"]:
                if prev[metric] > 0.0001:  # Avoid division by very small numbers
                    time_ratio = curr[metric] / prev[metric]
                    # Estimate complexity: O(n^k) where k = log(time_ratio) / log(n_ratio)
                    import math

                    if time_ratio > 0 and n_ratio > 0:
                        k = math.log(time_ratio) / math.log(n_ratio)
                        complexity = f"O(n^{k:.2f})"
                    else:
                        complexity = "N/A"
                    print(
                        f"  {metric}: {prev['n_components']} -> {curr['n_components']} "
                        f"= {time_ratio:.2f}x ({complexity})"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile netlist generation")
    parser.add_argument(
        "--sizes",
        type=str,
        default="10,100,500,1000,2000",
        help="Comma-separated list of component counts to test",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on largest size and show top functions",
    )
    parser.add_argument(
        "--circuit",
        choices=["chain", "ladder"],
        default="chain",
        help="Circuit topology to test",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print(f"Testing circuit topology: {args.circuit}")
    print(f"Component counts: {sizes}")

    # Run measurements
    results = []
    for n in sizes:
        print(f"\nMeasuring {n} components...", end=" ", flush=True)
        r = measure_circuit_operations(n, args.circuit)
        results.append(r)
        print(f"done ({r['netlist_time'] * 1000:.2f} ms)")

    print_results(results)

    # Detailed profiling if requested
    if args.profile:
        largest = max(sizes)
        print(f"\n\nDETAILED PROFILING ({largest} components)")
        print("=" * 80)

        stats = profile_netlist_generation(largest)
        stats.print_stats(30)  # Top 30 functions


if __name__ == "__main__":
    main()
