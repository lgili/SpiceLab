#!/usr/bin/env python3
"""Profile RAW file parsing performance.

This script profiles the performance of RAW file parsing for various file sizes
to identify bottlenecks and establish performance baselines.

Usage:
    python tools/profile_raw.py [--sizes 10,100,1000] [--profile]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import struct
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spicelab.io.raw_reader import parse_ngspice_raw, parse_ngspice_ascii_raw


def create_test_raw_ascii(
    path: str,
    n_signals: int,
    n_points: int,
    *,
    complex_data: bool = False,
) -> None:
    """Create a test ASCII RAW file with specified dimensions.

    Args:
        path: Output file path
        n_signals: Number of signals (including time/freq)
        n_points: Number of data points per signal
        complex_data: If True, generate complex (AC) data
    """
    with open(path, "w") as f:
        # Write header
        f.write("Title: Test RAW file for profiling\n")
        f.write("Date: Thu Nov 28 00:00:00 2024\n")
        if complex_data:
            f.write("Plotname: AC Analysis\n")
            f.write("Flags: complex\n")
        else:
            f.write("Plotname: Transient Analysis\n")
            f.write("Flags: real\n")
        f.write(f"No. Variables: {n_signals}\n")
        f.write(f"No. Points: {n_points}\n")
        f.write("Variables:\n")

        # Write variable definitions
        if complex_data:
            f.write("\t0\tfrequency\tfrequency\n")
        else:
            f.write("\t0\ttime\ttime\n")

        for i in range(1, n_signals):
            f.write(f"\t{i}\tv(n{i})\tvoltage\n")

        # Write values
        f.write("Values:\n")
        for pt in range(n_points):
            # Time/frequency value
            if complex_data:
                freq = 1.0 + pt * 10.0
                f.write(f"{pt}\t{freq}\n")
            else:
                t = pt * 1e-6
                f.write(f"{pt}\t{t}\n")

            # Signal values
            for sig in range(1, n_signals):
                if complex_data:
                    re = 1.0 / (1.0 + sig * 0.1)
                    im = -0.1 * sig
                    f.write(f"\t{re},{im}\n")
                else:
                    val = 1.0 * sig + 0.001 * pt
                    f.write(f"\t{val}\n")


def create_test_raw_binary(
    path: str,
    n_signals: int,
    n_points: int,
    *,
    complex_data: bool = False,
) -> None:
    """Create a test binary RAW file with specified dimensions.

    Args:
        path: Output file path
        n_signals: Number of signals (including time/freq)
        n_points: Number of data points per signal
        complex_data: If True, generate complex (AC) data
    """
    with open(path, "wb") as f:
        # Write header (ASCII text)
        header_lines = [
            "Title: Test Binary RAW file for profiling",
            "Date: Thu Nov 28 00:00:00 2024",
        ]

        if complex_data:
            header_lines.append("Plotname: AC Analysis")
            header_lines.append("Flags: complex")
        else:
            header_lines.append("Plotname: Transient Analysis")
            header_lines.append("Flags: real")

        header_lines.append(f"No. Variables: {n_signals}")
        header_lines.append(f"No. Points: {n_points}")
        header_lines.append("Variables:")

        if complex_data:
            header_lines.append("\t0\tfrequency\tfrequency")
        else:
            header_lines.append("\t0\ttime\ttime")

        for i in range(1, n_signals):
            header_lines.append(f"\t{i}\tv(n{i})\tvoltage")

        header_lines.append("Binary:")

        header_text = "\n".join(header_lines) + "\n"
        f.write(header_text.encode("utf-8"))

        # Write binary data (float64 little-endian)
        # Layout: point-major (for each point, write all signals)
        for pt in range(n_points):
            # Time/frequency value
            if complex_data:
                freq = 1.0 + pt * 10.0
                f.write(struct.pack("<d", freq))
            else:
                t = pt * 1e-6
                f.write(struct.pack("<d", t))

            # Signal values
            for sig in range(1, n_signals):
                if complex_data:
                    re = 1.0 / (1.0 + sig * 0.1)
                    im = -0.1 * sig
                    f.write(struct.pack("<d", re))
                    f.write(struct.pack("<d", im))
                else:
                    val = 1.0 * sig + 0.001 * pt
                    f.write(struct.pack("<d", val))


def measure_raw_parsing(
    n_signals: int,
    n_points: int,
    file_type: str = "ascii",
    complex_data: bool = False,
) -> dict:
    """Measure time for RAW file parsing.

    Returns dict with timing data.
    """
    results: dict[str, float | int | str] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "test.raw")

        # Create test file
        start = time.perf_counter()
        if file_type == "binary":
            create_test_raw_binary(raw_path, n_signals, n_points, complex_data=complex_data)
        else:
            create_test_raw_ascii(raw_path, n_signals, n_points, complex_data=complex_data)
        results["create_time"] = time.perf_counter() - start

        # Get file size
        results["file_size"] = os.path.getsize(raw_path)

        # Parse file
        start = time.perf_counter()
        if file_type == "binary":
            trace_set = parse_ngspice_raw(raw_path)
        else:
            trace_set = parse_ngspice_ascii_raw(raw_path)
        results["parse_time"] = time.perf_counter() - start

        # Measure trace access
        start = time.perf_counter()
        for name in trace_set.names[:min(10, len(trace_set.names))]:
            _ = trace_set[name].values
        results["access_time"] = time.perf_counter() - start

        # Metadata
        results["n_signals"] = n_signals
        results["n_points"] = n_points
        results["file_type"] = file_type
        results["complex"] = complex_data
        results["n_traces"] = len(trace_set.names)

    return results


def profile_raw_parsing(
    n_signals: int,
    n_points: int,
    file_type: str = "ascii",
) -> pstats.Stats:
    """Profile RAW parsing with cProfile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "test.raw")

        # Create test file
        if file_type == "binary":
            create_test_raw_binary(raw_path, n_signals, n_points)
        else:
            create_test_raw_ascii(raw_path, n_signals, n_points)

        profiler = cProfile.Profile()
        profiler.enable()

        # Parse the file
        if file_type == "binary":
            _ = parse_ngspice_raw(raw_path)
        else:
            _ = parse_ngspice_ascii_raw(raw_path)

        profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")

    return stats


def print_results(results: list[dict]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("RAW PARSING PERFORMANCE RESULTS")
    print("=" * 100)

    # Header
    print(
        f"{'Signals':>10} | "
        f"{'Points':>10} | "
        f"{'Type':>8} | "
        f"{'Complex':>8} | "
        f"{'Size (KB)':>12} | "
        f"{'Parse (ms)':>12} | "
        f"{'MB/s':>10}"
    )
    print("-" * 100)

    for r in results:
        size_kb = r["file_size"] / 1024
        parse_ms = r["parse_time"] * 1000
        mb_per_s = (r["file_size"] / 1024 / 1024) / r["parse_time"] if r["parse_time"] > 0 else 0

        print(
            f"{r['n_signals']:>10} | "
            f"{r['n_points']:>10} | "
            f"{r['file_type']:>8} | "
            f"{str(r['complex']):>8} | "
            f"{size_kb:>12.1f} | "
            f"{parse_ms:>12.2f} | "
            f"{mb_per_s:>10.1f}"
        )

    print("=" * 100)

    # Scaling analysis
    if len(results) >= 2:
        print("\nSCALING ANALYSIS (Parse Time):")
        for i in range(1, len(results)):
            prev = results[i - 1]
            curr = results[i]

            if prev["file_type"] != curr["file_type"]:
                continue

            total_prev = prev["n_signals"] * prev["n_points"]
            total_curr = curr["n_signals"] * curr["n_points"]
            n_ratio = total_curr / total_prev

            if prev["parse_time"] > 0.0001:
                time_ratio = curr["parse_time"] / prev["parse_time"]
                import math

                if time_ratio > 0 and n_ratio > 0:
                    k = math.log(time_ratio) / math.log(n_ratio)
                    complexity = f"O(n^{k:.2f})"
                else:
                    complexity = "N/A"

                print(
                    f"  {total_prev:,} -> {total_curr:,} values = "
                    f"{time_ratio:.2f}x ({complexity})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile RAW file parsing")
    parser.add_argument(
        "--signals",
        type=str,
        default="10,50,100,500",
        help="Comma-separated list of signal counts to test",
    )
    parser.add_argument(
        "--points",
        type=str,
        default="100,1000,10000",
        help="Comma-separated list of point counts to test",
    )
    parser.add_argument(
        "--type",
        choices=["ascii", "binary", "both"],
        default="both",
        help="File type to test",
    )
    parser.add_argument(
        "--complex",
        action="store_true",
        help="Test complex (AC) data",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on largest size and show top functions",
    )
    args = parser.parse_args()

    signal_counts = [int(s.strip()) for s in args.signals.split(",")]
    point_counts = [int(s.strip()) for s in args.points.split(",")]

    file_types = ["ascii", "binary"] if args.type == "both" else [args.type]

    print(f"Signal counts: {signal_counts}")
    print(f"Point counts: {point_counts}")
    print(f"File types: {file_types}")
    print(f"Complex data: {args.complex}")

    # Run measurements
    results = []
    for file_type in file_types:
        for n_signals in signal_counts:
            for n_points in point_counts:
                print(
                    f"\nMeasuring {n_signals} signals x {n_points} points ({file_type})...",
                    end=" ",
                    flush=True,
                )
                try:
                    r = measure_raw_parsing(
                        n_signals, n_points, file_type, complex_data=args.complex
                    )
                    results.append(r)
                    print(f"done ({r['parse_time'] * 1000:.2f} ms)")
                except Exception as e:
                    print(f"FAILED: {e}")

    if results:
        print_results(results)

    # Detailed profiling if requested
    if args.profile and results:
        # Find largest successful test
        largest = max(results, key=lambda r: r["n_signals"] * r["n_points"])
        print(
            f"\n\nDETAILED PROFILING ({largest['n_signals']} signals x {largest['n_points']} points)"
        )
        print("=" * 80)

        stats = profile_raw_parsing(
            largest["n_signals"],
            largest["n_points"],
            largest["file_type"],
        )
        stats.print_stats(30)


if __name__ == "__main__":
    main()
