"""Performance benchmarks for RAW file parsing.

These benchmarks measure the performance of RAW file parsing operations
to track performance over time and catch regressions.

Run with: pytest tests/benchmarks/test_raw_benchmarks.py --benchmark-only
"""

from __future__ import annotations

import os
import struct
import tempfile
from typing import Generator

import pytest


def _create_ascii_raw(path: str, n_signals: int, n_points: int) -> None:
    """Create a test ASCII RAW file."""
    with open(path, "w") as f:
        f.write("Title: Benchmark RAW file\n")
        f.write("Date: Thu Nov 28 00:00:00 2024\n")
        f.write("Plotname: Transient Analysis\n")
        f.write("Flags: real\n")
        f.write(f"No. Variables: {n_signals}\n")
        f.write(f"No. Points: {n_points}\n")
        f.write("Variables:\n")
        f.write("\t0\ttime\ttime\n")
        for i in range(1, n_signals):
            f.write(f"\t{i}\tv(n{i})\tvoltage\n")
        f.write("Values:\n")
        for pt in range(n_points):
            t = pt * 1e-6
            f.write(f"{pt}\t{t}\n")
            for sig in range(1, n_signals):
                val = 1.0 * sig + 0.001 * pt
                f.write(f"\t{val}\n")


def _create_binary_raw(path: str, n_signals: int, n_points: int) -> None:
    """Create a test binary RAW file."""
    with open(path, "wb") as f:
        header = f"""Title: Benchmark RAW file
Date: Thu Nov 28 00:00:00 2024
Plotname: Transient Analysis
Flags: real
No. Variables: {n_signals}
No. Points: {n_points}
Variables:
\t0\ttime\ttime
"""
        for i in range(1, n_signals):
            header += f"\t{i}\tv(n{i})\tvoltage\n"
        header += "Binary:\n"
        f.write(header.encode("utf-8"))

        for pt in range(n_points):
            t = pt * 1e-6
            f.write(struct.pack("<d", t))
            for sig in range(1, n_signals):
                val = 1.0 * sig + 0.001 * pt
                f.write(struct.pack("<d", val))


@pytest.fixture
def ascii_raw_10x100(tmp_path: str) -> Generator[str, None, None]:
    """Create a 10 signals x 100 points ASCII RAW file."""
    path = os.path.join(tmp_path, "test_10x100.raw")
    _create_ascii_raw(path, 10, 100)
    yield path


@pytest.fixture
def ascii_raw_10x1000(tmp_path: str) -> Generator[str, None, None]:
    """Create a 10 signals x 1000 points ASCII RAW file."""
    path = os.path.join(tmp_path, "test_10x1000.raw")
    _create_ascii_raw(path, 10, 1000)
    yield path


@pytest.fixture
def ascii_raw_100x1000(tmp_path: str) -> Generator[str, None, None]:
    """Create a 100 signals x 1000 points ASCII RAW file."""
    path = os.path.join(tmp_path, "test_100x1000.raw")
    _create_ascii_raw(path, 100, 1000)
    yield path


@pytest.fixture
def binary_raw_10x100(tmp_path: str) -> Generator[str, None, None]:
    """Create a 10 signals x 100 points binary RAW file."""
    path = os.path.join(tmp_path, "test_10x100_bin.raw")
    _create_binary_raw(path, 10, 100)
    yield path


@pytest.fixture
def binary_raw_10x1000(tmp_path: str) -> Generator[str, None, None]:
    """Create a 10 signals x 1000 points binary RAW file."""
    path = os.path.join(tmp_path, "test_10x1000_bin.raw")
    _create_binary_raw(path, 10, 1000)
    yield path


@pytest.fixture
def binary_raw_100x1000(tmp_path: str) -> Generator[str, None, None]:
    """Create a 100 signals x 1000 points binary RAW file."""
    path = os.path.join(tmp_path, "test_100x1000_bin.raw")
    _create_binary_raw(path, 100, 1000)
    yield path


# ==============================================================================
# ASCII RAW Parsing Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_ascii_raw_10x100(benchmark, ascii_raw_10x100: str) -> None:
    """Benchmark ASCII RAW parsing: 10 signals x 100 points."""
    from spicelab.io.raw_reader import parse_ngspice_ascii_raw

    result = benchmark(parse_ngspice_ascii_raw, ascii_raw_10x100)
    assert len(result.names) == 10


@pytest.mark.benchmark
def test_benchmark_ascii_raw_10x1000(benchmark, ascii_raw_10x1000: str) -> None:
    """Benchmark ASCII RAW parsing: 10 signals x 1000 points."""
    from spicelab.io.raw_reader import parse_ngspice_ascii_raw

    result = benchmark(parse_ngspice_ascii_raw, ascii_raw_10x1000)
    assert len(result.names) == 10


@pytest.mark.benchmark
def test_benchmark_ascii_raw_100x1000(benchmark, ascii_raw_100x1000: str) -> None:
    """Benchmark ASCII RAW parsing: 100 signals x 1000 points."""
    from spicelab.io.raw_reader import parse_ngspice_ascii_raw

    result = benchmark(parse_ngspice_ascii_raw, ascii_raw_100x1000)
    assert len(result.names) == 100


# ==============================================================================
# Binary RAW Parsing Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_binary_raw_10x100(benchmark, binary_raw_10x100: str) -> None:
    """Benchmark binary RAW parsing: 10 signals x 100 points."""
    from spicelab.io.raw_reader import parse_ngspice_raw

    result = benchmark(parse_ngspice_raw, binary_raw_10x100)
    assert len(result.names) == 10


@pytest.mark.benchmark
def test_benchmark_binary_raw_10x1000(benchmark, binary_raw_10x1000: str) -> None:
    """Benchmark binary RAW parsing: 10 signals x 1000 points."""
    from spicelab.io.raw_reader import parse_ngspice_raw

    result = benchmark(parse_ngspice_raw, binary_raw_10x1000)
    assert len(result.names) == 10


@pytest.mark.benchmark
def test_benchmark_binary_raw_100x1000(benchmark, binary_raw_100x1000: str) -> None:
    """Benchmark binary RAW parsing: 100 signals x 1000 points."""
    from spicelab.io.raw_reader import parse_ngspice_raw

    result = benchmark(parse_ngspice_raw, binary_raw_100x1000)
    assert len(result.names) == 100


# ==============================================================================
# Trace Access Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_trace_access(benchmark, binary_raw_100x1000: str) -> None:
    """Benchmark accessing trace values after parsing."""
    from spicelab.io.raw_reader import parse_ngspice_raw

    trace_set = parse_ngspice_raw(binary_raw_100x1000)

    def access_traces() -> int:
        total = 0
        for name in trace_set.names[:10]:
            total += len(trace_set[name].values)
        return total

    result = benchmark(access_traces)
    assert result == 10000  # 10 traces x 1000 points


@pytest.mark.benchmark
def test_benchmark_trace_magnitude(benchmark, binary_raw_100x1000: str) -> None:
    """Benchmark computing trace magnitude."""
    from spicelab.io.raw_reader import parse_ngspice_raw

    trace_set = parse_ngspice_raw(binary_raw_100x1000)
    trace = trace_set.names[1]  # First non-time trace

    def compute_magnitude():
        return trace_set[trace].magnitude()

    result = benchmark(compute_magnitude)
    assert len(result) == 1000


# ==============================================================================
# DataFrame Conversion Benchmarks
# ==============================================================================


@pytest.mark.benchmark
def test_benchmark_to_dataframe(benchmark, binary_raw_100x1000: str) -> None:
    """Benchmark converting TraceSet to DataFrame."""
    from spicelab.io.raw_reader import parse_ngspice_raw

    trace_set = parse_ngspice_raw(binary_raw_100x1000)

    result = benchmark(trace_set.to_dataframe)
    assert result.shape == (1000, 100)
