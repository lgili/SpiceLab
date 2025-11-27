# SpiceLab Performance Baseline

This document establishes performance baselines for SpiceLab's core operations.
All measurements were taken on a standard development machine (Apple Silicon M1/M2)
running Python 3.10+.

**Last Updated:** 2025-11-27

## Executive Summary

| Operation | Target | Current | Status |
|-----------|--------|---------|--------|
| Netlist (5k components) | <100ms | ~120ms | Close |
| RAW Parse (500 signals x 10k points, binary) | <500ms | ~656ms | Close |
| Monte Carlo (500 runs, simple circuit) | <500MB peak | <2MB | Excellent |
| Memory Leak Detection | No growth | 0 MB growth | Excellent |

## Circuit Operations

### Netlist Generation

The Union-Find optimization (M2) achieved O(n) scaling for netlist generation.

| Components | Build (ms) | Netlist (ms) | Hash (ms) | Total (ms) |
|------------|------------|--------------|-----------|------------|
| 10 | 0.08 | 0.08 | 0.14 | 0.41 |
| 100 | 0.38 | 0.55 | 0.58 | 2.43 |
| 500 | 2.35 | 2.75 | 2.89 | 11.98 |
| 1,000 | 3.51 | 6.04 | 5.71 | 24.13 |
| 5,000 | 19.65 | 29.31 | 29.00 | 119.79 |

**Scaling Analysis:**
- Build: O(n^1.07) - slightly superlinear due to dict operations
- Netlist: O(n^0.98) - linear scaling achieved
- Hash: O(n^1.01) - linear scaling achieved

### Circuit Building Performance

The Union-Find data structure (`spicelab/core/union_find.py`) provides:
- O(alpha(n)) amortized time for union/find operations
- Path compression for optimal tree flattening
- Union by rank for balanced trees
- Canonical value preservation for named nets

## RAW File Parsing

### ASCII vs Binary Performance

| Signals | Points | ASCII (ms) | Binary (ms) | Speedup |
|---------|--------|------------|-------------|---------|
| 10 | 100 | 0.91 | 0.29 | 3.1x |
| 10 | 1,000 | 5.69 | 1.32 | 4.3x |
| 10 | 10,000 | 55.82 | 10.45 | 5.3x |
| 100 | 1,000 | 45.39 | 9.79 | 4.6x |
| 100 | 10,000 | 481.46 | 102.22 | 4.7x |
| 500 | 10,000 | 2,609.55 | 656.29 | 4.0x |

**Key Findings:**
- Binary parsing is 4-5x faster than ASCII
- Both formats show O(n) scaling
- ASCII throughput: ~17 MB/s
- Binary throughput: ~60-75 MB/s

### Scaling Analysis

Both ASCII and binary parsing demonstrate linear O(n) complexity:
- ASCII: O(n^1.00) to O(n^1.06)
- Binary: O(n^0.90) to O(n^1.10)

## Monte Carlo Memory Usage

### Memory Per Run

| Runs | Circuit Size | Peak (MB) | Per Run (KB) | Time (s) |
|------|--------------|-----------|--------------|----------|
| 10 | 3 | 0.05 | 5.10 | 0.00 |
| 100 | 3 | 0.34 | 3.47 | 0.03 |
| 500 | 3 | 1.68 | 3.44 | 0.17 |
| 10 | 50 | 0.42 | 42.97 | 0.03 |
| 100 | 50 | 4.13 | 42.26 | 0.30 |
| 500 | 50 | 20.60 | 42.19 | 1.42 |

**Key Findings:**
- Memory scales linearly with run count
- Memory per run is stable (~3.5 KB for simple circuits, ~42 KB for 50-component circuits)
- 500 runs with 50 components uses only 20 MB peak

### Memory Leak Detection

Batch testing (10 batches x 100 runs each) shows:
- Memory growth: 0.00 MB across all batches
- No memory leaks detected
- Peak memory stable at ~0.13 MB per batch

## Profiling Tools

SpiceLab includes profiling scripts in the `tools/` directory:

```bash
# Profile netlist generation
python tools/profile_netlist.py --sizes 10,100,1000,5000 --profile

# Profile RAW file parsing
python tools/profile_raw.py --signals 10,100,500 --points 100,1000,10000 --profile

# Profile Monte Carlo memory
python tools/profile_monte_carlo.py --runs 10,100,500 --circuits 3,10,50
```

## Optimization History

### M2: Union-Find Optimization (2025-11-24)

**Problem:** O(n^2) net merging in `Circuit.connect()` caused exponential slowdown.

**Solution:** Implemented Union-Find data structure with:
- Path compression
- Union by rank
- Canonical value preservation

**Results:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Build (5k) | ~109ms | ~18ms | 6x |
| Netlist (5k) | O(n^2) | O(n) | Linear |
| Hash (5k) | O(n^2) | O(n) | Linear |

## Future Optimizations

### Planned (M2 Phase 4+)

1. **Parallel RAW Parsing**
   - Target: 3-4x speedup on multi-core systems
   - Approach: ProcessPoolExecutor with chunked signal parsing

2. **Memory-Mapped RAW Files**
   - Target: Reduced memory footprint for large files
   - Approach: Use mmap for lazy loading

3. **Lazy Component Evaluation**
   - Target: 30% memory reduction
   - Approach: `@cached_property` for expensive computations

### Nice to Have

- GPU acceleration for FFT/signal processing
- Distributed computing for massive Monte Carlo
- JIT compilation for hot paths (Numba/Cython)

## Benchmarking in CI

Performance is tracked in CI with:
- pytest-benchmark for automated benchmarks
- Regression alerts at >150% slowdown
- Benchmark results uploaded as artifacts

See `.github/workflows/ci.yml` for configuration.

## Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# Run with comparison to baseline
pytest tests/benchmarks/ --benchmark-compare

# Generate JSON output for CI
pytest tests/benchmarks/ --benchmark-json=benchmark.json
```

## Hardware Reference

Baseline measurements were taken on:
- **CPU:** Apple M1/M2 (8-core)
- **RAM:** 16GB unified memory
- **Python:** 3.10, 3.11, 3.12
- **OS:** macOS 14.x

Performance may vary on different hardware. Linux/Windows benchmarks are available in CI artifacts.
