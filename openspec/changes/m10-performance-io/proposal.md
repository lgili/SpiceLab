# M10 - Performance and Zero-Copy I/O

## Why
As simulations grow larger (millions of time points, thousands of Monte Carlo trials), current I/O becomes a bottleneck. SpiceLab needs memory-efficient file reading (mmap), lazy loading, chunked access, and compressed columnar storage to handle professional-scale datasets without memory exhaustion.

## What Changes
- Implement memory-mapped reading for binary LTspice and ngspice .raw files
- Add lazy loading with column-on-demand access to xarray datasets
- Implement chunked time/frequency window access: `ds.window(time=(t0, t1))`
- Add native Parquet export with columnar compression (zstd) and metadata preservation
- Optimize cache system with content-addressable storage keyed by netlist+params hash
- Add benchmarking infrastructure to track performance improvements

## Impact
- **Affected specs**: io-performance, caching
- **Affected code**:
  - Modified: `spicelab/io/readers.py` (add mmap and lazy loading)
  - Modified: `spicelab/orchestrator.py` (improve cache efficiency)
  - New: `benchmarks/` directory with performance tests
  - Modified: `tests/test_io_performance.py`
- **Dependencies**: PyArrow (for Parquet), mmap (stdlib)
