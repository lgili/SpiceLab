# I/O Performance Specification

## ADDED Requirements

### Requirement: Memory-Mapped File Reading
The system SHALL support memory-mapped file reading for large binary result files.

#### Scenario: Read large .raw file with mmap
- **GIVEN** LTspice .raw file with 10 million data points (500 MB)
- **WHEN** reading with mmap enabled
- **THEN** system SHALL map file to virtual memory without loading entire file into RAM

#### Scenario: Lazy signal loading
- **GIVEN** .raw file with 100 signals
- **WHEN** user requests only `V(out)` and `I(R1)`
- **THEN** system SHALL load only those 2 signals, not all 100

#### Scenario: Fallback to ASCII for unsupported formats
- **GIVEN** .raw file variant not supporting mmap
- **WHEN** binary read fails
- **THEN** system SHALL automatically fallback to ASCII parsing with warning

### Requirement: Chunked Data Access
The system SHALL provide windowed access to large time-series and frequency data.

#### Scenario: Time window extraction
- **GIVEN** transient simulation from 0 to 1 second
- **WHEN** user requests `ds.sel(time=slice(0.1, 0.2))`
- **THEN** system SHALL return only data in [0.1, 0.2] second window without loading full dataset

#### Scenario: Frequency band extraction
- **GIVEN** AC analysis from 1 Hz to 1 GHz
- **WHEN** user requests `ds.sel(freq=slice(1e6, 10e6))`
- **THEN** system SHALL return only 1-10 MHz band

#### Scenario: Chunked processing for memory control
- **GIVEN** dataset too large for available RAM
- **WHEN** user iterates with `ds.chunks(time_chunk=1000)`
- **THEN** system SHALL yield chunks of 1000 time points each for processing

### Requirement: Parquet Columnar Storage
The system SHALL support Parquet format for efficient compressed storage of results.

#### Scenario: Export xarray Dataset to Parquet
- **GIVEN** simulation results as xarray.Dataset
- **WHEN** user calls `ds.to_parquet("results.parquet", compression="zstd")`
- **THEN** system SHALL save columnar format with zstd compression

#### Scenario: Preserve metadata in Parquet
- **GIVEN** dataset with attrs: `{engine: "ngspice", version: "42", netlist_hash: "abc123"}`
- **WHEN** exporting to Parquet
- **THEN** metadata SHALL be preserved in Parquet footer and restored on read

#### Scenario: Read Parquet back to xarray
- **GIVEN** Parquet file created by SpiceLab
- **WHEN** user calls `xr.open_dataset("results.parquet", engine="parquet")`
- **THEN** system SHALL reconstruct xarray.Dataset with all dimensions and metadata

### Requirement: Content-Addressable Caching
The system SHALL implement hash-based caching to avoid redundant simulations.

#### Scenario: Cache by circuit hash
- **GIVEN** circuit netlist and parameters
- **WHEN** running simulation
- **THEN** system SHALL compute hash from netlist + params + engine and check cache before simulating

#### Scenario: Cache hit avoids re-simulation
- **GIVEN** cached result for parameter set `{R1: 1k, C1: 10p}`
- **WHEN** requesting same simulation
- **THEN** system SHALL return cached result without invoking engine

#### Scenario: Cache invalidation on netlist change
- **GIVEN** cached results for circuit
- **WHEN** netlist topology changes
- **THEN** system SHALL compute new hash, detect mismatch, and re-simulate

### Requirement: Cache Management
The system SHALL provide cache management tools for disk space control.

#### Scenario: Cache size limit with LRU eviction
- **GIVEN** cache configured with 10 GB limit
- **WHEN** cache exceeds limit
- **THEN** system SHALL evict least recently used entries until under limit

#### Scenario: Cache statistics reporting
- **GIVEN** cache with multiple entries
- **WHEN** user requests `cache.stats()`
- **THEN** system SHALL return: total size, entry count, hit rate, oldest entry age

#### Scenario: Manual cache clearing
- **GIVEN** cache directory with old results
- **WHEN** user calls `cache.clear(older_than="30 days")`
- **THEN** system SHALL remove entries older than 30 days

### Requirement: Performance Benchmarking
The system SHALL include benchmarking tools to measure and track performance.

#### Scenario: Benchmark file I/O speed
- **GIVEN** standard test .raw file (100 MB)
- **WHEN** running benchmark
- **THEN** system SHALL measure and report read time, throughput (MB/s), memory usage

#### Scenario: Benchmark cache performance
- **GIVEN** Monte Carlo with 1000 trials
- **WHEN** running with and without cache
- **THEN** benchmark SHALL report speedup factor and cache hit rate

#### Scenario: Track performance over versions
- **GIVEN** benchmark results from previous versions
- **WHEN** running current version benchmarks
- **THEN** system SHALL compare and report performance deltas: "I/O 23% faster than v1.0"
