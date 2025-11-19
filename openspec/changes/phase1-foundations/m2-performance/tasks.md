# M2: Performance Baseline - Tasks

**Status:** Proposed
**Start Date:** TBD
**Target Completion:** TBD (4-6 weeks)
**Dependencies:** M1 (stress test infrastructure)

## Task Breakdown

### Phase 1: Profiling & Baseline (Week 1)
- [ ] Setup profiling infrastructure
  - [ ] Install cProfile, py-spy, memory_profiler
  - [ ] Create `tools/profile_netlist_build.py`
  - [ ] Create `tools/profile_raw_parsing.py`
  - [ ] Create `tools/detect_memory_leaks.py`
- [ ] Profile netlist generation
  - [ ] Profile 10 components (baseline)
  - [ ] Profile 100 components
  - [ ] Profile 1,000 components
  - [ ] Profile 5,000 components
  - [ ] Profile 10,000 components
  - [ ] Save profiles as `.prof` files for analysis
- [ ] Profile RAW file parsing
  - [ ] Create test RAW files (10, 100, 1k signals)
  - [ ] Profile binary RAW reader
  - [ ] Profile ASCII RAW reader
  - [ ] Profile xarray Dataset construction
- [ ] Profile Monte Carlo
  - [ ] Profile 100 runs (memory usage)
  - [ ] Profile 1,000 runs (memory leaks?)
  - [ ] Track memory over time with tracemalloc
- [ ] Document baseline metrics
  - [ ] Create `docs/performance_baseline.md`
  - [ ] Record all current timings
  - [ ] Identify top 10 bottlenecks
  - [ ] Prioritize optimization targets

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 2: Comparative Benchmarks (Week 2)
- [ ] Setup benchmark environment
  - [ ] Create `benchmarks/` directory
  - [ ] Install pytest-benchmark
  - [ ] Configure benchmark settings (iterations, warmup)
  - [ ] Add pytest markers (`@pytest.mark.benchmark`)
- [ ] Install competitor tools
  - [ ] Install PySpice (optional, for comparison)
  - [ ] Install spicelib (optional, for comparison)
  - [ ] Document installation process
- [ ] Write netlist benchmarks
  - [ ] `bench_netlist_generation.py` (SpiceLab)
  - [ ] Add PySpice comparison (if available)
  - [ ] Add spicelib comparison (if available)
  - [ ] Test 10, 100, 1k, 5k, 10k components
- [ ] Write analysis benchmarks
  - [ ] `bench_transient_analysis.py`
  - [ ] `bench_ac_analysis.py`
  - [ ] `bench_dc_sweep.py`
  - [ ] Compare against PySpice (if available)
- [ ] Write Monte Carlo benchmarks
  - [ ] `bench_monte_carlo.py` (10, 100, 1k runs)
  - [ ] Memory usage tracking
  - [ ] Parallel efficiency test
- [ ] Setup benchmark CI
  - [ ] Create `.github/workflows/benchmark.yml`
  - [ ] Configure benchmark-action/github-action-benchmark
  - [ ] Setup GitHub Pages for benchmark dashboard
  - [ ] Add alert thresholds (>150% regression)
- [ ] Generate comparison report
  - [ ] Run all benchmarks locally
  - [ ] Create comparison table (SpiceLab vs PySpice/spicelib)
  - [ ] Document findings in `docs/benchmark_results.md`

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 3: Netlist Optimization (Week 3)
- [ ] Implement graph-based circuit
  - [ ] Add NetworkX dependency to `pyproject.toml`
  - [ ] Add `_graph` attribute to Circuit class
  - [ ] Implement `_rebuild_graph()` method
  - [ ] Add graph invalidation on modifications
- [ ] Optimize connectivity queries
  - [ ] Use graph for net traversal
  - [ ] Cache frequently accessed paths
  - [ ] Implement topological sort for components
- [ ] Optimize netlist generation
  - [ ] Implement `_topological_sort_components()`
  - [ ] Use graph order for netlist output
  - [ ] Avoid redundant connectivity checks
- [ ] Implement lazy evaluation
  - [ ] Add `@cached_property` for expensive attributes
  - [ ] Lazy parse component values (resistance, capacitance)
  - [ ] Cache SPICE card generation
  - [ ] Add `invalidate_cache()` method
- [ ] Add cache management
  - [ ] Implement cache invalidation on component changes
  - [ ] Add cache statistics (hit rate)
  - [ ] Document cache behavior
- [ ] Benchmark improvements
  - [ ] Re-run netlist benchmarks
  - [ ] Validate 10x improvement target
  - [ ] Document optimization techniques
  - [ ] Add performance tips to docs

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 4: Parallel & Memory Optimization (Week 4)
- [ ] Implement parallel RAW parsing
  - [ ] Refactor `read_raw()` for chunking
  - [ ] Create `read_raw_parallel()` function
  - [ ] Implement `_parse_signal_chunk()`
  - [ ] Add ProcessPoolExecutor integration
  - [ ] Auto-detect optimal worker count (CPU cores)
- [ ] Optimize RAW file I/O
  - [ ] Use memory-mapped files for large RAW files
  - [ ] Implement chunked reading (avoid loading all to RAM)
  - [ ] Add progress reporting for large files
- [ ] Memory leak detection
  - [ ] Run memory_profiler on Monte Carlo
  - [ ] Use tracemalloc to track allocations
  - [ ] Identify leak sources (circular refs, file handles)
- [ ] Fix identified leaks
  - [ ] Ensure proper file handle cleanup (context managers)
  - [ ] Break circular references in circuit graph
  - [ ] Explicitly delete large datasets in loops
  - [ ] Call `gc.collect()` after batch operations
- [ ] Optimize memory usage
  - [ ] Use NumPy views instead of copies
  - [ ] Lazy load xarray DataArrays
  - [ ] Compress cached results (optional)
- [ ] Validate memory reduction
  - [ ] Re-run Monte Carlo memory tests
  - [ ] Validate 50% reduction target
  - [ ] Document memory optimization tips

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 5: Integration & Validation (Week 5)
- [ ] Run comprehensive benchmarks
  - [ ] Execute full benchmark suite
  - [ ] Compare before/after performance
  - [ ] Generate detailed report
- [ ] Validate correctness
  - [ ] Run all unit tests (no regressions)
  - [ ] Run regression tests
  - [ ] Validate Monte Carlo results match original
  - [ ] Check netlist output identical (order may differ)
- [ ] Performance testing
  - [ ] Stress test optimized code (10k components)
  - [ ] Test parallel RAW parsing (4 workers vs 1)
  - [ ] Memory test large Monte Carlo (no leaks)
- [ ] Update documentation
  - [ ] Add performance tips to user guide
  - [ ] Document lazy loading behavior
  - [ ] Explain graph-based architecture
  - [ ] Create troubleshooting guide (performance)
- [ ] Create performance tuning guide
  - [ ] When to use parallel RAW parsing
  - [ ] How to reduce memory usage
  - [ ] Netlist optimization best practices
  - [ ] Profiling your own circuits

**Estimated Time:** 1 week
**Assignee:** TBD

---

### Phase 6: CI & Monitoring (Week 6)
- [ ] Configure benchmark CI
  - [ ] Finalize `.github/workflows/benchmark.yml`
  - [ ] Test CI on pull request
  - [ ] Validate benchmark data storage
- [ ] Setup regression alerts
  - [ ] Configure alert thresholds (>150% slowdown)
  - [ ] Add @maintainer notifications
  - [ ] Test alert system (intentional regression)
- [ ] Create performance dashboard
  - [ ] Setup GitHub Pages site
  - [ ] Add benchmark visualization charts
  - [ ] Show historical trends
  - [ ] Add comparison to PySpice/spicelib
- [ ] Add performance badges
  - [ ] Benchmark status badge to README
  - [ ] Link to performance dashboard
  - [ ] Document badge meaning
- [ ] Document CI process
  - [ ] How benchmarks run in CI
  - [ ] How to interpret results
  - [ ] How to investigate regressions
  - [ ] When to update thresholds

**Estimated Time:** 1 week
**Assignee:** TBD

---

## Acceptance Criteria

### Must Have
- [ ] Netlist generation (10k components): **<100ms** (10x improvement)
- [ ] RAW parsing (1k signals): **<500ms** (parallel speedup)
- [ ] Monte Carlo (1k runs): **<500MB** peak memory (50% reduction)
- [ ] 20+ benchmark tests implemented
- [ ] Benchmark CI active with regression alerts
- [ ] Performance dashboard deployed

### Should Have
- [ ] PySpice comparison benchmarks
- [ ] Memory profiling automated in CI
- [ ] Performance tuning guide written
- [ ] Graph-based circuit validated (correctness)

### Nice to Have
- [ ] spicelib comparison benchmarks
- [ ] GPU acceleration POC (future M10)
- [ ] Distributed Monte Carlo POC (future M15)

## Testing Checklist

Before marking M2 as complete:
- [ ] Run full benchmark suite (validate all improvements)
- [ ] Compare before/after metrics (document in report)
- [ ] Verify CI benchmarks run successfully
- [ ] Check regression alerts trigger correctly
- [ ] Validate performance dashboard displays correctly
- [ ] Run stress tests (10k components, 1k MC runs)
- [ ] Check no memory leaks (memory_profiler)
- [ ] Verify all unit tests pass (no correctness regressions)

## Performance Targets Summary

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Netlist (10k comp) | ~1000ms | <100ms (10x) | <50ms (20x) |
| RAW parse (1k sig) | ~2000ms | <500ms (4x) | <250ms (8x) |
| MC memory (1k runs) | ~1000MB | <500MB (50%) | <300MB (70%) |
| Overall speedup | 1x | 5-10x | 10-20x |

## Notes

- **NetworkX:** Make optional dependency with fallback to naive implementation
- **Parallel Parsing:** Auto-detect CPU cores, disable if single-core
- **Cache Invalidation:** Critical for correctness - test thoroughly
- **Benchmark Noise:** Use median of multiple runs, set generous thresholds
- **Memory Profiling:** Run separately from CI (overhead too high)

## Dependencies

- M1 (Stability) - stress test infrastructure needed for profiling

## Blocking

- M10 (Performance & I/O Overhaul) - advanced optimizations depend on M2 baseline
- M15 (Distributed Computing) - parallelization patterns established here

---

**Last Updated:** 2025-01-19
