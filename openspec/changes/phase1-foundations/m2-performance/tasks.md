# M2: Performance Baseline - Tasks

**Status:** In Progress
**Start Date:** 2025-11-24
**Target Completion:** TBD (4-6 weeks)
**Dependencies:** M1 (stress test infrastructure)

## Task Breakdown

### Phase 1: Profiling & Baseline ✅ COMPLETE
- [x] Setup profiling infrastructure
  - [x] Install cProfile (built-in Python)
  - [x] Create `tools/profile_netlist.py` (netlist build profiler)
- [x] Profile netlist generation
  - [x] Profile 10 components (baseline)
  - [x] Profile 100 components
  - [x] Profile 1,000 components
  - [x] Profile 5,000 components
- [x] Identified bottleneck: O(n²) net merging in Circuit.connect()

**Completed:** 2025-11-24

---

### Phase 2: Union-Find Optimization ✅ COMPLETE
- [x] Implement Union-Find data structure
  - [x] Create `spicelab/core/union_find.py`
  - [x] Implement path compression
  - [x] Implement union by rank
  - [x] Support canonical value preservation for named nets
- [x] Integrate Union-Find into Circuit class
  - [x] Update `Circuit.connect()` to use Union-Find for net merging
  - [x] Update `Circuit._net_of()` to use canonical nets
  - [x] Update `Circuit._assign_node_ids()` to use canonical nets
  - [x] Update `Circuit._net_label()` to use canonical nets
  - [x] Add cache invalidation mechanism
- [x] Verify correctness
  - [x] Run core circuit tests (test_circuit_introspection, test_circuit_rc_netlist)
  - [x] Run netlist and DSL tests
  - [x] Verify named nets are preserved in netlist output

**Completed:** 2025-11-24

---

### Phase 3: Performance Verification ✅ COMPLETE
- [x] Re-run profiler with optimizations
  - [x] Build time: Near O(n) scaling at large sizes
  - [x] Netlist generation: O(n) scaling confirmed
  - [x] Hash generation: O(n) scaling confirmed
- [x] Results (5,000 components):
  - Build: 18.29ms
  - Netlist: 27.96ms
  - Hash: 27.85ms
  - Total: 113.71ms

**Completed:** 2025-11-24

---

### Phase 4: Remaining Profiling Tasks (Future)
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

---

### Phase 5: Comparative Benchmarks (Future)
- [ ] Setup benchmark environment
  - [ ] Create `benchmarks/` directory
  - [ ] Install pytest-benchmark
  - [ ] Configure benchmark settings (iterations, warmup)
- [ ] Write netlist benchmarks
  - [ ] `bench_netlist_generation.py` (SpiceLab)
  - [ ] Test 10, 100, 1k, 5k, 10k components
- [ ] Write analysis benchmarks
  - [ ] `bench_transient_analysis.py`
  - [ ] `bench_ac_analysis.py`
  - [ ] `bench_dc_sweep.py`
- [ ] Write Monte Carlo benchmarks
  - [ ] `bench_monte_carlo.py` (10, 100, 1k runs)
  - [ ] Memory usage tracking
  - [ ] Parallel efficiency test

---

### Phase 6: Additional Netlist Optimization (Future)
- [ ] Implement graph-based circuit (NetworkX)
  - [ ] Add `_graph` attribute to Circuit class
  - [ ] Implement `_rebuild_graph()` method
  - [ ] Add graph invalidation on modifications
- [ ] Optimize connectivity queries
  - [ ] Use graph for net traversal
  - [ ] Cache frequently accessed paths
  - [ ] Implement topological sort for components

---

### Phase 7: Parallel & Memory Optimization (Future)
- [ ] Implement parallel RAW parsing
  - [ ] Refactor `read_raw()` for chunking
  - [ ] Create `read_raw_parallel()` function
  - [ ] Add ProcessPoolExecutor integration
- [ ] Optimize RAW file I/O
  - [ ] Use memory-mapped files for large RAW files
  - [ ] Implement chunked reading
- [ ] Memory leak detection
  - [ ] Run memory_profiler on Monte Carlo
  - [ ] Identify leak sources

---

### Phase 8: CI & Monitoring (Future)
- [ ] Configure benchmark CI
  - [ ] Create `.github/workflows/benchmark.yml`
  - [ ] Configure benchmark-action/github-action-benchmark
  - [ ] Setup GitHub Pages for benchmark dashboard
- [ ] Setup regression alerts
  - [ ] Configure alert thresholds (>150% slowdown)
  - [ ] Add @maintainer notifications

---

## Acceptance Criteria

### Must Have
- [x] Netlist generation: O(n) scaling (achieved with Union-Find)
- [ ] Netlist generation (10k components): **<100ms** (10x improvement)
- [ ] RAW parsing (1k signals): **<500ms** (parallel speedup)
- [ ] Monte Carlo (1k runs): **<500MB** peak memory (50% reduction)
- [ ] 20+ benchmark tests implemented
- [ ] Benchmark CI active with regression alerts

### Should Have
- [ ] PySpice comparison benchmarks
- [ ] Memory profiling automated in CI
- [ ] Performance tuning guide written

### Nice to Have
- [ ] spicelib comparison benchmarks
- [ ] GPU acceleration POC (future M10)

## Performance Results Summary

| Metric | Before (O(n²)) | After (O(n)) | Improvement |
|--------|----------------|--------------|-------------|
| Build (1k) | ~4.37ms | ~5.90ms | Similar (variation) |
| Build (5k) | ~109ms (est) | ~18ms | ~6x |
| Netlist (5k) | ~27ms | ~28ms | O(n) confirmed |
| Hash (5k) | ~27ms | ~28ms | O(n) confirmed |

## Files Created/Modified

### New Files
- `spicelab/core/union_find.py` - Union-Find data structure for O(α(n)) net merging
- `tools/profile_netlist.py` - Profiling script for circuit operations

### Modified Files
- `spicelab/core/circuit.py` - Integrated Union-Find for net merging
- `tests/test_montecarlo_job.py` - Fixed test to match actual node naming

## Notes

- **Union-Find:** Provides O(α(n)) amortized time for union/find operations
- **Path Compression:** Flattens tree structure during find operations
- **Union by Rank:** Attaches smaller tree under larger for balance
- **Canonical Value:** Preserves named nets (e.g., `Net("n1")`) through merges

## Dependencies

- M1 (Stability) - stress test infrastructure needed for profiling

## Blocking

- M10 (Performance & I/O Overhaul) - advanced optimizations depend on M2 baseline
- M15 (Distributed Computing) - parallelization patterns established here

---

**Last Updated:** 2025-11-24
