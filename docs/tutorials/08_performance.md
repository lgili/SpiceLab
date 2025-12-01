# Chapter 8: Performance Optimization

This chapter covers techniques for optimizing simulation performance: parallel execution, caching, and handling large circuits.

## Parallel Execution

### Multi-Worker Simulation

Run multiple simulations in parallel:

```python
from spicelab.analysis import monte_carlo, NormalPct

# Use multiple workers for Monte Carlo
result = monte_carlo(
    circuit,
    mapping={r1: NormalPct(5)},
    n=1000,
    analyses=[...],
    engine="ngspice",
    workers=4  # Use 4 parallel workers
)
```

**Worker guidelines:**
- Start with `workers = cpu_count - 1`
- Monitor memory usage (each worker loads full results)
- Diminishing returns beyond 8-16 workers for typical circuits

### Parameter Sweeps

```python
from spicelab.analysis.sweep_grid import run_param_grid

# Parallel parameter grid sweep
result = run_param_grid(
    circuit,
    variables=[(r1, ["1k", "2k", "5k"]), (c1, ["100n", "200n"])],
    analyses=[...],
    engine="ngspice",
    workers=4
)
```

## Result Caching

### Enable Caching

```python
from spicelab.analysis import monte_carlo

# Cache results to disk
result = monte_carlo(
    circuit,
    mapping={...},
    n=100,
    analyses=[...],
    engine="ngspice",
    cache_dir="./sim_cache"
)
```

### Reuse Cached Results

```python
# Second run reuses cached results
result = monte_carlo(
    circuit,
    mapping={...},  # Same parameters
    n=100,
    analyses=[...],
    engine="ngspice",
    cache_dir="./sim_cache",
    reuse_cache=True  # Reuse if netlists match
)
```

### Cache Management

```python
import shutil

# Clear cache
shutil.rmtree("./sim_cache", ignore_errors=True)

# Check cache size
import os
def get_cache_size(cache_dir):
    total = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)  # MB

print(f"Cache size: {get_cache_size('./sim_cache'):.1f} MB")
```

## Large Circuit Strategies

### Reduce Output Data

```python
# Only probe nodes of interest
circuit.add_directive(".probe v(out)")  # Instead of v(*)

# Limit saved data points
circuit.add_directive(".options numdgt=4")  # Reduce precision
```

### Optimize Timestep

```python
# Transient analysis with larger timestep
tran = AnalysisSpec("tran", {
    "tstep": "100u",   # Larger step (less data)
    "tstop": "10m",
    "tmax": "100u"     # Maximum timestep
})
```

### Use Skip Initial

```python
# Skip initial transient settling
tran = AnalysisSpec("tran", {
    "tstep": "1u",
    "tstop": "10m",
    "tstart": "1m"  # Start saving at 1ms
})
```

## Memory Management

### Process Results in Batches

```python
# For large Monte Carlo, process in batches
batch_size = 100
total_runs = 1000
all_results = []

for i in range(0, total_runs, batch_size):
    result = monte_carlo(
        circuit,
        mapping={...},
        n=batch_size,
        analyses=[...],
        engine="ngspice",
        seed=i  # Different seed per batch
    )

    # Process and extract only needed data
    batch_data = [
        float(run.handle.dataset()["V(out)"].values[-1])
        for run in result.runs
    ]
    all_results.extend(batch_data)

    # Explicit cleanup
    del result
```

### Lazy Loading

```python
# ResultHandle uses lazy loading
handle = run_simulation(circuit, analyses, engine="ngspice")

# Data not loaded yet
print(handle)  # Shows metadata only

# Data loaded on first access
ds = handle.dataset()  # Now data is loaded
```

## Profiling Simulations

### Timing Analysis

```python
import time

# Time simulation
start = time.perf_counter()
handle = run_simulation(circuit, analyses, engine="ngspice")
sim_time = time.perf_counter() - start

# Time data loading
start = time.perf_counter()
ds = handle.dataset()
load_time = time.perf_counter() - start

print(f"Simulation: {sim_time:.2f}s")
print(f"Data loading: {load_time:.2f}s")
```

### Identify Bottlenecks

```python
import cProfile
import pstats

# Profile Monte Carlo
profiler = cProfile.Profile()
profiler.enable()

result = monte_carlo(circuit, mapping, n=10, analyses=[...])

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Engine-Specific Optimizations

### ngspice

```python
# Use shared memory mode (if available)
from spicelab.engines import NgSpiceEngine

engine = NgSpiceEngine(shared_mode=True)

# Increase iteration limits for complex circuits
circuit.add_directive(".options itl1=500")  # DC iterations
circuit.add_directive(".options itl4=100")  # Transient iterations
```

### LTspice

```python
# Use batch mode for headless operation
from spicelab.engines import LtSpiceEngine

engine = LtSpiceEngine(batch_mode=True)

# Optimize for speed
circuit.add_directive(".options plotwinsize=0")  # Disable plot window
```

## Simulation Options Reference

```python
# Convergence helpers
circuit.add_directive(".options reltol=0.01")    # Relaxed tolerance
circuit.add_directive(".options abstol=1e-10")   # Absolute tolerance
circuit.add_directive(".options gmin=1e-12")     # Minimum conductance

# Performance tuning
circuit.add_directive(".options method=gear")     # GEAR integration
circuit.add_directive(".options maxstep=1u")      # Maximum timestep
circuit.add_directive(".options numdgt=4")        # Output precision

# Memory management
circuit.add_directive(".options nopage")          # No paging
```

## Exercises

### Exercise 8.1: Parallel Scaling
Run Monte Carlo with 100 iterations using 1, 2, 4, and 8 workers.
Plot execution time vs workers. What's the speedup?

### Exercise 8.2: Cache Effectiveness
Run the same simulation twice with caching enabled.
Compare first-run vs cached-run times.

### Exercise 8.3: Data Reduction
Compare simulation with `.probe v(*)` vs `.probe v(out)`.
How much smaller is the output file?

### Exercise 8.4: Timestep Optimization
Find the largest timestep that still captures a 1kHz signal accurately.
Balance speed vs accuracy.

### Exercise 8.5: Memory Profiling
Use `memory_profiler` to track memory usage during a large Monte Carlo.
Identify the peak memory point.

## Performance Checklist

| Optimization | Benefit | Trade-off |
|--------------|---------|-----------|
| Parallel workers | Linear speedup | Memory per worker |
| Result caching | Skip repeated runs | Disk space |
| Selective probing | Smaller output | Less data |
| Larger timestep | Faster simulation | Less accuracy |
| Batch processing | Lower peak memory | More complex code |
| Lazy loading | Deferred memory | Access latency |

## Next Steps

- [Chapter 9: Integration](09_integration.md) - CI/CD and automation
- [Chapter 10: Troubleshooting](10_troubleshooting.md) - Debugging techniques

---

**See also:**
- [Performance Baseline](../performance_baseline.md) - Benchmark reference
- [Troubleshooting Guide](../troubleshooting_guide.md) - Convergence issues
