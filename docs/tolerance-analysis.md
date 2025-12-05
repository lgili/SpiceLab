# Production Tolerance Analysis

This guide covers spicelab's enhanced tolerance analysis features for production-ready
circuit validation. These tools help you answer critical questions:

- Will my circuit work across component tolerances?
- What is my expected production yield?
- Which components most affect performance?
- What are the absolute worst-case bounds?

## Quick Start

```python
from spicelab.analysis import (
    monte_carlo, run_wca,
    NormalPct, NormalAbs, UniformPct, CorrelatedGroup,
    generate_report,
)
```

## Distribution Types

### Percentage-Based Distributions

For components with relative tolerances (most resistors, capacitors):

```python
from spicelab.analysis import NormalPct, UniformPct, TriangularPct

# 1% resistor (3-sigma = 1%)
R1_dist = NormalPct(0.01 / 3)  # sigma = 0.33%

# 5% uniform tolerance
C1_dist = UniformPct(0.05)  # ±5%

# Triangular distribution centered at nominal
R2_dist = TriangularPct(0.01)  # ±1%
```

### Absolute Distributions

For parameters specified in absolute units (op-amp offset voltage, reference voltage):

```python
from spicelab.analysis import NormalAbs, UniformAbs, TriangularAbs

# Op-amp offset: ±2mV max, 3-sigma spec
Vos_dist = NormalAbs(0.002 / 3)  # sigma = 0.67mV

# Reference voltage ±5mV uniform
Vref_dist = UniformAbs(0.005)  # ±5mV absolute
```

## Correlated Groups (Same-Lot Components)

Components from the same manufacturing batch tend to have correlated variations.
Use `CorrelatedGroup` to model this:

```python
from spicelab.analysis import monte_carlo, CorrelatedGroup, NormalPct

# Resistors from same batch vary together
same_lot_resistors = CorrelatedGroup([R1, R2, R3, R4], NormalPct(0.001))

# Independent components
mc_result = monte_carlo(
    circuit,
    mapping={
        same_lot_resistors: None,  # Correlated group (dist already specified)
        Vos1: NormalAbs(0.0007),   # Independent
        Vos2: NormalAbs(0.0007),   # Independent
    },
    n=10000,
    analyses=[AnalysisSpec("op")],
)
```

**When to use correlated groups:**
- Resistor networks/arrays from same package
- Matched transistor pairs
- Components from same reel/batch
- Any components you expect to track together

## Monte Carlo Analysis

### Basic Usage

```python
from spicelab.analysis import monte_carlo, NormalPct
from spicelab.core.types import AnalysisSpec

result = monte_carlo(
    circuit,
    mapping={
        R1: NormalPct(0.01),
        R2: NormalPct(0.01),
        C1: UniformPct(0.10),
    },
    n=10000,
    seed=42,  # Reproducibility
    analyses=[AnalysisSpec("op")],
    workers=4,  # Parallel execution
)
```

### Extracting Results

```python
# Extract metric from each run
def get_vout(run):
    return run.traces['V(vout)'].values[-1]

# Get statistics
stats = result.statistics(get_vout)
print(f"Mean: {stats.mean:.4f}")
print(f"Std:  {stats.std:.4f}")
print(f"Min:  {stats.min:.4f}")
print(f"Max:  {stats.max:.4f}")
```

### Process Capability Metrics

```python
# Define spec limits
LSL, USL = 2.4, 2.6  # Vout spec: 2.4V to 2.6V

# Calculate Cpk
cpk = result.cpk(get_vout, LSL, USL)
print(f"Cpk: {cpk:.2f}")

# Cpk interpretation:
# >= 2.00: Excellent (6-sigma quality)
# >= 1.67: Good
# >= 1.33: Acceptable
# >= 1.00: Marginal
# <  1.00: Poor

# Yield estimates
yield_est = result.yield_estimate(get_vout, LSL, USL)
actual_yield = result.actual_yield(get_vout, LSL, USL)
print(f"Estimated yield: {yield_est * 100:.2f}%")
print(f"Actual yield:    {actual_yield * 100:.2f}%")

# Sigma level
sigma = result.sigma_level(get_vout, LSL, USL)
print(f"Sigma level: {sigma:.1f}σ")
```

### Complete Process Summary

```python
summary = result.process_summary(
    get_vout,
    lsl=2.4,
    usl=2.6,
    metric_name="Vout",
)

print(f"""
Process Capability Report
=========================
Metric: {summary['metric_name']}
Samples: {summary['n']}

Statistics:
  Mean: {summary['mean']:.4f}
  Std:  {summary['std']:.4f}
  Range: [{summary['min']:.4f}, {summary['max']:.4f}]

Specification:
  LSL: {summary['lsl']}
  USL: {summary['usl']}

Process Capability:
  Cpk: {summary['cpk']:.3f}
  Sigma Level: {summary['sigma_level']:.2f}σ

Yield:
  Estimated: {summary['yield_pct']:.2f}%
  Actual: {summary['actual_yield_pct']:.2f}%
  Failures: {summary['failures']}
""")
```

## Worst-Case Analysis (WCA)

WCA evaluates all corner combinations to find absolute bounds:

```python
from spicelab.analysis import run_wca

# Mixed tolerances: relative and absolute
result = run_wca(
    circuit,
    tolerances={
        R1: 0.01,              # 1% relative
        R2: 0.05,              # 5% relative
        Vos: (0.002, 'abs'),   # ±2mV absolute
    },
    analyses=[AnalysisSpec("op")],
)

# Find worst cases
def get_vout(corner):
    return float(corner.dataset()['V(vout)'].values[-1])

worst_high = result.find_extreme(get_vout, kind='max')
worst_low = result.find_extreme(get_vout, kind='min')

print(f"Worst-case high: {get_vout(worst_high):.4f} at {worst_high.corner_name}")
print(f"Worst-case low:  {get_vout(worst_low):.4f} at {worst_low.corner_name}")
```

## Temperature Sweep Integration

Run tolerance analysis across multiple conditions:

```python
from spicelab.analysis import monte_carlo_sweep, wca_sweep

# Monte Carlo across temperatures
mc_sweep = monte_carlo_sweep(
    circuit,
    mapping={R1: NormalPct(0.01), R2: NormalPct(0.01)},
    conditions={'temp': [-40, 25, 85]},
    n=1000,
    analyses=[AnalysisSpec("op")],
)

# Summary across all conditions
summary_df = mc_sweep.summary_dataframe(
    get_vout,
    lsl=2.4,
    usl=2.6,
)
print(summary_df)

# WCA across temperatures
wca_sweep_result = wca_sweep(
    circuit,
    tolerances={R1: 0.01, R2: 0.01},
    conditions={'temp': [-40, 25, 85]},
    analyses=[AnalysisSpec("op")],
)

# Bounds table
bounds_df = wca_sweep_result.bounds_dataframe(get_vout, metric_name='Vout')
print(bounds_df)
```

## Simplified Result Extraction

Helper functions for common extraction patterns:

```python
from spicelab.analysis import get_voltage, get_current, get_power

# Extract from traces (case-insensitive)
vout = get_voltage(result.traces, 'vout')
i_r1 = get_current(result.traces, 'R1')

# At specific time
vout_1ms = get_voltage(result.traces, 'vout', at_time=1e-3)
```

## Report Generation

Generate professional HTML reports:

```python
from spicelab.analysis import generate_report

html = generate_report(
    mc_result,
    metric=get_vout,
    lsl=2.4,
    usl=2.6,
    output_path="tolerance_report.html",
    title="Vout Tolerance Analysis",
    metric_name="Output Voltage (V)",
)
```

The report includes:
- Summary cards (n, mean, std, Cpk, yield)
- Distribution histogram with spec limits
- Statistical summary table
- Process capability metrics (Cpk, sigma level, yield)
- Varied parameters table

## Best Practices

### 1. Choose Appropriate Sample Size

```python
# Quick check: n=1000
# Production validation: n=10000
# High-reliability: n=100000
```

### 2. Use Seeds for Reproducibility

```python
mc = monte_carlo(..., seed=42)  # Always reproducible
```

### 3. Model Real-World Correlations

```python
# Components from same batch -> CorrelatedGroup
# Different manufacturers -> Independent
```

### 4. Validate Against WCA

```python
# Monte Carlo gives statistical picture
# WCA gives absolute bounds
# Use both for complete validation
```

### 5. Consider Temperature Effects

```python
# Don't just analyze at room temperature
# Sweep across operating range: -40°C to +85°C
```

## Migration from Manual Approach

Before (manual):
```python
# Old way: manual sampling loop
for i in range(1000):
    r1_val = R1.value * random.gauss(1, 0.01)
    r2_val = R2.value * random.gauss(1, 0.01)
    # ... modify circuit and run simulation
```

After (spicelab API):
```python
# New way: declarative specification
result = monte_carlo(
    circuit,
    mapping={
        R1: NormalPct(0.01),
        R2: NormalPct(0.01),
    },
    n=1000,
    analyses=[AnalysisSpec("op")],
)
```

Benefits:
- Automatic parallelization (`workers=N`)
- Result caching (no re-running identical simulations)
- Built-in statistics and Cpk calculation
- HTML report generation
- Correlated group support
- Temperature sweep integration

## API Reference

See the [Analysis API Reference](reference/analysis.md) for complete documentation of:

- `monte_carlo()` - Monte Carlo analysis
- `run_wca()` - Worst-case analysis
- `monte_carlo_sweep()` - MC across conditions
- `wca_sweep()` - WCA across conditions
- `generate_report()` - HTML report generation
- Distribution classes: `NormalPct`, `NormalAbs`, `UniformPct`, etc.
- `CorrelatedGroup` - Same-lot component correlation
- `Statistics` - Statistical summary dataclass
- `MonteCarloResult` - Result container with process capability methods
