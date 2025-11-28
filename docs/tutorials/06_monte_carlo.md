# Chapter 6: Monte Carlo Analysis

This chapter covers statistical simulation using Monte Carlo analysis to account for component tolerances and manufacturing variations.

## Why Monte Carlo?

Real components have tolerances:
- Resistors: ±1%, ±5%, ±10%
- Capacitors: ±10%, ±20%
- Transistors: β varies 50-300

Monte Carlo simulates many circuit instances with random component variations to predict:
- **Yield**: Percentage of circuits meeting specifications
- **Sensitivity**: Which components most affect performance
- **Worst case**: Extreme performance bounds

## Basic Monte Carlo Setup

```python
from spicelab.analysis import monte_carlo, NormalPct, UniformPct
from spicelab.core.types import AnalysisSpec

# Define component variations
variations = {
    r1: NormalPct(5),    # R1 has ±5% tolerance (normal distribution)
    c1: UniformPct(10),  # C1 has ±10% tolerance (uniform distribution)
}

# Run Monte Carlo
result = monte_carlo(
    circuit,
    mapping=variations,
    n=100,                                    # Number of runs
    analyses=[AnalysisSpec("tran", {...})],
    engine="ngspice",
    seed=42                                   # For reproducibility
)
```

## Distribution Types

### NormalPct (Gaussian Distribution)

Most components follow normal distribution:

```python
from spicelab.analysis import NormalPct

# ±5% tolerance (3σ = 5%)
r_var = NormalPct(5)

# Custom sigma factor
r_var = NormalPct(5, sigma=2)  # 2σ = 5% (wider spread)
```

**Characteristics:**
- Bell curve centered on nominal value
- 68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ
- Best for: resistors, common capacitors

### UniformPct (Flat Distribution)

All values equally likely within range:

```python
from spicelab.analysis import UniformPct

# ±10% tolerance, flat distribution
c_var = UniformPct(10)
```

**Characteristics:**
- All values within ±10% equally probable
- Sharp cutoff at limits
- Best for: worst-case analysis, unknown distributions

### UniformAbs (Absolute Bounds)

Absolute variation (not percentage):

```python
from spicelab.analysis import UniformAbs

# ±100 ohms from nominal
r_var = UniformAbs(100)
```

**Use for:** Fixed absolute tolerances, temperature offsets

## Complete Example: RC Filter Analysis

```python
"""Monte Carlo analysis of RC lowpass filter"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vpulse
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.analysis import monte_carlo, NormalPct
import numpy as np

# Build RC filter
circuit = Circuit("rc_filter_mc")

vin = Vpulse("IN", v1=0, v2=1, td="0", tr="1n", tf="1n", pw="10m", per="20m")
r1 = Resistor("1", "10k")    # 10k ±5%
c1 = Capacitor("1", "100n")  # 100nF ±10%

circuit.add(vin, r1, c1)

input_net = Net("in")
output_net = Net("out")
circuit.connect(vin.ports[0], input_net)
circuit.connect(vin.ports[1], GND)
circuit.connect(r1.ports[0], input_net)
circuit.connect(r1.ports[1], output_net)
circuit.connect(c1.ports[0], output_net)
circuit.connect(c1.ports[1], GND)

# Define tolerances
variations = {
    r1: NormalPct(5),   # ±5% resistor
    c1: NormalPct(10),  # ±10% capacitor
}

# Run Monte Carlo (100 iterations)
result = monte_carlo(
    circuit,
    mapping=variations,
    n=100,
    analyses=[AnalysisSpec("tran", {"tstep": "10u", "tstop": "5m"})],
    engine="ngspice",
    seed=42
)

# Analyze results
rise_times = []
for run in result.runs:
    ds = run.handle.dataset()
    time = ds["time"].values
    v_out = ds["V(out)"].values

    # Find rise time (10% to 90%)
    v_10 = 0.1
    v_90 = 0.9
    t_10 = time[np.argmin(np.abs(v_out - v_10))]
    t_90 = time[np.argmin(np.abs(v_out - v_90))]
    rise_times.append(t_90 - t_10)

# Statistics
rise_times = np.array(rise_times)
print(f"Rise time statistics:")
print(f"  Mean: {np.mean(rise_times)*1e6:.2f} µs")
print(f"  Std:  {np.std(rise_times)*1e6:.2f} µs")
print(f"  Min:  {np.min(rise_times)*1e6:.2f} µs")
print(f"  Max:  {np.max(rise_times)*1e6:.2f} µs")
```

## Accessing Run Details

Each Monte Carlo run contains the varied parameters:

```python
for i, run in enumerate(result.runs):
    # Get the varied component values for this run
    params = run.parameters

    # Get the simulation results
    ds = run.handle.dataset()

    print(f"Run {i}:")
    print(f"  R1 = {params['R1']:.1f} Ω")
    print(f"  C1 = {params['C1']*1e9:.2f} nF")
    print(f"  V(out) final = {ds['V(out)'].values[-1]:.3f} V")
```

## Yield Analysis

Calculate percentage of circuits meeting specifications:

```python
def calculate_yield(result, spec_min, spec_max, measurement_func):
    """Calculate yield based on specification limits.

    Args:
        result: Monte Carlo result
        spec_min: Minimum acceptable value
        spec_max: Maximum acceptable value
        measurement_func: Function that extracts measurement from dataset

    Returns:
        Yield percentage (0-100)
    """
    passing = 0
    total = len(result.runs)

    for run in result.runs:
        ds = run.handle.dataset()
        value = measurement_func(ds)

        if spec_min <= value <= spec_max:
            passing += 1

    return (passing / total) * 100

# Example: Rise time must be between 0.5ms and 2ms
def get_rise_time(ds):
    time = ds["time"].values
    v_out = ds["V(out)"].values
    t_10 = time[np.argmin(np.abs(v_out - 0.1))]
    t_90 = time[np.argmin(np.abs(v_out - 0.9))]
    return t_90 - t_10

yield_pct = calculate_yield(result, 0.5e-3, 2e-3, get_rise_time)
print(f"Yield: {yield_pct:.1f}%")
```

## Sensitivity Analysis

Identify which components most affect performance:

```python
from scipy.stats import pearsonr

def sensitivity_analysis(result, measurement_func):
    """Compute correlation between parameters and measurement."""
    measurements = []
    param_values = {key: [] for key in result.runs[0].parameters.keys()}

    for run in result.runs:
        ds = run.handle.dataset()
        measurements.append(measurement_func(ds))

        for key, value in run.parameters.items():
            param_values[key].append(value)

    # Calculate correlation coefficients
    measurements = np.array(measurements)
    sensitivities = {}

    for param, values in param_values.items():
        corr, _ = pearsonr(values, measurements)
        sensitivities[param] = corr

    return sensitivities

sens = sensitivity_analysis(result, get_rise_time)
print("Sensitivity analysis:")
for param, corr in sorted(sens.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {param}: {corr:+.3f}")
```

## Histogram Visualization

```python
import matplotlib.pyplot as plt

# Collect measurements
measurements = [get_rise_time(run.handle.dataset()) for run in result.runs]
measurements_us = np.array(measurements) * 1e6

# Create histogram
fig, ax = plt.subplots()
ax.hist(measurements_us, bins=20, edgecolor='black')
ax.axvline(np.mean(measurements_us), color='r', linestyle='--', label=f'Mean: {np.mean(measurements_us):.1f} µs')
ax.axvline(np.mean(measurements_us) + np.std(measurements_us), color='g', linestyle=':', label=f'+1σ')
ax.axvline(np.mean(measurements_us) - np.std(measurements_us), color='g', linestyle=':')
ax.set_xlabel("Rise Time (µs)")
ax.set_ylabel("Count")
ax.set_title("Monte Carlo: Rise Time Distribution")
ax.legend()
plt.show()
```

## Corner Analysis vs Monte Carlo

For quick worst-case analysis, test corners instead of full Monte Carlo:

```python
# Corner combinations for 2 components
corners = [
    {"r1": "min", "c1": "min"},  # Fast corner
    {"r1": "min", "c1": "max"},
    {"r1": "max", "c1": "min"},
    {"r1": "max", "c1": "max"},  # Slow corner
]

# Run each corner manually or use WCA analysis
from spicelab.analysis import worst_case_analysis

wca_result = worst_case_analysis(
    circuit,
    tolerances={r1: 0.05, c1: 0.10},  # ±5%, ±10%
    analyses=[AnalysisSpec("tran", {...})],
    engine="ngspice"
)
```

## Best Practices

1. **Start small**: Begin with 50-100 runs, increase if needed
2. **Use seed**: Set seed for reproducibility
3. **Check convergence**: Ensure statistics stabilize with more runs
4. **Realistic tolerances**: Use actual component tolerances from datasheets
5. **Key parameters**: Focus on components that most affect performance

## Performance Tips

```python
# Use parallel execution
result = monte_carlo(
    circuit,
    mapping=variations,
    n=1000,
    analyses=[...],
    engine="ngspice",
    workers=4  # Parallel workers
)

# Use caching for repeated analysis
result = monte_carlo(
    circuit,
    mapping=variations,
    n=1000,
    analyses=[...],
    engine="ngspice",
    cache_dir="./mc_cache",  # Cache results
    reuse_cache=True
)
```

## Exercises

### Exercise 6.1: Voltage Divider Tolerance
Create a voltage divider (10k + 10k) and run Monte Carlo with ±5% resistors.
What's the worst-case output voltage variation?

### Exercise 6.2: Filter Bandwidth
Analyze an RC filter with ±10% components.
Plot histogram of -3dB bandwidth across 100 runs.

### Exercise 6.3: Yield Optimization
For a filter with fc spec = 1kHz ±20%, what component tolerance is needed for >95% yield?

### Exercise 6.4: Sensitivity Ranking
Build a 3-stage RC filter and determine which resistor/capacitor most affects total delay.

### Exercise 6.5: Temperature Effects
Model temperature variation as ±10% on all resistors (typical tempco).
Compare room temperature vs full temperature range yield.

## Quick Reference

```python
# Distribution types
from spicelab.analysis import NormalPct, UniformPct, UniformAbs

# NormalPct: Gaussian ±X%
# UniformPct: Flat ±X%
# UniformAbs: Flat ±X absolute

# Basic Monte Carlo
result = monte_carlo(
    circuit,
    mapping={component: distribution, ...},
    n=100,
    analyses=[AnalysisSpec(...)],
    engine="ngspice",
    seed=42
)

# Access runs
for run in result.runs:
    ds = run.handle.dataset()
    params = run.parameters
```

## Next Steps

- [Chapter 7: Advanced Components](07_advanced_components.md) - Vendor models and subcircuits
- [Chapter 8: Performance](08_performance.md) - Optimization techniques

---

**See also:**
- [Monte Carlo Examples](../../examples/monte_carlo/) - More examples
- [Statistics Helpers](../reference/stats.md) - Statistical functions
