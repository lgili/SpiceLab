# Chapter 5: Working with Results

This chapter covers how to access, process, and visualize simulation results using SpiceLab's xarray-based dataset system.

## Understanding ResultHandle

When you run a simulation, you get a `ResultHandle`:

```python
from spicelab.engines import run_simulation
from spicelab.core.types import AnalysisSpec

handle = run_simulation(circuit, [AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})], engine="ngspice")
```

The handle provides:
- **Lazy loading**: Data is only read when needed
- **Metadata access**: Analysis info, timestamps
- **Multiple formats**: xarray Dataset, pandas DataFrame

## Getting the Dataset

```python
# Get xarray Dataset
ds = handle.dataset()

# Dataset structure
print(ds)
```

Output:
```
<xarray.Dataset>
Dimensions:  (time: 1001)
Coordinates:
  * time     (time) float64 0.0 1e-06 2e-06 ... 0.000999 0.001
Data variables:
    V(in)    (time) float64 0.0 0.0 0.0 ... 1.0 1.0 1.0
    V(out)   (time) float64 0.0 0.0 0.0 ... 0.632 0.632 0.632
    I(V1)    (time) float64 -1e-04 -9.9e-05 ... -3.68e-05
```

## Accessing Variables

### By Name

```python
# Get a single variable
v_out = ds["V(out)"]
print(type(v_out))  # <class 'xarray.DataArray'>

# Get values as numpy array
values = ds["V(out)"].values
print(type(values))  # <class 'numpy.ndarray'>
```

### List All Variables

```python
# Data variables (voltages, currents)
print(list(ds.data_vars))
# ['V(in)', 'V(out)', 'I(V1)']

# Coordinates (time, frequency, etc.)
print(list(ds.coords))
# ['time']
```

### Variable Naming

SPICE variables follow naming conventions:
- `V(node)`: Voltage at node
- `I(Vname)`: Current through voltage source
- `I(Rname)`: Current through resistor (if requested)

## Working with Coordinates

### Time-Domain (Transient)

```python
# Get time array
time = ds["time"].values

# Convert to milliseconds
time_ms = time * 1000

# Find index at specific time
import numpy as np
idx = np.argmin(np.abs(time - 0.5e-3))  # Find index at t=0.5ms
v_at_500us = ds["V(out)"].values[idx]
```

### Frequency-Domain (AC)

```python
# Get frequency array
freq = ds["frequency"].values

# Results are complex
v_out_complex = ds["V(out)"].values

# Extract magnitude and phase
magnitude = np.abs(v_out_complex)
phase_rad = np.angle(v_out_complex)
phase_deg = np.degrees(phase_rad)

# Convert to dB
magnitude_db = 20 * np.log10(magnitude)
```

## Measurements

### Finding Maximum/Minimum

```python
v_out = ds["V(out)"].values

# Maximum value and location
v_max = np.max(v_out)
t_max = time[np.argmax(v_out)]

# Minimum value
v_min = np.min(v_out)

# Peak-to-peak
v_pp = v_max - v_min
```

### Finding Specific Values

```python
# Find time when V(out) crosses 0.5V (rising edge)
threshold = 0.5
crossings = np.where(np.diff(np.sign(v_out - threshold)))[0]
if len(crossings) > 0:
    t_cross = time[crossings[0]]
```

### Rise/Fall Time

```python
def find_rise_time(time, signal, low_pct=0.1, high_pct=0.9):
    """Calculate rise time (10% to 90% by default)."""
    v_final = signal[-1]
    v_low = v_final * low_pct
    v_high = v_final * high_pct

    t_low = time[np.argmin(np.abs(signal - v_low))]
    t_high = time[np.argmin(np.abs(signal - v_high))]

    return t_high - t_low

rise_time = find_rise_time(time, ds["V(out)"].values)
print(f"Rise time: {rise_time * 1e6:.2f} µs")
```

### -3dB Bandwidth

```python
def find_bandwidth(freq, magnitude_db):
    """Find -3dB bandwidth."""
    # Assuming magnitude_db[0] is the passband gain
    passband_gain = magnitude_db[0]
    threshold = passband_gain - 3

    # Find first crossing below -3dB
    below_3db = np.where(magnitude_db < threshold)[0]
    if len(below_3db) > 0:
        return freq[below_3db[0]]
    return None

bandwidth = find_bandwidth(freq, magnitude_db)
print(f"Bandwidth: {bandwidth:.1f} Hz")
```

## Converting to Pandas

For tabular operations:

```python
# Convert entire dataset to DataFrame
df = ds.to_dataframe()
print(df.head())

# Select specific variables
df_selected = ds[["V(in)", "V(out)"]].to_dataframe()

# Reset index for flat DataFrame
df_flat = df.reset_index()
```

## Plotting Results

### Basic Plotting with Matplotlib

```python
import matplotlib.pyplot as plt

# Time-domain plot
fig, ax = plt.subplots()
ax.plot(time * 1e3, ds["V(out)"].values)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (V)")
ax.set_title("Step Response")
ax.grid(True)
plt.show()
```

### Bode Plot (AC Analysis)

```python
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True)

# Magnitude plot
ax_mag.semilogx(freq, magnitude_db)
ax_mag.set_ylabel("Magnitude (dB)")
ax_mag.grid(True)
ax_mag.axhline(-3, color='r', linestyle='--', label='-3dB')
ax_mag.legend()

# Phase plot
ax_phase.semilogx(freq, phase_deg)
ax_phase.set_xlabel("Frequency (Hz)")
ax_phase.set_ylabel("Phase (degrees)")
ax_phase.grid(True)

plt.tight_layout()
plt.show()
```

### Multiple Signals

```python
fig, ax = plt.subplots()
ax.plot(time * 1e3, ds["V(in)"].values, label="Input")
ax.plot(time * 1e3, ds["V(out)"].values, label="Output")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (V)")
ax.legend()
ax.grid(True)
plt.show()
```

## Using SpiceLab Visualization

SpiceLab includes built-in plotting utilities:

```python
from spicelab.viz import plot_traces, plot_bode

# Plot time-domain traces
plot_traces(ds, signals=["V(in)", "V(out)"])

# Plot Bode diagram (for AC results)
plot_bode(ds, signal="V(out)")
```

## Exporting Data

### To CSV

```python
# Using pandas
df = ds.to_dataframe()
df.to_csv("results.csv")

# Specific columns
df[["V(out)"]].to_csv("output_only.csv")
```

### To NumPy

```python
# Save arrays
np.savez("results.npz",
         time=time,
         v_out=ds["V(out)"].values,
         v_in=ds["V(in)"].values)

# Load later
data = np.load("results.npz")
time_loaded = data["time"]
```

### To NetCDF (preserves xarray structure)

```python
ds.to_netcdf("results.nc")

# Load back
import xarray as xr
ds_loaded = xr.open_dataset("results.nc")
```

## Working with Multiple Runs

When running parameter sweeps or Monte Carlo:

```python
from spicelab.analysis import monte_carlo, NormalPct

# Run Monte Carlo
mc_result = monte_carlo(
    circuit,
    mapping={r1: NormalPct(5)},  # 5% tolerance
    n=100,
    analyses=[AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})],
    engine="ngspice"
)

# Access individual runs
for i, run in enumerate(mc_result.runs):
    ds = run.handle.dataset()
    final_value = ds["V(out)"].values[-1]
    print(f"Run {i}: V(out) = {final_value:.3f}V")

# Aggregate statistics
final_values = [run.handle.dataset()["V(out)"].values[-1]
                for run in mc_result.runs]
print(f"Mean: {np.mean(final_values):.3f}V")
print(f"Std: {np.std(final_values):.3f}V")
```

## Common Patterns

### Extract Single Value at End

```python
final_voltage = float(ds["V(out)"].values[-1])
```

### Find Settling Time

```python
def settling_time(time, signal, final_value=None, tolerance=0.02):
    """Time to settle within tolerance of final value."""
    if final_value is None:
        final_value = signal[-1]

    threshold = final_value * tolerance
    within_bounds = np.abs(signal - final_value) < threshold

    # Find last point outside bounds
    outside = np.where(~within_bounds)[0]
    if len(outside) == 0:
        return 0
    return time[outside[-1]]

ts = settling_time(time, ds["V(out)"].values)
print(f"Settling time (2%): {ts * 1e3:.2f} ms")
```

### Calculate Overshoot

```python
def overshoot_percent(signal, final_value=None):
    """Calculate percent overshoot."""
    if final_value is None:
        final_value = signal[-1]

    if final_value == 0:
        return 0

    peak = np.max(signal)
    return (peak - final_value) / final_value * 100

os = overshoot_percent(ds["V(out)"].values)
print(f"Overshoot: {os:.1f}%")
```

## Exercises

### Exercise 5.1: RC Time Constant
Run transient analysis on an RC circuit and measure:
- Time to reach 63.2% of final value (τ)
- Time to reach 99% of final value (5τ)

### Exercise 5.2: Filter Characterization
Run AC analysis on an RC filter and extract:
- DC gain (at lowest frequency)
- -3dB bandwidth
- Phase at cutoff frequency

### Exercise 5.3: Data Export
Simulate a circuit and export results to CSV with:
- Time in milliseconds
- All voltages in a single file

### Exercise 5.4: Statistical Analysis
Run 50 Monte Carlo simulations and:
- Calculate mean and standard deviation of output
- Find min/max values
- Plot histogram of results

### Exercise 5.5: Custom Measurement
Create a function that measures the slew rate (dV/dt) of a signal.

## Result Methods Quick Reference

```python
# Get dataset
ds = handle.dataset()

# Get metadata
attrs = handle.attrs()

# Common operations
ds.data_vars         # List data variables
ds.coords            # List coordinates
ds["V(out)"]         # Get single variable
ds["V(out)"].values  # Get numpy array
ds.to_dataframe()    # Convert to pandas
ds.to_netcdf("f.nc") # Save to file
```

## Next Steps

- [Chapter 6: Monte Carlo Analysis](06_monte_carlo.md) - Statistical simulations
- [Chapter 7: Advanced Components](07_advanced_components.md) - Vendor models

---

**See also:**
- [xarray Documentation](https://docs.xarray.dev/) - Full xarray reference
- [Troubleshooting Guide](../troubleshooting_guide.md) - Empty results debugging
