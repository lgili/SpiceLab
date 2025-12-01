# Chapter 4: Analysis Types

This chapter covers the different types of SPICE analyses available in SpiceLab: DC, AC, transient, and more.

## Overview of Analysis Types

| Analysis | Purpose | Use Case |
|----------|---------|----------|
| `.op` | DC operating point | Bias point, quiescent conditions |
| `.dc` | DC sweep | Transfer curves, I-V characteristics |
| `.ac` | AC frequency sweep | Frequency response, filters, stability |
| `.tran` | Transient | Time-domain behavior, switching |

## Running Simulations

All analyses follow the same pattern:

```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

# Define analysis
analysis = AnalysisSpec("type", {"param1": value1, ...})

# Run simulation
handle = run_simulation(circuit, [analysis], engine="ngspice")

# Get results
ds = handle.dataset()
```

## DC Operating Point (.op)

Calculates the DC bias point of the circuit with all capacitors open and inductors shorted.

### Basic Usage

```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

# No parameters needed for .op
op = AnalysisSpec("op", {})

handle = run_simulation(circuit, [op], engine="ngspice")
ds = handle.dataset()

# Access node voltages
print(ds.data_vars)  # Shows all voltages and currents
```

### Example: Bias Point Analysis

```python
"""Find operating point of voltage divider"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

# Build circuit
circuit = Circuit("bias_point")
v1 = Vdc("1", 10.0)
r1 = Resistor("1", "10k")
r2 = Resistor("2", "10k")
circuit.add(v1, r1, r2)

vin = Net("vin")
vout = Net("vout")
circuit.connect(v1.ports[0], vin)
circuit.connect(v1.ports[1], GND)
circuit.connect(r1.ports[0], vin)
circuit.connect(r1.ports[1], vout)
circuit.connect(r2.ports[0], vout)
circuit.connect(r2.ports[1], GND)

# Run operating point analysis
op = AnalysisSpec("op", {})
handle = run_simulation(circuit, [op], engine="ngspice")
ds = handle.dataset()

# Get output voltage
v_out = float(ds["V(vout)"].values)
print(f"Vout = {v_out}V")  # Should be 5V
```

## DC Sweep (.dc)

Sweeps a DC source and records circuit response.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | str | Source name to sweep (e.g., "V1") |
| `start` | float | Start value |
| `stop` | float | Stop value |
| `step` | float | Step size |

### Basic Usage

```python
# Sweep V1 from 0V to 5V in 0.1V steps
dc = AnalysisSpec("dc", {
    "source": "V1",
    "start": 0,
    "stop": 5,
    "step": 0.1
})

handle = run_simulation(circuit, [dc], engine="ngspice")
ds = handle.dataset()

# Results have a sweep dimension
print(ds["V(out)"].values)  # Array of output voltages
```

### Example: Diode I-V Curve

```python
"""Sweep voltage across a diode to get I-V characteristic"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc, Diode
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

circuit = Circuit("diode_iv")

v1 = Vdc("1", 0)  # Will be swept
r1 = Resistor("1", "100")  # Current limiting
d1 = Diode("1", model="1N4148")

circuit.add(v1, r1, d1)

vin = Net("vin")
vd = Net("vd")
circuit.connect(v1.ports[0], vin)
circuit.connect(v1.ports[1], GND)
circuit.connect(r1.ports[0], vin)
circuit.connect(r1.ports[1], vd)
circuit.connect(d1.ports[0], vd)  # Anode
circuit.connect(d1.ports[1], GND)  # Cathode

# DC sweep
dc = AnalysisSpec("dc", {
    "source": "V1",
    "start": 0,
    "stop": 1.0,
    "step": 0.01
})

handle = run_simulation(circuit, [dc], engine="ngspice")
ds = handle.dataset()

# Get diode voltage and current
v_diode = ds["V(vd)"]
i_diode = ds["I(V1)"]  # Current through voltage source
```

## AC Analysis (.ac)

Performs small-signal frequency-domain analysis.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `variation` | str | Sweep type: "dec", "oct", or "lin" |
| `npoints` | int | Points per decade/octave or total points |
| `fstart` | float | Start frequency (Hz) |
| `fstop` | float | Stop frequency (Hz) |

### Sweep Types

- **dec**: Logarithmic, points per decade (most common)
- **oct**: Logarithmic, points per octave
- **lin**: Linear sweep, total number of points

### Basic Usage

```python
# AC analysis: 100 points/decade, 1Hz to 1GHz
ac = AnalysisSpec("ac", {
    "variation": "dec",
    "npoints": 100,
    "fstart": 1,
    "fstop": 1e9
})

handle = run_simulation(circuit, [ac], engine="ngspice")
ds = handle.dataset()

# Results are complex (magnitude and phase)
import numpy as np
v_out = ds["V(out)"]
magnitude = np.abs(v_out.values)
phase = np.angle(v_out.values, deg=True)
```

### Example: RC Filter Response

```python
"""Analyze RC lowpass filter frequency response"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation
import numpy as np

# Build RC lowpass
circuit = Circuit("rc_filter")

vin = Vac("IN", ac_mag=1.0)  # 1V AC source
r1 = Resistor("1", "10k")
c1 = Capacitor("1", "100n")  # fc ≈ 159 Hz

circuit.add(vin, r1, c1)

input_net = Net("in")
output_net = Net("out")
circuit.connect(vin.ports[0], input_net)
circuit.connect(vin.ports[1], GND)
circuit.connect(r1.ports[0], input_net)
circuit.connect(r1.ports[1], output_net)
circuit.connect(c1.ports[0], output_net)
circuit.connect(c1.ports[1], GND)

# AC analysis
ac = AnalysisSpec("ac", {
    "variation": "dec",
    "npoints": 100,
    "fstart": 1,
    "fstop": 100_000
})

handle = run_simulation(circuit, [ac], engine="ngspice")
ds = handle.dataset()

# Get frequency response
freq = ds["frequency"].values
v_out = ds["V(out)"]
magnitude_db = 20 * np.log10(np.abs(v_out.values))
phase_deg = np.angle(v_out.values, deg=True)

# Find -3dB point (cutoff frequency)
idx_3db = np.argmin(np.abs(magnitude_db - (-3)))
fc_measured = freq[idx_3db]
print(f"Measured cutoff: {fc_measured:.1f} Hz")
```

## Transient Analysis (.tran)

Simulates circuit behavior over time.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tstep` | str/float | Suggested time step |
| `tstop` | str/float | Stop time |
| `tstart` | str/float | Start saving data (optional, default 0) |
| `tmax` | str/float | Maximum timestep (optional) |

### Basic Usage

```python
# Transient: 10µs step, 10ms duration
tran = AnalysisSpec("tran", {
    "tstep": "10u",
    "tstop": "10m"
})

handle = run_simulation(circuit, [tran], engine="ngspice")
ds = handle.dataset()

# Results are time-domain
time = ds["time"].values
v_out = ds["V(out)"].values
```

### Example: RC Step Response

```python
"""Simulate RC filter step response"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vpulse
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

circuit = Circuit("rc_step")

# Step input (0V to 1V at t=0)
vin = Vpulse("IN", v1=0, v2=1, td="0", tr="1n", tf="1n", pw="10m", per="20m")
r1 = Resistor("1", "10k")
c1 = Capacitor("1", "100n")  # tau = RC = 1ms

circuit.add(vin, r1, c1)

input_net = Net("in")
output_net = Net("out")
circuit.connect(vin.ports[0], input_net)
circuit.connect(vin.ports[1], GND)
circuit.connect(r1.ports[0], input_net)
circuit.connect(r1.ports[1], output_net)
circuit.connect(c1.ports[0], output_net)
circuit.connect(c1.ports[1], GND)

# Transient analysis
tran = AnalysisSpec("tran", {
    "tstep": "10u",
    "tstop": "5m"  # 5 time constants
})

handle = run_simulation(circuit, [tran], engine="ngspice")
ds = handle.dataset()

# Get time-domain response
time = ds["time"].values
v_out = ds["V(out)"].values

# Find rise time (10% to 90%)
v_10 = 0.1
v_90 = 0.9
t_10 = time[np.argmin(np.abs(v_out - v_10))]
t_90 = time[np.argmin(np.abs(v_out - v_90))]
rise_time = t_90 - t_10
print(f"Rise time: {rise_time*1000:.3f} ms")
```

## Multiple Analyses

Run multiple analyses in one simulation:

```python
# Run both AC and transient
ac = AnalysisSpec("ac", {"variation": "dec", "npoints": 100, "fstart": 1, "fstop": 1e6})
tran = AnalysisSpec("tran", {"tstep": "1u", "tstop": "1m"})

handle = run_simulation(circuit, [ac, tran], engine="ngspice")
ds = handle.dataset()

# Dataset contains results from both analyses
```

## Shortcuts for Quick Analysis

SpiceLab provides convenient shortcuts:

```python
from spicelab.shortcuts import quick_ac, quick_tran

# Quick AC with sensible defaults
result = quick_ac(circuit, start=1, stop=1e9)

# Quick transient
result = quick_tran(circuit, duration="10m")
```

## Exercises

### Exercise 4.1: Op-amp Gain
Build an inverting amplifier with gain=-10 and:
1. Run .op to find DC bias
2. Run .ac to measure frequency response
3. Find the -3dB bandwidth

### Exercise 4.2: LC Resonance
Build an RLC circuit and use .ac analysis to find the resonant frequency.
Compare with theory: f = 1/(2π√LC)

### Exercise 4.3: RC Time Constant
Build an RC circuit with R=1k, C=1µF (τ=1ms).
Use .tran to verify the capacitor reaches 63% of final value at t=τ.

### Exercise 4.4: Diode Rectifier
Build a half-wave rectifier and use .tran to simulate with a 1kHz sine input.
Add a filter capacitor and observe the ripple.

### Exercise 4.5: Parameter Sweep
Use .dc to sweep the feedback resistor in an op-amp circuit and observe gain variation.

## Analysis Quick Reference

```python
# Operating Point
AnalysisSpec("op", {})

# DC Sweep
AnalysisSpec("dc", {"source": "V1", "start": 0, "stop": 5, "step": 0.1})

# AC Analysis
AnalysisSpec("ac", {"variation": "dec", "npoints": 100, "fstart": 1, "fstop": 1e9})

# Transient
AnalysisSpec("tran", {"tstep": "10u", "tstop": "10m"})
```

## Next Steps

- [Chapter 5: Working with Results](05_results.md) - Processing and plotting data
- [Chapter 6: Monte Carlo Analysis](06_monte_carlo.md) - Statistical simulations

---

**See also:**
- [Troubleshooting Guide](../troubleshooting_guide.md) - Convergence issues
- [Template Catalog](../template_catalog.md) - Pre-built circuits for testing
