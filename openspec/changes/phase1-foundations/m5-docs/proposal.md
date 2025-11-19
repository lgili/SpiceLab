# M5: Documentation & Examples

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 6-8 weeks
**Dependencies:** M3 (component library), M4 (templates for examples)

## Problem Statement

SpiceLab currently has basic documentation (~30 pages) but lacks comprehensive tutorials, real-world examples, and migration guides. Without thorough documentation, users struggle to learn the library effectively, limiting adoption and increasing support burden.

### Current Gaps
- ❌ No beginner-to-advanced tutorial progression
- ❌ Only ~20 basic examples (mostly synthetic)
- ❌ No video tutorials or screencasts
- ❌ Incomplete API reference (~70% coverage)
- ❌ No migration guides from PySpice/spicelib
- ❌ No troubleshooting cookbook
- ❌ No performance tuning guide

### Impact
- **Learning Curve:** Steep, discourages beginners
- **Adoption:** Users can't find answers, abandon library
- **Support Burden:** Repetitive questions on basic usage
- **SEO:** Poor Google ranking (docs not comprehensive)
- **Professional Use:** Lack of best practices documentation

## Objectives

1. **Create complete tutorial** from beginner to advanced (10+ chapters)
2. **Add 50+ practical examples** covering real-world circuits
3. **Produce video tutorials** (5-10 videos, 5-15 min each)
4. **Complete API reference** (100% coverage with examples)
5. **Write migration guides** from PySpice and spicelib
6. **Build troubleshooting cookbook** (20+ common issues)
7. **Document performance tuning** best practices
8. **Target:** Top 3 Google results for "Python SPICE simulation"

## Technical Design

### 1. Complete Tutorial (Beginner to Advanced)

**Structure:**
```
Tutorial Structure (10 chapters):

1. Getting Started
   - Installation (pip, conda, engines)
   - First circuit (RC filter)
   - Running first simulation
   - Understanding results

2. Basic Components
   - Passive components (R, L, C)
   - Sources (DC, AC, pulse)
   - Ground and nets
   - Component values and units

3. Building Circuits
   - Programmatic circuit building
   - DSL syntax
   - Connecting components
   - Subcircuits

4. Analysis Types
   - DC operating point
   - Transient analysis
   - AC analysis
   - DC sweep
   - Parameterized sweeps

5. Working with Results
   - xarray datasets
   - Plotting with matplotlib
   - Exporting data (CSV, HDF5)
   - Custom measurements

6. Monte Carlo & Statistics
   - Tolerance distributions
   - Running Monte Carlo
   - Statistical analysis
   - Yield estimation

7. Advanced Components
   - Vendor models (TI, ADI)
   - Custom subcircuits
   - Behavioral models
   - Model parameters

8. Performance Optimization
   - Large circuit strategies
   - Parallel execution
   - Caching results
   - Memory management

9. Integration & Automation
   - CI/CD pipelines
   - Batch processing
   - Command-line usage
   - Jupyter notebooks

10. Troubleshooting
    - Convergence issues
    - Debugging circuits
    - Validation
    - Best practices
```

#### Example Tutorial Chapter
```markdown
# Chapter 2: Basic Components

In this chapter, you'll learn how to use SpiceLab's basic components to build circuits.

## Resistors

A resistor is a passive component that opposes current flow.

### Creating a Resistor

python
from spicelab.core.components import Resistor

# Method 1: Direct value (ohms)
r1 = Resistor("R1", 1000)  # 1kΩ

# Method 2: String with SI prefix (recommended)
r2 = Resistor("R2", "10k")  # 10kΩ
r3 = Resistor("R3", "1M")   # 1MΩ


### E-Series Values

For realistic simulations, use standard E12/E24 values:

python
from spicelab.library import R_E12

r4 = R_E12(4.7, 3)  # 4.7kΩ (E12 series)


### Exercise

Build a voltage divider with 10kΩ and 1kΩ resistors. Calculate Vout for Vin=5V.

<details>
<summary>Solution</summary>

python
circuit = Circuit("voltage_divider")
vin = VoltageSource("Vin", dc=5)
r1 = Resistor("R1", "10k")
r2 = Resistor("R2", "1k")

circuit.add(vin, r1, r2)
circuit.connect(vin.p, "in")
circuit.connect(r1.ports[0], "in")
circuit.connect(r1.ports[1], "out")
circuit.connect(r2.ports[0], "out")
circuit.connect(r2.ports[1], "0")

result = run_simulation(circuit, [AnalysisSpec("op", {})])
print(result["out"])  # Should be ~0.45V


</details>
```

### 2. Practical Examples (50+)

**Example Categories:**
```
examples/
├── 01_basic/
│   ├── rc_filter.py
│   ├── voltage_divider.py
│   ├── led_circuit.py
│   └── ...
├── 02_analog/
│   ├── opamp_amplifier.py
│   ├── instrumentation_amp.py
│   ├── active_filter.py
│   └── ...
├── 03_power/
│   ├── buck_converter.py
│   ├── ldo_regulator.py
│   ├── battery_charger.py
│   └── ...
├── 04_signal_processing/
│   ├── audio_equalizer.py
│   ├── adc_sampling.py
│   ├── pll_circuit.py
│   └── ...
├── 05_rf/
│   ├── antenna_matching.py
│   ├── mixer_circuit.py
│   ├── oscillator_vco.py
│   └── ...
├── 06_monte_carlo/
│   ├── tolerance_analysis.py
│   ├── yield_estimation.py
│   ├── worst_case.py
│   └── ...
└── 07_automation/
    ├── batch_simulations.py
    ├── parameter_sweep.py
    ├── optimization.py
    └── ...
```

#### Example Circuit
```python
# examples/02_analog/opamp_amplifier.py
"""
Non-Inverting OpAmp Amplifier
==============================

This example demonstrates a non-inverting amplifier using a TI LM358 OpAmp.

Circuit: Vin -> OpAmp -> Vout
Gain: 1 + R2/R1 = 1 + 10k/1k = 11 (20.8dB)
"""

from spicelab import Circuit, run_simulation
from spicelab.core.components import VoltageSource, Resistor
from spicelab.models import ModelCatalog
from spicelab.analysis import AnalysisSpec
import matplotlib.pyplot as plt

# Load TI LM358 OpAmp model
catalog = ModelCatalog()
lm358 = catalog.load("TI", "LM358")

# Build circuit
circuit = Circuit("non_inverting_amp")

# Input source (1V AC for frequency response)
vin = VoltageSource("Vin", dc=0, ac=1)

# Feedback network
r1 = Resistor("R1", "1k")   # To ground
r2 = Resistor("R2", "10k")  # Feedback

# OpAmp
opamp = lm358("U1")

# Power supplies
vcc = VoltageSource("Vcc", dc=15)
vee = VoltageSource("Vee", dc=-15)

# Add components
circuit.add(vin, r1, r2, opamp, vcc, vee)

# Connect circuit
circuit.connect(vin.p, "in")
circuit.connect(opamp.ports["in+"], "in")
circuit.connect(opamp.ports["in-"], "fb")
circuit.connect(opamp.ports["out"], "out")
circuit.connect(r1.ports[0], "fb")
circuit.connect(r1.ports[1], "0")
circuit.connect(r2.ports[0], "fb")
circuit.connect(r2.ports[1], "out")
circuit.connect(opamp.ports["vcc"], vcc.p)
circuit.connect(opamp.ports["vee"], vee.p)

# Run AC analysis (frequency response)
ac = AnalysisSpec("ac", {"fstart": "1", "fstop": "1e6", "npoints": 100})
result = run_simulation(circuit, [ac])

# Plot frequency response
freq = result["frequency"]
vout = result["out"]
gain_db = 20 * np.log10(np.abs(vout))

plt.figure(figsize=(10, 6))
plt.semilogx(freq, gain_db)
plt.grid(True)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain (dB)")
plt.title("Non-Inverting Amplifier Frequency Response")
plt.axhline(y=20.8, color='r', linestyle='--', label='Expected Gain (20.8dB)')
plt.legend()
plt.show()
```

### 3. Video Tutorials

**Video Series (5-10 videos):**
1. **Introduction to SpiceLab** (10 min)
   - What is SPICE?
   - Why SpiceLab vs LTspice/PySpice?
   - Installation and setup
2. **Building Your First Circuit** (12 min)
   - RC low-pass filter
   - Running simulations
   - Plotting results
3. **Working with Vendor Models** (8 min)
   - Finding models
   - Loading TI/ADI OpAmps
   - Realistic simulations
4. **Monte Carlo Analysis** (15 min)
   - Tolerance analysis
   - Statistical distributions
   - Yield estimation
5. **Advanced Tips & Tricks** (10 min)
   - Performance optimization
   - Debugging techniques
   - Best practices

**Platform:** YouTube (SpiceLab channel)

### 4. Complete API Reference

**Coverage:** 100% of public API

**Format:**
```markdown
# spicelab.core.components.Resistor

class **Resistor**(ref: str, resistance: float | str)

A resistor component.

**Parameters:**
- `ref` (str): Component reference (e.g., "R1")
- `resistance` (float | str): Resistance value in ohms or SI string (e.g., "1k", "10M")

**Examples:**

python
# Create 1kΩ resistor
r1 = Resistor("R1", 1000)
r2 = Resistor("R2", "1k")  # Equivalent

# E-series resistor
r3 = R_E12(4.7, 3)  # 4.7kΩ


**Attributes:**
- `ref`: str - Component reference
- `resistance`: float - Resistance value (ohms)
- `ports`: list[Port] - Connection ports [0, 1]

**Methods:**
- `spice_card()` → str: Generate SPICE netlist card
```

### 5. Migration Guides

#### PySpice Migration Guide
```markdown
# Migrating from PySpice

## Key Differences

| PySpice | SpiceLab | Notes |
|---------|----------|-------|
| `Circuit('name')` | `Circuit("name")` | Same |
| `circuit.R(1, 'n1', 'n2', 1e3)` | `Resistor("R1", "1k")` | More explicit |
| `circuit.V(1, 'n1', 'gnd', 5)` | `VoltageSource("V1", dc=5)` | Named parameters |
| `simulator.operating_point()` | `run_simulation(circuit, [AnalysisSpec("op", {})])` | Type-safe |

## Example: RC Filter

### PySpice
python
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

circuit = Circuit('RC Filter')
circuit.V('input', 'in', circuit.gnd, 5@u_V)
circuit.R(1, 'in', 'out', 1@u_kΩ)
circuit.C(1, 'out', circuit.gnd, 100@u_nF)

simulator = circuit.simulator()
analysis = simulator.transient(step_time=1@u_us, end_time=1@u_ms)


### SpiceLab
python
from spicelab import Circuit, run_simulation
from spicelab.core.components import VoltageSource, Resistor, Capacitor
from spicelab.analysis import AnalysisSpec

circuit = Circuit("RC Filter")
vin = VoltageSource("Vin", dc=5)
r1 = Resistor("R1", "1k")
c1 = Capacitor("C1", "100n")

circuit.add(vin, r1, c1)
circuit.connect(vin.p, "in")
circuit.connect(r1.ports[0], "in")
circuit.connect(r1.ports[1], "out")
circuit.connect(c1.ports[0], "out")
circuit.connect(c1.ports[1], "0")

tran = AnalysisSpec("tran", {"tstep": "1us", "tstop": "1ms"})
result = run_simulation(circuit, [tran])

```

### 6. Troubleshooting Cookbook

**20+ Common Issues:**
```markdown
# Troubleshooting Cookbook

## Convergence Failures

### Problem: "Convergence failed"

**Symptoms:** Simulation exits with convergence error

**Causes:**
1. Floating nodes (disconnected components)
2. Unrealistic component values
3. Stiff circuits (fast + slow time constants)

**Solutions:**
1. Run circuit validation: `circuit.validate()`
2. Add .OPTIONS RELTOL=1e-4 (relaxed tolerance)
3. Add initial conditions (.IC)
4. Check for shorts/opens

**Example:**
python
# Add relaxed tolerance
circuit.add_option("RELTOL", 1e-4)

# Or use interactive troubleshooter
from spicelab.troubleshooting import InteractiveTroubleshooter
troubleshooter = InteractiveTroubleshooter()
troubleshooter.diagnose_convergence_failure(circuit)

```

## Implementation Plan

### Week 1-2: Tutorial (Chapters 1-5)
- [ ] Write chapters 1-5
- [ ] Create code examples for each
- [ ] Add exercises with solutions
- [ ] Review and edit

### Week 3-4: Tutorial (Chapters 6-10) + Examples
- [ ] Write chapters 6-10
- [ ] Create 30+ practical examples
- [ ] Test all example code
- [ ] Add README for each category

### Week 5: Video Production
- [ ] Script 5-10 videos
- [ ] Record screencasts
- [ ] Edit videos
- [ ] Upload to YouTube
- [ ] Add to documentation

### Week 6: API Reference
- [ ] Generate API stubs
- [ ] Add examples to all classes/functions
- [ ] Cross-reference related APIs
- [ ] Review for 100% coverage

### Week 7: Migration Guides & Cookbook
- [ ] Write PySpice migration guide
- [ ] Write spicelib migration guide
- [ ] Create troubleshooting cookbook (20+ issues)
- [ ] Add performance tuning guide

### Week 8: Polish & SEO
- [ ] Review all documentation
- [ ] Optimize for SEO (keywords, structure)
- [ ] Add search functionality
- [ ] Deploy to docs site

## Success Metrics

### Content Volume
- [ ] **10+ tutorial chapters** (comprehensive)
- [ ] **50+ practical examples** (real-world circuits)
- [ ] **5-10 video tutorials** (5-15 min each)
- [ ] **100% API coverage** (all public methods documented)
- [ ] **2 migration guides** (PySpice, spicelib)
- [ ] **20+ troubleshooting recipes**

### Quality
- [ ] All code examples tested and working
- [ ] Tutorial progression tested with beginners
- [ ] Videos professionally edited
- [ ] SEO: Top 3 Google results for target keywords

### Adoption Metrics
- [ ] Docs page views: **10k+/month** (6 months post-launch)
- [ ] Video views: **5k+** total
- [ ] Reduced support questions: **50% reduction**
- [ ] User satisfaction: **>4.5/5**

## Dependencies

- M3 (component library for examples)
- M4 (templates for tutorial)

## References

- [Divio Documentation System](https://documentation.divio.com/)
- [Write the Docs](https://www.writethedocs.org/)
