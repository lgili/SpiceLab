# Chapter 1: Getting Started with SpiceLab

Welcome to SpiceLab! This chapter will guide you through installation and building your first circuit.

## What is SpiceLab?

SpiceLab is a Python library for circuit simulation that provides:
- **Pythonic API**: Build circuits using intuitive Python code
- **Multiple Engines**: Support for ngspice, LTspice, and Xyce
- **Rich Results**: xarray-based datasets for easy analysis
- **Validation**: Automatic circuit checking before simulation
- **Templates**: Pre-built circuit patterns for common topologies

## Installation

### Prerequisites

- Python 3.10 or later
- A SPICE engine (ngspice recommended)

### Install SpiceLab

```bash
pip install spicelab
```

### Install a SPICE Engine

**macOS (Homebrew):**
```bash
brew install ngspice
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ngspice
```

**Windows (Chocolatey):**
```bash
choco install ngspice
```

### Verify Installation

```bash
spicelab doctor
```

This checks your Python environment and SPICE engines.

## Your First Circuit: Voltage Divider

Let's build a simple voltage divider - two resistors that divide an input voltage.

### Circuit Diagram

```
     Vin (10V)
        │
       ─┴─
      │   │ V1
       ─┬─
        │
       ─┴─
      │   │ R1 (10k)
       ─┬─
        ├──── Vout (5V)
       ─┴─
      │   │ R2 (10k)
       ─┬─
        │
       ─┴─
       GND
```

With equal resistors, Vout = Vin × R2/(R1+R2) = 10V × 0.5 = 5V

### Step 1: Import Required Modules

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
```

**Key imports:**
- `Circuit`: The container for your circuit
- `Resistor`, `Vdc`: Component classes
- `GND`: The ground reference (SPICE node 0)
- `Net`: Named connection points (wires)

### Step 2: Create the Circuit

```python
circuit = Circuit("voltage_divider")
```

Every circuit needs a name. This becomes the title in the SPICE netlist.

### Step 3: Create Components

```python
# 10V DC voltage source
v1 = Vdc("1", 10.0)

# Two 10k resistors
r1 = Resistor("1", resistance=10_000)
r2 = Resistor("2", resistance=10_000)

# Add components to circuit
circuit.add(v1, r1, r2)
```

**Component naming:**
- Each component needs a unique identifier
- The full SPICE name is prefix + identifier (e.g., "V1", "R1", "R2")
- Values can be numbers (10000) or strings ("10k")

### Step 4: Connect Components

```python
# Create named nets
vin = Net("vin")
vout = Net("vout")

# Connect voltage source
circuit.connect(v1.ports[0], vin)   # V1+ to vin
circuit.connect(v1.ports[1], GND)   # V1- to ground

# Connect R1 between vin and vout
circuit.connect(r1.ports[0], vin)
circuit.connect(r1.ports[1], vout)

# Connect R2 between vout and ground
circuit.connect(r2.ports[0], vout)
circuit.connect(r2.ports[1], GND)
```

**Understanding ports:**
- `ports[0]` and `ports[1]` are the component terminals
- For voltage sources: `ports[0]` is positive (+), `ports[1]` is negative (-)
- For resistors: either terminal can connect to either net

### Step 5: View the Netlist

```python
print(circuit.build_netlist())
```

Output:
```spice
* voltage_divider
V1 vin 0 10.0
R1 vin vout 10000.0
R2 vout 0 10000.0
.end
```

### Step 6: Validate the Circuit

```python
result = circuit.validate()
print(f"Valid: {result.is_valid}")

if result.warnings:
    for warn in result.warnings:
        print(f"Warning: {warn.message}")
```

Validation checks for:
- Missing ground reference
- Floating nodes (disconnected)
- Short circuits (voltage sources in parallel)
- Unusual component values

## Running Simulations

To simulate, you need a SPICE engine:

```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

# Define analysis: DC operating point
op_analysis = AnalysisSpec("op", {})

# Run simulation
handle = run_simulation(circuit, [op_analysis], engine="ngspice")

# Get results
ds = handle.dataset()
print(ds)
```

We'll cover simulations in detail in Chapter 4.

## Complete Example

Here's the full code:

```python
"""Tutorial 01: Voltage Divider"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net

# Create circuit
circuit = Circuit("voltage_divider")

# Create components
v1 = Vdc("1", 10.0)
r1 = Resistor("1", resistance=10_000)
r2 = Resistor("2", resistance=10_000)
circuit.add(v1, r1, r2)

# Create nets
vin = Net("vin")
vout = Net("vout")

# Connect components
circuit.connect(v1.ports[0], vin)
circuit.connect(v1.ports[1], GND)
circuit.connect(r1.ports[0], vin)
circuit.connect(r1.ports[1], vout)
circuit.connect(r2.ports[0], vout)
circuit.connect(r2.ports[1], GND)

# Validate
result = circuit.validate()
print(f"Valid: {result.is_valid}")

# Show netlist
print(circuit.build_netlist())
```

Run the example:
```bash
python examples/tutorials/01_first_circuit.py
```

## Exercises

### Exercise 1.1: 3:1 Voltage Divider
Modify the voltage divider to output 2.5V from 10V input.
- Hint: Use R1=30k, R2=10k

### Exercise 1.2: LED Current Limiter
Create a circuit with:
- 5V source
- 330 ohm resistor
- Calculate the current (I = V/R)

### Exercise 1.3: Series Resistors
Create three resistors in series (1k, 2k, 3k) and verify the total resistance = 6k.

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| `Circuit` | Container for components and connections |
| `Component` | Electronic part (resistor, source, etc.) |
| `Net` | Named connection point (wire) |
| `GND` | Ground reference (SPICE node 0) |
| `ports` | Component terminals |
| `connect()` | Links port to net |
| `validate()` | Checks circuit for errors |
| `build_netlist()` | Generates SPICE netlist |

## Next Steps

- [Chapter 2: Basic Components](02_basic_components.md) - Learn about all component types
- [Chapter 3: Building Circuits](03_building_circuits.md) - Advanced circuit construction

---

**See also:**
- [Component Catalog](../component_catalog.md)
- [Troubleshooting Guide](../troubleshooting_guide.md)
