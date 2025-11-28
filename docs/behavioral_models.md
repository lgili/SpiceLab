# SpiceLab Behavioral Models Guide

This guide explains the behavioral models available in SpiceLab and how to use them for system-level circuit simulation.

## Overview

Behavioral models provide idealized component behavior that simplifies simulation while capturing essential circuit function. They are useful for:

- **System-level design** - Focus on topology without parameter tuning
- **Fast simulation** - Simplified models converge quickly
- **Educational purposes** - Understand fundamental behavior
- **Initial prototyping** - Verify concepts before detailed modeling

## Available Behavioral Models

### Ideal Diodes

Ideal diodes provide perfect rectification without the complexity of real diode models.

#### `behavioral.d_ideal` - Perfect Diode

```python
from spicelab.library import create_component

d1 = create_component("behavioral.d_ideal", "D1")
```

**Characteristics:**
- Forward voltage drop: 0V
- Reverse leakage: 0A
- No reverse breakdown

**Use cases:** Ideal rectifiers, OR gates, protection circuits

#### `behavioral.d_ideal_schottky` - Ideal Schottky

```python
d1 = create_component("behavioral.d_ideal_schottky", "D1")
```

**Characteristics:**
- Forward voltage drop: 0.3V (typical Schottky Vf)
- Reverse leakage: 0A
- No reverse breakdown

**Use cases:** Low-loss rectifiers, freewheeling diodes

#### `behavioral.d_ideal_z5v1` - Ideal Zener (5.1V)

```python
d1 = create_component("behavioral.d_ideal_z5v1", "D1")
```

**Characteristics:**
- Forward voltage drop: 0.7V
- Zener breakdown: 5.1V (sharp knee)
- Ideal regulation in breakdown

**Use cases:** Voltage clamping, simple regulators, protection

### Ideal Switches

Ideal switches provide near-perfect on/off behavior controlled by voltage.

#### `behavioral.sw_ideal` - Perfect Switch

```python
sw = create_component("behavioral.sw_ideal", "S1")
```

**Characteristics:**
- On-resistance: 1mΩ (near short)
- Off-resistance: 1GΩ (near open)
- Control: Voltage threshold (Von=1V, Voff=0V)

**Ports:**
```
sw.ports[0]  # Terminal 1
sw.ports[1]  # Terminal 2
sw.ports[2]  # Control+ (positive control input)
sw.ports[3]  # Control- (negative control input)
```

**Usage Example:**
```python
from spicelab.core.circuit import Circuit
from spicelab.core.net import Net, GND
from spicelab.core.components import Vdc, Vpulse, Resistor
from spicelab.library import create_component

circuit = Circuit("switch_test")

# Power supply
vdd = Vdc("VDD", 5)
circuit.add(vdd)
circuit.connect(vdd.ports[0], Net("vdd"))
circuit.connect(vdd.ports[1], GND)

# Control signal
vctrl = Vpulse("VCTRL", v1=0, v2=3, td="1m", tr="1u", tf="1u", pw="2m", per="4m")
circuit.add(vctrl)
circuit.connect(vctrl.ports[0], Net("ctrl"))
circuit.connect(vctrl.ports[1], GND)

# Ideal switch with load
sw = create_component("behavioral.sw_ideal", "S1")
r_load = Resistor("RLOAD", "1k")

circuit.add(sw, r_load)
circuit.connect(sw.ports[0], Net("vdd"))      # Input from supply
circuit.connect(sw.ports[1], Net("out"))       # Output to load
circuit.connect(sw.ports[2], Net("ctrl"))      # Control+
circuit.connect(sw.ports[3], GND)              # Control-
circuit.connect(r_load.ports[0], Net("out"))
circuit.connect(r_load.ports[1], GND)
```

#### `behavioral.sw_ideal_fast` - Fast Switch

```python
sw = create_component("behavioral.sw_ideal_fast", "S1")
```

Same as `sw_ideal` but with faster transition times (1ns rise/fall).

**Use cases:** High-frequency switching, PWM simulation

#### `behavioral.sw_relay` - Relay Model

```python
sw = create_component("behavioral.sw_relay", "S1")
```

**Characteristics:**
- On-resistance: 100mΩ (typical relay contact)
- Off-resistance: 10MΩ
- Slower transition (relay-like bounce modeled)

**Use cases:** Relay-based circuits, contact simulation

### Ideal Transformers

Ideal transformers provide perfect magnetic coupling with specified turns ratios.

#### `behavioral.xfmr_1_1` - Isolation Transformer

```python
xfmr = create_component("behavioral.xfmr_1_1", "T1")
```

**Characteristics:**
- Turns ratio: 1:1
- Perfect coupling (k=1)
- No losses

**Ports:**
```
xfmr.ports[0]  # Primary +
xfmr.ports[1]  # Primary -
xfmr.ports[2]  # Secondary +
xfmr.ports[3]  # Secondary -
```

#### `behavioral.xfmr_2_1` - Step-Down Transformer

```python
xfmr = create_component("behavioral.xfmr_2_1", "T1")
```

**Characteristics:**
- Turns ratio: 2:1 (step-down)
- Vout = Vin / 2

#### `behavioral.xfmr_1_2` - Step-Up Transformer

```python
xfmr = create_component("behavioral.xfmr_1_2", "T1")
```

**Characteristics:**
- Turns ratio: 1:2 (step-up)
- Vout = Vin * 2

**Usage Example:**
```python
from spicelab.core.circuit import Circuit
from spicelab.core.net import Net, GND
from spicelab.core.components import Vac, Resistor
from spicelab.library import create_component

circuit = Circuit("transformer_test")

# AC source
vac = Vac("VAC", amplitude=10, freq=60)
circuit.add(vac)
circuit.connect(vac.ports[0], Net("pri_p"))
circuit.connect(vac.ports[1], GND)

# Step-down transformer
xfmr = create_component("behavioral.xfmr_2_1", "T1")
circuit.add(xfmr)
circuit.connect(xfmr.ports[0], Net("pri_p"))   # Primary +
circuit.connect(xfmr.ports[1], GND)             # Primary -
circuit.connect(xfmr.ports[2], Net("sec_p"))   # Secondary +
circuit.connect(xfmr.ports[3], Net("sec_n"))   # Secondary -

# Load on secondary
r_load = Resistor("RLOAD", "100")
circuit.add(r_load)
circuit.connect(r_load.ports[0], Net("sec_p"))
circuit.connect(r_load.ports[1], Net("sec_n"))
```

### Ideal Controlled Sources

#### `behavioral.vcvs_unity` - Voltage Buffer

```python
buf = create_component("behavioral.vcvs_unity", "E1")
```

**Characteristics:**
- Voltage gain: 1 (unity)
- Input impedance: Infinite
- Output impedance: 0

**Use cases:** Voltage followers, isolation buffers

#### `behavioral.cccs_unity` - Current Mirror

```python
mirror = create_component("behavioral.cccs_unity", "F1")
```

**Characteristics:**
- Current gain: 1 (unity)
- Perfect current replication

**Use cases:** Current mirrors, current sensing

## SPICE Implementation

Behavioral models use SPICE behavioral modeling features:

### Ideal Diode Implementation

```spice
* Ideal diode using B-source
.subckt D_IDEAL anode cathode
B1 anode cathode I=if(V(anode,cathode)>0, V(anode,cathode)/1m, 0)
.ends
```

### Ideal Switch Implementation

```spice
* Ideal voltage-controlled switch
.model SW_IDEAL VSWITCH(RON=1m ROFF=1G VON=1 VOFF=0)
S1 n1 n2 ctrl+ ctrl- SW_IDEAL
```

### Ideal Transformer Implementation

```spice
* Ideal 2:1 transformer using coupled inductors
.subckt XFMR_2_1 pri_p pri_n sec_p sec_n
L1 pri_p pri_n 1
L2 sec_p sec_n 0.25
K1 L1 L2 1
.ends
```

## Best Practices

### When to Use Behavioral Models

1. **Early design phase** - Verify topology before tuning
2. **System simulation** - Model subsystems simply
3. **Teaching/learning** - Focus on concepts
4. **Fast iteration** - Quick what-if analysis

### When to Use Real Models

1. **Final verification** - Match production components
2. **Worst-case analysis** - Real tolerances matter
3. **Timing analysis** - Need accurate delays
4. **Thermal analysis** - Real power dissipation

### Mixing Models

You can mix behavioral and real models:

```python
# Behavioral switch for control
sw = create_component("behavioral.sw_ideal", "S1")

# Real MOSFET for power stage
mosfet = create_component("transistor.irf540n", "Q1")

# Real diode for rectification
diode = create_component("diode.1n5819", "D1")
```

## Comparison: Behavioral vs Real

| Aspect | Behavioral | Real Model |
|--------|------------|------------|
| Convergence | Fast | May need tweaking |
| Accuracy | Approximate | Realistic |
| Parameters | Few/none | Many |
| Temperature | Ignored | Modeled |
| Noise | None | Included |
| Use case | Concept | Production |

## Creating Custom Behavioral Models

### Custom Ideal Diode

```python
from spicelab.library import register_component
from spicelab.library.behavioral import IdealDiode

def my_diode_factory(ref: str, *, vfwd: float = 0.5) -> IdealDiode:
    return IdealDiode(ref, vfwd=vfwd, vrev=float("inf"))

register_component(
    "behavioral.d_custom_0v5",
    my_diode_factory,
    category="behavioral",
    metadata={"description": "Custom ideal diode, Vf=0.5V"},
)
```

### Custom Transformer Ratio

```python
from spicelab.library.behavioral import IdealTransformer

def my_xfmr_factory(ref: str, *, ratio: float = 3.0) -> IdealTransformer:
    return IdealTransformer(ref, turns_ratio=ratio)

register_component(
    "behavioral.xfmr_3_1",
    my_xfmr_factory,
    category="behavioral",
    metadata={"description": "3:1 step-down transformer"},
)
```

---

**Last Updated:** 2025-11-27

**See Also:**
- [Component Catalog](component_catalog.md) - Full component reference
- [Circuit Building Guide](../README.md) - Getting started with SpiceLab
