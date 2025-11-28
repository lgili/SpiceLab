# Chapter 7: Advanced Components

This chapter covers using real-world components: vendor models, custom subcircuits, and behavioral models.

## Vendor Models

Real transistors and op-amps have SPICE models from manufacturers.

### Using the Component Library

```python
from spicelab.library import create_component, list_components

# List available components
diodes = list_components(category="diode")
print(diodes)  # ['1N4148', '1N4001', '1N5819', ...]

# Create component from library
d1 = create_component("diode.1n4148", "D1")

# Add to circuit
circuit.add(d1)
circuit.connect(d1.ports[0], anode_net)  # Anode
circuit.connect(d1.ports[1], cathode_net)  # Cathode
```

### Diodes

```python
from spicelab.library import create_component

# Signal diode
d1 = create_component("diode.1n4148", "D1")

# Rectifier diode
d2 = create_component("diode.1n4001", "D2")

# Schottky diode
d3 = create_component("diode.1n5819", "D3")

# Zener diode
d4 = create_component("diode.1n4733", "Dz")  # 5.1V Zener
```

### Transistors

```python
# NPN BJT
q1 = create_component("transistor.2n2222", "Q1")
# Ports: [collector, base, emitter]

# PNP BJT
q2 = create_component("transistor.2n2907", "Q2")

# N-channel MOSFET
m1 = create_component("transistor.2n7000", "M1")
# Ports: [drain, gate, source]

# P-channel MOSFET
m2 = create_component("transistor.bs250", "M2")
```

### Op-Amps

```python
# General purpose op-amp (5-terminal)
u1 = create_component("opamp.lm741", "U1")
# Ports: [in+, in-, out, vcc, vee]

# Rail-to-rail op-amp
u2 = create_component("opamp.opa340", "U2")

# Low noise op-amp
u3 = create_component("opamp.opa627", "U3")
```

## Custom Models

### Adding SPICE Model Definitions

```python
circuit = Circuit("with_model")

# Add model definition
circuit.add_directive(".model MyDiode D(Is=1e-14 N=1.05 Rs=0.5)")

# Use the model
from spicelab.core.components import Diode
d1 = Diode("1", model="MyDiode")
circuit.add(d1)
```

### Including External Model Files

```python
# Include model library file
circuit.add_directive(".include /path/to/models.lib")

# Or include specific model
circuit.add_directive(".lib /path/to/vendor.lib TL072")
```

## Behavioral Models

Model behavior without physical implementation.

### Behavioral Voltage Sources

```python
from spicelab.core.components import BehavioralVoltage

# Voltage controlled by expression
b1 = BehavioralVoltage("1", expr="V(in)*2")  # 2x gain

# More complex expression
b2 = BehavioralVoltage("2", expr="V(a) - V(b)")  # Differential

# With functions
b3 = BehavioralVoltage("3", expr="sin(2*pi*1000*time)")  # 1kHz sine
```

### Behavioral Current Sources

```python
from spicelab.core.components import BehavioralCurrent

# Voltage-controlled current source (transconductance)
b1 = BehavioralCurrent("1", expr="V(in)*0.001")  # gm = 1mS

# Current mirror
b2 = BehavioralCurrent("2", expr="I(Vref)*2")  # 2x mirror ratio
```

### Ideal Components

```python
from spicelab.library.behavioral import (
    create_ideal_opamp,
    create_ideal_switch,
    create_ideal_diode,
)

# Ideal op-amp (infinite gain, bandwidth)
opamp = create_ideal_opamp("U1")

# Ideal switch (on/off based on control voltage)
sw = create_ideal_switch("S1", control_net="ctrl", threshold=2.5)

# Ideal diode (no forward drop)
d_ideal = create_ideal_diode("D1")
```

## Example: BJT Amplifier with Real Model

```python
"""Common-emitter amplifier with 2N2222"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Capacitor, Vdc, Vac
from spicelab.core.net import GND, Net
from spicelab.library import create_component

circuit = Circuit("ce_amplifier")

# Power supply
vcc = Vdc("CC", 12.0)

# Input AC source (1mV amplitude for small-signal)
vin = Vac("IN", ac_mag=0.001)

# BJT from library
q1 = create_component("transistor.2n2222", "Q1")

# Bias resistors
r1 = Resistor("1", "47k")   # Base voltage divider
r2 = Resistor("2", "10k")   # Base voltage divider
rc = Resistor("C", "2.2k")  # Collector resistor
re = Resistor("E", "1k")    # Emitter resistor

# Coupling capacitors
cin = Capacitor("IN", "10u")
cout = Capacitor("OUT", "10u")
ce = Capacitor("E", "100u")  # Emitter bypass

circuit.add(vcc, vin, q1, r1, r2, rc, re, cin, cout, ce)

# Create nets
vcc_net = Net("vcc")
base = Net("base")
collector = Net("collector")
emitter = Net("emitter")
input_net = Net("in")
output_net = Net("out")

# Power connections
circuit.connect(vcc.ports[0], vcc_net)
circuit.connect(vcc.ports[1], GND)

# Bias network
circuit.connect(r1.ports[0], vcc_net)
circuit.connect(r1.ports[1], base)
circuit.connect(r2.ports[0], base)
circuit.connect(r2.ports[1], GND)

# Transistor
circuit.connect(q1.ports[0], collector)  # Collector
circuit.connect(q1.ports[1], base)       # Base
circuit.connect(q1.ports[2], emitter)    # Emitter

# Load resistors
circuit.connect(rc.ports[0], vcc_net)
circuit.connect(rc.ports[1], collector)
circuit.connect(re.ports[0], emitter)
circuit.connect(re.ports[1], GND)

# Bypass capacitor
circuit.connect(ce.ports[0], emitter)
circuit.connect(ce.ports[1], GND)

# Input coupling
circuit.connect(vin.ports[0], input_net)
circuit.connect(vin.ports[1], GND)
circuit.connect(cin.ports[0], input_net)
circuit.connect(cin.ports[1], base)

# Output coupling
circuit.connect(cout.ports[0], collector)
circuit.connect(cout.ports[1], output_net)

print(circuit.build_netlist())
```

## Subcircuits

Create reusable circuit blocks.

### Defining a Subcircuit

```python
# Define subcircuit using SPICE syntax
circuit.add_directive("""
.subckt voltage_regulator in out gnd
R1 in adj 240
R2 adj gnd 1.2k
Xreg in adj out lm317
.ends voltage_regulator
""")

# Use the subcircuit
circuit.add_directive("X1 vin vout 0 voltage_regulator")
```

### Creating Subcircuits Programmatically

```python
def create_diff_pair_subcircuit(name: str, r_load: float = 10_000) -> str:
    """Generate differential pair subcircuit definition."""
    return f"""
.subckt {name} vinp vinn vout vcc vee
Q1 c1 vinp ve 2n2222
Q2 c2 vinn ve 2n2222
Rc1 vcc c1 {r_load}
Rc2 vcc c2 {r_load}
Ibias ve vee 1m
.ends {name}
"""

# Add to circuit
circuit.add_directive(create_diff_pair_subcircuit("diffpair"))
circuit.add_directive("X1 inp inn out vcc vee diffpair")
```

## E-Series Component Selection

Select standard values from E-series:

```python
from spicelab.utils.e_series import nearest_e24, nearest_e96, e_series_range

# Find nearest E24 value to calculated result
r_calc = 4.7e3 * 2.2  # 10.34k
r_e24 = nearest_e24(r_calc)
print(f"Calculated: {r_calc}, E24: {r_e24}")  # E24: 10000

# Find nearest E96 value (more precise)
r_e96 = nearest_e96(r_calc)
print(f"E96: {r_e96}")  # E96: 10200

# Get range of E24 values
values = e_series_range("e24", 1000, 10000)
print(values)  # [1000, 1100, 1200, ..., 10000]
```

## Exercises

### Exercise 7.1: LED Driver
Create an LED circuit using a real diode model.
Calculate resistor for 20mA at 5V supply (LED Vf ≈ 2V).

### Exercise 7.2: BJT Switch
Build a transistor switch using 2N2222.
Drive an LED (or resistive load) from a 3.3V logic signal.

### Exercise 7.3: Op-Amp Filter
Build a Sallen-Key filter using LM741.
Compare ideal vs real op-amp frequency response.

### Exercise 7.4: Voltage Regulator
Create a simple Zener regulator using 1N4733 (5.1V Zener).
Analyze load regulation with varying load current.

### Exercise 7.5: Custom Model
Create a custom diode model with:
- Is = 1e-14 A
- N = 1.8 (ideality factor)
- Rs = 10 ohms

Simulate and compare with 1N4148.

## Component Library Reference

| Category | Examples | Ports |
|----------|----------|-------|
| Diode | 1N4148, 1N4001, 1N5819 | [anode, cathode] |
| NPN BJT | 2N2222, 2N3904 | [C, B, E] |
| PNP BJT | 2N2907, 2N3906 | [C, B, E] |
| N-MOS | 2N7000, IRF510 | [D, G, S] |
| P-MOS | BS250, IRF9510 | [D, G, S] |
| Op-Amp | LM741, TL072, OPA340 | [+, -, out, vcc, vee] |
| Zener | 1N47xx series | [anode, cathode] |

## Next Steps

- [Chapter 8: Performance](08_performance.md) - Optimization and caching
- [Chapter 9: Integration](09_integration.md) - CI/CD and automation

---

**See also:**
- [Component Catalog](../component_catalog.md) - Full component list
- [Behavioral Models Guide](../behavioral_models.md) - Behavioral modeling details
