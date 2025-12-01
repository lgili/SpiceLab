# Chapter 3: Building Circuits

This chapter covers advanced techniques for building circuits: programmatic construction, subcircuits, and using templates.

## Circuit Construction Patterns

### Basic Pattern: Add and Connect

The fundamental pattern for building circuits:

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net

# 1. Create circuit
circuit = Circuit("my_circuit")

# 2. Create components
v1 = Vdc("1", 5.0)
r1 = Resistor("1", "10k")

# 3. Add components to circuit
circuit.add(v1, r1)

# 4. Create nets
vcc = Net("vcc")

# 5. Connect ports to nets
circuit.connect(v1.ports[0], vcc)
circuit.connect(v1.ports[1], GND)
circuit.connect(r1.ports[0], vcc)
circuit.connect(r1.ports[1], GND)
```

### Bulk Add

Add multiple components at once:

```python
# Add multiple components in one call
circuit.add(v1, r1, r2, r3, c1)

# Or from a list
components = [Resistor(str(i), "1k") for i in range(10)]
circuit.add(*components)
```

### Net Reuse

Reuse nets across multiple connections:

```python
# Create shared nets once
vcc = Net("vcc")
signal = Net("signal")
out = Net("out")

# Reuse for multiple connections
circuit.connect(v1.ports[0], vcc)
circuit.connect(r1.ports[0], vcc)  # Same net as v1
circuit.connect(r2.ports[0], vcc)  # Same net as v1 and r1
```

## Programmatic Circuit Building

### Loop-Based Construction

Build repetitive structures with loops:

```python
"""Create a resistor ladder (R-2R DAC)"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import GND, Net

circuit = Circuit("r2r_ladder")
bits = 4

# Create resistors programmatically
r_resistors = []  # Horizontal R resistors
r2_resistors = []  # Vertical 2R resistors

for i in range(bits):
    r_resistors.append(Resistor(f"R{i}", "10k"))
    r2_resistors.append(Resistor(f"2R{i}", "20k"))

# Add all resistors
circuit.add(*r_resistors, *r2_resistors)

# Create nets for each bit
bit_nets = [Net(f"b{i}") for i in range(bits)]
node_nets = [Net(f"n{i}") for i in range(bits)]
output = Net("out")

# Connect the ladder structure
for i in range(bits):
    # Vertical 2R to bit input
    circuit.connect(r2_resistors[i].ports[0], bit_nets[i])
    circuit.connect(r2_resistors[i].ports[1], node_nets[i])

    # Horizontal R connections
    if i == 0:
        circuit.connect(r_resistors[i].ports[0], node_nets[i])
        circuit.connect(r_resistors[i].ports[1], GND)
    else:
        circuit.connect(r_resistors[i].ports[0], node_nets[i])
        circuit.connect(r_resistors[i].ports[1], node_nets[i-1])

# Connect last node to output
circuit.connect(node_nets[-1], output)

print(circuit.build_netlist())
```

### Factory Functions

Create reusable circuit builders:

```python
def create_voltage_divider(name: str, vin: float, ratio: float,
                           total_r: float = 10_000) -> Circuit:
    """Create a voltage divider circuit.

    Args:
        name: Circuit name
        vin: Input voltage
        ratio: Output/Input ratio (0 to 1)
        total_r: Total resistance R1+R2

    Returns:
        Circuit with voltage divider
    """
    circuit = Circuit(name)

    # Calculate resistor values
    r2 = total_r * ratio
    r1 = total_r - r2

    # Create components
    v1 = Vdc("1", vin)
    r_top = Resistor("1", r1)
    r_bot = Resistor("2", r2)

    circuit.add(v1, r_top, r_bot)

    # Create nets
    input_net = Net("vin")
    output_net = Net("vout")

    # Connect
    circuit.connect(v1.ports[0], input_net)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r_top.ports[0], input_net)
    circuit.connect(r_top.ports[1], output_net)
    circuit.connect(r_bot.ports[0], output_net)
    circuit.connect(r_bot.ports[1], GND)

    return circuit

# Use the factory
div_3v3 = create_voltage_divider("div_3v3", vin=5.0, ratio=0.66)
div_1v8 = create_voltage_divider("div_1v8", vin=3.3, ratio=0.55)
```

### Parameterized Components

Create components with calculated values:

```python
import math

def create_rc_filter(fc: float, impedance: float = 10_000) -> tuple:
    """Create RC filter components for given cutoff frequency.

    Args:
        fc: Cutoff frequency in Hz
        impedance: Filter impedance (R value)

    Returns:
        Tuple of (resistor, capacitor) components
    """
    # Calculate C from fc = 1/(2*pi*R*C)
    c_value = 1 / (2 * math.pi * impedance * fc)

    r = Resistor("FILT", impedance)
    c = Capacitor("FILT", c_value)

    return r, c

# Create filter components for 1kHz cutoff
r_filt, c_filt = create_rc_filter(fc=1000)
print(f"R = {r_filt.resistance}, C = {c_filt.capacitance}")
```

## Using Circuit Templates

SpiceLab provides pre-built templates for common circuits.

### Filter Templates

```python
from spicelab.templates import (
    rc_lowpass,
    rc_highpass,
    butterworth_lowpass,
    sallen_key_lowpass,
)

# Simple RC lowpass (1kHz cutoff)
lpf = rc_lowpass(fc=1000, impedance=10_000)
print(lpf.build_netlist())

# Second-order Butterworth
result = butterworth_lowpass(fc=1000, order=2)
circuit = result.circuit
print(f"Cutoff: {result.cutoff_frequency} Hz")
print(f"Q factor: {result.q_factor}")

# Sallen-Key with custom Q
result = sallen_key_lowpass(fc=1000, q=0.707)  # Butterworth Q
circuit = result.circuit
```

### Amplifier Templates

```python
from spicelab.templates import (
    inverting_amplifier,
    non_inverting_amplifier,
    voltage_follower,
    differential_amplifier,
)

# Inverting amp with gain of -10
inv_amp = inverting_amplifier(gain=10, input_impedance=10_000)

# Non-inverting amp with gain of 11
noninv_amp = non_inverting_amplifier(gain=11, feedback_resistance=10_000)

# Unity-gain buffer
buffer = voltage_follower()

# Differential amp
diff_result = differential_amplifier(gain=10)
circuit = diff_result.circuit
```

### Power Templates

```python
from spicelab.templates import voltage_divider

# 50% voltage divider
div = voltage_divider(ratio=0.5, total_resistance=10_000)
```

### Template Results

Many templates return a result object with additional information:

```python
from spicelab.templates import butterworth_lowpass

result = butterworth_lowpass(fc=1000, order=4)

# Access the circuit
circuit = result.circuit

# Access components by name
components = result.components
print(components.keys())  # dict_keys(['R1', 'C1', 'R2', 'C2', ...])

# Access design parameters
print(f"Cutoff frequency: {result.cutoff_frequency}")
print(f"Q factor: {result.q_factor}")
```

## Combining Circuits

### Manual Combination

Connect outputs of one circuit to inputs of another:

```python
# Create input filter
input_filter = rc_lowpass(fc=10_000, impedance=1000)

# Create main circuit
main = Circuit("signal_chain")

# Add filter components to main circuit manually
# (Templates return simple circuits, extract components)
```

### Hierarchical Design

Build complex circuits from smaller blocks:

```python
def create_two_stage_filter(fc1: float, fc2: float) -> Circuit:
    """Create cascaded RC filter."""
    circuit = Circuit("two_stage_filter")

    # First stage
    r1 = Resistor("1", "10k")
    c1 = Capacitor("1", 1 / (2 * math.pi * 10_000 * fc1))

    # Second stage
    r2 = Resistor("2", "10k")
    c2 = Capacitor("2", 1 / (2 * math.pi * 10_000 * fc2))

    circuit.add(r1, c1, r2, c2)

    # Nets
    input_net = Net("in")
    mid = Net("mid")
    output = Net("out")

    # First stage connections
    circuit.connect(r1.ports[0], input_net)
    circuit.connect(r1.ports[1], mid)
    circuit.connect(c1.ports[0], mid)
    circuit.connect(c1.ports[1], GND)

    # Second stage connections
    circuit.connect(r2.ports[0], mid)
    circuit.connect(r2.ports[1], output)
    circuit.connect(c2.ports[0], output)
    circuit.connect(c2.ports[1], GND)

    return circuit

# Create filter with different corner frequencies
filter_2stage = create_two_stage_filter(fc1=1000, fc2=1000)
```

## Adding SPICE Directives

Add analysis commands and options directly to the circuit:

```python
circuit = Circuit("with_directives")

# Add components...

# Add analysis directive
circuit.add_directive(".tran 1u 10m")

# Add probe directive
circuit.add_directive(".probe v(out)")

# Add simulation options
circuit.add_directive(".options reltol=0.001")

# Add initial conditions
circuit.add_directive(".ic V(out)=0")

# Show netlist with directives
print(circuit.build_netlist())
```

## Circuit Validation

Always validate before simulation:

```python
result = circuit.validate()

if not result.is_valid:
    print("Errors:")
    for error in result.errors:
        print(f"  {error.message}")

if result.warnings:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  [{warning.severity}] {warning.message}")
        if warning.suggestion:
            print(f"    Suggestion: {warning.suggestion}")
```

### Strict Mode

Treat warnings as errors:

```python
result = circuit.validate(strict=True)
# Now warnings appear in result.errors
```

## Exercises

### Exercise 3.1: RC Filter Chain
Create a function that generates an N-stage RC lowpass filter where all stages have the same cutoff frequency.

### Exercise 3.2: Resistor Network
Create a circuit with 5 resistors in a star configuration (all connected to a common center node).

### Exercise 3.3: Template Modification
Use the `voltage_divider` template, then add a buffer capacitor (100nF) across the output.

### Exercise 3.4: Parameterized Amplifier
Create a function `create_gain_stage(gain: float)` that returns an inverting amplifier circuit with the specified gain.

### Exercise 3.5: Circuit with Directives
Build an RC circuit and add directives for:
- Transient analysis (10ms)
- Probing the output voltage
- Setting initial capacitor voltage to 0V

## Best Practices

1. **Use meaningful names**: Name nets after their function (`vcc`, `signal_in`, `output`)
2. **Validate early**: Call `validate()` before simulation
3. **Use templates**: Start with templates for common topologies
4. **Factor out common patterns**: Create factory functions for reusable circuits
5. **Document calculations**: Include comments showing design equations

## Next Steps

- [Chapter 4: Analysis Types](04_analysis_types.md) - Running different simulations
- [Chapter 5: Working with Results](05_results.md) - Processing simulation output

---

**See also:**
- [Template Catalog](../template_catalog.md) - All available templates
- [Component Catalog](../component_catalog.md) - Component reference
