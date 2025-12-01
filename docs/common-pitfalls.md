# Common Pitfalls and Solutions

This guide covers common mistakes when using SpiceLab and how to avoid them.

## Circuit Building

### 1. Forgetting Ground Connection

**Problem**: Circuit validation fails with "No ground reference"

```python
# Wrong - no ground connection
circuit = Circuit("test")
R1 = Resistor(ref="1", resistance=1000)
circuit.add(R1)
vin = Net("vin")
vout = Net("vout")
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], vout)  # No ground!
```

**Solution**: Always connect at least one node to GND

```python
from spicelab.core.net import GND

circuit.connect(R1.ports[1], GND)  # Connect to ground
```

### 2. Floating Nodes

**Problem**: Component terminals not fully connected

```python
# Wrong - capacitor has floating terminal
circuit.add(R1, C1)
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], vout)
circuit.connect(C1.ports[0], vout)
# C1.ports[1] is floating!
```

**Solution**: Ensure all ports are connected

```python
circuit.connect(C1.ports[1], GND)
```

**Tip**: Use `circuit.validate()` to catch floating nodes before simulation.

### 3. Parallel Voltage Sources

**Problem**: Two voltage sources connected in parallel (short circuit)

```python
# Wrong - voltage source loop
V1 = Vdc(ref="1", value=5)
V2 = Vdc(ref="2", value=5)
circuit.add(V1, V2)
# Connecting both + terminals together and both - terminals together
```

**Solution**: Remove one source or add series resistance

```python
# Add small series resistance if needed
R_series = Resistor(ref="s", resistance=0.001)  # 1mΩ
```

### 4. Wrong Port Order

**Problem**: Connecting component ports in wrong order

```python
# Common confusion - which port is positive?
V1 = Vdc(ref="1", value=5)
circuit.connect(V1.ports[0], GND)  # Port 0 is positive!
circuit.connect(V1.ports[1], vin)  # Port 1 is negative
```

**Solution**: Check port conventions or use the help system

```python
from spicelab.help import get_help

help_obj = get_help(V1)
print(help_obj.full())  # Shows port conventions
```

**Convention**:
- Voltage sources: port[0] = positive (+), port[1] = negative (-)
- Resistors/Capacitors/Inductors: port[0] and port[1] are interchangeable

## Values and Units

### 5. Wrong Unit Prefix

**Problem**: Confusing µ (micro) with m (milli)

```python
# Wrong - 1mF is 0.001F, very large!
C1 = Capacitor(ref="1", capacitance=1e-3)  # Did you mean 1µF?
```

**Solution**: Use explicit engineering notation

```python
# Clear notation
C1 = Capacitor(ref="1", capacitance=1e-6)   # 1µF = 1e-6
C2 = Capacitor(ref="2", capacitance=100e-9)  # 100nF = 100e-9
```

**Common prefixes**:
- M (mega) = 1e6
- k (kilo) = 1e3
- m (milli) = 1e-3
- µ (micro) = 1e-6
- n (nano) = 1e-9
- p (pico) = 1e-12

### 6. Zero or Negative Values

**Problem**: Component with zero or negative value

```python
# Wrong - resistance can't be zero
R1 = Resistor(ref="1", resistance=0)
```

**Solution**: Use realistic positive values

```python
# Use a small value instead of zero
R1 = Resistor(ref="1", resistance=1e-6)  # 1µΩ minimum
```

### 7. Unrealistic Component Values

**Problem**: Values outside practical range

```python
# Warning - 1pΩ resistor is unrealistic
R1 = Resistor(ref="1", resistance=1e-12)

# Warning - 1F capacitor is very large
C1 = Capacitor(ref="1", capacitance=1)
```

**Solution**: Use practical values or acknowledge simulation-only

```python
# Typical ranges:
# Resistors: 1Ω to 10MΩ
# Capacitors: 1pF to 10000µF
# Inductors: 1nH to 100mH
```

## Simulation

### 8. Analysis Without Source

**Problem**: AC analysis without AC source

```python
# Wrong - no AC source for AC analysis
circuit = Circuit("test")
V1 = Vdc(ref="1", value=5)  # DC only!
circuit.add(V1)
# AC analysis won't show anything useful
```

**Solution**: Add AC source for AC analysis

```python
V1 = Vac(ref="1", value=0, ac_mag=1)  # AC source
# Or use templates which set this up correctly
```

### 9. Wrong Frequency Range

**Problem**: AC analysis with inappropriate frequency range

```python
# Wrong - missing interesting frequencies
# For a 1kHz filter, starting at 1MHz misses everything
result = quick_ac(circuit, start=1e6, stop=1e9)
```

**Solution**: Choose frequency range around expected behavior

```python
# For 1kHz filter, sweep around cutoff
result = quick_ac(circuit, start=1, stop=1e6)  # 1Hz to 1MHz
```

### 10. Timestep Too Large

**Problem**: Transient analysis with too large timestep

```python
# Wrong - 1ms step for 1MHz signal misses everything
result = quick_tran(circuit, duration="1ms", step="1ms")
```

**Solution**: Use at least 10 points per cycle

```python
# For 1MHz signal, use <100ns step
result = quick_tran(circuit, duration="10us", step="10ns")
# Or let SpiceLab auto-calculate
result = quick_tran(circuit, duration="10us")  # Auto step
```

## Results and Measurements

### 11. Wrong Signal Name

**Problem**: Requesting non-existent signal

```python
# Wrong - signal name doesn't exist
ds = result.dataset()
vout = ds["vout"]  # KeyError!
```

**Solution**: Check available variables first

```python
# List available signals
print(ds.data_vars)
# Use correct SPICE naming: V(node_name)
vout = ds["V(vout)"]
```

### 12. Complex vs Magnitude

**Problem**: Using complex numbers when magnitude expected

```python
# AC results are complex
vout = ds["V(vout)"]  # Complex array
print(f"Output: {vout[0]}")  # Shows (0.5+0.3j)
```

**Solution**: Extract magnitude or phase explicitly

```python
import numpy as np

magnitude = np.abs(vout)
phase = np.angle(vout, deg=True)
db = 20 * np.log10(magnitude)
```

## Template Usage

### 13. Template Parameter Confusion

**Problem**: Misunderstanding template parameters

```python
# Wrong - ratio is Vout/Vin, not R1/R2
divider = voltage_divider(ratio=2.0)  # Invalid! ratio must be 0-1
```

**Solution**: Read template documentation

```python
# ratio = Vout/Vin (0 to 1)
divider = voltage_divider(ratio=0.5)  # 50% divider
divider = voltage_divider(ratio=0.1)  # 10% divider

# For gain > 1, use amplifier templates
amp = non_inverting_amp(gain=2.0)  # Gain of 2
```

### 14. Missing Template Dependencies

**Problem**: Template requires model that's not included

```python
# Some templates need op-amp models
amp = inverting_amp(gain=10)
# If simulation fails, check for missing .model statements
```

**Solution**: Check template documentation and add models

```python
# Add model directive if needed
circuit.add_directive(".model TL072 opamp")
```

## Debugging Tips

### Use Dry Run First

```python
from spicelab.debug import dry_run

result = dry_run(circuit, analyses)
print(result)
```

### Enable Verbose Mode

```python
from spicelab.debug import VerboseSimulation

with VerboseSimulation():
    result = quick_ac(circuit, start=1, stop=1e6)
```

### Check Netlist Output

```python
print(circuit.preview_netlist())
# Verify the netlist looks correct before simulation
```

### Validate Circuit

```python
result = circuit.validate()
if not result.is_valid:
    for error in result.errors:
        print(f"ERROR: {error}")
for warning in result.warnings:
    print(f"WARNING: {warning}")
```

## See Also

- [Usability Guide](usability-guide.md) - Best practices
- [Troubleshooting](troubleshooting.md) - Error messages and solutions
- [Templates Reference](templates-reference.md) - Template documentation
