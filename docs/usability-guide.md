# SpiceLab Usability Guide

This guide covers best practices and productivity tips for using SpiceLab effectively.

## Quick Start

### Creating Your First Circuit

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Capacitor, Vac
from spicelab.core.net import Net, GND

# Create circuit
circuit = Circuit("rc_filter")

# Create components
R1 = Resistor(ref="1", resistance=1000)  # 1kΩ
C1 = Capacitor(ref="1", capacitance=100e-9)  # 100nF
V1 = Vac(ref="1", value=0, ac_mag=1)  # 1V AC source

# Add to circuit
circuit.add(R1, C1, V1)

# Create nets and connect
vin = Net("vin")
vout = Net("vout")

circuit.connect(V1.ports[0], vin)
circuit.connect(V1.ports[1], GND)
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], vout)
circuit.connect(C1.ports[0], vout)
circuit.connect(C1.ports[1], GND)
```

### Using Templates (Recommended)

For common circuits, use templates instead of building from scratch:

```python
from spicelab.templates import rc_lowpass, voltage_divider, inverting_amp

# Create an RC filter with 1kHz cutoff
filter_circuit = rc_lowpass(fc=1000)

# Create a voltage divider with 50% ratio
divider = voltage_divider(ratio=0.5)

# Create an inverting amplifier with gain of 10
amp = inverting_amp(gain=10)
```

## Best Practices

### 1. Always Validate Before Simulation

```python
result = circuit.validate()
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

### 2. Use Engineering Notation

SpiceLab supports flexible value formats:

```python
# All equivalent
R1 = Resistor(ref="1", resistance=10_000)
R2 = Resistor(ref="2", resistance=10e3)
R3 = Resistor(ref="3", resistance=10000)

# Capacitors
C1 = Capacitor(ref="1", capacitance=100e-9)  # 100nF
C2 = Capacitor(ref="2", capacitance=1e-6)    # 1µF
```

### 3. Use Dry-Run for Debugging

Before running long simulations, validate the setup:

```python
from spicelab.debug import dry_run

result = dry_run(circuit, analyses)
if result.valid:
    print("Ready to simulate!")
else:
    for error in result.errors:
        print(f"Issue: {error}")
```

### 4. Use Verbose Mode for Troubleshooting

```python
from spicelab.debug import VerboseSimulation

with VerboseSimulation():
    result = quick_ac(circuit, start=1, stop=1e6)
```

### 5. Preview Netlist Before Simulation

```python
# View what will be sent to the simulator
print(circuit.preview_netlist())
```

## Workflow Shortcuts

### Quick Simulations

```python
from spicelab.shortcuts.simulation import quick_ac, quick_tran, quick_op

# AC analysis with smart defaults
result = quick_ac(circuit, start=1, stop=1e6)

# Transient analysis
result = quick_tran(circuit, duration="10ms")

# DC operating point
result = quick_op(circuit)
```

### Method Chaining

```python
from spicelab.shortcuts.chainable import SimulationBuilder

result = (
    SimulationBuilder(circuit)
    .ac(start=1, stop=1e6)
    .measure("V(vout)")
    .run()
)
```

## Getting Help

### Context-Sensitive Help

```python
from spicelab.help import get_help, show_help

# Get help for any object
help_obj = get_help(circuit)
print(help_obj.summary())
print(help_obj.methods())

# Or use show_help for immediate output
show_help(circuit)
```

### Interactive Tutorials

```python
from spicelab.help import run_tutorial, list_tutorials

# See available tutorials
for t in list_tutorials():
    print(t)

# Run a tutorial
run_tutorial("basics")
```

### API Cheat Sheet

```python
from spicelab.help import generate_cheatsheet, CheatsheetFormat

# Generate markdown cheat sheet
print(generate_cheatsheet(CheatsheetFormat.MARKDOWN))

# Or save to file
from spicelab.help.cheatsheet import save_cheatsheet
save_cheatsheet("cheatsheet.html")
```

## User Experience Features

### Progress Bars

For long simulations, progress bars show ETA:

```python
from spicelab.ux import ProgressBar, ProgressStyle

with ProgressBar(total=100, desc="Simulating") as pbar:
    for i in range(100):
        # do work
        pbar.update(1)
```

### Undo/Redo

Track circuit modifications:

```python
from spicelab.ux import CircuitHistory

history = CircuitHistory(circuit)
history.save("Initial state")

# Make changes
circuit.add(new_component)
history.save("Added component")

# Undo
history.undo()
```

### Clipboard

Copy and paste circuits:

```python
from spicelab.ux import CircuitClipboard

clipboard = CircuitClipboard()
clipboard.copy_circuit(circuit, "My filter design")

# Later, paste as new circuit
new_circuit = clipboard.paste_circuit(name="Copy of filter")
```

### Circuit Diff

Compare two circuit versions:

```python
from spicelab.ux import diff_circuits

diff = diff_circuits(circuit_v1, circuit_v2)
print(diff.summary())
```

### Bookmarks

Save frequently used configurations:

```python
from spicelab.ux import BookmarkManager

manager = BookmarkManager()
manager.save_circuit("my_filter", circuit, "RC lowpass 1kHz")

# Later, load it
loaded = manager.load_circuit("my_filter")
```

## Type Safety

SpiceLab includes type validation for catching errors early:

```python
from spicelab.core.typing import validate_types, positive, in_range

@validate_types
def create_filter(cutoff: float, order: int) -> Circuit:
    cutoff = positive(cutoff)  # Ensures > 0
    order = in_range(order, 1, 10)  # Ensures 1-10
    # ...
```

## Common Tasks

### Calculate Component Values

```python
from spicelab.utils.design import rc_cutoff_frequency, voltage_divider_ratio

# Find cutoff frequency
fc = rc_cutoff_frequency(R=1000, C=1e-6)  # 159 Hz

# Calculate divider ratio
ratio = voltage_divider_ratio(R1=10000, R2=10000)  # 0.5
```

### Use E-Series Values

```python
from spicelab.utils.e_series import find_nearest_e24, e24_series

# Find nearest E24 resistor
nearest = find_nearest_e24(4700)  # Returns 4.7k

# Get all E24 values in a range
values = e24_series(1000, 10000)
```

## See Also

- [Getting Started](getting-started.md) - Installation and first steps
- [Templates Reference](templates-reference.md) - All available templates
- [Common Pitfalls](common-pitfalls.md) - Avoid common mistakes
- [API Reference](api-reference.md) - Complete API documentation
