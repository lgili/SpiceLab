# Chapter 2: Basic Components

This chapter covers the fundamental components available in SpiceLab: passive components and voltage/current sources.

## Passive Components

### Resistors

Resistors limit current flow according to Ohm's law: V = I × R

```python
from spicelab.core.components import Resistor

# Different ways to specify resistance
r1 = Resistor("1", resistance=10_000)  # 10k ohm (numeric)
r2 = Resistor("2", "10k")              # 10k ohm (string with suffix)
r3 = Resistor("3", "10000")            # 10k ohm (string numeric)
r4 = Resistor("4", 1e4)                # 10k ohm (scientific notation)
```

**Value suffixes:**
| Suffix | Multiplier | Example |
|--------|------------|---------|
| `f` | 10⁻¹⁵ | "1f" = 1 femto |
| `p` | 10⁻¹² | "100p" = 100 pico |
| `n` | 10⁻⁹ | "10n" = 10 nano |
| `u` | 10⁻⁶ | "1u" = 1 micro |
| `m` | 10⁻³ | "1m" = 1 milli |
| `k` | 10³ | "10k" = 10 kilo |
| `M` or `Meg` | 10⁶ | "1M" = 1 mega |
| `G` | 10⁹ | "1G" = 1 giga |

**Ports:**
- `ports[0]`: Terminal 1
- `ports[1]`: Terminal 2

### Capacitors

Capacitors store energy in an electric field: I = C × dV/dt

```python
from spicelab.core.components import Capacitor

# Different ways to specify capacitance
c1 = Capacitor("1", capacitance=100e-9)  # 100nF (numeric)
c2 = Capacitor("2", "100n")              # 100nF (string)
c3 = Capacitor("3", "0.1u")              # 100nF (0.1µF)
```

**Common values:**
- Decoupling: 100nF, 10µF
- Filtering: 1µF - 1000µF
- Timing: 1nF - 100nF

**Ports:**
- `ports[0]`: Positive terminal
- `ports[1]`: Negative terminal

### Inductors

Inductors store energy in a magnetic field: V = L × dI/dt

```python
from spicelab.core.components import Inductor

# Different ways to specify inductance
l1 = Inductor("1", inductance=10e-6)  # 10µH (numeric)
l2 = Inductor("2", "10u")             # 10µH (string)
l3 = Inductor("3", "100n")            # 100nH
```

**Common values:**
- RF chokes: 10nH - 1µH
- Power inductors: 1µH - 100µH
- Filter inductors: 100µH - 10mH

**Ports:**
- `ports[0]`: Terminal 1
- `ports[1]`: Terminal 2

## Voltage Sources

### DC Voltage Source (Vdc)

Provides constant DC voltage.

```python
from spicelab.core.components import Vdc

# 5V DC supply
vcc = Vdc("CC", 5.0)

# Negative supply
vee = Vdc("EE", -5.0)
```

**Parameters:**
- `voltage`: DC voltage in Volts

**Ports:**
- `ports[0]`: Positive terminal (+)
- `ports[1]`: Negative terminal (-), usually connected to GND

### AC Voltage Source (Vac)

Used for AC (frequency domain) analysis.

```python
from spicelab.core.components import Vac

# 1V AC source for frequency response analysis
vin = Vac("IN", ac_mag=1.0)

# With phase shift
vin_phase = Vac("IN2", ac_mag=1.0, ac_phase=90)  # 90 degree phase
```

**Parameters:**
- `ac_mag`: AC magnitude in Volts (default 1.0)
- `ac_phase`: AC phase in degrees (default 0)

**Use case:** Filter frequency response, amplifier gain measurement

### Pulse Voltage Source (Vpulse)

Generates square waves and pulses for transient analysis.

```python
from spicelab.core.components import Vpulse

# Clock signal: 0V to 5V, 1MHz
clk = Vpulse(
    "CLK",
    v1=0,          # Low voltage
    v2=5,          # High voltage
    td="0",        # Delay before first pulse
    tr="1n",       # Rise time
    tf="1n",       # Fall time
    pw="500n",     # Pulse width
    per="1u"       # Period (1MHz = 1µs period)
)

# Step input
step = Vpulse(
    "STEP",
    v1=0,
    v2=1,
    td="1m",       # Step occurs at 1ms
    tr="1n",
    tf="1n",
    pw="10m",      # Long pulse width
    per="20m"
)
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `v1` | Initial/low voltage |
| `v2` | Pulsed/high voltage |
| `td` | Delay time |
| `tr` | Rise time |
| `tf` | Fall time |
| `pw` | Pulse width |
| `per` | Period |

### Sinusoidal Voltage Source (Vsin)

Generates sine waves for transient analysis.

```python
from spicelab.core.components import Vsin

# 1kHz sine wave, 1V amplitude
sine = Vsin(
    "SIN",
    vo=0,          # DC offset
    va=1.0,        # Amplitude
    freq=1000      # Frequency in Hz
)

# With DC offset
sine_offset = Vsin(
    "SIN2",
    vo=2.5,        # 2.5V DC offset
    va=1.0,        # 1V amplitude (swings 1.5V to 3.5V)
    freq=1000
)
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `vo` | DC offset voltage |
| `va` | Amplitude |
| `freq` | Frequency in Hz |
| `td` | Delay before start (optional) |
| `theta` | Damping factor (optional) |
| `phase` | Initial phase in degrees (optional) |

## Current Sources

### DC Current Source (Idc)

Provides constant DC current.

```python
from spicelab.core.components import Idc

# 1mA current source
i1 = Idc("1", 1e-3)
```

**Ports:**
- `ports[0]`: Current flows INTO this terminal
- `ports[1]`: Current flows OUT of this terminal

### Pulse Current Source (Ipulse)

Current pulse for transient analysis.

```python
from spicelab.core.components import Ipulse

# Current pulse
ipulse = Ipulse(
    "PULSE",
    i1=0,
    i2=1e-3,      # 1mA pulse
    td="0",
    tr="1u",
    tf="1u",
    pw="10u",
    per="100u"
)
```

## Complete Example: RC Filter

Let's build an RC lowpass filter using the components we've learned:

```python
"""RC Lowpass Filter"""
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vac
from spicelab.core.net import GND, Net

# Create circuit
circuit = Circuit("rc_lowpass")

# Components
vin = Vac("IN", ac_mag=1.0)
r1 = Resistor("1", "10k")
c1 = Capacitor("1", "100n")

circuit.add(vin, r1, c1)

# Nets
input_net = Net("in")
output_net = Net("out")

# Connections
circuit.connect(vin.ports[0], input_net)
circuit.connect(vin.ports[1], GND)
circuit.connect(r1.ports[0], input_net)
circuit.connect(r1.ports[1], output_net)
circuit.connect(c1.ports[0], output_net)
circuit.connect(c1.ports[1], GND)

# Calculate cutoff frequency
import math
R = 10_000  # 10k
C = 100e-9  # 100n
fc = 1 / (2 * math.pi * R * C)
print(f"Cutoff frequency: {fc:.1f} Hz")

# Validate and show netlist
result = circuit.validate()
print(f"Valid: {result.is_valid}")
print(circuit.build_netlist())
```

Output:
```
Cutoff frequency: 159.2 Hz
Valid: True
* rc_lowpass
VIN in 0 AC 1.0
R1 in out 10000.0
C1 out 0 1e-07
.end
```

## Exercises

### Exercise 2.1: RLC Circuit
Create an RLC series circuit with:
- 10V DC source
- 100 ohm resistor
- 10mH inductor
- 100nF capacitor

Calculate the resonant frequency: f = 1/(2π√LC)

### Exercise 2.2: Voltage Divider with Different Ratios
Create voltage dividers that output:
- 3.3V from 5V (for 3.3V devices)
- 1.65V from 3.3V (for ADC reference)

### Exercise 2.3: Clock Generator
Create a 10kHz clock signal using Vpulse with:
- 0V low, 3.3V high
- 50% duty cycle
- 10ns rise/fall times

### Exercise 2.4: Sine Wave with DC Offset
Create a 1kHz sine wave that:
- Has 2.5V DC offset
- Swings between 0V and 5V
- Hint: Use Vsin with vo=2.5, va=2.5

### Exercise 2.5: Current Source Load
Create a circuit where a 1mA current source drives a 1k resistor.
Calculate the voltage across the resistor: V = I × R = 1V

## Component Reference Table

| Component | Class | Key Parameters | Ports |
|-----------|-------|----------------|-------|
| Resistor | `Resistor` | resistance | 2 |
| Capacitor | `Capacitor` | capacitance | 2 |
| Inductor | `Inductor` | inductance | 2 |
| DC Voltage | `Vdc` | voltage | 2 |
| AC Voltage | `Vac` | ac_mag, ac_phase | 2 |
| Pulse Voltage | `Vpulse` | v1, v2, td, tr, tf, pw, per | 2 |
| Sine Voltage | `Vsin` | vo, va, freq | 2 |
| DC Current | `Idc` | current | 2 |
| Pulse Current | `Ipulse` | i1, i2, td, tr, tf, pw, per | 2 |

## Next Steps

- [Chapter 3: Building Circuits](03_building_circuits.md) - Advanced circuit construction
- [Chapter 4: Analysis Types](04_analysis_types.md) - Running simulations

---

**See also:**
- [Component Catalog](../component_catalog.md) - Complete component reference
- [Template Catalog](../template_catalog.md) - Pre-built circuit templates
