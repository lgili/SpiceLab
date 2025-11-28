# SpiceLab Template Catalog

Circuit templates provide pre-built topologies for rapid prototyping. All templates return ready-to-simulate circuits with correctly sized components.

## Quick Start

```python
from spicelab.templates import (
    # Filters
    rc_lowpass, rc_highpass, rlc_bandpass, sallen_key_lowpass,
    butterworth_lowpass, chebyshev_lowpass, bessel_lowpass,
    # Amplifiers
    voltage_follower, inverting_amplifier, non_inverting_amplifier,
    differential_amplifier, summing_amplifier,
    # Power
    voltage_divider,
    # Bias
    current_mirror,
    # Compensation
    dominant_pole_compensation, lead_compensation, miller_compensation,
    # ADC/DAC
    r2r_dac_ladder, sample_and_hold, comparator_bank,
)

# Create a 1kHz Butterworth lowpass
result = butterworth_lowpass(fc=1000, order=4)
circuit = result.circuit
print(circuit.build_netlist())
```

---

## Filter Templates

### First-Order Filters

#### `rc_lowpass(fc, impedance=10000, name="RC_Filter")`

Creates a simple RC lowpass filter.

**Parameters:**
- `fc`: Cutoff frequency in Hz (-3dB point)
- `impedance`: Input impedance (default 10kΩ)

**Design equation:** `fc = 1 / (2π RC)`

**Topology:**
```
vin ---[R]---+--- vout
             |
            [C]
             |
            GND
```

**Example:**
```python
circuit = rc_lowpass(fc=1000, impedance=10_000)
# R = 10kΩ, C ≈ 15.9nF for fc = 1kHz
```

---

#### `rc_highpass(fc, impedance=10000, name="RC_Highpass")`

Creates a simple RC highpass filter.

**Parameters:**
- `fc`: Cutoff frequency in Hz (-3dB point)
- `impedance`: Input impedance (default 10kΩ)

**Topology:**
```
vin ---||---+--- vout
       C    |
           [R]
            |
           GND
```

**Example:**
```python
circuit = rc_highpass(fc=100, impedance=10_000)
# R = 10kΩ, C ≈ 159nF for fc = 100Hz
```

---

### Second-Order Filters

#### `sallen_key_lowpass(fc, q=0.707, impedance=10000)`

Creates a 2nd-order Sallen-Key lowpass filter.

**Parameters:**
- `fc`: Cutoff frequency in Hz
- `q`: Quality factor (0.707 = Butterworth, maximally flat)
- `impedance`: Base resistance (R1 = R2)

**Returns:** `FilterResult` with circuit, components, fc, q_factor

**Topology:**
```
vin ---[R1]---+---[R2]---+--- vout
              |         |
             [C1]      [C2]
              |         |
             GND       GND
```

**Example:**
```python
result = sallen_key_lowpass(fc=1000, q=0.707)
print(f"Q factor: {result.q_factor}")
print(f"Components: {result.components.keys()}")
```

---

#### `rlc_bandpass(fc, bandwidth, impedance=1000)`

Creates a series RLC bandpass filter.

**Parameters:**
- `fc`: Center frequency in Hz
- `bandwidth`: 3dB bandwidth in Hz
- `impedance`: Characteristic impedance

**Returns:** `FilterResult` with Q factor = fc / bandwidth

**Topology:**
```
vin ---[R]---[L]---+--- vout
                   |
                  [C]
                   |
                  GND
```

**Example:**
```python
result = rlc_bandpass(fc=1000, bandwidth=100)
print(f"Q factor: {result.q_factor}")  # Q = 10
```

---

### Higher-Order Filters (Butterworth, Chebyshev, Bessel)

#### `butterworth_lowpass(fc, order=2, impedance=10000)`

Creates a Butterworth lowpass filter using cascaded Sallen-Key stages.

**Characteristics:**
- Maximally flat passband (no ripple)
- -20 × order dB/decade rolloff
- Good balance of flatness and rolloff

**Supported orders:** 1-6

**Example:**
```python
result = butterworth_lowpass(fc=1000, order=4)
# Creates 4th-order filter: -80 dB/decade rolloff
```

---

#### `chebyshev_lowpass(fc, order=2, ripple_db=0.5, impedance=10000)`

Creates a Chebyshev Type I lowpass filter.

**Characteristics:**
- Steeper rolloff than Butterworth
- Passband ripple (controlled by `ripple_db`)
- Best for maximum selectivity

**Parameters:**
- `ripple_db`: Passband ripple in dB (default 0.5)

**Example:**
```python
result = chebyshev_lowpass(fc=1000, order=4, ripple_db=0.5)
# Creates filter with 0.5dB passband ripple
```

---

#### `bessel_lowpass(fc, order=2, impedance=10000)`

Creates a Bessel lowpass filter.

**Characteristics:**
- Maximally flat group delay (linear phase)
- Preserves waveform shape
- Slowest rolloff of the three types

**Use cases:** Pulse/transient applications, audio

**Example:**
```python
result = bessel_lowpass(fc=1000, order=4)
# Creates filter with linear phase response
```

---

### Filter Comparison

| Filter Type | Passband | Rolloff | Phase | Best For |
|-------------|----------|---------|-------|----------|
| Butterworth | Maximally flat | Moderate | Nonlinear | General purpose |
| Chebyshev | Ripple | Steep | Nonlinear | High selectivity |
| Bessel | Rounded | Slow | Linear | Pulse preservation |

---

## Amplifier Templates

### `voltage_follower(name="Voltage_Follower", vcc=15.0, vee=-15.0)`

Creates a unity-gain voltage buffer with ideal opamp.

**Returns:** `AmplifierResult` with gain = 1.0

**Topology:**
```
vin -----(+)\
            >---- vout
     +--(-)/      |
     |            |
     +------------+
```

**Example:**
```python
result = voltage_follower()
print(f"Gain: {result.gain}")  # 1.0
```

---

### `inverting_amplifier(gain, input_impedance=10000)`

Creates an inverting amplifier resistor network.

**Parameters:**
- `gain`: Magnitude of voltage gain (positive number)
- `input_impedance`: Rin value

**Design equation:** `Gain = -Rf / Rin`

**Topology:**
```
vin ---[Rin]---+---[Rf]--- vout
               |
            (opamp -)
```

**Example:**
```python
circuit = inverting_amplifier(gain=10, input_impedance=10_000)
# Rin = 10kΩ, Rf = 100kΩ for gain of -10
```

---

### `non_inverting_amplifier(gain, feedback_resistance=10000)`

Creates a non-inverting amplifier resistor network.

**Parameters:**
- `gain`: Voltage gain (must be >= 1)
- `feedback_resistance`: Rf value

**Design equation:** `Gain = 1 + Rf / R1`

**Topology:**
```
vin --- (opamp +)
                    vout
                      |
       GND---[R1]---+-[Rf]
                    |
                 (opamp -)
```

**Example:**
```python
circuit = non_inverting_amplifier(gain=11, feedback_resistance=10_000)
# Rf = 10kΩ, R1 = 1kΩ for gain of 11
```

---

### `differential_amplifier(gain=1.0, input_impedance=10000)`

Creates a differential amplifier with ideal opamp.

**Parameters:**
- `gain`: Differential voltage gain
- `input_impedance`: R1 = R2 value

**Output equation:** `Vout = gain × (Vp - Vn)`

**Returns:** `AmplifierResult` with vinp, vinn, vout nets

**Example:**
```python
result = differential_amplifier(gain=10)
# Vout = 10 × (Vp - Vn)
```

---

### `summing_amplifier(num_inputs=2, gain=1.0, input_impedance=10000)`

Creates an inverting summing amplifier.

**Parameters:**
- `num_inputs`: Number of input channels (1-8)
- `gain`: Gain per input channel
- `input_impedance`: Rin for each channel

**Output equation:** `Vout = -gain × (V1 + V2 + ... + Vn)`

**Example:**
```python
result = summing_amplifier(num_inputs=3, gain=2)
# Vout = -2 × (V1 + V2 + V3)
```

---

## Power Templates

### `voltage_divider(ratio, total_resistance=10000)`

Creates a resistive voltage divider.

**Parameters:**
- `ratio`: Output voltage ratio (0 < ratio < 1)
- `total_resistance`: R1 + R2

**Design equation:** `Vout = Vin × ratio`

**Example:**
```python
circuit = voltage_divider(ratio=0.5, total_resistance=10_000)
# R1 = 5kΩ, R2 = 5kΩ for 50% division
```

---

## Bias Templates

### `current_mirror(reference_current, mirror_ratio=1.0, vcc=5.0)`

Creates a simple current mirror.

**Parameters:**
- `reference_current`: Reference current in Amps
- `mirror_ratio`: Iout / Iref ratio

**Returns:** `BiasResult` with output_current

**Example:**
```python
result = current_mirror(reference_current=1e-3, mirror_ratio=2)
# Iref = 1mA, Iout = 2mA
print(f"Output current: {result.output_current}")
```

---

## Compensation Templates

### `dominant_pole_compensation(pole_frequency, impedance=10000)`

Creates a simple RC dominant pole network.

**Parameters:**
- `pole_frequency`: Desired pole frequency in Hz
- `impedance`: Resistor value

**Returns:** `CompensationResult`

**Example:**
```python
result = dominant_pole_compensation(pole_frequency=100)
# Creates pole at 100Hz
```

---

### `lead_compensation(zero_frequency, pole_frequency, impedance=10000)`

Creates a lead compensation network for phase boost.

**Parameters:**
- `zero_frequency`: Zero frequency (must be < pole_frequency)
- `pole_frequency`: Pole frequency

**Returns:** `CompensationResult` with phase_margin_boost

**Phase boost formula:** `φmax = arcsin((fp-fz)/(fp+fz))`

**Example:**
```python
result = lead_compensation(zero_frequency=1000, pole_frequency=10000)
print(f"Phase boost: {result.phase_margin_boost:.1f}°")  # ~55°
```

---

### `lead_lag_compensation(lead_zero_freq, lead_pole_freq, lag_pole_freq)`

Creates combined lead-lag compensation network.

**Example:**
```python
result = lead_lag_compensation(
    lead_zero_freq=1000,
    lead_pole_freq=10000,
    lag_pole_freq=10
)
```

---

### `miller_compensation(pole_frequency, gain=100, impedance=10000)`

Creates Miller compensation network for pole splitting.

**Parameters:**
- `pole_frequency`: Desired dominant pole frequency
- `gain`: Stage gain (affects effective capacitance)

**Example:**
```python
result = miller_compensation(pole_frequency=10, gain=100)
# Creates 10Hz dominant pole with Miller effect
```

---

## ADC/DAC Templates

### `r2r_dac_ladder(bits=4, r_value=10000)`

Creates an R-2R resistor ladder DAC.

**Parameters:**
- `bits`: Number of bits (1-8)
- `r_value`: Base resistance R

**Returns:** `ConverterResult` with input_nets (bit inputs) and output_net

**Topology (4-bit):**
```
B3 --[2R]--+--[R]--+--[R]--+--[R]--+-- Vout
           |       |       |       |
B2 --[2R]--+       |       |       |
                   |       |       |
B1 --------[2R]----+       |       |
                           |       |
B0 ----------------[2R]----+       |
                                   |
                           [2R]----+
                                   |
                                  GND
```

**Example:**
```python
result = r2r_dac_ladder(bits=8)
print(f"Input nets: {[n.name for n in result.input_nets]}")
# ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
```

---

### `sample_and_hold(hold_capacitance=100e-12, buffer_impedance=10000)`

Creates a sample-and-hold circuit.

**Parameters:**
- `hold_capacitance`: Hold capacitor value (default 100pF)
- `buffer_impedance`: Switch/buffer impedance

**Returns:** `ConverterResult`

**Example:**
```python
result = sample_and_hold(hold_capacitance=100e-12)
# Creates S/H with 100pF hold capacitor
```

---

### `comparator_bank(bits=3, vref_range=(0.0, 5.0))`

Creates a comparator reference ladder for flash ADC.

**Parameters:**
- `bits`: ADC bits (1-4)
- `vref_range`: Reference voltage range (vlow, vhigh)

**Returns:** `ConverterResult` with reference voltage nets

**Example:**
```python
result = comparator_bank(bits=3, vref_range=(0, 3.3))
# Creates 7 reference voltages for 3-bit flash ADC
```

---

## Result Types

### FilterResult

```python
@dataclass
class FilterResult:
    circuit: Circuit
    components: dict[str, Component]
    cutoff_frequency: float
    q_factor: float | None
```

### AmplifierResult

```python
@dataclass
class AmplifierResult:
    circuit: Circuit
    components: dict[str, Component]
    gain: float
    input_net: Net
    output_net: Net
```

### BiasResult

```python
@dataclass
class BiasResult:
    circuit: Circuit
    components: dict[str, Component]
    output_current: float
    mirror_ratio: float
```

### CompensationResult

```python
@dataclass
class CompensationResult:
    circuit: Circuit
    components: dict[str, Component]
    pole_frequency: float | None
    zero_frequency: float | None
    phase_margin_boost: float | None  # Degrees
```

### ConverterResult

```python
@dataclass
class ConverterResult:
    circuit: Circuit
    components: dict[str, Component]
    resolution_bits: int
    input_nets: list[Net]
    output_net: Net
```

---

## Complete Example: Audio Crossover

```python
from spicelab.templates import butterworth_lowpass, butterworth_highpass
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vac

# Create 2-way crossover at 2kHz
lpf = butterworth_lowpass(fc=2000, order=4)  # Woofer path
hpf_circuit = rc_highpass(fc=2000)  # Tweeter path (simplified)

# Add to main circuit with input source
main = Circuit("audio_crossover")
vsrc = Vac("IN", amplitude=1, freq=1000)
main.add(vsrc)

# Connect and simulate...
print(lpf.circuit.build_netlist())
```

---

**Last Updated:** 2025-11-27

**See Also:**
- [Component Catalog](component_catalog.md) - Component reference
- [Troubleshooting Guide](troubleshooting_guide.md) - Debug help
