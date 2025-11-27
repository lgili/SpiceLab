# Circuit Templates Reference

SpiceLab provides pre-built circuit templates for common designs. Templates handle component values, connections, and analysis setup automatically.

## Filter Templates

### RC Lowpass Filter

```python
from spicelab.templates import rc_lowpass

# Create RC lowpass with 1kHz cutoff
circuit = rc_lowpass(fc=1000)
```

**Parameters**:
- `fc`: Cutoff frequency in Hz

**Components created**:
- R1: Calculated for desired fc with C=100nF
- C1: 100nF (fixed)
- V1: AC source

**Nodes**:
- `vin`: Input
- `vout`: Output (across capacitor)

---

### RC Highpass Filter

```python
from spicelab.templates import rc_highpass

circuit = rc_highpass(fc=100)  # 100Hz cutoff
```

**Parameters**:
- `fc`: Cutoff frequency in Hz

**Components created**:
- C1: Calculated for desired fc
- R1: 10kΩ (fixed)
- V1: AC source

**Nodes**:
- `vin`: Input
- `vout`: Output (across resistor)

---

### RL Lowpass Filter

```python
from spicelab.templates import rl_lowpass

circuit = rl_lowpass(fc=10000)  # 10kHz cutoff
```

**Parameters**:
- `fc`: Cutoff frequency in Hz

---

### RL Highpass Filter

```python
from spicelab.templates import rl_highpass

circuit = rl_highpass(fc=1000)
```

---

### RLC Bandpass Filter

```python
from spicelab.templates import rlc_bandpass

circuit = rlc_bandpass(f0=1000, Q=10)  # 1kHz center, Q=10
```

**Parameters**:
- `f0`: Center frequency in Hz
- `Q`: Quality factor

---

### Butterworth Lowpass (2nd Order)

```python
from spicelab.templates import butterworth_lowpass

circuit = butterworth_lowpass(fc=1000)
```

**Parameters**:
- `fc`: Cutoff frequency in Hz

Creates a 2nd order Butterworth lowpass with maximally flat response.

---

### Chebyshev Lowpass

```python
from spicelab.templates import chebyshev_lowpass

circuit = chebyshev_lowpass(fc=1000, ripple_db=1)
```

**Parameters**:
- `fc`: Cutoff frequency in Hz
- `ripple_db`: Passband ripple in dB

---

## Amplifier Templates

### Voltage Divider

```python
from spicelab.templates import voltage_divider

circuit = voltage_divider(ratio=0.5)  # Vout = 0.5 * Vin
```

**Parameters**:
- `ratio`: Output/input ratio (0 to 1)

**Formula**: `ratio = R2 / (R1 + R2)`

---

### Inverting Amplifier

```python
from spicelab.templates import inverting_amp

circuit = inverting_amp(gain=10)  # Vout = -10 * Vin
```

**Parameters**:
- `gain`: Magnitude of voltage gain

**Note**: Output is inverted (180° phase shift)

---

### Non-Inverting Amplifier

```python
from spicelab.templates import non_inverting_amp

circuit = non_inverting_amp(gain=2)  # Vout = 2 * Vin
```

**Parameters**:
- `gain`: Voltage gain (must be >= 1)

**Formula**: `gain = 1 + Rf/Rin`

---

### Voltage Follower (Buffer)

```python
from spicelab.templates import voltage_follower

circuit = voltage_follower()
```

Unity gain buffer with high input impedance and low output impedance.

---

### Differential Amplifier

```python
from spicelab.templates import differential_amp

circuit = differential_amp(gain=5)
```

**Parameters**:
- `gain`: Differential gain

**Output**: `Vout = gain * (Vp - Vn)`

---

### Summing Amplifier

```python
from spicelab.templates import summing_amp

circuit = summing_amp(weights=[1, 1, 1])  # Equal weights
```

**Parameters**:
- `weights`: List of input weights

**Output**: `Vout = -(w1*V1 + w2*V2 + ...)`

---

### Integrator

```python
from spicelab.templates import integrator

circuit = integrator(time_constant=1e-3)  # 1ms
```

**Parameters**:
- `time_constant`: RC time constant in seconds

---

### Differentiator

```python
from spicelab.templates import differentiator

circuit = differentiator(time_constant=1e-3)
```

---

## Compensation Networks

### Lead Compensation

```python
from spicelab.templates import lead_compensation

circuit = lead_compensation(
    zero_frequency=1000,  # Zero at 1kHz
    pole_frequency=10000,  # Pole at 10kHz
    impedance=10000       # 10kΩ base impedance
)
```

Provides phase lead for stability improvement.

---

### Lag Compensation

```python
from spicelab.templates import lag_compensation

circuit = lag_compensation(
    pole_frequency=100,
    zero_frequency=1000,
    impedance=10000
)
```

Provides phase lag to reduce gain at high frequencies.

---

### Lead-Lag Compensation

```python
from spicelab.templates import lead_lag_compensation

circuit = lead_lag_compensation(
    lead_zero=1000,
    lead_pole=10000,
    lag_pole=10,
    lag_zero=100
)
```

Combination of lead and lag networks.

---

## Bias Networks

### Current Mirror

```python
from spicelab.templates import current_mirror

circuit = current_mirror(
    reference_current=1e-3,  # 1mA reference
    output_ratio=1.0         # 1:1 mirror ratio
)
```

---

### Bandgap Reference

```python
from spicelab.templates import bandgap_reference

circuit = bandgap_reference(
    output_voltage=1.25  # Target output voltage
)
```

---

## Oscillators

### Wien Bridge Oscillator

```python
from spicelab.templates import wien_bridge_oscillator

circuit = wien_bridge_oscillator(frequency=1000)  # 1kHz
```

---

### Phase Shift Oscillator

```python
from spicelab.templates import phase_shift_oscillator

circuit = phase_shift_oscillator(frequency=5000)
```

---

## Power Supply

### Half-Wave Rectifier

```python
from spicelab.templates import half_wave_rectifier

circuit = half_wave_rectifier(load_resistance=1000)
```

---

### Full-Wave Rectifier

```python
from spicelab.templates import full_wave_rectifier

circuit = full_wave_rectifier(load_resistance=1000)
```

---

## Accessing Template Components

All templates return a `TemplateResult` with access to individual components:

```python
result = rc_lowpass(fc=1000)

# Access the circuit
circuit = result.circuit

# Access individual components
R1 = result.components["R1"]
C1 = result.components["C1"]

# Access nets
vin = result.nets["vin"]
vout = result.nets["vout"]

# Get component values
print(f"R1 = {R1.resistance} Ω")
print(f"C1 = {C1.capacitance} F")
```

## Modifying Templates

Templates create standard circuits, but you can modify them:

```python
result = rc_lowpass(fc=1000)
circuit = result.circuit

# Add additional components
R2 = Resistor(ref="2", resistance=1000)
circuit.add(R2)

# Modify connections
circuit.connect(R2.ports[0], result.nets["vout"])
circuit.connect(R2.ports[1], GND)
```

## Creating Custom Templates

See [Creating Templates](creating-templates.md) for how to create your own reusable circuit templates.

## See Also

- [Usability Guide](usability-guide.md) - Best practices
- [Common Pitfalls](common-pitfalls.md) - Avoid common mistakes
- [API Reference](api-reference.md) - Complete API documentation
