# SpiceLab Component Catalog

This document provides a complete reference for all pre-built components available in the SpiceLab component library.

## Overview

SpiceLab includes a comprehensive component library with 68+ ready-to-use components across multiple categories. Components are accessed via the registry system:

```python
from spicelab.library import create_component, list_components, get_metadata

# List all components
all_components = list_components()

# List components by category
diodes = list_components(category="diode")

# Create a component
d1 = create_component("diode.1n4148", "D1")

# Get component metadata
info = get_metadata("diode.1n4148")
```

## Component Categories

| Category | Count | Description |
|----------|-------|-------------|
| [Diodes](#diodes) | 18 | Signal, rectifier, Schottky, Zener, and LED diodes |
| [Transistors](#transistors) | 26 | MOSFETs (N/P) and BJTs (NPN/PNP) |
| [Behavioral](#behavioral-models) | 11 | Ideal diodes, switches, and transformers |
| [Op-Amps](#operational-amplifiers) | 3 | Ideal and subcircuit-based op-amps |
| [Capacitors](#capacitors) | 3 | Ceramic, electrolytic, and film capacitors |
| [Inductors](#inductors) | 2 | Power and RF inductors |
| [Resistors](#resistors) | 3 | Precision, general-purpose, and shunt resistors |
| [Switches](#switches) | 2 | Voltage and current-controlled switches |

---

## Diodes

### Signal Diodes

| Component ID | Part | Description |
|--------------|------|-------------|
| `diode.1n4148` | 1N4148 | High-speed switching diode, 100V, 200mA |
| `diode.1n914` | 1N914 | General-purpose switching diode, 100V, 200mA |

**Usage:**
```python
d1 = create_component("diode.1n4148", "D1")
```

### Rectifier Diodes

| Component ID | Part | Vrrm | If(avg) | Description |
|--------------|------|------|---------|-------------|
| `diode.1n4001` | 1N4001 | 50V | 1A | General-purpose rectifier |
| `diode.1n4004` | 1N4004 | 400V | 1A | Medium-voltage rectifier |
| `diode.1n4007` | 1N4007 | 1000V | 1A | High-voltage rectifier |

### Schottky Diodes

| Component ID | Part | Vrrm | If(avg) | Vf(typ) | Description |
|--------------|------|------|---------|---------|-------------|
| `diode.1n5817` | 1N5817 | 20V | 1A | 0.45V | Low-drop Schottky |
| `diode.1n5818` | 1N5818 | 30V | 1A | 0.55V | Medium-voltage Schottky |
| `diode.1n5819` | 1N5819 | 40V | 1A | 0.60V | Standard Schottky |
| `diode.bat54` | BAT54 | 30V | 200mA | 0.24V | Small-signal Schottky |

### Zener Diodes

| Component ID | Part | Vz | Power | Description |
|--------------|------|-----|-------|-------------|
| `diode.bzt52c3v3` | BZT52C3V3 | 3.3V | 500mW | 3.3V Zener for logic rails |
| `diode.bzt52c5v1` | BZT52C5V1 | 5.1V | 500mW | 5V USB/TTL protection |
| `diode.1n4733a` | 1N4733A | 5.1V | 1W | Power Zener for 5V regulation |
| `diode.1n4742a` | 1N4742A | 12V | 1W | Power Zener for 12V regulation |
| `diode.1n4744a` | 1N4744A | 15V | 1W | Power Zener for 15V regulation |

### LEDs

| Component ID | Color | Vf(typ) | Description |
|--------------|-------|---------|-------------|
| `diode.led_red` | Red | 1.8V | Standard red LED |
| `diode.led_green` | Green | 2.1V | Standard green LED |
| `diode.led_blue` | Blue | 3.2V | Standard blue LED |

**Usage:**
```python
led = create_component("diode.led_red", "LED1")
```

---

## Transistors

### N-Channel MOSFETs

| Component ID | Part | Vds | Id | Rds(on) | Description |
|--------------|------|-----|-----|---------|-------------|
| `transistor.2n7000` | 2N7000 | 60V | 200mA | 5Ω | Logic-level NMOS |
| `transistor.bss138` | BSS138 | 50V | 200mA | 3.5Ω | SOT-23 low-side switch |
| `transistor.irf540n` | IRF540N | 100V | 33A | 44mΩ | TO-220 power MOSFET |
| `transistor.irf3205` | IRF3205 | 55V | 110A | 8mΩ | High-current MOSFET |
| `transistor.irlz44n` | IRLZ44N | 55V | 47A | 22mΩ | Logic-level power MOSFET |
| `transistor.ao3400` | AO3400 | 30V | 5.8A | 26mΩ | SOT-23 N-MOSFET |

**Usage:**
```python
q1 = create_component("transistor.2n7000", "Q1")
# Access: q1.ports -> (gate, drain, source)
```

### P-Channel MOSFETs

| Component ID | Part | Vds | Id | Rds(on) | Description |
|--------------|------|-----|-----|---------|-------------|
| `transistor.irf9540n` | IRF9540N | -100V | -23A | 117mΩ | TO-220 P-MOSFET |
| `transistor.irf4905` | IRF4905 | -55V | -74A | 20mΩ | High-current P-MOSFET |
| `transistor.ao3401` | AO3401 | -30V | -4A | 55mΩ | SOT-23 P-MOSFET |
| `transistor.si2301cds` | Si2301CDS | -20V | -2.3A | 112mΩ | Low-Vgs P-MOSFET |
| `transistor.fqp27p06` | FQP27P06 | -60V | -27A | 70mΩ | TO-220 P-MOSFET |

### NPN BJTs

| Component ID | Part | Vceo | Ic | hFE | ft | Description |
|--------------|------|------|-----|-----|-----|-------------|
| `transistor.2n2222` | 2N2222 | 40V | 800mA | 100-300 | 300MHz | GP switching transistor |
| `transistor.2n3904` | 2N3904 | 40V | 200mA | 100-300 | 300MHz | Small-signal NPN |
| `transistor.2n4401` | 2N4401 | 40V | 600mA | 100-300 | 250MHz | Medium-power NPN |
| `transistor.bc547` | BC547 | 45V | 100mA | 110-800 | 300MHz | Low-noise NPN |
| `transistor.bc337` | BC337 | 45V | 800mA | 100-600 | 100MHz | Higher-current NPN |
| `transistor.mmbt3904` | MMBT3904 | 40V | 200mA | 100-300 | 300MHz | SOT-23 version of 2N3904 |
| `transistor.tip31c` | TIP31C | 100V | 3A | 25-50 | 3MHz | TO-220 power NPN |
| `transistor.tip41c` | TIP41C | 100V | 6A | 15-75 | 3MHz | High-power NPN |

**Usage:**
```python
q1 = create_component("transistor.2n3904", "Q1")
# Access: q1.ports -> (base, collector, emitter)
```

### PNP BJTs

| Component ID | Part | Vceo | Ic | hFE | ft | Description |
|--------------|------|------|-----|-----|-----|-------------|
| `transistor.2n2907` | 2N2907 | -60V | -600mA | 100-300 | 200MHz | GP PNP, complement to 2N2222 |
| `transistor.2n3906` | 2N3906 | -40V | -200mA | 100-300 | 250MHz | Small-signal PNP |
| `transistor.2n4403` | 2N4403 | -40V | -600mA | 100-300 | 200MHz | Medium-power PNP |
| `transistor.bc557` | BC557 | -45V | -100mA | 125-800 | 150MHz | Low-noise PNP |
| `transistor.bc327` | BC327 | -45V | -800mA | 100-600 | 100MHz | Higher-current PNP |
| `transistor.mmbt3906` | MMBT3906 | -40V | -200mA | 100-300 | 250MHz | SOT-23 PNP |
| `transistor.tip32c` | TIP32C | -100V | -3A | 25-50 | 3MHz | TO-220 power PNP |

---

## Behavioral Models

Behavioral models provide idealized component behavior for system-level simulation.

### Ideal Diodes

| Component ID | Von | Vz | Description |
|--------------|-----|-----|-------------|
| `behavioral.d_ideal` | 0V | ∞ | Perfect diode, no forward drop |
| `behavioral.d_ideal_schottky` | 0.3V | ∞ | Ideal Schottky with 0.3V drop |
| `behavioral.d_ideal_z5v1` | 0.7V | 5.1V | Ideal Zener, Vz=5.1V |

**Usage:**
```python
d1 = create_component("behavioral.d_ideal", "D1")
```

### Ideal Switches

| Component ID | Ron | Roff | Description |
|--------------|-----|------|-------------|
| `behavioral.sw_ideal` | 1mΩ | 1GΩ | Perfect SPST switch |
| `behavioral.sw_ideal_fast` | 1mΩ | 1GΩ | Fast switching (1ns rise/fall) |
| `behavioral.sw_relay` | 100mΩ | 10MΩ | Relay-like characteristics |

**Usage:**
```python
sw = create_component("behavioral.sw_ideal", "S1")
# Control with: sw.ctrl -> control voltage node
```

### Ideal Transformers

| Component ID | Ratio | Description |
|--------------|-------|-------------|
| `behavioral.xfmr_1_1` | 1:1 | Isolation transformer |
| `behavioral.xfmr_2_1` | 2:1 | Step-down 2:1 |
| `behavioral.xfmr_1_2` | 1:2 | Step-up 1:2 |

**Usage:**
```python
xfmr = create_component("behavioral.xfmr_2_1", "T1")
# Ports: primary+, primary-, secondary+, secondary-
```

### Ideal Sources

| Component ID | Description |
|--------------|-------------|
| `behavioral.vcvs_unity` | Unity-gain voltage follower (buffer) |
| `behavioral.cccs_unity` | Unity-gain current mirror |

---

## Operational Amplifiers

### Available Op-Amps

| Component ID | Type | Description |
|--------------|------|-------------|
| `opamp.ideal` | Ideal | Ideal op-amp, gain=1e6 |
| `opamp.lm741` | Subckt | LM741 macro-model (classic op-amp) |
| `opamp.tl081` | Subckt | TL081 FET-input op-amp |

**Usage:**
```python
# Ideal op-amp
op1 = create_component("opamp.ideal", "U1")

# Real op-amp with subcircuit
op2 = create_component("opamp.lm741", "U2")
# Ports: non-inv input, inv input, output, V+, V-
```

**Subcircuit Op-Amps:**
Subcircuit-based op-amps include supply pins (V+, V-) and use manufacturer-provided macro-models. Include files are automatically referenced.

---

## Capacitors

| Component ID | Value | Dielectric | Voltage | Description |
|--------------|-------|------------|---------|-------------|
| `capacitor.ceramic_100n_50v_x7r` | 100nF | X7R | 50V | Decoupling for MCU rails |
| `capacitor.electrolytic_10u_25v` | 10µF | Al. Electrolytic | 25V | Bulk DC storage |
| `capacitor.film_1u_250v` | 1µF | Polypropylene | 250V | Audio coupling/snubbers |

**Usage:**
```python
c1 = create_component("capacitor.ceramic_100n_50v_x7r", "C1")

# Override value:
c2 = create_component("capacitor.ceramic_100n_50v_x7r", "C2", value="220n")
```

---

## Inductors

| Component ID | Value | Imax | DCR | Description |
|--------------|-------|------|-----|-------------|
| `inductor.power_10u_3a` | 10µH | 3A | 45mΩ | Buck converter inductor |
| `inductor.rf_100n_100ma` | 100nH | 100mA | 1.2Ω | RF choke/bias tee |

**Usage:**
```python
l1 = create_component("inductor.power_10u_3a", "L1")
```

---

## Resistors

| Component ID | Value | Tolerance | Power | Description |
|--------------|-------|-----------|-------|-------------|
| `resistor.e24.1k_1pct_0.25w` | 1kΩ | 1% | 0.25W | Precision metal film |
| `resistor.e24.4k7_5pct_0.5w` | 4.7kΩ | 5% | 0.5W | GP carbon film pull-up |
| `resistor.shunt_0r01_1pct_1w` | 10mΩ | 1% | 1W | Current sensing shunt |

**Usage:**
```python
r1 = create_component("resistor.e24.1k_1pct_0.25w", "R1")

# Override value:
r2 = create_component("resistor.e24.1k_1pct_0.25w", "R2", value="2.2k")
```

---

## Switches

### Voltage-Controlled Switch

| Component ID | Model | Ron | Roff | Description |
|--------------|-------|-----|------|-------------|
| `switch.vsw_spst_fast` | SW_SPST_FAST | 0.1Ω | 1MΩ | Voltage-controlled SPST |

**Model Card:**
```spice
.model SW_SPST_FAST VSWITCH(RON=0.1 ROFF=1e6 VON=2 VOFF=0)
```

### Current-Controlled Switch

| Component ID | Model | Ron | Roff | Description |
|--------------|-------|-----|------|-------------|
| `switch.isw_spst_sense` | ISW_SPST_SLOW | 0.2Ω | 5MΩ | Current-controlled SPST |

**Model Card:**
```spice
.model ISW_SPST_SLOW ISWITCH(RON=0.2 ROFF=5e6 ION=20mA IOFF=5mA)
```

---

## Registry API Reference

### List Components

```python
from spicelab.library import list_components

# All components
all_comps = list_components()

# By category
diodes = list_components(category="diode")
transistors = list_components(category="transistor")
```

### Get Metadata

```python
from spicelab.library import get_metadata

info = get_metadata("diode.1n4148")
# Returns: {'part': '1N4148', 'description': '...', 'model': '...', ...}
```

### Create Component

```python
from spicelab.library import create_component

# Basic usage
d1 = create_component("diode.1n4148", "D1")

# With parameter override
r1 = create_component("resistor.e24.1k_1pct_0.25w", "R1", value="2.2k")
```

### Register Custom Component

```python
from spicelab.library import register_component
from spicelab.core.components import Resistor

def my_resistor_factory(ref: str, *, value: str | float = "10k") -> Resistor:
    return Resistor(ref, value)

register_component(
    "resistor.custom.10k_special",
    my_resistor_factory,
    category="resistor",
    metadata={"description": "My special 10k resistor"},
)
```

---

## Adding New Components

To add components to the library:

1. **Create entry in appropriate module** (`spicelab/library/diodes.py`, etc.)
2. **Define metadata** (part number, specs, model card)
3. **Create factory function** that returns the component
4. **Register with category** for organization

See `spicelab/library/diodes.py` for a complete example.

---

## Model Files

SPICE model files are stored in `spicelab/library/data/`:

```
spicelab/library/data/
├── diodes/
│   ├── 1n4148.mod
│   ├── 1n4001.mod
│   └── ...
├── transistors/
│   ├── 2n2222.mod
│   ├── irf540n.mod
│   └── ...
└── opamps/
    ├── lm741.sub
    └── tl081.sub
```

Model files are automatically included when generating netlists.

---

**Last Updated:** 2025-11-27
