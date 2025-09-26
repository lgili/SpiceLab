# Components Overview

Circuit Toolkit ships a typed component library. Each component exposes
`ports`, participates in the circuit connectivity graph, and knows how to format
its SPICE card when you call `Circuit.build_netlist()`.

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND

c = Circuit("rc")
V1 = Vdc("VIN", 5.0)
R1 = Resistor("R", "1k")
C1 = Capacitor("C", "100n")
for comp in (V1, R1, C1):
    c.add(comp)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)
print(c.build_netlist())
```

## Common parts

| Component | Helper | Notes |
|-----------|--------|-------|
| `Resistor(ref, value)` | `R(value)` | Value accepts numbers or suffixed strings (`"1k"`, `"10m"`). |
| `Capacitor(ref, value)` | `C(value)` | Two-terminal capacitor. |
| `Inductor(ref, value)` | `L(value)` | Two-terminal inductor. |
| `Vdc`, `Idc` | `V`, `I` | Independent DC sources. |
| `Vpulse`, `Ipulse` | `VP`, `IP` | Pulse sources. |
| `Vsin`, `Isin` | `VSIN`, `ISIN` | Sine sources. |
| `Vpwl`, `Ipwl` | `VPWL`, `IPWL` | Piecewise-linear sources. |
| `VA`, `IA` | Small-signal AC helpers. |
| `VCVS`, `VCCS`, `CCCS`, `CCVS` | `E`, `G`, `F`, `H` | Controlled sources. |
| `OpAmpIdeal` | `OA` | Three-port ideal op-amp (VCVS). |
| `AnalogMux8` | – | 8:1 analog multiplexer with optional enable pins. |

All helpers live in `spicelab.core.components`. Import only what you need, or
use the shorthand constructors (`R`, `C`, `V`, etc.) for quick sketches.

## Directives & models

When you need SPICE directives (e.g. `.model`, `.include`) call
`circuit.add_directive(...)`. Components such as `Diode` or `AnalogMux8`
expose convenience booleans to emit models automatically; check their docstrings
for arguments.

Example:
```python
from spicelab.core.components import Diode

D1 = Diode("D1", model="DFAST")
c.add_directive(".model DFAST D(Is=1e-14 Rs=0.5)")
```

## Visual preview

Use `Circuit.summary()` to inspect connectivity and `Circuit.render_svg()` to
export a quick Graphviz diagram (see `examples/circuit_preview.py`).

For full workflows (sweeps, Monte Carlo, engines) read the dedicated guides –
components slot directly into those orchestration helpers.
