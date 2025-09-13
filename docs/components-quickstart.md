## Components Quickstart

A compact cheat‑sheet for the main components exposed by PyCircuitKit. The
table shows the Python class, an optional helper function, the SPICE card that
is generated, port names, and relevant notes.

Import helpers used below:

```
from cat.core.circuit import Circuit
from cat.core.net import GND, Net
from cat.core.components import *
```

| Python (Class) | Helper | SPICE Card (shape) | Ports | Notes |
| --- | --- | --- | --- | --- |
| `Resistor` | `R(value)` | `Rref a b value` | `a, b` | Value accepts suffixes (`"1k"`, `"100n"`). |
| `Capacitor` | `C(value)` | `Cref a b value` | `a, b` |  |
| `Inductor` | `L(value)` | `Lref a b value` | `a, b` |  |
| `Vdc` | `V(value)` | `Vref p n value` | `p, n` | DC source. |
| `Vac` | `VA(ac_mag, ac_phase=0)` | `Vref p n AC mag [phase]` | `p, n` | Small‑signal AC. |
| `Vpulse` | `VP(v1, v2, td, tr, tf, pw, per)` | `Vref p n PULSE(...)` | `p, n` |  |
| `Vsin` | — | `Vref p n SIN(vdc vac freq td theta)` | `p, n` | Sine. |
| `Vpwl` | — | `Vref p n PWL(<args_raw>)` | `p, n` | Raw args string. |
| `Idc` | `I(value)` | `Iref p n value` | `p, n` | DC current source. |
| `Iac` | `IA(ac_mag, ac_phase=0)` | `Iref p n AC mag [phase]` | `p, n` | Small‑signal AC. |
| `Ipulse` | `IP(i1, i2, td, tr, tf, pw, per)` | `Iref p n PULSE(...)` | `p, n` |  |
| `Isin` | — | `Iref p n SIN(idc iac freq td theta)` | `p, n` |  |
| `Ipwl` | — | `Iref p n PWL(<args_raw>)` | `p, n` |  |
| `VCVS` | `E(gain)` | `Eref p n cp cn gain` | `p, n, cp, cn` | Voltage‑controlled voltage src. |
| `VCCS` | `G(gm)` | `Gref p n cp cn gm` | `p, n, cp, cn` | Voltage‑controlled current src. |
| `CCCS` | `F(ctrl_vsrc, gain)` | `Fref p n Vsrc gain` | `p, n` | Current‑controlled current src. |
| `CCVS` | `H(ctrl_vsrc, r)` | `Href p n Vsrc r` | `p, n` | Current‑controlled voltage src. |
| `Diode` | `D(model)` | `Dref a c model` | `a, c` | Requires matching `.model` directive. |
| `VSwitch` | `SW(model)` | `Sref p n cp cn model` | `p, n, cp, cn` | Needs `.model <name> SW(...)`. |
| `ISwitch` | `SWI(ctrl_vsrc, model)` | `Wref p n Vsrc model` | `p, n` | Current‑controlled switch. |
| `OpAmpIdeal` | `OA(gain=1e6)` | `Eref out 0 inp inn gain` | `inp, inn, out` | 3‑pin ideal OA, VCVS‑based. |
| `AnalogMux8` | `MUX8(r_series, sel)` | — | `in, out0..out7, [en0..en7]` | Emits Rs (static `sel`) or `S...` + R when `enable_ports=True`. |

Notes

- Numeric strings accept engineering suffixes (k, m, u/µ, n, p, g, t, `meg`).
- Controlled sources `F`/`H` use a voltage source name (`V1` etc.) as current sensor.
- `AnalogMux8` parameters: `r_series`, `sel` (0..7), `off_resistance`, `enable_ports`,
  `emit_model`, `sw_model`. When using switches, add a suitable `.model` line for the
  `SW` element (or pass `emit_model=True`).
