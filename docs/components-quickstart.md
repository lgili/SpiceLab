# Components quick reference

Short aliases for the most common Circuit Toolkit components:

| Helper | Component | Example |
|--------|-----------|---------|
| `R(value)` | `Resistor` | `R("10k")` |
| `C(value)` | `Capacitor` | `C("100n")` |
| `L(value)` | `Inductor` | `L(1e-3)` |
| `V(value)` | `Vdc` | `V(5.0)` |
| `I(value)` | `Idc` | `I(1e-3)` |
| `VP(...)` | `Vpulse` | `VP(0, 5, td=0, tr=1e-6, tf=1e-6, pw=1e-3, per=2e-3)` |
| `VSIN(...)` | `Vsin` | `VSIN(vdc=0, vac=1, freq=1e3)` |
| `VPWL(points)` | `Vpwl` | `VPWL([(0,0), (1e-3,5)])` |
| `OA(gain)` | `OpAmpIdeal` | `OA(1e6)` |
| `E(ref, gain)` | `VCVS` | `E("1", 2.0)` |
| `G(ref, gm)` | `VCCS` | `G("1", 1e-3)` |

All helpers come from `spicelab.core.components`. They return fully fledged
component instances that you add to a `Circuit` and connect via ports.

Remember to add any required `.model` directives for diodes, BJTs, switches, or
your own subcircuits:
```python
c.add_directive(".model SWMOD VSWITCH(Ron=1 Roff=1Meg Vt=2 Vh=0.5)")
```
