## Components Guide

This page shows how to instantiate and use the available components in
PyCircuitKit, with small runnable snippets. All examples use the same pattern:

- Create a `Circuit` and add components
- Connect ports to named nets (or `GND`)
- Optionally run an analysis (OP/TRAN/AC/DC) to verify behavior

Minimal scaffolding used below:

```
from spicelab.core.circuit import Circuit
from spicelab.core.net import GND, Net
from spicelab.core.components import *  # convenient in examples
from spicelab.analysis import OP, TRAN, AC, DC
```

Tip: Run examples from docs by copying blocks into a file and using
`python your_file.py` (after `pip install -e .`) or `PYTHONPATH=. python your_file.py` from repo root.

### Passive Components

- `Resistor(ref, value)` — two-terminal R. Value accepts numeric or suffixed string (e.g. "1k").
- `Capacitor(ref, value)` — two-terminal C.
- `Inductor(ref, value)` — two-terminal L.

```
c = Circuit("rc")
vin = Net("vin"); vout = Net("vout")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")
c.add(V1, R1, C1)
c.connect(V1.ports[0], vin); c.connect(V1.ports[1], GND)
c.connect(R1.ports[0], vin); c.connect(R1.ports[1], vout)
c.connect(C1.ports[0], vout); c.connect(C1.ports[1], GND)
res = OP().run(c)
print("V(vout)=", float(res.traces["v(vout)"].values[-1]))
```

### Independent Sources (Voltage/Current)

- DC: `Vdc`, `Idc`
- AC (small-signal): `Vac(ac_mag=..., ac_phase=...)`, `Iac(...)`
- PULSE: `Vpulse(v1, v2, td, tr, tf, pw, per)`, `Ipulse(i1, i2, td, tr, tf, pw, per)`
- SIN: `Vsin(vdc, vac, freq, td, theta)`, `Isin(...)`
- PWL: `Vpwl(args_raw)`, `Ipwl(args_raw)` where `args_raw` is the inside of PWL(...)

Helpers with auto-references are available: `V`, `VA`, `VP`, `I`, `IA`, `IP`.

```
# Transient: step 0->1V into RC
c = Circuit("tran_step")
vin = Net("vin"); vout = Net("vout")
Vp = Vpulse("1", 0.0, 1.0, 0.0, 1e-6, 1e-6, 1e-3, 2e-3)
R = Resistor("1", 1000.0); C = Capacitor("1", 1e-6)
c.add(Vp, R, C)
c.connect(Vp.ports[0], vin); c.connect(Vp.ports[1], GND)
c.connect(R.ports[0], vin); c.connect(R.ports[1], vout)
c.connect(C.ports[0], vout); c.connect(C.ports[1], GND)
res = TRAN("0.1ms", "5ms").run(c)
print("last V(vout)=", float(res.traces["v(vout)"].values[-1]))
```

### Controlled Sources (E/G/F/H)

- `VCVS(ref, gain)` → E: voltage-controlled voltage source
- `VCCS(ref, gm)` → G: voltage-controlled current source
- `CCCS(ref, ctrl_vsrc, gain)` → F: current-controlled current source (uses a voltage source name as sensor)
- `CCVS(ref, ctrl_vsrc, r)` → H: current-controlled voltage source

```
# VCVS gain=2 driving 1k load
c = Circuit("vcvs")
vin = Net("vin"); vout = Net("vout")
V1 = Vdc("1", 1.0); E1 = VCVS("1", 2.0); RL = Resistor("L", 1000.0)
c.add(V1, E1, RL)
c.connect(V1.ports[0], vin); c.connect(V1.ports[1], GND)
c.connect(E1.ports[2], vin); c.connect(E1.ports[3], GND)  # control
c.connect(E1.ports[0], vout); c.connect(E1.ports[1], GND)  # output
c.connect(RL.ports[0], vout); c.connect(RL.ports[1], GND)
print(float(OP().run(c).traces["v(vout)"].values[-1]))  # ≈ 2.0 V
```

### Diodes and Switches

- `Diode(ref, model)` and add a matching `.model` directive to the circuit.
- Voltage/Current-controlled switches: `VSwitch`, `ISwitch` (require `.model <name> SW(...)`).

```
# Diode biased from 1V through 1k
c = Circuit("d_fwd")
n1 = Net("n1")
V = Vdc("1", 1.0); R = Resistor("1", 1000.0); D = Diode("1", "D1")
c.add_directive(".model D1 D(Is=1e-14)")
c.add(V, R, D)
c.connect(V.ports[0], R.ports[0]); c.connect(R.ports[1], n1)
c.connect(D.ports[0], n1); c.connect(D.ports[1], GND)
c.connect(V.ports[1], GND)
print(float(OP().run(c).traces["v(n1)"].values[-1]))
```

### Ideal Op-Amp

- `OpAmpIdeal(ref, gain)` or helper `OA(gain)`; three ports `(inp, inn, out)`, modeled as VCVS.

```
# Unity-gain buffer
c = Circuit("oa_buf")
vin = Net("vin"); vout = Net("vout")
V = Vdc("1", 1.0); u = OpAmpIdeal("1", 1e6)
c.add(V, u)
c.connect(V.ports[0], vin); c.connect(V.ports[1], GND)
c.connect(u.ports[0], vin); c.connect(u.ports[2], vout); c.connect(u.ports[1], vout)
print(float(OP().run(c).traces["v(vout)"].values[-1]))  # ≈ 1.0 V
```

### Analog Multiplexer (1-to-8)

Use `AnalogMux8` to connect a single input to one of eight outputs.

- Static selection: `sel=N` chooses `outN` with series resistance `r_series` (others at `off_resistance`).
- Dynamic enable ports: `enable_ports=True` exposes `en0..en7` and emits `S...` elements; add a `.model` for the switch (or `emit_model=True`).

```
from spicelab.core.components import AnalogMux8
mux = AnalogMux8(ref="MU1", r_series=100, sel=4)
print(mux.spice_card(lambda p: p.name))
```

### Running Analyses

```
# OP (DC operating point)
res = OP().run(c)
# TRAN (time-domain)
res = TRAN("1us", "1ms").run(c)
# AC (small-signal)
res = AC("dec", 20, 10.0, 1e6).run(c)
# DC sweep
res = DC("1", 0.0, 5.0, 0.1).run(c)
```

### Notes

- Ensure `ngspice` is installed and available in PATH.
- Numeric strings with suffixes are accepted (e.g. `"1k"`, `"100n"`). See `spicelab.utils.units`.
- Controlled sources `F/H` require the name of a (dummy) voltage source for current sensing.
