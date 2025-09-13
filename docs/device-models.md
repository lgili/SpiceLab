## Device Models and `.model` Directives

Some components in PyCircuitKit rely on SPICE device models defined via
`.model` lines in the netlist. This page shows how to add models and wire
typical circuits using the high‑level API.

Key points:

- Use `Circuit.add_directive(line)` to append raw SPICE directives.
- Diodes must reference a named model (e.g. `D1` → `.model D1 D(...)`).
- Voltage/Current‑controlled switches (`VSwitch`, `ISwitch`) require a
  `.model <name> SW(...)` with thresholds and on/off resistances.
- Controlled sources `F`/`H` use a named voltage source as current sensor
  (no `.model`, but a topology note).

### Adding a diode model

```
from cat.core.circuit import Circuit
from cat.core.components import Vdc, Resistor, Diode
from cat.core.net import GND, Net
from cat.analysis import OP

c = Circuit("d_fwd")
n1 = Net("n1")
V = Vdc("1", 1.0)
R = Resistor("1", 1000.0)
D = Diode("1", "D1")  # references model D1

# Add a diode model (Shockley params shown as an example)
c.add_directive(".model D1 D(Is=1e-14 N=1.9 Rs=5 Cjo=2p)")

c.add(V, R, D)
c.connect(V.ports[0], R.ports[0])
c.connect(R.ports[1], n1)
c.connect(D.ports[0], n1)  # anode
c.connect(D.ports[1], GND)  # cathode
c.connect(V.ports[1], GND)

res = OP().run(c)
print("V(n1)=", float(res.traces["v(n1)"].values[-1]))
```

Typical parameters for the SPICE diode (`D`) model include `Is`, `N`, `Rs`,
`Cjo`, `Bv`, etc. Choose values appropriate to your target device.

### Voltage‑controlled switch (`VSwitch`)

```
from cat.core.circuit import Circuit
from cat.core.components import Vdc, Vpulse, VSwitch, Resistor
from cat.core.net import GND, Net
from cat.analysis import TRAN

c = Circuit("vswitch_demo")
vin = Net("vin")
vout = Net("vout")
en = Net("en")

# Control signal: 0→1 V pulse
EN = Vpulse("en", 0.0, 1.0, 0.0, 1e-6, 1e-6, 1e-3, 2e-3)
V1 = Vdc("1", 5.0)
S1 = VSwitch("1", model="SW1")
RL = Resistor("L", 1000.0)

# Recommended switch model (tune as needed)
c.add_directive(".model SW1 SW(RON=10 ROFF=1e9 Vt=0.5 VH=0.1)")

c.add(EN, V1, S1, RL)

# Source side
c.connect(V1.ports[0], vin)
c.connect(V1.ports[1], GND)

# Switch: p→vin, n→vout, control cp→en, cn→GND (active high)
c.connect(S1.ports[0], vin)   # p
c.connect(S1.ports[1], vout)  # n
c.connect(S1.ports[2], en)    # cp
c.connect(S1.ports[3], GND)   # cn

# Load
c.connect(RL.ports[0], vout)
c.connect(RL.ports[1], GND)

# Control signal wiring
c.connect(EN.ports[0], en)
c.connect(EN.ports[1], GND)

res = TRAN("0.1ms", "5ms").run(c)
print("final V(vout)=", float(res.traces["v(vout)"].values[-1]))
```

Notes for switches:

- The switch turns on when `(V(cp) − V(cn)) > Vt`. `VH` sets hysteresis.
- Choose `RON` and `ROFF` appropriate to your application.
- For current‑controlled switches (`ISwitch`), the control is the current
  through a named voltage source `Vref` (current sensor) provided via the
  `ctrl_vsrc` argument.

### Current‑controlled elements (`F`/`H`) and sensors

`CCCS` (`F`) and `CCVS` (`H`) measure the current through an existing voltage
source. A common pattern is to insert a 0 V source as sensor:

```
from cat.core.components import Vdc, Resistor, CCCS

# Sensor source at the node of interest
VSENS = Vdc("S", 0.0)
F1 = CCCS("1", ctrl_vsrc="S", gain=2.0)  # 2× current mirror

# Wire VSENS in series with the branch you want to sense
# Then connect F1.p/n to the output branch where mirrored current flows.
```

This pattern avoids modifying the DC operating point (0 V across the sensor)
while providing a reference for `F`/`H` controlled elements.
