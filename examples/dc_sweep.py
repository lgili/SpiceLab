from __future__ import annotations

import matplotlib.pyplot as plt
from _common import savefig
from cat.analysis import DC
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND

c = Circuit("ex_dc_sweep")
V1 = Vdc("1", 0.0)
R1 = Resistor("1", "1k")
c.add(V1, R1)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], GND)
c.connect(V1.ports[1], GND)

res = DC("1", 0.0, 5.0, 0.1).run(c)

# Plota V(vin)
ts = res.traces
x = ts.x.values
vin = ts["v(n1)"].values if "v(n1)" in ts.names else ts.names[-1]

fig = plt.figure()
ax = fig.gca()
ax.plot(x, ts[vin].values)
ax.set_xlabel(ts.x.name)
ax.set_ylabel(vin)
ax.set_title("DC Sweep")
ax.grid(True)
fig.tight_layout()
savefig(fig, "dc_sweep.png")
