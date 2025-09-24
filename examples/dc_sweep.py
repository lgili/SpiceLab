from __future__ import annotations

import matplotlib.pyplot as plt
from _common import savefig
from spicelab.analysis import DC
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.spice import ngspice_cli

c = Circuit("ex_dc_sweep")
V1 = Vdc("1", 0.0)
R1 = Resistor("1", "1k")
c.add(V1, R1)
vin = Net("vin")
c.connect(V1.ports[0], vin)
c.connect(R1.ports[0], vin)
c.connect(R1.ports[1], GND)
c.connect(V1.ports[1], GND)

res = DC("1", 0.0, 5.0, 0.1).run(c)

# Plota V(vin)
ts = res.traces
x = ts.x.values
name = "v(vin)" if "v(vin)" in ts.names else ("v(n1)" if "v(n1)" in ts.names else ts.names[-1])

fig = plt.figure()
ax = fig.gca()
ax.plot(x, ts[name].values)
ax.set_xlabel(ts.x.name)
ax.set_ylabel(name)
ax.set_title("DC Sweep")
ax.grid(True)
fig.tight_layout()
savefig(fig, "dc_sweep.png")
ngspice_cli.cleanup_artifacts(res.run.artifacts)
