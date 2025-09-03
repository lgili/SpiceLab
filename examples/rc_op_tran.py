from __future__ import annotations

from _common import savefig
from cat.analysis import OP, TRAN
from cat.analysis.viz.plot import plot_traces
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND

# RC simples
c = Circuit("ex_rc_op_tran")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")

c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])  # vin
c.connect(R1.ports[1], C1.ports[0])  # vout
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

# OP
op = OP().run(c)
print("OP traces:", op.traces.names)

# TRAN
tran = TRAN("50us", "2ms").run(c)
print("TRAN traces:", tran.traces.names)

# Plota saída no tempo
fig = plot_traces(tran.traces, ys=["v(out)"], title="RC step response")
savefig(fig, "rc_tran.png")
