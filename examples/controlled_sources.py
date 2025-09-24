from __future__ import annotations

from spicelab.core.circuit import Circuit
from spicelab.core.components import D, E, F, G, H, R, V
from spicelab.core.net import GND, Net

# Simple demo with controlled sources and a diode
c = Circuit("controlled_sources_demo")

vin = Net("vin")
vout = Net("vout")

V1 = V(1.0)  # DC 1 V source
E1 = E(10.0)  # VCVS gain 10
G1 = G("1e-3")  # VCCS gm 1 mS
F1 = F("V1", 2.0)  # CCCS controlled by V1 current, gain 2
H1 = H("V1", 1000.0)  # CCVS controlled by V1 current, 1 kOhm
D1 = D("Dmod")  # diode with model name Dmod

c.add(V1, E1, G1, F1, H1, D1, R("1k"), R("2k"))

# Wire basics: V1 -> vin
c.connect(V1.ports[0], vin)
c.connect(V1.ports[1], GND)

# E1 output between vout and GND, control between vin and GND
p, n, cp, cn = E1.ports
c.connect(p, vout)
c.connect(n, GND)
c.connect(cp, vin)
c.connect(cn, GND)

# G1 output between vout and GND, control between vin and GND
p, n, cp, cn = G1.ports
c.connect(p, vout)
c.connect(n, GND)
c.connect(cp, vin)
c.connect(cn, GND)

# F1 between vout and GND; controlled by V1
p, n = F1.ports
c.connect(p, vout)
c.connect(n, GND)

# H1 between vout and GND; controlled by V1
p, n = H1.ports
c.connect(p, vout)
c.connect(n, GND)

# D1 between vout and GND
a, c_ = D1.ports
c.connect(a, vout)
c.connect(c_, GND)

print(c.build_netlist())
