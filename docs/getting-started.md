# Getting started

Create a Circuit and build a netlist:

```python
from cat.core.circuit import Circuit
from cat.core.components import Vdc, Resistor, Capacitor
from cat.core.net import GND

c = Circuit("rc_lowpass")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")

c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

print(c.build_netlist())
```

This will print a SPICE netlist ready to be executed by a SPICE runner.
