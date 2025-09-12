from cat.analysis import TRAN
from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND, Net
from cat.spice import ngspice_cli

c = Circuit("rc_lowpass")
vin, vout = Net("vin"), Net("vout")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "100n")
c.add(V1, R1, C1)
c.connect(V1.ports[0], vin)
c.connect(V1.ports[1], GND)
c.connect(R1.ports[0], vin)
c.connect(R1.ports[1], vout)
c.connect(C1.ports[0], vout)
c.connect(C1.ports[1], GND)

res = TRAN("10us", "5ms").run(c)
print("traces:", res.traces.names)
ngspice_cli.cleanup_artifacts(res.run.artifacts)
