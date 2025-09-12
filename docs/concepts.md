# Core Concepts

- Ports: typed terminals on components (positive/negative/node).
- Nets: logical nodes (use `GND` for ground). A `Net` can be named for readability.
- Components: objects with `ports` and a `spice_card(net_of)` method.
- Circuit: container that owns components, connections, and optional SPICE directives.

## Wiring rules
- Connect `Port` to `Net`, or `Port` to `Port` (auto-creates a shared Net).
- All ports must be connected before building the netlist.
- `GND` is reserved (node "0").

## Minimal RC example
```python
from cat.core.circuit import Circuit
from cat.core.components import Vdc, Resistor, Capacitor
from cat.core.net import GND

c = Circuit("rc")
V1, R1, C1 = Vdc("1", 5.0), Resistor("1", "1k"), Capacitor("1", "100n")
c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])  # vin
c.connect(R1.ports[1], C1.ports[0])  # vout
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

print(c.build_netlist())
```

## Directives
Use `circuit.add_directive(".model ...")` or `.param`, `.include`, etc., to embed raw SPICE lines.
