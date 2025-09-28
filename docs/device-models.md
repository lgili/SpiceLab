# Device models

Many components require SPICE `.model` cards or `.include` files. spicelab
simply embeds the directives you provide – it does not ship vendor models.

## Adding a model
```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Diode

c = Circuit("diode_demo")
D1 = Diode("D1", model="DFAST")
c.add(D1)
c.add_directive(".model DFAST D(Is=1e-14 Rs=0.5 N=1.1)")
```

The directive is emitted above the rest of the netlist when you call
`circuit.build_netlist()`.

## Including external libraries
```python
c.add_directive('.include "./models/opamp.sub"')
```

Use `.lib`, `.param`, `.func`, etc. as needed. Directives are preserved verbatim.

## Organising models
- Keep vendor libraries under `models/` in your project and reference them with
  relative paths.
- Version-control the model source so Monte Carlo and regression tests remain deterministic.
- Pair directives with `netlist_hash` metadata when caching simulations – the hash
  should incorporate the model text to avoid stale caches.
