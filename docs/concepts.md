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
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND

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

## Results and post-processing

After running an analysis, you receive a result handle that provides multiple views of the data and metadata.

### Data access

- `handle.dataset()` → `xarray.Dataset`
- `handle.to_pandas()` → `pandas.DataFrame`
- `handle.to_polars()` → `polars.DataFrame`

### Metadata

Call `handle.attrs()` to obtain a dict-like structure with descriptive attributes, for example:

```
attrs = handle.attrs()
print(attrs["engine"])           # "ngspice"
print(attrs.get("engine_version"))
print(attrs.get("netlist_hash"))
print(attrs.get("analysis_modes"))     # ["tran"], ["ac"], ["dc"], ...
print(attrs.get("analysis_params"))    # [{"mode": "tran", "tstep": ..., "tstop": ...}]
```

These normalized attributes make it easy to build reports and cache results deterministically.

For DC sweeps, the dataset also carries convenience attributes:

```
ds = handle.dataset()
print(ds.attrs.get("sweep_src"))    # e.g., "V1" (source name)
print(ds.attrs.get("sweep_unit"))   # "V" or "A" when inferable
print(ds.attrs.get("sweep_label"))  # original coordinate label before renaming to "sweep"
```
