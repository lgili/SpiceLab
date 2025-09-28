# Core concepts

spicelab revolves around a few simple abstractions:

- **Ports** – typed terminals on components.
- **Nets** – named nodes (use `GND` for ground). Connecting two ports creates or
  joins a net.
- **Components** – Python objects with `ports` and a `spice_card(...)` method.
- **Circuit** – owns components, nets, and free-form SPICE directives.
- **ResultHandle** – wrapper around simulation artefacts; exposes datasets and metadata.

## Wiring rules
- Connect `Port` to `Net`, or `Port` to `Port` (implicit net).
- All ports must be connected before building the netlist.
- Ground is the reserved node `0` (use `GND`).

## Minimal RC example
```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND

c = Circuit("rc")
V1, R1, C1 = Vdc("1", 5.0), Resistor("1", "1k"), Capacitor("1", "100n")
for comp in (V1, R1, C1):
    c.add(comp)

c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)
print(c.build_netlist())
```

## Directives
Add raw SPICE lines directly:
```python
c.add_directive(".include ./models/opamp.sub")
c.add_directive(".model SWMOD VSWITCH(Ron=1 Roff=1Meg Vt=2 Vh=0.5)")
```
Directives are emitted above the element cards in the generated netlist.

## Results & metadata
Simulations return `ResultHandle` objects. The API is intentionally small:

```python
handle.dataset()        # -> xarray.Dataset
handle.to_pandas()      # -> pandas.DataFrame
handle.to_polars()      # -> polars.DataFrame
attrs = handle.attrs()  # -> Mapping[str, Any]
```

Useful attributes include `engine`, `engine_version`, `netlist_hash`, and
`analysis_params`. Handles also carry paths to the generated netlist / log / RAW
artefacts when available.

### DC sweeps
DC datasets normalise the sweep coordinate under `sweep` and record the original
label in `dataset.attrs['dc_sweep_label']` together with the source name.

## Preview helpers
`Circuit.summary()` prints a net connectivity report and
`Circuit.render_svg()` produces a quick Graphviz diagram (see
`examples/circuit_preview.py`).

These primitives feed into the higher-level orchestration helpers documented in
[Engines](engines.md), [Sweeps](sweeps-step.md) and [Monte Carlo](monte-carlo.md).
