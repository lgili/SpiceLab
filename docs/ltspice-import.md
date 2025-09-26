# LTspice integration

LTspice schematic and netlist support is being revisited for the new Circuit Toolkit API. Until
that is complete you can:

1. Export a SPICE netlist from LTspice (`View → SPICE netlist`).
2. Load it with `spicelab.io.ltspice_parser.from_ltspice_file`.
3. Run it through `run_simulation(...)` with the desired engine.

```python
from spicelab.io.ltspice_parser import from_ltspice_file
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

circuit = from_ltspice_file("./my_filter.cir")
ac = AnalysisSpec("ac", {"sweep_type": "dec", "n": 40, "fstart": 10.0, "fstop": 1e6})
handle = run_simulation(circuit, [ac], engine="ltspice")
print(handle.dataset())
```

Round-trip schematic editing (`ltspice_asc`) and property preservation will be
documented after the API refresh.
