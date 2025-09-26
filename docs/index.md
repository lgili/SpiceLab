# Circuit Toolkit

Circuit Toolkit (Python package `spicelab`) is a modern interface for building
SPICE netlists, orchestrating simulations across multiple engines, and
post-processing results with familiar scientific Python tools.

![Circuit Toolkit logo](assets/logo.svg){ width="260" }

## Why Circuit Toolkit?
- **Unified orchestration** – drive NGSpice, LTspice CLI, and Xyce from one API.
- **Deterministic caching** – hashed jobs avoid rerunning the same sweep or Monte Carlo trial.
- **Typed circuits** – ports, nets, and components are Python objects, not stringly-typed blobs.
- **First-class data access** – result handles expose xarray, pandas, and polars views with rich metadata.
- **Docs & examples** – runnable scripts show how to wire circuits, sweeps, and metrics.

## Quick Preview
```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

c = Circuit("rc_demo")
V1 = Vdc("VIN", 5.0)
R1 = Resistor("R", "1k")
C1 = Capacitor("C", "100n")
for comp in (V1, R1, C1):
    c.add(comp)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

tran = AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})
handle = run_simulation(c, [tran], engine="ngspice")
print(handle.dataset()["V(R)"])
```

## Explore
- [Installation](installation.md) – set up UV/virtualenv and optional engines.
- [Getting Started](getting-started.md) – build a circuit, run an analysis, inspect datasets.
- [Concepts](concepts.md) – circuits, components, ports, nets, handles.
- [Engines](engines.md) – orchestration and caching model.
- [Sweeps](sweeps-step.md) · [Monte Carlo](monte-carlo.md) – parametric workflows.
- [Data I/O](unified-io.md) – ingestion of NGSpice/LTspice/Xyce RAW/PRN/CSV files.
- [Cookbook](cookbook.md) – copy/paste snippets for metrics and data handling.
- [Examples](examples.md) – runnable scripts with plots and figures.

## Package status
Circuit Toolkit is actively evolving. During the API freeze the package name
remains `spicelab`; future releases will publish under the `circuit-toolkit`
name on PyPI. Breaking changes are documented in the changelog and reflected in
these docs.
