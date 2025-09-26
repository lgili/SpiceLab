# Getting Started

This quick tour shows how to create a circuit, run an analysis via the unified
engine orchestrator, and inspect the results.

## 1. Build a circuit
```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Capacitor, Resistor, Vdc
from spicelab.core.net import GND

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
```

## 2. Run an analysis
```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

tran = AnalysisSpec("tran", {"tstep": "10us", "tstop": "5ms"})
handle = run_simulation(c, [tran], engine="ngspice")
```

`handle` is a `ResultHandle`. It keeps metadata and a lazy reference to the
engine output; call `handle.dataset()` to get an `xarray.Dataset` when you need
actual data.

## 3. Inspect the dataset
```python
import pandas as pd

ds = handle.dataset()
print(ds)
print("variables:", list(ds.data_vars))
print("attrs:", handle.attrs())

# Convert to pandas if you want tabular data
print(ds.to_dataframe().head())
```

## 4. Run the bundled examples

From the repository root:
```bash
uv run --active python examples/rc_tran.py
uv run --active python examples/sweep_value_unified.py
uv run --active python examples/step_sweep_grid.py
```
Each script prints basic information and (where relevant) saves artefacts next
to the script.
