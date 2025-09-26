# Sweeps

The orchestrator API lets you sweep component values across any supported engine
(NGSpice, LTspice CLI, Xyce) while reusing cache entries and working with typed
result handles. Two helpers cover the common cases:

- `run_value_sweep` — sweep a single component over a list of values.
- `run_param_grid` — evaluate the Cartesian product of multiple components/values.

Both helpers return lightweight objects containing `ResultHandle` instances. You
can inspect datasets on demand and reuse cached results transparently.

## Single-component value sweep

```python
from spicelab.analysis.sweep_grid import run_value_sweep
from spicelab.core.types import AnalysisSpec

sweep = run_value_sweep(
    circuit=c,
    component=R1,
    values=["1k", "2k", "5k"],
    analyses=[AnalysisSpec("tran", {"tstep": "50us", "tstop": "5ms"})],
    engine="ngspice",
    cache_dir=".spicelab_cache",
)

for sweep_run in sweep.runs:
    ds = sweep_run.handle.dataset()
    print(sweep_run.value, float(ds["V(R1)"][-1]))
```

## Multi-parameter grid

```python
from spicelab.analysis.sweep_grid import run_param_grid

variables = [
    (V1, [1.0, 5.0]),
    (R1, ["1k", "2k"]),
    (C1, ["100n", "220n"]),
]

result = run_param_grid(
    circuit=c,
    variables=variables,
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
    cache_dir=".spicelab_cache",
    workers=4,
)

for run in result.runs:
    print(run.combo, run.from_cache)
```

Each `run.combo` is a dictionary mapping component references to the value used
for that simulation. Call `run.handle.dataset()` when you need the full
`xarray.Dataset` and `run.handle.attrs()` for metadata (engine, netlist hash,
paths to artifacts, etc.).

## Native `.step`

If you already have a netlist that contains a `.step` directive the orchestrator
helpers are not required. Run the engine directly and parse the multi-plot RAW
file with the unified readers module.
