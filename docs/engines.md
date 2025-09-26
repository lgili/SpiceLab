# Engines

Circuit Toolkit abstracts SPICE engines behind a small orchestrator. You provide
a circuit plus one or more `AnalysisSpec`s; the library takes care of invoking
the chosen engine, collecting artefacts, and normalising the output.

## Selecting an engine
```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

tran = AnalysisSpec("tran", {"tstep": 1e-6, "tstop": 1e-3})
handle = run_simulation(circuit, [tran], engine="ngspice")
print(handle.dataset().data_vars)
```

`engine` accepts `"ngspice"`, `"ltspice"`, `"xyce"` (or the explicit `*-cli`
aliases). For lower-level access, call `get_simulator(name)`.

Command-line examples honour the `--engine` flag and the `SPICELAB_ENGINE`
environment variable.

## Binary discovery

The CLI adapters search the PATH and fall back to environment variables:

| Engine | Environment variable |
|--------|----------------------|
| NGSpice | `SPICELAB_NGSPICE` |
| LTspice | `SPICELAB_LTSPICE` or `LTSPICE_EXE` |
| Xyce | `SPICELAB_XYCE` |

Set the variable to the absolute path of the executable when auto-discovery
fails. Missing binaries raise `EngineBinaryNotFound` with install hints.

## Output formats

| Engine | Output | Notes |
|--------|--------|-------|
| NGSpice | ASCII RAW | Control block forces ASCII. |
| LTspice | ASCII RAW | The adapter passes `-b` and discovers the generated RAW/log files. |
| Xyce | PRN / CSV | Parsed into canonical xarray datasets. |

`handle.dataset()` always returns an `xarray.Dataset` with canonical coordinates
(`time`, `freq`, optional `step`) and signal names (`V(node)`, `I(element)`).

## Parallelism & caching
- pass `workers>1` to `run_simulation` (or `run_value_sweep` / `run_param_grid` /
  `monte_carlo`) to enable thread pools.
- set `cache_dir` to a persistent location to reuse hashed job results across runs.

## Troubleshooting
- **Binary not found** – install the engine and/or set the environment variable.
- **LTspice still writes binary RAW** – update to the latest LTspice build; on
  macOS the CLI always emits ASCII.
- **xarray missing** – install optional dependency (`pip install xarray`).
