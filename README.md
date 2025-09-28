# spicelab

[![Build](https://github.com/lgili/SpiceLab/actions/workflows/ci.yml/badge.svg)](https://github.com/lgili/SpiceLab/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://lgili.github.io/CircuitToolkit/)
[![PyPI](https://img.shields.io/pypi/v/spicelab.svg)](https://pypi.org/project/spicelab/)
[![Python](https://img.shields.io/pypi/pyversions/spicelab.svg)](https://pypi.org/project/spicelab/)
[![License](https://img.shields.io/github/license/lgili/circuit_toolkit.svg)](LICENSE)

spicelab is a typed Python layer for describing SPICE circuits, running
simulations against multiple engines (NGSpice, LTspice CLI, Xyce) and analysing
the results with familiar data libraries (xarray · pandas · polars).


---

## Highlights
- **Unified orchestrator** – run a circuit on any configured engine with one call.
- **Deterministic caching** – hashed jobs avoid re-running identical sweeps/Monte Carlo trials.
- **Typed circuits** – ports, nets and components are Python objects; no stringly-typed surprises.
- **xarray-first results** – datasets carry canonical signal names (`V(node)`, `I(element)`) and rich metadata.
- **Measurement helpers** – `.meas`-style gain/overshoot/settling specs return tidy polars DataFrames.
- **Extensible component library** – build, preview and export netlists (including Graphviz topology previews).
- **Reporting helpers** – turn simulation outputs into HTML/Markdown summaries with a few lines of code.
- **Environment doctor** – `python -m spicelab.doctor` validates engine/shared-library setup before long runs.

---

## Engine support matrix

| Feature | NGSpice | LTspice CLI | Xyce |
| --- | --- | --- | --- |
| Operating point / AC / Tran analyses | ✅ | ✅ | ✅ |
| Value/grid sweeps with caching | ✅ | ✅ | ✅ |
| Monte Carlo orchestrator | ✅ | ✅ | ✅ |
| Co-simulation callbacks | ✅ *(libngspice shared)* | — | — |
| HTML / Markdown reporting | ✅ | ✅ | ✅ |
| Plot helpers (Bode / Step / Nyquist) | ✅ | ✅ | ✅ |

LTspice and Xyce support rely on the respective CLI binaries being installed and discoverable.
Set `SPICELAB_LTSPICE` or `SPICELAB_XYCE` when the executables are not on `PATH`. Co-simulation
callbacks require the shared `libngspice` library.

---

## Quick start
Install the package straight from PyPI:

```bash
python -m pip install --upgrade pip
python -m pip install spicelab
```

Need optional helpers? Append extras such as `spicelab[viz]` for Plotly or
`spicelab[data]` for xarray/polars integrations.

Once installed, connect an engine (NGSpice, LTspice CLI, or Xyce) and run your
first transient analysis:

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

c = Circuit("rc_lowpass")
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
ds = handle.dataset()
print(list(ds.data_vars))
```

### Sweeps in one line
```python
from spicelab.analysis.sweep_grid import run_value_sweep

value_sweep = run_value_sweep(
    circuit=c,
    component=R1,
    values=["1k", "2k", "5k"],
    analyses=[tran],
    engine="ngspice",
)
for run in value_sweep.runs:
    ds = run.handle.dataset()
    print(run.value, list(ds.data_vars))
```

### Monte Carlo with typed metrics
```python
from spicelab.analysis import NormalPct, monte_carlo

mc = monte_carlo(
    circuit=c,
    mapping={R1: NormalPct(0.05)},
    n=64,
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
    seed=42,
)

print(mc.to_dataframe(metric=None, param_prefix="param_").head())
```

## Notebook workflows
- Build complex circuits quickly with the DSL:
  ```python
  from spicelab.dsl import CircuitBuilder

  builder = CircuitBuilder("rc_filter")
  builder.vdc("vin", "gnd", value="5")
  builder.resistor("vin", "vout", value="1k")
  builder.capacitor("vout", "gnd", value="220n")
  circuit = builder.build()
  circuit.connectivity_dataframe()  # pandas.DataFrame for rich display
  ```
- Use interactive widgets inside Jupyter/VS Code:
  ```python
  from spicelab.viz.notebook import connectivity_widget, dataset_plot_widget

  connectivity_widget(circuit)
  dataset_plot_widget(handle.dataset())
  ```
---

## Documentation
Full documentation lives at **https://lgili.github.io/CircuitToolkit/**:

- [Getting started](https://lgili.github.io/CircuitToolkit/getting-started/)
- [Engines & caching](https://lgili.github.io/CircuitToolkit/engines/)
- [Sweeps](https://lgili.github.io/CircuitToolkit/sweeps-step/)
- [Monte Carlo](https://lgili.github.io/CircuitToolkit/monte-carlo/)
- [Unified I/O](https://lgili.github.io/CircuitToolkit/unified-io/)
- [Cookbook snippets](https://lgili.github.io/CircuitToolkit/cookbook/)

Runnable demos are under [`examples/`](examples/) and can be executed with
`uv run --active python examples/<script>.py`. Highlights:

- `examples/closed_loop.py` – co-simulation loop where Python adjusts a source
  via the shared ngspice backend callbacks.

---

- Prefer working from source? Clone the repo and use [uv](https://github.com/astral-sh/uv):
  ```bash
  uv venv
  source .venv/bin/activate            # Linux/macOS
  # .\.venv\Scripts\activate.ps1       # Windows PowerShell
  uv pip install -e .[viz,data]
  ```

## Installation details
- Python **3.10+**
- Install from PyPI with `pip install spicelab`
- Optional extras: `spicelab[viz]` for Plotly output, `spicelab[data]` for xarray/polars helpers
- Engines (any subset): NGSpice · LTspice CLI · Xyce
- For ngspice co-simulation callbacks, also install the `libngspice` shared
  library and export `SPICELAB_NGSPICE_SHARED` (see [installation docs](https://lgili.github.io/CircuitToolkit/installation/)).
- Quick diagnostic: `python -m spicelab.doctor`

Environment overrides when binaries are not on PATH:

| Variable | Purpose |
|----------|---------|
| `SPICELAB_NGSPICE` | Absolute path to `ngspice` |
| `SPICELAB_NGSPICE_SHARED` | Absolute path to `libngspice` (`.so`/`.dylib`/`.dll`) |
| `SPICELAB_LTSPICE` | Absolute path to LTspice CLI (`LTspice`/`XVIIx64.exe`) |
| `SPICELAB_XYCE` | Absolute path to `Xyce` |
| `SPICELAB_ENGINE` | Default engine name for examples (`ngspice`, `ltspice`, `xyce`) |

---

## Contributing
- Run the formatting/lint suite: `ruff format . && ruff check . --fix`
- Run tests: `pytest`
- Static typing: `mypy`

Pull requests are welcome! Please open an issue if you plan a larger change so
we can discuss the design direction.

---

## License & acknowledgements
MIT License © Luiz Carlos Gili. spicelab stands on the shoulders of the
SPICE ecosystem (NGSpice, LTspice, Xyce) and scientific Python libraries. Many
thanks to their authors and maintainers.
