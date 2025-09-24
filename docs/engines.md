# Engines & Backends

This page shows how to pick a simulation engine (ngspice, LTspice, Xyce), how the CLI backends are discovered, and a quick feature matrix for the current M3 state.

## Selecting an engine

You can run the same circuit/analyses with different engines using the orchestrator:

```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import run_simulation

analyses = [
    AnalysisSpec("tran", {"tstep": 1e-6, "tstop": 1e-3}),
]
handle = run_simulation(circuit, analyses, engine="ngspice")  # or "ltspice" / "xyce"
ds = handle.dataset()
```

Or obtain the simulator explicitly:

```python
from spicelab.core.types import AnalysisSpec
from spicelab.engines import get_simulator

sim = get_simulator("xyce")
handle = sim.run(circuit, [AnalysisSpec("ac", {"sweep_type": "dec", "n": 10, "fstart": 10.0, "fstop": 1e6})])
```

Result datasets are normalized to xarray.Dataset with conventional coordinates:
- time for transient analyses
- freq for AC analyses

## Binary discovery and environment variables

The CLI adapters auto-detect engine binaries. You can override discovery with env vars:

- ngspice: set CAT_SPICE_NGSPICE to the full path of the ngspice executable.
- LTspice: set SPICELAB_LTSPICE to the LTspice executable.
  - macOS default: `/Applications/LTspice.app/Contents/MacOS/LTspice` (or `LTspiceXVII`).
- Xyce: set SPICELAB_XYCE to the Xyce executable.

If a binary cannot be found, the adapters raise a clear RuntimeError explaining how to configure it.

## Installation hints (quick)

- ngspice
  - macOS: `brew install ngspice`
  - Debian/Ubuntu: `sudo apt-get install ngspice`
  - Windows: installer from the ngspice project site; add to PATH
- LTspice
  - Download from Analog Devices (ADI). On macOS it is an app bundle at `/Applications/LTspice.app`.
  - If not on PATH, point SPICELAB_LTSPICE to the embedded binary.
- Xyce
  - Prebuilt packages exist for some platforms; alternatively build from source. Ensure the `Xyce` binary is on PATH or set SPICELAB_XYCE.

## Notes on outputs and readers

- ngspice and LTspice backends generate RAW files; our readers expect ASCII RAW. The adapters force ASCII where possible (ngspice via control block; LTspice via `-ascii`).
- Xyce writes `.prn` or `.csv` tables; these are read with `read_xyce_table()`.
- Dataset attrs include engine metadata and normalized analysis summaries.

## Mini feature matrix (M3)

| Engine     | Interface | Parallel | Callbacks (shared lib) | Noise | Output format |
|------------|-----------|----------|------------------------|-------|----------------|
| ngspice    | CLI       | No       | No                     | No    | RAW (ASCII)    |
| LTspice    | CLI       | No       | No                     | No    | RAW (ASCII)    |
| Xyce       | CLI       | Yes      | No                     | No    | PRN/CSV        |

Notes:
- “Parallel” reflects whether the engine commonly supports parallel modes in CLI; the Xyce backend does not configure MPI here, but Xyce itself supports parallel execution.
- Callback/Shared Lib support is planned for a future ngspice shared backend (M5).

## Troubleshooting

- “binary not found” → set the appropriate env var or install the engine on PATH.
- LTspice RAW not recognized → ensure ASCII output (the adapter passes `-ascii`). If your build still writes binary RAW, convert to ASCII or switch to ngspice/Xyce for now.
- Missing xarray → install optional dependency: `pip install xarray`.
