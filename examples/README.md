# Circuit Toolkit examples

A handful of scripts demonstrating the unified orchestrator and helper APIs.
Run them from the repository root with `uv run --active python examples/<script>.py`.

| Script | Highlights |
| --- | --- |
| `rc_tran.py` | Simple RC transient using `run_simulation` |
| `rc_ac_unified.py` | AC sweep orchestrated through any engine |
| `sweep_value_unified.py` | Single-component sweep with caching |
| `step_sweep_grid.py` | Multi-parameter grid sweep and CSV export |
| `step_sweep_fig.py` | Parameter grid plotted with Plotly helpers |
| `mc_demo_plots.py` | Generates Monte Carlo demo figures (synthetic data) |
| `analog_mux_demo.py` | Inspect the analog mux topology and generate DOT |
| `engine_ac_demo.py` | Minimal AC analysis using `run_simulation` |

Most scripts save output data/plots next to the script. A working NGSpice
installation on PATH is recommended for the ones that invoke an engine.
