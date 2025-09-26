# Examples Overview

The repository ships a small set of runnable scripts under `examples/`.
All of them use the unified orchestrator (`run_simulation`,
`run_value_sweep`, `run_param_grid`) and can target any supported engine
(NGSpice, LTspice CLI, Xyce).

| Script | Description |
| --- | --- |
| `rc_tran.py` | RC transient simulation using `run_simulation`. |
| `rc_ac_unified.py` | AC analysis with engine selection via CLI flag. |
| `sweep_value_unified.py` | Single-component sweep with caching. |
| `step_sweep_grid.py` | Multi-parameter sweep; exports a CSV table. |
| `step_sweep_fig.py` | Same sweep rendered with Plotly. |
| `mc_demo_plots.py` | Generates synthetic Monte Carlo plots for docs. |
| `monte_carlo_demo.py` | Monte Carlo demo with orchestrator fallback and HTML/PNG exports. |
| `analog_mux_demo.py` | Inspects the `AnalogMux8` component topology. |
| `engine_ac_demo.py` | Minimal AC example using the orchestrator. |
| `xyce_tran.py` | RC transient aimed at Xyce with optional Plotly exports. |

Run them from the repository root:
```bash
uv run --active python examples/rc_tran.py
```
