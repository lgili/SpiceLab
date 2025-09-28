# spicelab examples

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
| `rc_ac_tran.py` | RC transient + AC demo (uses real engine when available, otherwise synthetic) |
| `monte_carlo_demo.py` | Monte Carlo demo (synthetic by default; can be wired to orchestrator) |
| `read_raw_demo.py` | Demonstrates reading .raw/.prn files via readers |

Most scripts save output data/plots next to the script. A working NGSpice
installation on PATH is recommended for the ones that invoke an engine.

monte_carlo_demo.py options
---------------------------

The `examples/monte_carlo_demo.py` script supports the following CLI options (in addition to -n):

- `--real` and `--engine <name>`: attempt real simulations via the orchestrator. If the orchestrator
	helper is not available the script falls back to running single-sample simulations (if `run_simulation` is present) or synthetic sampling.
- `--workers <num>`: number of workers to use when calling the orchestrator-level `monte_carlo` helper.
- `--cache-dir <path>`: optional cache directory passed to the orchestrator helper.
- `--metric-col <name>`: prefer the specified column name from the result dataframe when extracting the scalar metric (for example: `--metric-col "V(vout)"`). This makes runs reproducible when the automatic heuristic picks the wrong column.
- `--out-html <dir>` / `--out-img <dir>`: export interactive HTML and static images of the generated plots. HTML exports include `mc_hist.html`, `mc_param_scatter.html` (first numeric parameter vs metric) and `mc_params_matrix.html` when at least two numeric parameters are available. Static images mirror the same names/extensions.

The orchestrator path enriches the sampled parameter dictionaries with friendly aliases (for example, the varied resistor `R1` is exposed as both `R1.R` and `R`) so downstream plots can always render a sensible scatter axis even when only a single component is varied.

Run example:

```bash
PYTHONPATH=. uv run --active python examples/monte_carlo_demo.py --real --engine ngspice --workers 2 --metric-col "V(vout)" --out-html out
```
