# Using the examples

Circuit Toolkit ships runnable demos under `examples/`. Execute them from the
repository root with `uv run --active python examples/<script>.py`.

1. Ensure an engine is available. NGSpice is the default:
   ```bash
   brew install ngspice           # macOS
   sudo apt install ngspice      # Debian/Ubuntu
   ```

   Xyce users can install the binary from <https://xyce.sandia.gov> and then export
   `SPICELAB_XYCE=/path/to/xyce` so Circuit Toolkit can find it.

2. Install optional plotting dependencies if you want PNG/HTML output:
   ```bash
   uv run --active pip install matplotlib pandas
   uv run --active pip install "spicelab[viz]"   # PyPI install with Plotly extras
   # working from a clone? alternatively: uv run --active pip install -e '.[viz]'
   ```

3. Run any script from the repository root. Examples:
   ```bash
   uv run --active python examples/rc_tran.py
   uv run --active python examples/sweep_value_unified.py
   uv run --active python examples/step_sweep_grid.py
   uv run --active python examples/xyce_tran.py --engine xyce
   uv run --active python examples/monte_carlo_demo.py --real --engine ngspice --workers 2 --metric-col "V(vout)"
   ```

Most scripts print the engine they used, dataset coordinates/variables, and
persist plots or CSVs next to the script.

Monte Carlo tip: pass `--metric-col` to `monte_carlo_demo.py` when you want to force a
specific column from the orchestrator dataframe (for example, `V(vout)`), ensuring
the comparison plots use the metric you expect. HTML exports land in the directory
you provide via `--out-html` (`mc_hist.html`, `mc_param_scatter.html`, `mc_params_matrix.html`).

Xyce transient tip: `examples/xyce_tran.py` falls back gracefully when the engine
is missing, so you can run it even before installing Xyce to verify your setup steps.
