# Using the examples

Circuit Toolkit ships runnable demos under `examples/`. Execute them from the
repository root with `uv run --active python examples/<script>.py`.

1. Ensure an engine is available. NGSpice is the default:
   ```bash
   brew install ngspice           # macOS
   sudo apt install ngspice      # Debian/Ubuntu
   ```

2. Install optional plotting dependencies if you want PNG/HTML output:
   ```bash
   uv run --active pip install matplotlib pandas
   uv run --active pip install -e '.[viz]'  # for Plotly demos
   ```

3. Run any script from the repository root. Examples:
   ```bash
   uv run --active python examples/rc_tran.py
   uv run --active python examples/sweep_value_unified.py
   uv run --active python examples/step_sweep_grid.py
   ```

Most scripts print the engine they used, dataset coordinates/variables, and
persist plots or CSVs next to the script.
