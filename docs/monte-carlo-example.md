# Monte Carlo example

`examples/mc_demo_plots.py` generates the static assets used throughout the
Circuit Toolkit Monte Carlo documentation. It does not require a simulator: the script creates
synthetic samples and feeds them to the plotting helpers.

```bash
uv run --active python examples/mc_demo_plots.py
```

The script writes the following PNG files under `docs/assets/examples/`:

- `mc_hist.png` – histogram of a synthetic metric.
- `mc_metric_kde.png` – kernel density estimate of the same metric.
- `mc_R1_vs_metric.png` – scatter plot correlating a parameter with the metric.
- `mc_params_matrix.png` – pairwise parameter scatter matrix.

Use these images directly in your docs or replace them with plots generated from
real Monte Carlo runs (`MonteCarloResult.to_dataframe()` + Plotly, Matplotlib,
etc.).

## Real orchestrator demo

For an end-to-end example that talks to a real simulator (when available), run
`examples/monte_carlo_demo.py`. The script attempts three modes, in order:

1. Orchestrator-backed Monte Carlo via `spicelab.analysis.monte_carlo` (supports
	 workers, caching, synthetic fallbacks).
2. Per-trial `run_simulation` if the orchestrator helper is unavailable.
3. Synthetic sampling when no engine is present.

Key CLI flags:

- `--real --engine <name>`: request an actual engine (for example, `ngspice`).
- `--workers` / `--cache-dir`: forwarded to the orchestrator helper.
- `--metric-col <column>`: prefer a specific metric column from the dataframe.
- `--out-html <dir>` / `--out-img <dir>`: store `mc_hist`, `mc_param_scatter`, and
	`mc_params_matrix` in interactive HTML and/or static PNG/SVG format.

Example run:

```bash
PYTHONPATH=. uv run --active python examples/monte_carlo_demo.py \
		--real --engine ngspice --workers 2 --metric-col "V(vout)" --out-html out
```

When the orchestrator path is used, the sampled parameter dictionaries are augmented
with aliases (e.g., `R`, `C`) so the plotting helpers always have a numeric axis to
render even if only a single component varies.
