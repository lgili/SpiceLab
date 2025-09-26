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
