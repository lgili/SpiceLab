# Plotting

CAT ships interactive Plotly helpers for common visualization tasks. Each helper
returns a `VizFigure`, which wraps the underlying Plotly figure and exposes
`show()`, `to_html()`, and `to_image()` convenience methods.

Install the optional ``viz`` extra to pull in Plotly and Kaleido (for static image
export):

```bash
pip install spicelab[viz]
```

## Time-domain traces
```python
from spicelab.analysis.viz.plot import plot_traces

fig = plot_traces(res.traces, ys=["v(n1)", "i(R1)"], title="RC response")
fig.to_html("rc_response.html")  # standalone HTML with embedded Plotly figure
```

## Bode plots
```python
from spicelab.analysis.viz.plot import plot_bode

# 'y' must reference a complex trace (AC/Small-signal analysis)
bode_fig = plot_bode(res.traces, y="v(vout)")
bode_fig.show()  # open an interactive browser window
```

## Parameter sweeps
```python
from spicelab.analysis.viz.plot import plot_sweep_df

fig = plot_sweep_df(df, x="time", y="v(vout)", hue="R", title="Parameter sweep")
fig.to_image("sweep.png")  # requires kaleido; falls back to HTML otherwise
```

Explore ``spicelab.viz`` for lower-level building blocks (`time_series_view`, `bode_view`,
`monte_carlo_histogram`, `monte_carlo_param_scatter`, etc.) when you want to assemble
custom dashboards or compose multiple views manually.
