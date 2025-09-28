# Plotting

spicelab ships interactive Plotly helpers for common visualization tasks. Each helper
returns a `VizFigure`, which wraps the underlying Plotly figure and exposes
`show()`, `to_html()`, and `to_image()` convenience methods.

Install the optional ``viz`` extra to pull in Plotly and Kaleido (for static image
export):

```bash
pip install spicelab[viz]
```

## Time-domain traces
```python
from spicelab.viz import plot_traces

ds = handle.dataset()
fig = plot_traces(ds, ys=["V(R)", "I(R)"] , title="RC response")
fig.to_html("rc_response.html")  # standalone HTML with embedded Plotly figure
```

## Bode plots
```python
from spicelab.viz import plot_bode

# 'y' must reference a complex trace (AC/Small-signal analysis)
ac_ds = ac_handle.dataset()
bode_fig = plot_bode(ac_ds, y="V(vout)")
bode_fig.show()  # open an interactive browser window
```

## Step responses
```python
from spicelab.viz import plot_step_response

tran_ds = tran_handle.dataset()  # time-domain dataset
step_fig = plot_step_response(
	tran_ds,
	y="V(vout)",
	settle_tolerance=0.02,
	title="RC step response",
)
step_fig.to_html("rc_step_response.html")
```

`plot_step_response` annotates rise time, overshoot and settling bands automatically when
enough data is available. Pass `x="time"` to force a specific coordinate or override
`steady_state`/`initial_value` when the default heuristics are not sufficient.

## Nyquist plots
```python
from spicelab.viz import plot_nyquist

loop_ds = ac_loop_handle.dataset()  # AC dataset with loop gain trace
nyquist_fig = plot_nyquist(loop_ds, y="loop_gain", title="Loop Nyquist")
nyquist_fig.show()
```

The Nyquist helper draws the complex trajectory of the selected trace and highlights the
last segment direction to simplify stability inspections. Provide `xlabel`/`ylabel` to
rename axes or disable `show_arrow` when exporting static reports.

## Parameter sweeps
```python
from spicelab.viz import plot_sweep_df

fig = plot_sweep_df(df, x="time", y="v(vout)", hue="R", title="Parameter sweep")
fig.to_image("sweep.png")  # requires kaleido; falls back to HTML otherwise
```

Explore ``spicelab.viz`` for lower-level building blocks (`time_series_view`, `bode_view`,
`monte_carlo_histogram`, `monte_carlo_param_scatter`, etc.) when you want to assemble
custom dashboards or compose multiple views manually.
