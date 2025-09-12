# Plotting

CAT includes light plotting helpers that use Matplotlib when available.

## Plot traces
```python
from cat.analysis.viz.plot import plot_traces
fig = plot_traces(res.traces, ys=["v(out)"], title="Vout vs time")
```

## Bode plot
```python
from cat.analysis.viz.plot import plot_bode
# y must be a complex trace (e.g., AC result)
fig_mag, fig_phase = plot_bode(res.traces, y="v(out)")
```

## Plot sweeps (stacked DataFrame)
```python
from cat.analysis.viz.plot import plot_sweep_df
fig = plot_sweep_df(df, x="time", y="v(out)", hue="R", title="Sweep R")
```

If Matplotlib is not installed, these helpers raise a clear error.
