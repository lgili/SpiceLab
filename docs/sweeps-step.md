# Sweeps and .STEP

There are two ways to sweep parameters:

- Python-side sweeps (`step_param`, `step_grid`) — easiest to combine with Pandas.
- Native SPICE `.step` directives — use `run_step_native` to parse multi-plot RAW.

![STEP grid](assets/examples/step_grid.png)

## Python sweeps

Single parameter:
```python
from cat.analysis import step_param, TRAN
res = step_param(c, name="R", values=["1k","2k","5k"], analysis_factory=lambda: TRAN("50us","5ms"))
```

Grid of parameters:
```python
from cat.analysis import ParamGrid, step_grid
grid: ParamGrid = {"R": ["1k","2k"], "C": ["100n","220n"]}
step = step_grid(c, grid, analysis_factory=lambda: TRAN("50us","5ms"))
```

Stacking to DataFrame:
```python
from cat.analysis import stack_step_to_df
df = stack_step_to_df(step, y=["v(out)"], with_x=True)
```

Plotting grouped curves (optional):
```python
from cat.analysis.viz.plot import plot_sweep_df
fig = plot_sweep_df(df, x="time", y="v(out)", hue="R")
```

## Native .STEP

Provide directives that include `.step` and parse all plots:
```python
from cat.analysis import run_step_native
out = run_step_native(c, directives=[".step param R list 1k 2k 5k", ".tran 50us 5ms"])
sets = out.tracesets  # list[TraceSet]
```
