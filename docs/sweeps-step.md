# Sweeps and .STEP

There are two ways to sweep parameters:

- Python-side sweeps (`step_param`, `step_grid`) — easiest to combine with Pandas.
- Native SPICE `.step` directives — use `run_step_native` to parse multi-plot RAW.

![STEP grid](assets/examples/step_grid.png)

## Python sweeps

Single parameter:
```python
from spicelab.analysis import step_param, TRAN
res = step_param(
    c,
    name="R",
    values=["1k","2k","5k"],
    analysis_factory=lambda: TRAN("50us","5ms"),
    progress=True,   # prints progress to stderr
)
```

Grid of parameters:
```python
from spicelab.analysis import ParamGrid, step_grid
grid: ParamGrid = {"R": ["1k","2k"], "C": ["100n","220n"]}
step = step_grid(c, grid, analysis_factory=lambda: TRAN("50us","5ms"), progress=True)
```

Stacking to DataFrame:
```python
from spicelab.analysis import stack_step_to_df
df = stack_step_to_df(step, y=["v(n1)"], with_x=True)
```

Plotting grouped curves (optional):
```python
from spicelab.analysis.viz.plot import plot_sweep_df
fig = plot_sweep_df(df, x="time", y="v(n1)", hue="R")

### Custom progress callback

You can pass a function `callback(done, total)` to receive updates instead of the default bar:

```python
def cb(done: int, total: int) -> None:
    print(f"{done}/{total}", end="\r")

step = step_grid(c, grid, analysis_factory=lambda: TRAN("50us","5ms"), progress=cb)
```
```

## Native .STEP

Provide directives that include `.step` and parse all plots:
```python
from spicelab.analysis import run_step_native
out = run_step_native(c, directives=[".step param R list 1k 2k 5k", ".tran 50us 5ms"])
sets = out.tracesets  # list[TraceSet]
```
