# Cookbook

Short, copy-paste recipes for common tasks.

## −3 dB bandwidth
```python
from cat.analysis import bandwidth_3db
bw = bandwidth_3db(res.traces, y_out="v(out)")
```

## 0 dB crossover and margins
```python
from cat.analysis import crossover_freq_0db, phase_margin, gain_margin_db
wc = crossover_freq_0db(res.traces, y_out="v(out)")
pm = phase_margin(res.traces, y_out="v(out)")
gm = gain_margin_db(res.traces, y_out="v(out)")
```

## Overshoot and settling time
```python
from cat.analysis import overshoot_pct, settling_time
ov = overshoot_pct(res.traces, "v(out)")  # %
st = settling_time(res.traces, "v(out)")
```

## Interpolate at a given time (per run)
```python
import numpy as np

def sample_at(ts, name: str, t: float) -> float:
    t_arr = ts["time"].values
    y_arr = ts[name].values
    return float(np.interp(t, t_arr, y_arr))
```

## Stack multiple runs into a DataFrame
```python
from cat.analysis import stack_runs_to_df
# runs: list[AnalysisResult]
df = stack_runs_to_df(runs, y=["v(out)"], with_x=True)
```
