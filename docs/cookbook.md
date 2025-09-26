# Cookbook

Short, copy-paste recipes for common tasks.

## Basic measurements (gain, overshoot, settling)
```python
from spicelab.analysis import GainSpec, OvershootSpec, SettlingTimeSpec, measure

specs = [
    GainSpec(name="gain@1k", numerator="V(out)", denominator="V(in)", freq=1_000.0),
    OvershootSpec(name="overshoot", signal="V(out)", target=1.0),
    SettlingTimeSpec(name="settle", signal="V(out)", target=1.0, tolerance=0.02),
]
df = measure(result_handle, specs)
```

## Sample a trace at a given time
```python
import numpy as np

def sample_at(ts, name: str, t: float) -> float:
    t_arr = ts["time"].values
    y_arr = ts[name].values
    return float(np.interp(t, t_arr, y_arr))
```

## Monte Carlo → DataFrame
```python
from spicelab.analysis import NormalPct, monte_carlo
from spicelab.core.types import AnalysisSpec

mc = monte_carlo(
    circuit=c,
    mapping={R1: NormalPct(0.05)},
    n=32,
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
)
df = mc.to_dataframe(metric=None, param_prefix="param_")
```
