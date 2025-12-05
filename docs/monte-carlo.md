# Monte Carlo

spicelab runs Monte Carlo jobs by mapping components to distributions and
supplying per-trial analyses. The orchestrator handles caching,
multithreading, and xarray/polars exports.

> **New in M16**: Enhanced tolerance analysis with correlated groups, absolute
> distributions, process capability metrics (Cpk), and HTML report generation.
> See [Tolerance Analysis](tolerance-analysis.md) for the complete guide.

![Monte Carlo histogram](assets/examples/mc_hist.png)

```python
from spicelab.core.circuit import Circuit
from spicelab.core.components import Vdc, Resistor, Capacitor
from spicelab.core.net import GND
from spicelab.analysis import monte_carlo, NormalPct
from spicelab.core.types import AnalysisSpec

c = Circuit("rc")
V1, R1, C1 = Vdc("1", 5.0), Resistor("1", "1k"), Capacitor("1", "100n")
c.add(V1, R1, C1)
c.connect(V1.ports[0], R1.ports[0])
c.connect(R1.ports[1], C1.ports[0])
c.connect(V1.ports[1], GND)
c.connect(C1.ports[1], GND)

mc = monte_carlo(
    circuit=c,
    mapping={R1: NormalPct(0.05)},
    n=16,
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
    cache_dir=".spicelab_cache",
    seed=123,
)

df = mc.to_dataframe(param_prefix="param_")
print(df.head())

# Access cached handles / job metadata if needed
handles = mc.result_handles()
if mc.job:
    print("cache dir:", mc.job.cache_dir)
```

Tips:
- Use `workers` for parallel execution and optionally point `cache_dir` to reuse results between runs.
- Provide a metric function to compute scalar KPIs per trial, or set `y=[...]` and `sample_at` to extract trace values.
- `mc.result_handles()` and `mc.job` expose the underlying result handles and cache metadata when you need xarray datasets or cache diagnostics.
- When consuming results via the CLI demo (`examples/monte_carlo_demo.py`), pass `--metric-col` to enforce which dataframe column becomes the plotted metric (for example, `V(vout)`). The script exports `mc_hist`, `mc_param_scatter`, and `mc_params_matrix` HTML/PNG files when you supply `--out-html` / `--out-img`.

### Progress bar / callback

Add a lightweight progress bar to stderr:

```python
mc = monte_carlo(
    circuit=c,
    mapping={R1: NormalPct(0.01)},
    n=1000,
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
    progress=True,   # prints: "MC: 317/1000 (31%)"
)
```

Or provide your own callback (`done`, `total`):

```python
def cb(done: int, total: int) -> None:
    print(f"{done}/{total}", end="\r")

mc = monte_carlo(
    c,
    mapping={R1: NormalPct(0.01)},
    n=1000,
    analyses=[AnalysisSpec("op", {})],
    engine="ngspice",
    progress=cb,
)
```

### Correlated Groups (Same-Lot Components)

Components from the same manufacturing batch vary together:

```python
from spicelab.analysis import CorrelatedGroup, NormalPct

# Resistors from same reel share the same variation
same_lot = CorrelatedGroup([R1, R2, R3, R4], NormalPct(0.001))

mc = monte_carlo(
    c,
    mapping={
        same_lot: None,  # Group with its own distribution
        C1: UniformPct(0.10),  # Independent capacitor
    },
    n=10000,
    analyses=[AnalysisSpec("op", {})],
)
```

### Absolute Tolerance Distributions

For parameters specified in absolute units (op-amp offsets, reference voltages):

```python
from spicelab.analysis import NormalAbs, UniformAbs

mc = monte_carlo(
    c,
    mapping={
        R1: NormalPct(0.01),        # 1% relative
        Vos: NormalAbs(0.002 / 3),  # ±2mV, 3-sigma
        Vref: UniformAbs(0.005),    # ±5mV uniform
    },
    n=10000,
    analyses=[AnalysisSpec("op", {})],
)
```

### Process Capability Metrics

Calculate Cpk, yield, and sigma level:

```python
def get_vout(run):
    return run.traces['V(vout)'].values[-1]

# Cpk (process capability index)
cpk = mc.cpk(get_vout, lsl=2.4, usl=2.6)
print(f"Cpk: {cpk:.2f}")  # >= 1.33 is acceptable

# Yield estimates
yield_est = mc.yield_estimate(get_vout, lsl=2.4, usl=2.6)
actual_yield = mc.actual_yield(get_vout, lsl=2.4, usl=2.6)

# Full process summary
summary = mc.process_summary(get_vout, lsl=2.4, usl=2.6)
print(f"Cpk: {summary['cpk']:.2f}")
print(f"Yield: {summary['yield_pct']:.1f}%")
print(f"Sigma: {summary['sigma_level']:.1f}σ")
```

### HTML Report Generation

Generate professional reports with histograms and process metrics:

```python
from spicelab.analysis import generate_report

html = generate_report(
    mc,
    metric=get_vout,
    lsl=2.4,
    usl=2.6,
    output_path="mc_report.html",
    title="Output Voltage Tolerance Analysis",
    metric_name="Vout (V)",
)
```

The report includes:
- Summary cards (n, mean, std, Cpk, yield)
- Distribution histogram with spec limits
- Statistical summary table
- Process capability analysis
- Varied parameters table
