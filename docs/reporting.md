# Reporting Helpers

spicelab ships with a minimal reporting layer so simulation scripts can
produce shareable artefacts (HTML or Markdown) without pulling in a full templating
engine. The core entry point is `spicelab.reporting.ReportBuilder`.

```python
import numpy as np
from pathlib import Path
from spicelab.io.raw_reader import Trace, TraceSet
from spicelab.reporting import ReportBuilder
from spicelab.viz.plotly import time_series_view

# Assume ``handle`` is a ResultHandle from run_simulation()
ds = handle.dataset()
time = ds.coords["time"].values
vout = ds["V(out)"].values
traces = TraceSet([
    Trace("time", "s", np.asarray(time)),
    Trace("V(out)", "V", np.asarray(vout)),
])
figure = time_series_view(traces, ys=["V(out)"])

report = ReportBuilder("RC transient", subtitle="Monte Carlo sample #12")
summary = report.add_section("Summary")
summary.add_markdown("Circuit converged in 1.2 ms with no warnings.")
summary.add_table([
    {"element": "R1", "value": "1 kOhm"},
    {"element": "C1", "value": "100 nF"},
], caption="Component values")
summary.add_figure(figure, caption="Output voltage vs. time")

outputs = report.write(Path("reports"))
print("HTML report at", outputs["html"])
```

`ReportBuilder` manages sections, tables, inline markdown, and embedded figures.
Figure blocks accept `VizFigure` instances (from Plotly helpers) or raw HTML
strings; assets are written under `reports/figures/` and linked from the main
report. Tables accept `polars.DataFrame`, pandas data frames, or simple
``list[dict]`` structures.

## Dataset summary

To turn an `xarray.Dataset` into a compact table, use
`spicelab.reporting.report.dataset_summary`:

```python
from spicelab.reporting import ReportBuilder
from spicelab.reporting.report import dataset_summary

report = ReportBuilder("AC sweep")
section = report.add_section("Dataset")
section.add_table(dataset_summary(handle.dataset()), caption="Variables & dimensions")
```

The helper returns `(headers, rows)` ready to feed into `ReportSection.add_table`.

## Output formats

`ReportBuilder.write()` accepts any combination of `html` and `md` formats.
HTML output includes a lightweight stylesheet for readability, while the Markdown
variant is primed for changelog attachments or quick email summaries.

Set `include_figures=False` if you only need textual output.
