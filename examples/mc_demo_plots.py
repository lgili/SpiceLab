"""Generate demo Monte Carlo plots used in docs.

This script is intentionally standalone and uses the plotting helpers from
`cat.analysis.viz.plot`. It creates synthetic samples and metrics so docs can
include images without running NGSpice.
"""

from __future__ import annotations

import random
from pathlib import Path

from cat.analysis.viz.plot import plot_mc_kde, plot_param_vs_metric, plot_params_matrix

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "examples"
OUT.mkdir(parents=True, exist_ok=True)

# synthetic MC samples
random.seed(0)
N = 500
samples = []
metrics = []
for _ in range(N):
    r = random.gauss(100.0, 10.0)
    v = random.gauss(5.0, 0.2)
    samples.append({"R1": r, "V1": v, "R2": random.uniform(10, 20)})
    # simple metric: R1 / V1
    metrics.append(r / max(1e-12, v))

# KDE histogram
fig = plot_mc_kde(metrics, title="Metric KDE (R1/V1)")
fig.savefig(OUT / "mc_metric_kde.png")

# param vs metric scatter
fig2 = plot_param_vs_metric(samples, metrics, param="R1", title="R1 vs Metric")
fig2.savefig(OUT / "mc_R1_vs_metric.png")

# params matrix
fig3 = plot_params_matrix(samples, params=["R1", "V1", "R2"])
fig3.savefig(OUT / "mc_params_matrix.png")

print("Wrote demo images to", OUT)
