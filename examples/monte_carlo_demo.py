import argparse
import importlib
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from cat.analysis import OP, UniformAbs, monte_carlo
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND
from cat.spice.base import RunArtifacts, RunResult
from cat.spice.registry import get_run_directives, set_run_directives
from examples._common import savefig

plt: Any | None
try:
    plt = importlib.import_module("matplotlib.pyplot")
except ModuleNotFoundError:
    plt = None

try:
    pandas_plotting = importlib.import_module("pandas.plotting")
    scatter_matrix = getattr(pandas_plotting, "scatter_matrix", None)
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    scatter_matrix = None

try:  # optional dependency for nicer KDE overlays; tests may not have scipy
    scipy_stats = importlib.import_module("scipy.stats")
    gaussian_kde = getattr(scipy_stats, "gaussian_kde", None)
except ModuleNotFoundError:
    gaussian_kde = None

ASCII_TEMPLATE = """Title:  op
Date:   Thu Sep  1 12:00:00 2025
Plotname: Operating Point
Flags: real
No. Variables: 2
No. Points: 1
Variables:
        0       time    time
        1       v(n1)   voltage
Values:
        0       0.0     {value}
"""


def _fake_runner_value(val: float) -> Callable[[str, Sequence[str]], RunResult]:
    def _runner(netlist: str, directives: Sequence[str]) -> RunResult:
        td = tempfile.mkdtemp()
        raw = os.path.join(td, "sim.raw")
        with open(raw, "w") as f:
            f.write(ASCII_TEMPLATE.format(value=val))
        log = os.path.join(td, "ngspice.log")
        with open(log, "w") as f:
            f.write("ok\n")
        art = RunArtifacts(netlist_path=os.path.join(td, "deck.cir"), log_path=log, raw_path=raw)
        return RunResult(artifacts=art, returncode=0, stdout="", stderr="")

    return _runner


def build_demo_circuit() -> tuple[Circuit, Resistor]:
    c = Circuit("mc_demo")
    V1 = Vdc("1", 1.0)
    R1 = Resistor("1", 1000.0)
    c.add(V1, R1)
    c.connect(V1.ports[0], R1.ports[0])
    c.connect(R1.ports[1], GND)
    c.connect(V1.ports[1], GND)
    return c, R1


def run_demo(n: int = 3, outdir: str | Path | None = None, use_real_runner: bool = False) -> Any:
    """Run a monte_carlo and return the resulting dataframe.

    If `use_real_runner` is False the example substitutes a fake runner so
    the demo runs quickly in CI. Set `outdir` to save plots.
    """
    outdir = Path(outdir) if outdir is not None else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    old = get_run_directives()
    try:
        if not use_real_runner:
            vals = [0.1 * (i + 1) for i in range(n)]
            runners = [_fake_runner_value(v) for v in vals]
            idx = {"i": 0}

            def mux(net: str, dirs: Sequence[str]) -> RunResult:
                i = idx["i"]
                idx["i"] = (i + 1) % len(runners)
                return runners[i](net, dirs)

            set_run_directives(mux)

        c, r = build_demo_circuit()
        mc = monte_carlo(
            c, {r: UniformAbs(0.0)}, n=n, analysis_factory=lambda: OP(), seed=1, workers=1
        )

        # Convert to dataframe and compute basic metrics
        df = mc.to_dataframe(metric=None)
        # pick numeric columns to analyze
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return df

        # compute metrics for each numeric column
        metrics: dict[str, dict[str, float]] = {}
        for col in numeric_cols:
            mean = float(df[col].mean())
            std = float(df[col].std())
            pcts = df[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
            cv = float(std / mean) if mean != 0 else float("nan")
            metrics[col] = {
                "mean": mean,
                "std": std,
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "cv": cv,
                "p05": float(pcts.get(0.05, float("nan"))),
                "p25": float(pcts.get(0.25, float("nan"))),
                "p50": float(pcts.get(0.5, float("nan"))),
                "p75": float(pcts.get(0.75, float("nan"))),
                "p95": float(pcts.get(0.95, float("nan"))),
            }
        print("Monte Carlo metrics:")
        for k, v in metrics.items():
            print(k, v)

        # Plot histogram for the first numeric column; optionally overlay a KDE
        col0 = numeric_cols[0]
        out_name = outdir / f"monte_carlo_{col0}_hist.png"
        if plt is None:
            print("matplotlib not available; skipping plot generation")
        else:
            fig, ax = cast(Any, plt).subplots(figsize=(6, 4))
            df[col0].plot(kind="hist", density=True, bins=20, ax=ax, alpha=0.6, color="C0")
            if gaussian_kde is not None and len(df[col0].dropna()) > 1:
                kde = gaussian_kde(df[col0].dropna())
                xs = np.linspace(df[col0].min(), df[col0].max(), 200)
                ax.plot(xs, kde(xs), color="C1")
            ax.set_title(f"Monte Carlo distribution: {col0}")
            ax.set_xlabel(col0)
            ax.set_ylabel("Density")
            savefig(fig, str(out_name))

        # Scatter matrix for numeric parameters
        out_matrix: Path | None = None
        if len(numeric_cols) > 1 and scatter_matrix is not None and plt is not None:
            axes = scatter_matrix(
                df[numeric_cols],
                diagonal="kde" if gaussian_kde is not None else "hist",
                figsize=(6, 6),
            )
            fig_parent = axes[0, 0].get_figure()
            out_matrix = outdir / "monte_carlo_scatter_matrix.png"
            savefig(fig_parent, str(out_matrix))

        # Create a simple HTML report with embedded images and metrics
        report_lines = [
            "<html>",
            "<head><meta charset='utf-8'><title>Monte Carlo Report</title></head>",
            "<body>",
            f"<h1>Monte Carlo Report ({col0})</h1>",
            "<h2>Metrics</h2>",
            "<pre>",
            str(metrics),
            "</pre>",
            "<h2>Plots</h2>",
            f"<img src='{out_name.name}' alt='hist' width='600'/>",
        ]
        if out_matrix is not None:
            report_lines.append(
                f"<h3>Scatter matrix</h3><img src='{out_matrix.name}' width='600' />"
            )
        report_lines.extend(["</body>", "</html>"])
        report_path = outdir / "monte_carlo_report.html"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        return df
    finally:
        set_run_directives(old)

    outdir = Path(outdir) if outdir is not None else Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo demo")
    parser.add_argument("--n", type=int, default=10, help="number of MC runs")
    parser.add_argument("--outdir", type=str, default=".", help="output directory for plots")
    parser.add_argument(
        "--real-run", action="store_true", help="use real ngspice runner instead of fake runner"
    )
    args = parser.parse_args()

    df = run_demo(args.n, outdir=args.outdir, use_real_runner=args.real_run)
    print(df.head())
