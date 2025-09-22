import argparse
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cat.analysis import OP, UniformAbs, monte_carlo
from cat.analysis.viz.plot import (
    plot_mc_kde,
    plot_mc_metric_hist,
    plot_param_vs_metric,
    plot_params_matrix,
)
from cat.core.circuit import Circuit
from cat.core.components import Resistor, Vdc
from cat.core.net import GND
from cat.spice.base import RunArtifacts, RunResult
from cat.spice.registry import get_run_directives, set_run_directives

if TYPE_CHECKING:
    from cat.viz.plotly import _PlotlyNotAvailable as PlotlyNotAvailable
else:
    try:
        from cat.viz.plotly import _PlotlyNotAvailable as PlotlyNotAvailable
    except Exception:  # pragma: no cover - fallback type

        class PlotlyNotAvailable(RuntimeError):
            """Fallback exception when Plotly helpers are unavailable."""

            pass


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


def _emit_plotly_report(
    df: Any,
    numeric_cols: list[str],
    metrics: dict[str, dict[str, float]],
    outdir: Path,
) -> None:
    col0 = numeric_cols[0]
    hist_fig = plot_mc_metric_hist(
        df[col0].dropna().tolist(),
        title=f"Distribution of {col0}",
        xlabel=col0,
    )
    hist_path = outdir / f"monte_carlo_{col0}_hist.html"
    hist_fig.to_html(hist_path, include_plotlyjs="cdn")

    kde_fig = plot_mc_kde(df[col0].dropna().tolist(), title=f"KDE of {col0}", xlabel=col0)
    (outdir / f"monte_carlo_{col0}_kde.html").write_text(
        kde_fig.figure.to_html(include_plotlyjs="cdn", full_html=True),
        encoding="utf-8",
    )

    out_matrix: Path | None = None
    matrix_fig = None
    scatter_fig = None
    if len(numeric_cols) > 1:
        samples = df[numeric_cols].to_dict("records")
        matrix_fig = plot_params_matrix(samples, params=numeric_cols)
        out_matrix = outdir / "monte_carlo_params_matrix.html"
        matrix_fig.to_html(out_matrix, include_plotlyjs="cdn")

        scatter_fig = plot_param_vs_metric(
            samples,
            df[col0].tolist(),
            param=numeric_cols[1],
            title=f"{numeric_cols[1]} vs {col0}",
            xlabel=numeric_cols[1],
            ylabel=col0,
        )
        scatter_fig.to_html(outdir / "monte_carlo_param_metric.html", include_plotlyjs="cdn")

    report_path = outdir / "monte_carlo_report.html"
    html_sections = [
        "<html>",
        "<head><meta charset='utf-8'><title>Monte Carlo Report</title></head>",
        "<body>",
        f"<h1>Monte Carlo Report ({col0})</h1>",
        "<h2>Metrics</h2>",
        "<pre>",
        str(metrics),
        "</pre>",
        "<h2>Histogram</h2>",
        hist_fig.figure.to_html(include_plotlyjs="cdn", full_html=False),
        "<h2>KDE</h2>",
        kde_fig.figure.to_html(include_plotlyjs=False, full_html=False),
    ]
    if out_matrix is not None:
        html_sections.append("<h2>Parameter scatter matrix</h2>")
        assert matrix_fig is not None
        html_sections.append(matrix_fig.figure.to_html(include_plotlyjs="cdn", full_html=False))
        if scatter_fig is not None:
            html_sections.append("<h2>Parameter vs metric</h2>")
            html_sections.append(
                scatter_fig.figure.to_html(include_plotlyjs="cdn", full_html=False)
            )
    html_sections.append("</body></html>")
    report_path.write_text("\n".join(html_sections), encoding="utf-8")


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

        try:
            _emit_plotly_report(df, numeric_cols, metrics, outdir)
        except PlotlyNotAvailable:
            print("Plotly not available; skipping interactive outputs")

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
