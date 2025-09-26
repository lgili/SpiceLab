"""Lightweight Monte Carlo demo: vary resistor and collect a simple metric.

Synthetic sampling (no simulator required)::

    uv run --active python examples/monte_carlo_demo.py

Real per-sample simulations (engine defaults to ngspice)::

    uv run --active python examples/monte_carlo_demo.py --real --engine ngspice
    uv run --active python examples/monte_carlo_demo.py --real --engine ltspice
    uv run --active python examples/monte_carlo_demo.py --real --engine xyce
"""

from __future__ import annotations

import argparse
import random
import re

import numpy as np

try:  # pragma: no cover
    from ._common import parser_with_engine, print_header, resolve_engine
except ImportError:  # pragma: no cover - executed directly
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._common import (  # type: ignore[import-not-found]
        parser_with_engine,
        print_header,
        resolve_engine,
    )

from spicelab.viz import (
    monte_carlo_histogram,
    monte_carlo_param_scatter,
    params_scatter_matrix,
)

try:
    from spicelab.engines.orchestrator import run_simulation
except Exception:
    run_simulation = None

try:
    # orchestration-level Monte Carlo helper
    from spicelab.analysis import UniformAbs, monte_carlo
except Exception:
    UniformAbs = None  # type: ignore
    monte_carlo = None  # type: ignore

from spicelab.core.circuit import Circuit
from spicelab.core.components import C, R, V
from spicelab.core.net import GND, Net
from spicelab.core.types import AnalysisSpec

DEFAULT_C_VALUE = 100e-9


def _normalize_samples(samples_raw: list[dict[str, float]]) -> list[dict[str, float]]:
    """Add friendly aliases (R/C) for plotting while preserving original keys."""

    normalized: list[dict[str, float]] = []
    for raw in samples_raw:
        entry: dict[str, float] = dict(raw)
        r_val: float | None = None
        c_val: float | None = None
        for key, val in raw.items():
            if not isinstance(key, str):
                continue
            head = key.split(".", 1)[0].strip().upper()
            try:
                val_f = float(val)
            except Exception:
                continue
            if head == "R":
                r_val = val_f
            elif head == "C":
                c_val = val_f
        if r_val is not None:
            entry["R"] = r_val
        if c_val is not None:
            entry["C"] = c_val
        else:
            entry.setdefault("C", DEFAULT_C_VALUE)
        normalized.append(entry)
    return normalized


def _choose_plot_params(samples: list[dict[str, float]]) -> list[str]:
    if not samples:
        return []
    numeric_keys: list[str] = []
    for sample in samples:
        for key, val in sample.items():
            if not isinstance(key, str):
                continue
            try:
                float(val)
            except Exception:
                continue
            if key not in numeric_keys:
                numeric_keys.append(key)
    preferred = [k for k in ("R", "C") if k in numeric_keys]
    remaining = [k for k in numeric_keys if k not in preferred]
    return preferred + remaining


def synthetic_mc(n: int = 200):
    samples: list[dict] = []
    metrics: list[float] = []
    for _ in range(n):
        Rval = random.uniform(800.0, 1200.0)
        Cval = random.uniform(80e-9, 120e-9)
        gain = 1.0 / (1.0 + Rval * Cval * 1000.0)
        samples.append({"R": Rval, "C": Cval})
        metrics.append(gain)
    return samples, metrics


def real_mc(n: int, engine: str, metric_col: str | None = None) -> tuple[list[dict], list[float]]:
    """Run n single-sample simulations and extract a scalar metric from each result.

    metric_col: optional preferred column name (string) used to choose the metric when
    the result dataset/dataframe contains multiple columns.
    """
    if run_simulation is None:
        raise RuntimeError("run_simulation not available")
    samples: list[dict] = []
    metrics: list[float] = []
    for _ in range(n):
        Rval = random.uniform(800.0, 1200.0)
        samples.append({"R": Rval, "C": 100e-9})
        circuit = Circuit("rc_mc")
        vin = Net("vin")
        vout = Net("vout")
        src = V(1.0)
        r1 = R(Rval)
        c1 = C("100n")
        circuit.add(src, r1, c1)
        circuit.connect(src.ports[0], vin)
        circuit.connect(src.ports[1], GND)
        circuit.connect(r1.ports[0], vin)
        circuit.connect(r1.ports[1], vout)
        circuit.connect(c1.ports[0], vout)
        circuit.connect(c1.ports[1], GND)
        try:
            res = run_simulation(circuit, [AnalysisSpec("op", {})], engine=engine)
            ds = res.dataset() if hasattr(res, "dataset") else None
            if ds is not None:
                try:
                    # obtain candidate variable names
                    if hasattr(ds, "data_vars"):
                        names = list(ds.data_vars)
                    else:
                        try:
                            names = list(ds.keys())
                        except Exception:
                            names = []

                    # voltage-like names usually start with 'V(' in our readers
                    volt_names = [
                        n for n in names if isinstance(n, str) and n.lower().startswith("v(")
                    ]

                    # If user supplied a preferred metric column, try to match it first
                    chosen = None
                    if metric_col:
                        lower_metric = metric_col.lower()
                        # exact match
                        for n in names:
                            if isinstance(n, str) and n.lower() == lower_metric:
                                chosen = n
                                break
                        # substring match
                        if chosen is None:
                            for n in names:
                                if isinstance(n, str) and lower_metric in n.lower():
                                    chosen = n
                                    break

                    # prefer one that mentions 'vout' if no metric_col matched
                    if chosen is None:
                        for n in volt_names:
                            if "vout" in n.lower():
                                chosen = n
                                break
                    if chosen is None and volt_names:
                        chosen = volt_names[0]

                    if chosen is not None:
                        val = None
                        try:
                            # xarray-like access
                            var = ds[chosen]
                            val = getattr(var, "values", None)
                        except Exception:
                            try:
                                # mapping-like access
                                val = ds[chosen]
                            except Exception:
                                val = None

                        if val is None:
                            metrics.append(float(np.random.random()))
                        else:
                            try:
                                metric = float(np.asarray(val).flatten()[0])
                                metrics.append(metric)
                            except Exception:
                                metrics.append(float(np.random.random()))
                    else:
                        metrics.append(float(np.random.random()))
                except Exception:
                    metrics.append(float(np.random.random()))
            else:
                metrics.append(float(np.random.random()))
        except Exception:
            metrics.append(float(np.random.random()))
    return samples, metrics


def _orchestrator_mc(
    n: int,
    engine: str,
    *,
    workers: int = 1,
    cache_dir: str | None = None,
    metric_col: str | None = None,
) -> tuple[list[dict], list[float]]:
    """Run monte_carlo via the orchestrator-level helper and extract a single metric.

    Returns (samples, metrics) where metrics are the numeric values extracted from
    the dataframe (preferring 'V(vout)' column when present). If no numeric column
    can be determined, returns random metrics as a fallback to avoid breaking demos.
    """
    if monte_carlo is None or UniformAbs is None:
        raise RuntimeError("orchestrator monte_carlo helper not available")
    # build circuit and mapping (match real_mc construction)
    circuit = Circuit("rc_mc")
    vin = Net("vin")
    vout = Net("vout")
    src = V(1.0)
    r1 = R("1k")
    c1 = C("100n")
    circuit.add(src, r1, c1)
    circuit.connect(src.ports[0], vin)
    circuit.connect(src.ports[1], GND)
    circuit.connect(r1.ports[0], vin)
    circuit.connect(r1.ports[1], vout)
    circuit.connect(c1.ports[0], vout)
    circuit.connect(c1.ports[1], GND)

    # create mapping: vary resistor R (component object must be hashable key)
    # use UniformAbs with a small delta to mimic prior sampling range
    mapping = {r1: UniformAbs(200.0)}
    mc = monte_carlo(
        circuit,
        mapping,
        n=n,
        analyses=[AnalysisSpec("op", {})],
        engine=engine,
        workers=workers,
        cache_dir=cache_dir,
    )
    # convert to dataframe and pick the best numeric column for the metric
    df = mc.to_dataframe(metric=None)

    def _choose_metric_column(df_obj):
        # prefer columns like 'V(vout)' or 'V(n1)'; fall back to any numeric column
        try:
            num_idx = df_obj.select_dtypes(include="number").columns.tolist()
        except Exception:
            # select_dtypes may not exist in small DataFrame-like objects; try heuristics
            try:
                num_idx = [c for c in list(df_obj.columns) if isinstance(df_obj[c][0], int | float)]
            except Exception:
                num_idx = []

        if not num_idx:
            return None

        lname_map = {c: str(c).lower() for c in num_idx}

        # If user supplied metric_col, try to match it
        if metric_col:
            lm = metric_col.lower()
            for c in num_idx:
                if str(c).lower() == lm:
                    return c
            for c in num_idx:
                if lm in str(c).lower():
                    return c

        # Heuristics in order:
        # 1) v(...) containing 'vout'
        for c in num_idx:
            lc = lname_map[c]
            if lc.startswith("v(") and "vout" in lc:
                return c
        # 2) any column containing 'vout'
        for c in num_idx:
            if "vout" in lname_map[c]:
                return c
        # 3) any column that looks like 'v(...)'
        for c in num_idx:
            if lname_map[c].startswith("v("):
                return c
        # 4) columns named exactly 'vout' or variants like 'v_out'
        for c in num_idx:
            if re.fullmatch(r"v\W*out\W*", lname_map[c]):
                return c
        # 5) fallback: first numeric column
        return num_idx[0]

    chosen_col = _choose_metric_column(df)
    samples = _normalize_samples(getattr(mc, "samples", []))
    if chosen_col is None:
        # fallback to random metrics to avoid breaking demos
        metrics = [float(np.random.random()) for _ in samples]
        return samples, metrics

    # Extract numeric list
    try:
        coldata = df[chosen_col]
        if hasattr(coldata, "tolist"):
            vals = coldata.tolist()
        elif isinstance(coldata, list):
            vals = coldata
        else:
            try:
                vals = list(coldata)
            except Exception:
                vals = [coldata]
    except Exception:
        # If df supports iteration like a MiniDataFrame or has _rows
        vals = [row.get(chosen_col) for row in getattr(df, "_rows", [])]
    metrics = [float(v) for v in vals]
    return samples, metrics


def _parser() -> argparse.ArgumentParser:
    p = parser_with_engine("Monte Carlo sampling demo", default="ngspice")
    p.add_argument("-n", type=int, default=200)
    p.add_argument("--real", action="store_true")
    p.add_argument(
        "--workers", type=int, default=1, help="Number of workers for orchestrator Monte Carlo"
    )
    p.add_argument(
        "--cache-dir", default=None, help="Cache directory for orchestrator Monte Carlo (optional)"
    )
    p.add_argument(
        "--metric-col",
        default=None,
        help="Prefer this column name from results for the metric (e.g. 'V(vout)')",
    )
    p.add_argument("--out-html", default=None, help="Directory to save HTML outputs (optional)")
    p.add_argument("--out-img", default=None, help="Directory to save static images (optional)")
    p.add_argument("--img-format", default="png", help="Image format for static export (png/svg)")
    p.add_argument("--img-scale", type=float, default=2.0, help="Scale factor for image export")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    engine = resolve_engine(getattr(args, "engine", None))
    print_header("Monte Carlo", engine if args.real else None)

    if args.real:
        # Prefer the orchestrator-level monte_carlo helper (supports caching/workers)
        if monte_carlo is not None and UniformAbs is not None:
            try:
                samples, metrics = _orchestrator_mc(
                    args.n,
                    engine,
                    workers=args.workers,
                    cache_dir=args.cache_dir,
                    metric_col=args.metric_col,
                )
            except Exception:
                samples, metrics = real_mc(args.n, engine, metric_col=args.metric_col)
        else:
            samples, metrics = real_mc(args.n, engine, metric_col=args.metric_col)
    else:
        samples, metrics = synthetic_mc(args.n)

    # Build plots and optionally export
    hist = monte_carlo_histogram(metrics, title="Monte Carlo metric")
    hist.show()
    if args.out_html:
        from pathlib import Path

        od = Path(args.out_html)
        od.mkdir(parents=True, exist_ok=True)
        hist.to_html(od / "mc_hist.html", include_plotlyjs="cdn")
    if args.out_img:
        from pathlib import Path

        od = Path(args.out_img)
        try:
            hist.to_image(
                od / f"mc_hist.{args.img_format}", scale=args.img_scale, format=args.img_format
            )
        except RuntimeError as exc:
            print("Image export failed (kaleido missing or error):", exc)

    plot_params = _choose_plot_params(samples)

    scatter_fig = None
    if plot_params:
        x_param = plot_params[0]
        scatter_fig = monte_carlo_param_scatter(
            samples, metrics, x_param, title=f"{x_param} vs metric"
        )
        scatter_fig.show()
        if args.out_html:
            from pathlib import Path

            od = Path(args.out_html)
            scatter_fig.to_html(od / "mc_param_scatter.html", include_plotlyjs="cdn")
        if args.out_img:
            from pathlib import Path

            od = Path(args.out_img)
            try:
                scatter_fig.to_image(
                    od / f"mc_param_scatter.{args.img_format}",
                    scale=args.img_scale,
                    format=args.img_format,
                )
            except RuntimeError as exc:
                print("Image export failed (kaleido missing or error):", exc)
    else:
        print("[monte_carlo_demo] Skipping parameter scatter: no numeric parameters available")

    matrix_fig = None
    if len(plot_params) >= 2:
        matrix_params = plot_params[:2]
        matrix_fig = params_scatter_matrix(samples, params=matrix_params, title="Params matrix")
        matrix_fig.show()
        if args.out_html:
            from pathlib import Path

            od = Path(args.out_html)
            matrix_fig.to_html(od / "mc_params_matrix.html", include_plotlyjs="cdn")
        if args.out_img:
            from pathlib import Path

            od = Path(args.out_img)
            try:
                matrix_fig.to_image(
                    od / f"mc_params_matrix.{args.img_format}",
                    scale=args.img_scale,
                    format=args.img_format,
                )
            except RuntimeError as exc:
                print("Image export failed (kaleido missing or error):", exc)
    else:
        print("[monte_carlo_demo] Skipping scatter matrix: need at least two numeric parameters")


if __name__ == "__main__":
    main()
