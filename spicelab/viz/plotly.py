"""Plotly-based visualization primitives for CAT analyses."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..io.raw_reader import TraceSet


class _PlotlyNotAvailable(RuntimeError):
    pass


def _ensure_plotly() -> tuple[Any, Any, Any]:
    try:
        go = importlib.import_module("plotly.graph_objects")
        make_subplots = importlib.import_module("plotly.subplots").make_subplots
        px = importlib.import_module("plotly.express")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise _PlotlyNotAvailable(
            "Plotly is required for interactive visualization. Install the 'viz' extra:"
            " pip install spicelab[viz]"
        ) from exc
    return go, make_subplots, px


def _numeric(values: Iterable[Any]) -> NDArray[np.floating[Any]]:
    return np.asarray(list(values), dtype=float)


def _resolve_trace_name(ts: TraceSet, name: str) -> str:
    if name in ts.names:
        return name
    low = name.lower()
    for candidate in ts.names:
        if candidate.lower() == low:
            return candidate
    raise KeyError(f"Trace '{name}' not found. Available: {ts.names}")


def _pick_x(ts: TraceSet) -> tuple[NDArray[np.floating[Any]], str]:
    x_attr = getattr(ts, "x", None)
    if x_attr is not None:
        try:
            return _numeric(x_attr.values), getattr(x_attr, "name", "x")
        except Exception:  # pragma: no cover - fallback to heuristics
            pass

    names_lower = [name.lower() for name in ts.names]
    for candidate in ("time", "frequency"):
        if candidate in names_lower:
            name = ts.names[names_lower.index(candidate)]
            return _numeric(ts[name].values), name

    first = ts.names[0]
    return _numeric(ts[first].values), first


def _first_crossing_time(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    target: float,
    *,
    rising: bool,
) -> float | None:
    if x.size == 0:
        return None
    mask = y >= target if rising else y <= target
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(x[0])
    x0 = float(x[i - 1])
    x1 = float(x[i])
    y0 = float(y[i - 1])
    y1 = float(y[i])
    if y1 == y0:
        return float(x1)
    ratio = (target - y0) / (y1 - y0)
    return float(x0 + ratio * (x1 - x0))


def _settling_time(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    final_value: float,
    tolerance: float,
) -> float | None:
    if tolerance <= 0.0 or x.size == 0:
        return None
    diffs = np.abs(y - final_value)
    for i in range(x.size):
        if np.all(diffs[i:] <= tolerance):
            return float(x[i])
    return None


@dataclass
class VizFigure:
    figure: Any
    metadata: Mapping[str, Any] | None = field(default=None)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.figure, item)

    def to_html(
        self,
        path: str | Path,
        *,
        include_plotlyjs: str = "cdn",
        auto_open: bool = False,
    ) -> Path:
        out = Path(path)
        out.write_text(
            self.figure.to_html(full_html=True, include_plotlyjs=include_plotlyjs),
            encoding="utf-8",
        )
        if auto_open:
            try:  # pragma: no cover - optional open in browser
                import webbrowser

                webbrowser.open(out.as_uri())
            except Exception:
                pass
        return out

    def to_image(self, path: str | Path, *, scale: float = 2.0, format: str = "png") -> Path:
        out = Path(path)
        try:
            self.figure.write_image(str(out), scale=scale, format=format)
        except ValueError as exc:  # pragma: no cover - kaleido missing
            raise RuntimeError(
                "Plotly static image export requires 'kaleido'. Install with pip install kaleido."
            ) from exc
        return out

    def to_csv(self, path: str | Path, *, include_metadata: bool = False) -> Path:
        """Export trace data to CSV format.

        Args:
            path: Output file path.
            include_metadata: If True, include metadata as comments at the top.

        Returns:
            Path to the written file.
        """
        import csv
        import json

        out = Path(path)

        # Extract data from figure traces
        traces_data: list[dict[str, Any]] = []
        for trace in self.figure.data:
            trace_dict: dict[str, Any] = {"name": getattr(trace, "name", "unnamed")}
            if hasattr(trace, "x") and trace.x is not None:
                trace_dict["x"] = list(trace.x)
            if hasattr(trace, "y") and trace.y is not None:
                trace_dict["y"] = list(trace.y)
            traces_data.append(trace_dict)

        with out.open("w", newline="", encoding="utf-8") as f:
            if include_metadata and self.metadata:
                f.write(f"# Metadata: {json.dumps(dict(self.metadata))}\n")

            # Write each trace as x,y columns
            if traces_data:
                writer = csv.writer(f)
                # Header row
                headers = []
                for trace in traces_data:
                    name = trace.get("name", "trace")
                    if "x" in trace:
                        headers.append(f"{name}_x")
                    if "y" in trace:
                        headers.append(f"{name}_y")
                writer.writerow(headers)

                # Data rows - find max length
                max_len = (
                    max(len(trace.get("x", [])) for trace in traces_data) if traces_data else 0
                )
                max_len = max(
                    max_len,
                    max(len(trace.get("y", [])) for trace in traces_data) if traces_data else 0,
                )

                for i in range(max_len):
                    row = []
                    for trace in traces_data:
                        if "x" in trace:
                            row.append(trace["x"][i] if i < len(trace["x"]) else "")
                        if "y" in trace:
                            row.append(trace["y"][i] if i < len(trace["y"]) else "")
                    writer.writerow(row)

        return out

    def to_json(self, path: str | Path, *, include_figure: bool = True) -> Path:
        """Export figure data to JSON format.

        Args:
            path: Output file path.
            include_figure: If True, include full Plotly figure JSON.

        Returns:
            Path to the written file.
        """
        import json

        out = Path(path)

        data: dict[str, Any] = {}

        # Add metadata
        if self.metadata:
            data["metadata"] = dict(self.metadata)

        # Add trace data in a simplified format
        traces_data = []
        for trace in self.figure.data:
            trace_dict: dict[str, Any] = {"name": getattr(trace, "name", "unnamed")}
            if hasattr(trace, "x") and trace.x is not None:
                trace_dict["x"] = [float(v) if isinstance(v, int | float) else v for v in trace.x]
            if hasattr(trace, "y") and trace.y is not None:
                trace_dict["y"] = [float(v) if isinstance(v, int | float) else v for v in trace.y]
            traces_data.append(trace_dict)
        data["traces"] = traces_data

        # Optionally include full Plotly figure
        if include_figure:
            data["plotly_figure"] = self.figure.to_dict()

        out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return out

    def show(self, **kwargs: Any) -> None:
        try:
            self.figure.show(**kwargs)
        except ValueError as exc:
            message = str(exc)
            if "nbformat" not in message.lower():
                raise
            if importlib.util.find_spec("nbformat") is None:
                raise RuntimeError(
                    "Plotly inline rendering requires nbformat>=4.2. Install it with "
                    "'pip install nbformat'."
                ) from exc
            raise
        except Exception:
            raise

    # Jupyter display hook for nicer UX in notebooks
    def _ipython_display_(self) -> None:  # pragma: no cover - interactive hook
        try:
            # Prefer Plotly's HTML renderer for notebooks
            self.show()
        except RuntimeError as exc:
            print(exc)
        except Exception:
            # Fallback to printing a minimal representation
            print(f"VizFigure(kind={self.metadata.get('kind') if self.metadata else 'unknown'})")


def time_series_view(
    ts: TraceSet,
    ys: Sequence[str] | None = None,
    x: str | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    grid: bool = True,
    template: str | None = "plotly_white",
    markers: bool = False,
    color_map: Mapping[str, str] | None = None,
    line_width: float | None = None,
    marker_size: int | None = None,
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    # allow caller to override which column to use for x
    if x is not None:
        try:
            x_name = _resolve_trace_name(ts, x)
            x_vals = _numeric(ts[x_name].values)
            xname = x_name
        except Exception:  # fallback to heuristics
            x_vals, xname = _pick_x(ts)
    else:
        x_vals, xname = _pick_x(ts)
    if ys is None:
        names = [n for n in ts.names if n != xname]
    else:
        names = [_resolve_trace_name(ts, n) for n in ys]

    fig = go.Figure()
    mode = "lines+markers" if markers else "lines"
    for name in names:
        values = ts[name].values
        line: dict[str, Any] | None = {"width": line_width} if line_width is not None else None
        if color_map and name in color_map:
            if line is None:
                line = {"color": color_map[name]}
            else:
                line = {**line, "color": color_map[name]}
        marker = dict(size=marker_size) if marker_size is not None else None
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=values,
                mode=mode,
                name=name,
                line=line,
                marker=marker,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or xname,
        yaxis_title=ylabel,
        template=template,
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=grid, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=grid, gridcolor="rgba(150,150,150,0.2)")
    return VizFigure(fig, metadata={"kind": "time_series", "traces": names})


def bode_view(
    ts: TraceSet,
    y: str,
    x: str | None = None,
    *,
    unwrap_phase: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    template: str | None = "plotly_white",
) -> VizFigure:
    go, make_subplots, _ = _ensure_plotly()
    if x is not None:
        try:
            x_name = _resolve_trace_name(ts, x)
            x_vals = _numeric(ts[x_name].values)
            xname = x_name
        except Exception:  # fallback
            x_vals, xname = _pick_x(ts)
    else:
        x_vals, xname = _pick_x(ts)
    y_name = _resolve_trace_name(ts, y)
    z = np.asarray(ts[y_name].values)
    if not np.iscomplexobj(z):
        raise ValueError(f"Trace '{y}' is not complex; AC analysis is required for Bode plots.")

    mag_db = 20.0 * np.log10(np.abs(z))
    phase = np.angle(z, deg=True)
    if unwrap_phase:
        phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=x_vals, y=mag_db, mode="lines", name="Magnitude [dB]"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=phase, mode="lines", name="Phase [deg]", line=dict(color="indianred")
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Magnitude [dB]", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Phase [deg]", row=2, col=1, showgrid=True)
    fig.update_xaxes(title_text=xlabel or xname, row=2, col=1, showgrid=True)
    fig.update_layout(
        title=title or f"Bode plot for {y}",
        template=template,
        legend=dict(orientation="h"),
    )
    return VizFigure(fig, metadata={"kind": "bode", "trace": y})


def nyquist_view(
    ts: TraceSet,
    y: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
    show_arrow: bool = True,
) -> VizFigure:
    """Plot Nyquist curve (Re vs Im) for a complex trace.

    Accepts the same TraceSet used for Bode; y must be complex-valued.
    """
    go, _, _ = _ensure_plotly()
    x_vals, _ = _pick_x(ts)
    y_name = _resolve_trace_name(ts, y)
    z = np.asarray(ts[y_name].values)
    if not np.iscomplexobj(z):
        raise ValueError(f"Trace '{y}' is not complex; AC analysis is required for Nyquist plots.")

    re = np.real(z)
    im = np.imag(z)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=re, y=im, mode="lines+markers", name=y))
    # optional arrow showing direction (last segment)
    if show_arrow and re.size >= 2:
        fig.add_trace(
            go.Scatter(
                x=[re[-2], re[-1]],
                y=[im[-2], im[-1]],
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False,
            )
        )

    fig.update_layout(title=title or f"Nyquist plot for {y}", template=template)
    fig.update_xaxes(title_text=xlabel or "Re")
    fig.update_yaxes(title_text=ylabel or "Im")
    return VizFigure(fig, metadata={"kind": "nyquist", "trace": y})


def step_response_view(
    ts: TraceSet,
    y: str,
    x: str | None = None,
    *,
    steady_state: float | None = None,
    initial_value: float | None = None,
    settle_tolerance: float = 0.02,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
    show_annotations: bool = True,
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    if x is not None:
        try:
            x_name = _resolve_trace_name(ts, x)
            x_vals = _numeric(ts[x_name].values)
            xname = x_name
        except Exception:
            x_vals, xname = _pick_x(ts)
    else:
        x_vals, xname = _pick_x(ts)
    y_name = _resolve_trace_name(ts, y)
    y_vals = _numeric(ts[y_name].values)
    if y_vals.size == 0:
        raise ValueError("Trace is empty; cannot build step response plot")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name=y))

    init_val = float(initial_value) if initial_value is not None else float(y_vals[0])
    final_val = float(steady_state) if steady_state is not None else float(y_vals[-1])
    delta = final_val - init_val
    rising = delta >= 0.0
    magnitude = abs(delta)

    t10 = t90 = rise_time = None
    overshoot_pct = None
    settling_time = None

    if magnitude > 0:
        target10 = init_val + 0.1 * delta
        target90 = init_val + 0.9 * delta
        t10 = _first_crossing_time(x_vals, y_vals, target10, rising=rising)
        t90 = _first_crossing_time(x_vals, y_vals, target90, rising=rising)
        if t10 is not None and t90 is not None:
            rise_time = max(0.0, t90 - t10)

        if rising:
            overshoot = float(y_vals.max()) - final_val
        else:
            overshoot = final_val - float(y_vals.min())
        overshoot_pct = max(0.0, overshoot) / magnitude * 100.0

        tol = abs(settle_tolerance) * magnitude
        settling_time = _settling_time(x_vals, y_vals, final_val, tol)

        band_label = f"±{settle_tolerance * 100:.1f}%"
        if tol > 0:
            upper = final_val + tol
            lower = final_val - tol
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=np.full_like(x_vals, upper),
                    mode="lines",
                    name=f"steady-state {band_label}",
                    line=dict(color="rgba(100,100,100,0.4)", dash="dot"),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=np.full_like(x_vals, lower),
                    mode="lines",
                    name=None,
                    line=dict(color="rgba(100,100,100,0.4)", dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.full_like(x_vals, final_val),
                mode="lines",
                name="steady-state",
                line=dict(color="rgba(120,120,120,0.7)", dash="dash"),
                hoverinfo="skip",
                showlegend=True,
            )
        )

        if show_annotations:
            for t, _label in ((t10, "10%"), (t90, "90%"), (settling_time, "settling")):
                if t is None:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[t, t],
                        y=[min(y_vals.min(), final_val), max(y_vals.max(), final_val)],
                        mode="lines",
                        name=None,
                        line=dict(color="rgba(200,200,200,0.6)", dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.full_like(x_vals, final_val),
                mode="lines",
                name="steady-state",
                line=dict(color="rgba(120,120,120,0.7)", dash="dash"),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title or f"Step response for {y}",
        template=template,
        xaxis_title=xlabel or xname,
        yaxis_title=ylabel or y,
        showlegend=True,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")

    metadata = {
        "kind": "step_response",
        "trace": y,
        "initial_value": init_val,
        "final_value": final_val,
        "t10": t10,
        "t90": t90,
        "rise_time": rise_time,
        "settling_time": settling_time,
        "settle_tolerance": settle_tolerance,
        "overshoot_pct": overshoot_pct,
    }
    return VizFigure(fig, metadata=metadata)


def sweep_curve(
    df: Any,
    x: str,
    y: str,
    hue: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    template: str | None = "plotly_white",
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    fig = go.Figure()
    for hue_value, group in df.groupby(hue):
        fig.add_trace(
            go.Scatter(
                x=group[x],
                y=group[y],
                mode="lines",
                name=f"{hue}={hue_value}",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or x,
        yaxis_title=ylabel,
        template=template,
        showlegend=legend,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    return VizFigure(fig, metadata={"kind": "sweep", "x": x, "y": y, "hue": hue})


def monte_carlo_histogram(
    metrics: Sequence[float],
    *,
    title: str | None = None,
    bins: int = 50,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
    show_normal_fit: bool = True,
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    values = np.asarray(list(metrics), dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=bins, marker=dict(color="#2E93fA"), opacity=0.85))

    normal_metadata: dict[str, float] | None = None
    if show_normal_fit and values.size >= 2:
        sigma = float(values.std(ddof=0))
        if sigma > 0.0:
            mu = float(values.mean())
            edges = np.histogram_bin_edges(values, bins=bins)
            xs = np.linspace(edges[0], edges[-1], 256)
            bin_width = (edges[-1] - edges[0]) / max(bins, 1)
            if bin_width > 0.0:
                coeff = values.size * bin_width
                ys = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
                    -0.5 * ((xs - mu) / sigma) ** 2
                )
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys * coeff,
                        mode="lines",
                        name="normal fit",
                        line=dict(color="#FF9F43", width=3),
                    )
                )
                normal_metadata = {"mean": mu, "std": sigma}
    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "metric",
        yaxis_title=ylabel or "count",
        template=template,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    metadata: dict[str, float | str | dict[str, float]] = {"kind": "mc_hist"}
    if normal_metadata is not None:
        metadata["normal_fit"] = normal_metadata
    return VizFigure(fig, metadata=metadata)


def monte_carlo_param_scatter(
    samples: Sequence[Mapping[str, float]],
    metrics: Sequence[float],
    param: str,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    xs = [sample.get(param, 0.0) for sample in samples]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=list(metrics),
            mode="markers",
            marker=dict(size=8, opacity=0.8, color=list(metrics), colorscale="Viridis"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel or param,
        yaxis_title=ylabel or "metric",
        template=template,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    return VizFigure(fig, metadata={"kind": "mc_param_scatter", "param": param})


def monte_carlo_kde(
    metrics: Sequence[float],
    *,
    title: str | None = None,
    xlabel: str | None = None,
    template: str | None = "plotly_white",
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    values = np.asarray(list(metrics), dtype=float)
    fig = go.Figure()

    kde = None
    if values.size >= 2:
        try:
            scipy_stats = importlib.import_module("scipy.stats")
            gaussian_kde = getattr(scipy_stats, "gaussian_kde", None)
        except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
            gaussian_kde = None
        if gaussian_kde is not None:
            try:
                kde = gaussian_kde(values)
            except Exception:  # pragma: no cover - scipy errors fall back to hist
                kde = None
    if kde is not None:
        xs = np.linspace(values.min(), values.max(), 256)
        fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", name="KDE"))
    else:
        fig.add_trace(go.Histogram(x=values, nbinsx=50, opacity=0.85, name="hist"))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "metric",
        yaxis_title="density",
        template=template,
        showlegend=kde is not None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    return VizFigure(fig, metadata={"kind": "mc_kde"})


def params_scatter_matrix(
    samples: Sequence[Mapping[str, float]],
    params: Sequence[str] | None = None,
    *,
    title: str | None = None,
    template: str | None = "plotly_white",
) -> VizFigure:
    if not samples:
        raise ValueError("No Monte Carlo samples provided for scatter matrix")

    go, _, px = _ensure_plotly()
    pd = importlib.import_module("pandas")

    df = pd.DataFrame(samples)
    if params is not None:
        df = df.loc[:, list(params)]
    fig = px.scatter_matrix(df, dimensions=df.columns, title=title)
    fig.update_layout(template=template)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")
    return VizFigure(fig, metadata={"kind": "params_matrix", "columns": list(df.columns)})


def wca_corners_bar(
    corner_names: Sequence[str],
    values: Sequence[float],
    *,
    nominal: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
    highlight_extremes: bool = True,
) -> VizFigure:
    """Plot WCA corner results as a horizontal bar chart.

    Args:
        corner_names: List of corner names (e.g., ['R1+, R2-', 'R1-, R2+']).
        values: Corresponding metric values for each corner.
        nominal: Optional nominal value to show as reference line.
        title: Plot title.
        xlabel: X-axis label (metric name).
        ylabel: Y-axis label.
        template: Plotly template.
        highlight_extremes: If True, highlight min/max bars in red/green.

    Returns:
        VizFigure with the bar chart.
    """
    go, _, _ = _ensure_plotly()

    values_arr = np.asarray(list(values), dtype=float)
    min_val = float(values_arr.min())
    max_val = float(values_arr.max())

    # Assign colors
    colors: list[str] | str
    if highlight_extremes:
        colors = []
        for v in values_arr:
            if v == min_val:
                colors.append("#E74C3C")  # Red for min
            elif v == max_val:
                colors.append("#27AE60")  # Green for max
            else:
                colors.append("#3498DB")  # Blue for others
    else:
        colors = "#3498DB"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=list(corner_names),
            x=values_arr,
            orientation="h",
            marker=dict(color=colors, opacity=0.85),
            text=[f"{v:.4f}" for v in values_arr],
            textposition="auto",
        )
    )

    if nominal is not None:
        fig.add_vline(
            x=nominal,
            line=dict(color="#F39C12", width=2, dash="dash"),
            annotation_text=f"Nominal: {nominal:.4f}",
            annotation_position="top",
        )

    fig.update_layout(
        title=title or "WCA Corner Analysis",
        xaxis_title=xlabel or "Value",
        yaxis_title=ylabel or "Corner",
        template=template,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")

    return VizFigure(
        fig,
        metadata={
            "kind": "wca_corners",
            "min": min_val,
            "max": max_val,
            "nominal": nominal,
            "n_corners": len(corner_names),
        },
    )


def monte_carlo_histogram_with_specs(
    metrics: Sequence[float],
    *,
    lsl: float | None = None,
    usl: float | None = None,
    nominal: float | None = None,
    title: str | None = None,
    bins: int = 50,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
    show_normal_fit: bool = True,
    show_stats: bool = True,
) -> VizFigure:
    """Plot Monte Carlo histogram with specification limits and statistics.

    Args:
        metrics: Sequence of metric values from Monte Carlo runs.
        lsl: Lower specification limit.
        usl: Upper specification limit.
        nominal: Nominal/target value.
        title: Plot title.
        bins: Number of histogram bins.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        template: Plotly template.
        show_normal_fit: Show normal distribution fit curve.
        show_stats: Show statistics annotation (mean, std, Cpk, yield).

    Returns:
        VizFigure with the histogram.
    """
    go, _, _ = _ensure_plotly()

    values = np.asarray(list(metrics), dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    n = len(values)

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            marker=dict(color="#3498DB", line=dict(color="white", width=1)),
            opacity=0.85,
            name="Distribution",
        )
    )

    # Normal fit
    if show_normal_fit and std > 0 and n >= 2:
        edges = np.histogram_bin_edges(values, bins=bins)
        xs = np.linspace(edges[0], edges[-1], 256)
        bin_width = (edges[-1] - edges[0]) / max(bins, 1)
        if bin_width > 0.0:
            coeff = n * bin_width
            ys = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys * coeff,
                    mode="lines",
                    name="Normal fit",
                    line=dict(color="#E67E22", width=3),
                )
            )

    # Mean line
    fig.add_vline(
        x=mean,
        line=dict(color="#E74C3C", width=2, dash="dash"),
        annotation_text=f"μ={mean:.4f}",
        annotation_position="top",
    )

    # 3-sigma bounds
    if std > 0:
        fig.add_vline(
            x=mean - 3 * std,
            line=dict(color="#9B59B6", width=1, dash="dot"),
        )
        fig.add_vline(
            x=mean + 3 * std,
            line=dict(color="#9B59B6", width=1, dash="dot"),
        )

    # Specification limits
    if lsl is not None:
        fig.add_vline(
            x=lsl,
            line=dict(color="#27AE60", width=2),
            annotation_text=f"LSL={lsl}",
            annotation_position="bottom left",
        )
    if usl is not None:
        fig.add_vline(
            x=usl,
            line=dict(color="#27AE60", width=2),
            annotation_text=f"USL={usl}",
            annotation_position="bottom right",
        )

    # Nominal
    if nominal is not None:
        fig.add_vline(
            x=nominal,
            line=dict(color="#F39C12", width=2, dash="dashdot"),
            annotation_text=f"Nom={nominal}",
            annotation_position="top right",
        )

    # Calculate Cpk and yield if spec limits provided
    cpk = None
    yield_pct = None
    failures = 0
    if lsl is not None and usl is not None and std > 0:
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)

        # Count failures
        failures = int(np.sum((values < lsl) | (values > usl)))
        yield_pct = (n - failures) / n * 100

    # Stats annotation
    if show_stats:
        stats_text = f"n={n}<br>μ={mean:.4f}<br>σ={std:.4f}"
        if cpk is not None:
            stats_text += f"<br>Cpk={cpk:.2f}"
        if yield_pct is not None:
            stats_text += f"<br>Yield={yield_pct:.1f}%"

        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=11),
            align="right",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        title=title or "Monte Carlo Distribution",
        xaxis_title=xlabel or "Value",
        yaxis_title=ylabel or "Count",
        template=template,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")

    return VizFigure(
        fig,
        metadata={
            "kind": "mc_hist_specs",
            "n": n,
            "mean": mean,
            "std": std,
            "cpk": cpk,
            "yield_pct": yield_pct,
            "failures": failures,
            "lsl": lsl,
            "usl": usl,
        },
    )


def monte_carlo_cumulative(
    metrics: Sequence[float],
    *,
    lsl: float | None = None,
    usl: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    template: str | None = "plotly_white",
) -> VizFigure:
    """Plot cumulative distribution function (CDF) for Monte Carlo results.

    Args:
        metrics: Sequence of metric values from Monte Carlo runs.
        lsl: Lower specification limit.
        usl: Upper specification limit.
        title: Plot title.
        xlabel: X-axis label.
        template: Plotly template.

    Returns:
        VizFigure with the CDF plot.
    """
    go, _, _ = _ensure_plotly()

    values = np.asarray(list(metrics), dtype=float)
    sorted_vals = np.sort(values)
    cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sorted_vals,
            y=cumulative,
            mode="lines",
            name="CDF",
            line=dict(color="#3498DB", width=2),
        )
    )

    # 50% line
    fig.add_hline(y=50, line=dict(color="gray", width=1, dash="dash"))

    # Median annotation
    median = float(np.median(values))
    fig.add_vline(
        x=median,
        line=dict(color="#E74C3C", width=1, dash="dash"),
        annotation_text=f"Median={median:.4f}",
        annotation_position="top",
    )

    # Spec limits
    if lsl is not None:
        fig.add_vline(x=lsl, line=dict(color="#27AE60", width=2))
        # Find percentile at LSL
        pct_at_lsl = float(np.sum(values < lsl) / len(values) * 100)
        fig.add_annotation(
            x=lsl,
            y=pct_at_lsl,
            text=f"{pct_at_lsl:.1f}% below LSL",
            showarrow=True,
            arrowhead=2,
        )

    if usl is not None:
        fig.add_vline(x=usl, line=dict(color="#27AE60", width=2))
        # Find percentile at USL
        pct_at_usl = float(np.sum(values <= usl) / len(values) * 100)
        fig.add_annotation(
            x=usl,
            y=pct_at_usl,
            text=f"{100 - pct_at_usl:.1f}% above USL",
            showarrow=True,
            arrowhead=2,
        )

    fig.update_layout(
        title=title or "Cumulative Distribution",
        xaxis_title=xlabel or "Value",
        yaxis_title="Cumulative %",
        template=template,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")

    return VizFigure(fig, metadata={"kind": "mc_cdf", "median": median})


def compare_traces(
    traces: Sequence[tuple[TraceSet, str, str]],
    x: str | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    template: str | None = "plotly_white",
    normalize: bool = False,
    show_difference: bool = False,
    legend: bool = True,
) -> VizFigure:
    """Compare multiple waveforms from different simulations.

    This is useful for comparing simulation results before/after a change,
    or comparing different circuit configurations.

    Args:
        traces: Sequence of (TraceSet, trace_name, label) tuples.
                Each tuple contains the data source, which trace to extract,
                and a label for the legend.
        x: Name of x-axis variable (auto-detected if None).
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        template: Plotly template.
        normalize: If True, normalize all traces to [0, 1] range.
        show_difference: If True and exactly 2 traces, show difference plot.
        legend: Show legend.

    Returns:
        VizFigure with the comparison plot.

    Example:
        >>> compare_traces([
        ...     (ts_before, "V(out)", "Before"),
        ...     (ts_after, "V(out)", "After"),
        ... ], title="Output voltage comparison")
    """
    go, make_subplots, _ = _ensure_plotly()

    if not traces:
        raise ValueError("At least one trace must be provided for comparison")

    # Determine number of subplot rows
    n_rows = 2 if show_difference and len(traces) == 2 else 1
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3] if n_rows == 2 else [1.0],
    )

    colors = [
        "#3498DB",
        "#E74C3C",
        "#27AE60",
        "#9B59B6",
        "#F39C12",
        "#1ABC9C",
        "#E91E63",
        "#00BCD4",
    ]

    extracted_data: list[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], str]] = []
    xname = None

    for i, (ts, trace_name, label) in enumerate(traces):
        # Get x values
        if x is not None:
            try:
                x_name = _resolve_trace_name(ts, x)
                x_vals = _numeric(ts[x_name].values)
                xname = x_name
            except Exception:
                x_vals, xname = _pick_x(ts)
        else:
            x_vals, xname = _pick_x(ts)

        # Get y values
        y_name = _resolve_trace_name(ts, trace_name)
        y_vals = _numeric(ts[y_name].values)

        # Handle complex values (take magnitude)
        if np.iscomplexobj(y_vals):
            y_vals = np.abs(y_vals)

        # Normalize if requested
        if normalize:
            y_min, y_max = float(y_vals.min()), float(y_vals.max())
            if y_max > y_min:
                y_vals = (y_vals - y_min) / (y_max - y_min)

        extracted_data.append((x_vals, y_vals, label))

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=label,
                line=dict(color=color, width=2),
            ),
            row=1,
            col=1,
        )

    # Add difference plot if requested
    diff_data = None
    if show_difference and len(traces) == 2:
        x1, y1, label1 = extracted_data[0]
        x2, y2, label2 = extracted_data[1]

        # Interpolate to common x-axis if needed
        if len(x1) == len(x2) and np.allclose(x1, x2):
            diff = y2 - y1
            x_common = x1
        else:
            # Use the denser x-axis
            if len(x1) >= len(x2):
                x_common = x1
                y2_interp = np.interp(x1, x2, y2)
                diff = y2_interp - y1
            else:
                x_common = x2
                y1_interp = np.interp(x2, x1, y1)
                diff = y2 - y1_interp

        diff_data = {
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_diff": float(np.mean(diff)),
            "rms_diff": float(np.sqrt(np.mean(diff**2))),
        }

        fig.add_trace(
            go.Scatter(
                x=x_common,
                y=diff,
                mode="lines",
                name=f"Δ ({label2} - {label1})",
                line=dict(color="#7F8C8D", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(127, 140, 141, 0.3)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, row=2, col=1, line=dict(color="gray", width=1, dash="dash"))
        fig.update_yaxes(title_text="Difference", row=2, col=1, showgrid=True)

    fig.update_layout(
        title=title or "Waveform Comparison",
        template=template,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(title_text=xlabel or xname, row=n_rows, col=1, showgrid=True)
    fig.update_yaxes(title_text=ylabel, row=1, col=1, showgrid=True)

    metadata: dict[str, Any] = {
        "kind": "compare_traces",
        "n_traces": len(traces),
        "labels": [label for _, _, label in traces],
        "normalized": normalize,
    }
    if diff_data is not None:
        metadata["difference"] = diff_data

    return VizFigure(fig, metadata=metadata)


def bode_with_margins(
    ts: TraceSet,
    y: str,
    x: str | None = None,
    *,
    unwrap_phase: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    template: str | None = "plotly_white",
    unity_gain_level: float = 0.0,
    phase_margin_target: float = -180.0,
) -> VizFigure:
    """Plot Bode diagram with gain margin and phase margin annotations.

    Args:
        ts: TraceSet containing frequency-domain data.
        y: Name of the complex-valued trace (e.g., loop gain).
        x: Name of x-axis variable (auto-detected if None).
        unwrap_phase: Unwrap phase to avoid discontinuities.
        title: Plot title.
        xlabel: X-axis label.
        template: Plotly template.
        unity_gain_level: Reference level for gain margin (default 0 dB).
        phase_margin_target: Phase level for phase margin (default -180 deg).

    Returns:
        VizFigure with Bode plot and margin annotations.
    """
    go, make_subplots, _ = _ensure_plotly()

    if x is not None:
        try:
            x_name = _resolve_trace_name(ts, x)
            x_vals = _numeric(ts[x_name].values)
            xname = x_name
        except Exception:
            x_vals, xname = _pick_x(ts)
    else:
        x_vals, xname = _pick_x(ts)

    y_name = _resolve_trace_name(ts, y)
    z = np.asarray(ts[y_name].values)
    if not np.iscomplexobj(z):
        raise ValueError(f"Trace '{y}' is not complex; AC analysis is required for Bode plots.")

    mag_db = 20.0 * np.log10(np.abs(z))
    phase = np.angle(z, deg=True)
    if unwrap_phase:
        phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))

    # Find gain crossover frequency (where magnitude crosses unity_gain_level)
    gain_crossover_freq = None
    phase_at_crossover = None
    phase_margin = None

    crossings = np.where(np.diff(np.sign(mag_db - unity_gain_level)))[0]
    if len(crossings) > 0:
        # Use first crossing
        idx = crossings[0]
        # Linear interpolation for more accurate frequency
        x0, x1 = x_vals[idx], x_vals[idx + 1]
        y0, y1 = mag_db[idx], mag_db[idx + 1]
        if y1 != y0:
            ratio = (unity_gain_level - y0) / (y1 - y0)
            gain_crossover_freq = x0 + ratio * (x1 - x0)
            # Interpolate phase at crossover
            p0, p1 = phase[idx], phase[idx + 1]
            phase_at_crossover = p0 + ratio * (p1 - p0)
            phase_margin = phase_at_crossover - phase_margin_target

    # Find phase crossover frequency (where phase crosses phase_margin_target)
    phase_crossover_freq = None
    gain_at_phase_crossover = None
    gain_margin = None

    phase_crossings = np.where(np.diff(np.sign(phase - phase_margin_target)))[0]
    if len(phase_crossings) > 0:
        idx = phase_crossings[0]
        x0, x1 = x_vals[idx], x_vals[idx + 1]
        p0, p1 = phase[idx], phase[idx + 1]
        if p1 != p0:
            ratio = (phase_margin_target - p0) / (p1 - p0)
            phase_crossover_freq = x0 + ratio * (x1 - x0)
            # Interpolate magnitude at phase crossover
            g0, g1 = mag_db[idx], mag_db[idx + 1]
            gain_at_phase_crossover = g0 + ratio * (g1 - g0)
            gain_margin = unity_gain_level - gain_at_phase_crossover

    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)

    # Magnitude plot
    fig.add_trace(
        go.Scatter(x=x_vals, y=mag_db, mode="lines", name="Magnitude [dB]"),
        row=1,
        col=1,
    )

    # Unity gain reference line
    fig.add_hline(
        y=unity_gain_level,
        row=1,
        col=1,
        line=dict(color="gray", width=1, dash="dot"),
    )

    # Phase plot
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=phase, mode="lines", name="Phase [deg]", line=dict(color="indianred")
        ),
        row=2,
        col=1,
    )

    # Phase margin target line
    fig.add_hline(
        y=phase_margin_target,
        row=2,
        col=1,
        line=dict(color="gray", width=1, dash="dot"),
    )

    # Annotate gain crossover and phase margin
    if gain_crossover_freq is not None and phase_at_crossover is not None:
        # Vertical line at gain crossover
        fig.add_vline(
            x=gain_crossover_freq,
            line=dict(color="#27AE60", width=1.5, dash="dash"),
        )
        # Phase margin annotation
        if phase_margin is not None:
            fig.add_annotation(
                x=gain_crossover_freq,
                y=phase_at_crossover,
                text=f"PM = {phase_margin:.1f}°",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#27AE60",
                font=dict(color="#27AE60"),
                row=2,
                col=1,
            )

    # Annotate phase crossover and gain margin
    if phase_crossover_freq is not None and gain_at_phase_crossover is not None:
        # Vertical line at phase crossover
        fig.add_vline(
            x=phase_crossover_freq,
            line=dict(color="#E74C3C", width=1.5, dash="dash"),
        )
        # Gain margin annotation
        if gain_margin is not None:
            fig.add_annotation(
                x=phase_crossover_freq,
                y=gain_at_phase_crossover,
                text=f"GM = {gain_margin:.1f} dB",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#E74C3C",
                font=dict(color="#E74C3C"),
                row=1,
                col=1,
            )

    fig.update_yaxes(title_text="Magnitude [dB]", row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Phase [deg]", row=2, col=1, showgrid=True)
    fig.update_xaxes(title_text=xlabel or xname, row=2, col=1, showgrid=True, type="log")
    fig.update_layout(
        title=title or f"Bode plot with margins for {y}",
        template=template,
        legend=dict(orientation="h"),
    )

    return VizFigure(
        fig,
        metadata={
            "kind": "bode_margins",
            "trace": y,
            "gain_crossover_freq": gain_crossover_freq,
            "phase_margin": phase_margin,
            "phase_crossover_freq": phase_crossover_freq,
            "gain_margin": gain_margin,
        },
    )


def multi_axis_plot(
    ts: TraceSet,
    traces: Sequence[tuple[str, str | None]],
    x: str | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    template: str | None = "plotly_white",
    legend: bool = True,
) -> VizFigure:
    """Plot multiple traces with separate Y-axes.

    Useful for plotting signals with different units or scales on the same time axis.

    Args:
        ts: TraceSet containing the data.
        traces: Sequence of (trace_name, y_axis_label) tuples.
                y_axis_label can be None to use trace name.
        x: Name of x-axis variable (auto-detected if None).
        title: Plot title.
        xlabel: X-axis label.
        template: Plotly template.
        legend: Show legend.

    Returns:
        VizFigure with multi-axis plot.

    Example:
        >>> multi_axis_plot(ts, [
        ...     ("V(out)", "Voltage [V]"),
        ...     ("I(R1)", "Current [A]"),
        ... ])
    """
    go, _, _ = _ensure_plotly()

    if not traces:
        raise ValueError("At least one trace must be provided")

    if x is not None:
        try:
            x_name = _resolve_trace_name(ts, x)
            x_vals = _numeric(ts[x_name].values)
            xname = x_name
        except Exception:
            x_vals, xname = _pick_x(ts)
    else:
        x_vals, xname = _pick_x(ts)

    colors = [
        "#3498DB",
        "#E74C3C",
        "#27AE60",
        "#9B59B6",
        "#F39C12",
        "#1ABC9C",
    ]

    fig = go.Figure()

    for i, (trace_name, _ylabel) in enumerate(traces):
        y_name = _resolve_trace_name(ts, trace_name)
        y_vals = ts[y_name].values

        # Handle complex values
        if np.iscomplexobj(y_vals):
            y_vals = np.abs(y_vals)
        else:
            y_vals = _numeric(y_vals)

        color = colors[i % len(colors)]

        if i == 0:
            # First trace uses primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, width=2),
                )
            )
        else:
            # Subsequent traces use secondary y-axes
            yaxis_name = f"y{i + 1}"
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=trace_name,
                    line=dict(color=color, width=2),
                    yaxis=yaxis_name,
                )
            )

    # Configure layout with multiple y-axes
    layout_updates: dict[str, Any] = {
        "title": title or "Multi-axis Plot",
        "template": template,
        "showlegend": legend,
        "xaxis": {"title": xlabel or xname, "showgrid": True},
        "yaxis": {
            "title": {"text": traces[0][1] or traces[0][0], "font": {"color": colors[0]}},
            "tickfont": {"color": colors[0]},
            "showgrid": True,
        },
    }

    # Add secondary y-axes
    for i in range(1, len(traces)):
        color = colors[i % len(colors)]
        axis_key = f"yaxis{i + 1}"
        layout_updates[axis_key] = {
            "title": {"text": traces[i][1] or traces[i][0], "font": {"color": color}},
            "tickfont": {"color": color},
            "overlaying": "y",
            "side": "right" if i % 2 == 1 else "left",
            "position": 1.0 - (i // 2) * 0.1 if i % 2 == 1 else (i // 2) * 0.1,
            "showgrid": False,
        }

    fig.update_layout(**layout_updates)

    return VizFigure(
        fig,
        metadata={
            "kind": "multi_axis",
            "traces": [name for name, _ in traces],
        },
    )
