"""Plotly-based visualization primitives for CAT analyses."""

from __future__ import annotations

import importlib
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
            " pip install pycircuitkit[viz]"
        ) from exc
    return go, make_subplots, px


def _numeric(values: Iterable[Any]) -> NDArray[np.floating[Any]]:
    return np.asarray(list(values), dtype=float)


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

    def show(self, **kwargs: Any) -> None:
        self.figure.show(**kwargs)


def time_series_view(
    ts: TraceSet,
    ys: Sequence[str] | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    grid: bool = True,
    template: str | None = "plotly_white",
    markers: bool = False,
    color_map: Mapping[str, str] | None = None,
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    x, xname = _pick_x(ts)
    names = [n for n in ts.names if n != xname] if ys is None else list(ys)

    fig = go.Figure()
    mode = "lines+markers" if markers else "lines"
    for name in names:
        values = ts[name].values
        line = None
        if color_map and name in color_map:
            line = dict(color=color_map[name])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=values,
                mode=mode,
                name=name,
                line=line,
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
    *,
    unwrap_phase: bool = True,
    title: str | None = None,
    xlabel: str | None = None,
    template: str | None = "plotly_white",
) -> VizFigure:
    go, make_subplots, _ = _ensure_plotly()
    x, xname = _pick_x(ts)
    z = np.asarray(ts[y].values)
    if not np.iscomplexobj(z):
        raise ValueError(f"Trace '{y}' is not complex; AC analysis is required for Bode plots.")

    mag_db = 20.0 * np.log10(np.abs(z))
    phase = np.angle(z, deg=True)
    if unwrap_phase:
        phase = np.rad2deg(np.unwrap(np.deg2rad(phase)))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(x=x, y=mag_db, mode="lines", name="Magnitude [dB]"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=phase, mode="lines", name="Phase [deg]", line=dict(color="indianred")),
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
) -> VizFigure:
    go, _, _ = _ensure_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=list(metrics), nbinsx=bins, marker=dict(color="#2E93fA"), opacity=0.85)
    )
    fig.update_layout(
        title=title,
        xaxis_title=xlabel or "metric",
        yaxis_title=ylabel or "count",
        template=template,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.2)")
    return VizFigure(fig, metadata={"kind": "mc_hist"})


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
            kde = gaussian_kde(values)
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
