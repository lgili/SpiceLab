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
