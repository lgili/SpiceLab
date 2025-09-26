from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from spicelab.io.raw_reader import Trace, TraceSet


class FakeFigure:
    def __init__(self) -> None:
        self.data: list[Any] = []
        self.layout = SimpleNamespace(
            title=SimpleNamespace(text=None),
            template=SimpleNamespace(layout=SimpleNamespace(paper_bgcolor=None)),
        )
        self.layout_updates: list[dict[str, Any]] = []
        self.x_updates: list[dict[str, Any]] = []
        self.y_updates: list[dict[str, Any]] = []
        self.fail_image = False

    def add_trace(self, trace: Any, row: int | None = None, col: int | None = None) -> None:
        self.data.append(trace)

    def update_layout(self, **kwargs: Any) -> None:
        self.layout_updates.append(kwargs)
        if "title" in kwargs:
            self.layout.title.text = kwargs["title"]
        if "template" in kwargs:
            tpl = kwargs["template"]
            if isinstance(tpl, str):
                self.layout.template = SimpleNamespace(
                    name=tpl, layout=SimpleNamespace(paper_bgcolor="rgba(0,0,0,0)")
                )
            else:
                self.layout.template = tpl

    def update_xaxes(self, **kwargs: Any) -> None:
        self.x_updates.append(kwargs)

    def update_yaxes(self, **kwargs: Any) -> None:
        self.y_updates.append(kwargs)

    def to_html(self, full_html: bool = True, include_plotlyjs: str = "cdn") -> str:
        return "<html></html>"

    def write_image(self, path: str, scale: float = 2.0, format: str = "png") -> None:
        if self.fail_image:
            raise ValueError("missing kaleido")
        Path(path).write_text("ok", encoding="utf-8")

    def show(self, **kwargs: Any) -> None:  # pragma: no cover - benign
        self.layout_updates.append({"show": kwargs})


class FakeScatter:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.name = kwargs.get("name")


class FakeHistogram:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


go_module = cast(Any, ModuleType("plotly.graph_objects"))
go_module.Figure = FakeFigure
go_module.Scatter = FakeScatter
go_module.Histogram = FakeHistogram

subplots_module = cast(Any, ModuleType("plotly.subplots"))
subplots_module.make_subplots = lambda *args, **kwargs: FakeFigure()

express_module = cast(Any, ModuleType("plotly.express"))
express_module.scatter_matrix = lambda df, dimensions, title=None: FakeFigure()

plotly_module = ModuleType("plotly")
sys.modules["plotly"] = plotly_module
sys.modules["plotly.graph_objects"] = go_module
sys.modules["plotly.subplots"] = subplots_module
sys.modules["plotly.express"] = express_module


class _MiniFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def groupby(self, key: str) -> list[tuple[Any, _MiniFrame]]:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for row in self._rows:
            groups.setdefault(row[key], []).append(row)
        return [(k, _MiniFrame(v)) for k, v in groups.items()]

    def __getitem__(self, key: str) -> list[Any]:
        return [row[key] for row in self._rows]


from spicelab.analysis.viz import plot as plot_wrappers  # noqa: E402
from spicelab.viz.plotly import (  # noqa: E402
    VizFigure,
    bode_view,
    monte_carlo_histogram,
    monte_carlo_kde,
    monte_carlo_param_scatter,
    params_scatter_matrix,
    sweep_curve,
    time_series_view,
)


def _trace_set(
    values: Sequence[complex] | Sequence[float], *, complex_trace: bool = False
) -> TraceSet:
    x = np.array([0.0, 1.0, 2.0])
    traces = [Trace("time", "s", x)]
    arr = np.array(values)
    if complex_trace:
        traces.append(Trace("V(out)", None, arr))
    else:
        traces.append(Trace("V(out)", None, arr))
    return TraceSet(traces)


def test_time_series_view_and_wrapper() -> None:
    ts = _trace_set([0.0, 1.0, 2.0])
    fig = time_series_view(ts, ys=["V(out)"], title="demo")
    assert isinstance(fig, VizFigure)
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "time_series"
    wrapped = plot_wrappers.plot_traces(ts, ys=["V(out)"])
    wrapped_meta = wrapped.metadata
    assert wrapped_meta is not None
    assert wrapped_meta["kind"] == "time_series"


def test_bode_view_complex_trace() -> None:
    ts = _trace_set([1 + 1j, 1 + 2j, 1 + 3j], complex_trace=True)
    fig = bode_view(ts, "V(out)")
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "bode"
    wrapped = plot_wrappers.plot_bode(ts, "V(out)")
    assert isinstance(wrapped.figure, FakeFigure)


def test_sweep_curve_and_wrapper_grid_toggle() -> None:
    rows = [{"x": 0, "y": 1, "corner": "tt"}, {"x": 1, "y": 2, "corner": "ff"}]
    df = _MiniFrame(rows)
    fig = sweep_curve(df, "x", "y", "corner")
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "sweep"
    wrapped = plot_wrappers.plot_sweep_df(df, "x", "y", "corner", grid=False)
    assert isinstance(wrapped.figure, FakeFigure)


def test_monte_carlo_histogram_and_wrapper() -> None:
    metrics = [0.1, 0.2, 0.3]
    fig = monte_carlo_histogram(metrics, bins=10)
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "mc_hist"
    wrapped = plot_wrappers.plot_mc_metric_hist(metrics, bins=5)
    assert isinstance(wrapped.figure, FakeFigure)


def test_monte_carlo_param_scatter_and_wrapper_no_grid() -> None:
    samples: list[Mapping[str, float]] = [{"R": 100.0}, {"R": 200.0}]
    metrics = [1.0, 2.0]
    fig = monte_carlo_param_scatter(samples, metrics, "R")
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "mc_param_scatter"
    wrapped = plot_wrappers.plot_param_vs_metric(samples, metrics, "R", grid=False)
    assert isinstance(wrapped.figure, FakeFigure)


def test_monte_carlo_kde_hist_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    scipy_stats = ModuleType("scipy.stats")
    scipy_stats.gaussian_kde = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scipy.stats", scipy_stats)
    fig = monte_carlo_kde([0.0, 0.5, 1.0])
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "mc_kde"


def test_params_scatter_matrix_filters_columns() -> None:
    samples = [
        {"R": 100.0, "C": 1e-9},
        {"R": 110.0, "C": 1.1e-9},
    ]
    fig = params_scatter_matrix(samples, params=["R"])
    metadata = fig.metadata
    assert metadata is not None
    assert metadata["kind"] == "params_matrix"


def test_plot_wrappers_reject_matplotlib_axes() -> None:
    ts = _trace_set([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        plot_wrappers.plot_traces(ts, ax=object())

    df = _MiniFrame([{"x": 0, "y": 1, "corner": "tt"}, {"x": 1, "y": 2, "corner": "ff"}])
    with pytest.raises(ValueError):
        plot_wrappers.plot_sweep_df(df, "x", "y", "corner", ax=object())

    samples: list[Mapping[str, float]] = [{"R": 100.0}]
    metrics = [0.5]
    with pytest.raises(ValueError):
        plot_wrappers.plot_param_vs_metric(samples, metrics, "R", ax=object())


def test_vizfigure_export_helpers(tmp_path: Path) -> None:
    fig = VizFigure(FakeFigure())
    out_html = fig.to_html(tmp_path / "plot.html")
    assert out_html.exists()
    out_img = fig.to_image(tmp_path / "plot.png")
    assert out_img.exists()
    failing = VizFigure(FakeFigure())
    failing.figure.fail_image = True
    with pytest.raises(RuntimeError):
        failing.to_image(tmp_path / "fail.png")


def test_plot_wrappers_metric_none_error() -> None:
    bad_metrics: Any = None
    with pytest.raises(ValueError):
        plot_wrappers.plot_mc_metric_hist(bad_metrics)
