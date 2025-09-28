"""Interactive visualization helpers built on Plotly."""

from __future__ import annotations

from .circuit import components_for_net, connectivity_table, summary_table
from .notebook import connectivity_widget, dataset_plot_widget
from .plotly import (
    VizFigure,
    bode_view,
    monte_carlo_histogram,
    monte_carlo_kde,
    monte_carlo_param_scatter,
    nyquist_view,
    params_scatter_matrix,
    step_response_view,
    sweep_curve,
    time_series_view,
)
from .plots import (
    plot_bode,
    plot_mc_kde,
    plot_mc_metric_hist,
    plot_nyquist,
    plot_param_vs_metric,
    plot_params_matrix,
    plot_step_response,
    plot_sweep_df,
    plot_traces,
)

__all__ = [
    "VizFigure",
    "time_series_view",
    "bode_view",
    "step_response_view",
    "sweep_curve",
    "monte_carlo_histogram",
    "monte_carlo_param_scatter",
    "monte_carlo_kde",
    "params_scatter_matrix",
    "nyquist_view",
    "plot_traces",
    "plot_bode",
    "plot_step_response",
    "plot_nyquist",
    "plot_sweep_df",
    "plot_mc_metric_hist",
    "plot_mc_kde",
    "plot_param_vs_metric",
    "plot_params_matrix",
    "connectivity_table",
    "summary_table",
    "components_for_net",
    "connectivity_widget",
    "dataset_plot_widget",
]
