from __future__ import annotations

import ipywidgets as widgets
import numpy as np
import pytest
import xarray as xr
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.viz.notebook import (
    connectivity_widget,
    dataset_plot_widget,
    infer_x_dimension,
    should_enable_log_scale,
)


def _sample_circuit() -> Circuit:
    circuit = Circuit("widget_test")
    v1 = Vdc(ref="VIN", value="5")
    r1 = Resistor(ref="RLOAD", value="1k")
    circuit.add(v1, r1)
    in_net = Net("vin")
    circuit.connect(v1.ports[0], in_net)
    circuit.connect(v1.ports[1], GND)
    circuit.connect(r1.ports[0], in_net)
    circuit.connect(r1.ports[1], GND)
    return circuit


def test_connectivity_widget_returns_vbox() -> None:
    circ = _sample_circuit()
    widget = connectivity_widget(circ)
    assert isinstance(widget, widgets.VBox)
    # dropdown + html children expected
    assert len(widget.children) == 2


def test_dataset_plot_widget_basic() -> None:
    time = np.linspace(0, 1, 5)
    dataset = xr.Dataset({"Vout": ("time", np.sin(2 * np.pi * time))}, coords={"time": time})
    widget = dataset_plot_widget(dataset)
    assert isinstance(widget, widgets.VBox)
    assert widget.children  # figure and controls present


def test_dataset_plot_widget_requires_numeric_variable() -> None:
    dataset = xr.Dataset({"label": ("time", ["a", "b"])}, coords={"time": [0, 1]})
    try:
        dataset_plot_widget(dataset)
    except ValueError as exc:
        assert "numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for non-numeric dataset")


def test_infer_x_dimension_prefers_frequency_axis() -> None:
    freq = np.logspace(2, 5, 12)
    trials = np.arange(3)
    ds = xr.Dataset(
        {"gain": (("trial", "freq"), np.ones((trials.size, freq.size)))},
        coords={"trial": trials, "freq": freq},
    )
    assert infer_x_dimension(ds) == "freq"


def test_infer_x_dimension_validates_explicit_dimension() -> None:
    time = np.linspace(0, 1, 50)
    channels = np.arange(4)
    ds = xr.Dataset(
        {"signal": (("time", "channel"), np.ones((time.size, channels.size)))},
        coords={"time": time, "channel": channels},
    )
    assert infer_x_dimension(ds, "time") == "time"
    with pytest.raises(ValueError):
        infer_x_dimension(ds, "voltage")


def test_should_enable_log_scale_detects_frequency_like_axes() -> None:
    freq = np.logspace(1, 4, 20)
    assert should_enable_log_scale(freq, "freq") is True


def test_should_enable_log_scale_filters_invalid_axes() -> None:
    values = np.linspace(-1.0, 1.0, 10)
    assert should_enable_log_scale(values, "freq") is False
    short = np.array([1.0])
    assert should_enable_log_scale(short, "freq") is False


def test_dataset_plot_widget_adds_log_scale_toggle_for_frequency() -> None:
    freq = np.logspace(2, 5, 30)
    dataset = xr.Dataset({"gain": ("freq", np.ones_like(freq))}, coords={"freq": freq})
    widget = dataset_plot_widget(dataset)
    controls = widget.children[0]
    assert isinstance(controls, widgets.HBox)
    toggles = [child for child in controls.children if isinstance(child, widgets.ToggleButtons)]
    assert toggles, "Expected a log scale toggle when frequency axis is present"
    fig = widget.children[1]
    assert fig.layout.xaxis.type == "log"


def test_dataset_plot_widget_skips_log_scale_for_linear_axes() -> None:
    time = np.linspace(0, 1, 25)
    dataset = xr.Dataset({"gain": ("time", np.ones_like(time))}, coords={"time": time})
    widget = dataset_plot_widget(dataset)
    controls = widget.children[0]
    toggles = [child for child in controls.children if isinstance(child, widgets.ToggleButtons)]
    assert not toggles, "Did not expect a log scale toggle for linear time axis"
