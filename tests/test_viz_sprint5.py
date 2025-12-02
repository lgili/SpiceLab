"""Tests for Sprint 5 Visualization Features (M13).

Tests for:
- Waveform comparison (compare_traces)
- Bode plot with margins (bode_with_margins)
- Multi-axis plots (multi_axis_plot)
- Data export (to_csv, to_json)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


# Mock TraceSet for testing without simulation
class MockTrace:
    """Mock trace object."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    @property
    def values(self) -> np.ndarray:
        return self._values


class MockTraceSet:
    """Mock TraceSet for testing."""

    def __init__(self, data: dict[str, np.ndarray]) -> None:
        self._data = data
        self._names = list(data.keys())

    @property
    def names(self) -> list[str]:
        return self._names

    def __getitem__(self, name: str) -> MockTrace:
        return MockTrace(self._data[name])


# ============================================================================
# Compare Traces Tests
# ============================================================================


class TestCompareTraces:
    """Tests for waveform comparison view."""

    def test_compare_two_traces(self) -> None:
        """Test comparing two waveforms."""
        from spicelab.viz import compare_traces

        # Create two mock trace sets
        time = np.linspace(0, 1, 100)
        ts1 = MockTraceSet({"time": time, "V(out)": np.sin(2 * np.pi * time)})
        ts2 = MockTraceSet({"time": time, "V(out)": np.sin(2 * np.pi * time + 0.1)})

        fig = compare_traces(
            [
                (ts1, "V(out)", "Before"),  # type: ignore[arg-type]
                (ts2, "V(out)", "After"),  # type: ignore[arg-type]
            ],
            title="Comparison Test",
        )

        assert fig is not None
        assert fig.metadata is not None
        assert fig.metadata["kind"] == "compare_traces"
        assert fig.metadata["n_traces"] == 2
        assert fig.metadata["labels"] == ["Before", "After"]

    def test_compare_with_difference(self) -> None:
        """Test comparing two waveforms with difference plot."""
        from spicelab.viz import compare_traces

        time = np.linspace(0, 1, 100)
        ts1 = MockTraceSet({"time": time, "V(out)": np.ones(100) * 1.0})
        ts2 = MockTraceSet({"time": time, "V(out)": np.ones(100) * 1.1})

        fig = compare_traces(
            [
                (ts1, "V(out)", "Ref"),  # type: ignore[arg-type]
                (ts2, "V(out)", "Test"),  # type: ignore[arg-type]
            ],
            show_difference=True,
        )

        assert fig.metadata is not None
        assert "difference" in fig.metadata
        diff_info = fig.metadata["difference"]
        assert pytest.approx(diff_info["max_abs_diff"], rel=0.01) == 0.1
        assert pytest.approx(diff_info["mean_diff"], rel=0.01) == 0.1

    def test_compare_with_normalization(self) -> None:
        """Test comparing traces with normalization."""
        from spicelab.viz import compare_traces

        time = np.linspace(0, 1, 100)
        ts1 = MockTraceSet({"time": time, "V(out)": np.linspace(0, 1, 100)})
        ts2 = MockTraceSet({"time": time, "V(out)": np.linspace(0, 10, 100)})

        fig = compare_traces(
            [
                (ts1, "V(out)", "Small"),  # type: ignore[arg-type]
                (ts2, "V(out)", "Large"),  # type: ignore[arg-type]
            ],
            normalize=True,
        )

        assert fig.metadata is not None
        assert fig.metadata["normalized"] is True

    def test_compare_empty_raises(self) -> None:
        """Test that empty traces list raises error."""
        from spicelab.viz import compare_traces

        with pytest.raises(ValueError, match="At least one trace"):
            compare_traces([])

    def test_compare_multiple_traces(self) -> None:
        """Test comparing more than two traces."""
        from spicelab.viz import compare_traces

        time = np.linspace(0, 1, 100)
        ts1 = MockTraceSet({"time": time, "V(out)": np.sin(2 * np.pi * time)})
        ts2 = MockTraceSet({"time": time, "V(out)": np.cos(2 * np.pi * time)})
        ts3 = MockTraceSet({"time": time, "V(out)": np.sin(4 * np.pi * time)})

        fig = compare_traces(
            [
                (ts1, "V(out)", "Sin 1Hz"),  # type: ignore[arg-type]
                (ts2, "V(out)", "Cos 1Hz"),  # type: ignore[arg-type]
                (ts3, "V(out)", "Sin 2Hz"),  # type: ignore[arg-type]
            ],
        )

        assert fig.metadata["n_traces"] == 3


# ============================================================================
# Bode with Margins Tests
# ============================================================================


class TestBodeWithMargins:
    """Tests for Bode plot with gain/phase margin annotations."""

    def test_bode_margins_basic(self) -> None:
        """Test basic Bode with margins plot."""
        from spicelab.viz import bode_with_margins

        # Create a simple low-pass filter response
        freq = np.logspace(0, 6, 100)
        fc = 1e3  # 1kHz cutoff
        H = 1 / (1 + 1j * freq / fc)  # First-order LP

        ts = MockTraceSet({"frequency": freq, "V(out)": H})

        fig = bode_with_margins(ts, "V(out)", title="LP Filter Margins")  # type: ignore[arg-type]

        assert fig is not None
        assert fig.metadata is not None
        assert fig.metadata["kind"] == "bode_margins"

    def test_bode_margins_with_phase_margin(self) -> None:
        """Test Bode margins extraction from a system with phase margin."""
        from spicelab.viz import bode_with_margins

        # Create a second-order system with known crossover
        freq = np.logspace(0, 5, 500)
        s = 1j * 2 * np.pi * freq

        # Loop gain: G(s) = 1000 / (s * (1 + s/100) * (1 + s/1000))
        # This has gain crossover around 100 Hz
        omega0 = 2 * np.pi * 10
        omega1 = 2 * np.pi * 100
        omega2 = 2 * np.pi * 1000

        H = 1000 / (s / omega0 * (1 + s / omega1) * (1 + s / omega2))

        ts = MockTraceSet({"frequency": freq, "loop_gain": H})

        fig = bode_with_margins(ts, "loop_gain")  # type: ignore[arg-type]

        # Should have extracted some margin info
        assert fig.metadata is not None
        # May or may not have margins depending on crossover location
        assert "gain_crossover_freq" in fig.metadata
        assert "phase_margin" in fig.metadata

    def test_bode_non_complex_raises(self) -> None:
        """Test that non-complex trace raises error."""
        from spicelab.viz import bode_with_margins

        freq = np.logspace(0, 6, 100)
        ts = MockTraceSet({"frequency": freq, "V(out)": np.ones(100)})

        with pytest.raises(ValueError, match="not complex"):
            bode_with_margins(ts, "V(out)")  # type: ignore[arg-type]


# ============================================================================
# Multi-Axis Plot Tests
# ============================================================================


class TestMultiAxisPlot:
    """Tests for multi-axis plotting."""

    def test_multi_axis_two_traces(self) -> None:
        """Test multi-axis plot with two traces."""
        from spicelab.viz import multi_axis_plot

        time = np.linspace(0, 1, 100)
        voltage = np.sin(2 * np.pi * time)
        current = 0.1 * np.cos(2 * np.pi * time)

        ts = MockTraceSet(
            {
                "time": time,
                "V(out)": voltage,
                "I(R1)": current,
            }
        )

        fig = multi_axis_plot(
            ts,  # type: ignore[arg-type]
            [
                ("V(out)", "Voltage [V]"),
                ("I(R1)", "Current [A]"),
            ],
            title="Voltage and Current",
        )

        assert fig is not None
        assert fig.metadata is not None
        assert fig.metadata["kind"] == "multi_axis"
        assert len(fig.metadata["traces"]) == 2

    def test_multi_axis_empty_raises(self) -> None:
        """Test that empty traces list raises error."""
        from spicelab.viz import multi_axis_plot

        ts = MockTraceSet({"time": np.linspace(0, 1, 100)})

        with pytest.raises(ValueError, match="At least one trace"):
            multi_axis_plot(ts, [])  # type: ignore[arg-type]


# ============================================================================
# Data Export Tests
# ============================================================================


class TestDataExport:
    """Tests for VizFigure data export methods.

    Note: These tests require real Plotly. They are skipped if running with mocked Plotly.
    """

    @pytest.fixture
    def _skip_if_mocked(self) -> None:
        """Skip test if Plotly is mocked."""
        try:
            import plotly.graph_objects as go

            # Check if it's a real figure class
            fig = go.Figure()
            if not hasattr(fig, "to_dict") or not callable(fig.to_dict):
                pytest.skip("Plotly is mocked")
            # Check if the figure has proper structure
            if "data" not in fig.to_dict():
                pytest.skip("Plotly is mocked")
        except Exception:
            pytest.skip("Plotly not available")

    def test_to_csv_basic(self, _skip_if_mocked: None) -> None:
        """Test basic CSV export."""
        from spicelab.viz import time_series_view

        time = np.linspace(0, 1, 10)
        ts = MockTraceSet({"time": time, "V(out)": np.sin(time)})

        fig = time_series_view(ts, ["V(out)"])  # type: ignore[arg-type]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            result = fig.to_csv(path)
            assert result == path
            assert path.exists()

            content = path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) == 11  # header + 10 data rows
            assert "V(out)" in lines[0]  # header contains trace name
        finally:
            path.unlink(missing_ok=True)

    def test_to_csv_with_metadata(self, _skip_if_mocked: None) -> None:
        """Test CSV export with metadata comment."""
        from spicelab.viz import time_series_view

        time = np.linspace(0, 1, 5)
        ts = MockTraceSet({"time": time, "V(out)": np.ones(5)})

        fig = time_series_view(ts, ["V(out)"])  # type: ignore[arg-type]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)

        try:
            fig.to_csv(path, include_metadata=True)
            content = path.read_text()
            assert content.startswith("# Metadata:")
            assert '"kind": "time_series"' in content
        finally:
            path.unlink(missing_ok=True)

    def test_to_json_basic(self, _skip_if_mocked: None) -> None:
        """Test basic JSON export."""
        from spicelab.viz import time_series_view

        time = np.linspace(0, 1, 10)
        ts = MockTraceSet({"time": time, "V(out)": np.sin(time)})

        fig = time_series_view(ts, ["V(out)"])  # type: ignore[arg-type]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            result = fig.to_json(path)
            assert result == path
            assert path.exists()

            data = json.loads(path.read_text())
            assert "metadata" in data
            assert "traces" in data
            assert data["metadata"]["kind"] == "time_series"
            assert len(data["traces"]) >= 1
        finally:
            path.unlink(missing_ok=True)

    def test_to_json_without_figure(self, _skip_if_mocked: None) -> None:
        """Test JSON export without full Plotly figure."""
        from spicelab.viz import time_series_view

        time = np.linspace(0, 1, 5)
        ts = MockTraceSet({"time": time, "V(out)": np.ones(5)})

        fig = time_series_view(ts, ["V(out)"])  # type: ignore[arg-type]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            fig.to_json(path, include_figure=False)
            data = json.loads(path.read_text())
            assert "plotly_figure" not in data
            assert "traces" in data
        finally:
            path.unlink(missing_ok=True)


# ============================================================================
# Integration Tests
# ============================================================================


class TestVisualizationIntegration:
    """Integration tests for visualization features."""

    def test_compare_export_roundtrip(self) -> None:
        """Test creating comparison and exporting."""
        from spicelab.viz import compare_traces

        time = np.linspace(0, 1, 50)
        ts1 = MockTraceSet({"time": time, "V(out)": np.sin(time)})
        ts2 = MockTraceSet({"time": time, "V(out)": np.cos(time)})

        fig = compare_traces(
            [
                (ts1, "V(out)", "Sin"),  # type: ignore[arg-type]
                (ts2, "V(out)", "Cos"),  # type: ignore[arg-type]
            ],
            show_difference=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            fig.to_json(path)
            data = json.loads(path.read_text())

            # Verify metadata includes difference stats
            assert data["metadata"]["kind"] == "compare_traces"
            assert "difference" in data["metadata"]
        finally:
            path.unlink(missing_ok=True)

    def test_all_new_views_have_metadata(self) -> None:
        """Test that all new views produce proper metadata."""
        from spicelab.viz import bode_with_margins, compare_traces, multi_axis_plot

        # Setup data
        time = np.linspace(0, 1, 100)
        freq = np.logspace(0, 4, 100)

        ts_time = MockTraceSet({"time": time, "V(out)": np.sin(time), "I(R1)": np.cos(time)})
        ts_freq = MockTraceSet({"frequency": freq, "H": 1 / (1 + 1j * freq / 1000)})

        # Test compare_traces
        fig1 = compare_traces([(ts_time, "V(out)", "Test")])  # type: ignore[arg-type]
        assert fig1.metadata is not None
        assert fig1.metadata["kind"] == "compare_traces"

        # Test bode_with_margins
        fig2 = bode_with_margins(ts_freq, "H")  # type: ignore[arg-type]
        assert fig2.metadata is not None
        assert fig2.metadata["kind"] == "bode_margins"

        # Test multi_axis_plot
        fig3 = multi_axis_plot(ts_time, [("V(out)", None), ("I(R1)", None)])  # type: ignore[arg-type]
        assert fig3.metadata is not None
        assert fig3.metadata["kind"] == "multi_axis"
