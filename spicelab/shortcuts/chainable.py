"""Chainable result wrapper for fluent API workflows.

Provides a ChainableResult class that wraps ResultHandle and enables
method chaining for common workflows:

    result = quick_ac(circuit).pm().bw().plot()

This module implements tasks 4.3, 4.4, and 4.5 from usability-improvements spec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..core.types import ResultHandle
    from ..viz.plotly import VizFigure


__all__ = [
    "ChainableResult",
    "MeasurementResult",
    "wrap_result",
]


@dataclass
class MeasurementResult:
    """Container for a single measurement result."""

    name: str
    value: float
    units: str
    details: dict[str, Any]

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4g} {self.units}"


class ChainableResult:
    """Fluent wrapper around ResultHandle for method chaining.

    Enables workflows like:
        result = quick_ac(circuit).pm("vout", "vin").bw("vout").plot()

    All measurement methods return self for chaining.
    Use .value() or .values() to extract measurement results.

    Attributes:
        handle: The underlying ResultHandle
        measurements: List of MeasurementResult from method calls
        analysis_type: Detected analysis type ("ac", "tran", "op", "noise", None)
    """

    def __init__(self, handle: ResultHandle) -> None:
        """Initialize chainable wrapper.

        Args:
            handle: ResultHandle from simulation
        """
        self._handle = handle
        self._measurements: list[MeasurementResult] = []
        self._analysis_type: str | None = None
        self._detect_analysis_type()

    def _detect_analysis_type(self) -> None:
        """Detect analysis type from dataset coordinates."""
        try:
            ds = self._handle.dataset()
            coords = list(getattr(ds, "coords", {}).keys())
            if "frequency" in coords or "freq" in coords:
                self._analysis_type = "ac"
            elif "time" in coords:
                self._analysis_type = "tran"
            else:
                # Check attrs for analysis info
                attrs = self._handle.attrs()
                analyses = attrs.get("analyses", [])
                if analyses:
                    first_mode = analyses[0].get("mode") if isinstance(analyses[0], dict) else None
                    if first_mode:
                        self._analysis_type = first_mode
        except Exception:
            pass

    @property
    def handle(self) -> ResultHandle:
        """Access the underlying ResultHandle."""
        return self._handle

    @property
    def measurements(self) -> list[MeasurementResult]:
        """List of measurements performed."""
        return self._measurements.copy()

    @property
    def analysis_type(self) -> str | None:
        """Detected analysis type."""
        return self._analysis_type

    def dataset(self) -> Any:
        """Access the underlying xarray Dataset."""
        return self._handle.dataset()

    def to_polars(self) -> Any:
        """Convert to polars DataFrame."""
        return self._handle.to_polars()

    def attrs(self) -> Mapping[str, Any]:
        """Access result metadata."""
        return self._handle.attrs()

    # =========================================================================
    # Measurement Shortcuts (4.4)
    # =========================================================================

    def pm(
        self,
        numerator: str = "vout",
        denominator: str = "vin",
        name: str = "phase_margin",
    ) -> ChainableResult:
        """Measure phase margin at unity-gain crossover.

        Args:
            numerator: Output signal name (default "vout")
            denominator: Input signal name (default "vin")
            name: Measurement name for results

        Returns:
            self for method chaining

        Example:
            >>> result = quick_ac(circuit).pm("vout", "vin")
            >>> print(result.value("phase_margin"))
        """
        from ..analysis.measure import PhaseMarginSpec, measure

        spec = PhaseMarginSpec(name=name, numerator=numerator, denominator=denominator)
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units="deg",
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="deg",
                    details={"error": str(e)},
                )
            )
        return self

    def gm(
        self,
        numerator: str = "vout",
        denominator: str = "vin",
        name: str = "gain_margin",
        tolerance_deg: float = 15.0,
    ) -> ChainableResult:
        """Measure gain margin at phase = -180°.

        Args:
            numerator: Output signal name (default "vout")
            denominator: Input signal name (default "vin")
            name: Measurement name for results
            tolerance_deg: Tolerance for phase crossing detection

        Returns:
            self for method chaining
        """
        from ..analysis.measure import GainMarginSpec, measure

        spec = GainMarginSpec(
            name=name,
            numerator=numerator,
            denominator=denominator,
            tolerance_deg=tolerance_deg,
        )
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units="dB",
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="dB",
                    details={"error": str(e)},
                )
            )
        return self

    def bw(
        self,
        numerator: str = "vout",
        denominator: str = "vin",
        name: str = "bandwidth",
    ) -> ChainableResult:
        """Measure unity-gain bandwidth (GBW).

        Args:
            numerator: Output signal name (default "vout")
            denominator: Input signal name (default "vin")
            name: Measurement name for results

        Returns:
            self for method chaining

        Example:
            >>> result = quick_ac(circuit).bw("vout", "vin")
            >>> print(result.value("bandwidth"))
        """
        from ..analysis.measure import GainBandwidthSpec, measure

        spec = GainBandwidthSpec(name=name, numerator=numerator, denominator=denominator)
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units="Hz",
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="Hz",
                    details={"error": str(e)},
                )
            )
        return self

    def gain(
        self,
        numerator: str = "vout",
        freq: float = 1.0,
        denominator: str | None = None,
        kind: Literal["mag", "db"] = "db",
        name: str = "gain",
    ) -> ChainableResult:
        """Measure gain at a specific frequency.

        Args:
            numerator: Output signal name
            freq: Frequency in Hz
            denominator: Input signal name (optional)
            kind: "db" for dB, "mag" for linear magnitude
            name: Measurement name for results

        Returns:
            self for method chaining
        """
        from ..analysis.measure import GainSpec, measure

        spec = GainSpec(
            name=name,
            numerator=numerator,
            freq=freq,
            denominator=denominator,
            kind=kind,
        )
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units=str(row.get("units", "dB" if kind == "db" else "V/V")),
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="dB" if kind == "db" else "V/V",
                    details={"error": str(e)},
                )
            )
        return self

    def overshoot(
        self,
        signal: str = "vout",
        target: float = 1.0,
        reference: float | None = None,
        percent: bool = True,
        name: str = "overshoot",
    ) -> ChainableResult:
        """Measure peak overshoot in transient response.

        Args:
            signal: Signal name to measure
            target: Target steady-state value
            reference: Initial reference value (default: first sample)
            percent: If True, return percentage; else absolute
            name: Measurement name for results

        Returns:
            self for method chaining

        Example:
            >>> result = quick_tran(circuit, "1ms").overshoot("vout", target=1.0)
        """
        from ..analysis.measure import OvershootSpec, measure

        spec = OvershootSpec(
            name=name,
            signal=signal,
            target=target,
            reference=reference,
            percent=percent,
        )
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units="%" if percent else "V",
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="%" if percent else "V",
                    details={"error": str(e)},
                )
            )
        return self

    def settling_time(
        self,
        signal: str = "vout",
        target: float = 1.0,
        tolerance: float = 0.02,
        tolerance_kind: Literal["abs", "pct"] = "pct",
        name: str = "settling_time",
    ) -> ChainableResult:
        """Measure settling time to within tolerance of target.

        Args:
            signal: Signal name to measure
            target: Target steady-state value
            tolerance: Tolerance band (absolute or percentage)
            tolerance_kind: "abs" for absolute, "pct" for percentage
            name: Measurement name for results

        Returns:
            self for method chaining
        """
        from ..analysis.measure import SettlingTimeSpec, measure

        spec = SettlingTimeSpec(
            name=name,
            signal=signal,
            target=target,
            tolerance=tolerance,
            tolerance_kind=tolerance_kind,
        )
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units="s",
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="s",
                    details={"error": str(e)},
                )
            )
        return self

    def rise_time(
        self,
        signal: str = "vout",
        target: float | None = None,
        reference: float | None = None,
        low_pct: float = 0.1,
        high_pct: float = 0.9,
        name: str = "rise_time",
    ) -> ChainableResult:
        """Measure 10-90% (or custom) rise time.

        Args:
            signal: Signal name to measure
            target: Target value (default: max of signal)
            reference: Reference value (default: first sample)
            low_pct: Low threshold percentage (default 0.1 = 10%)
            high_pct: High threshold percentage (default 0.9 = 90%)
            name: Measurement name for results

        Returns:
            self for method chaining
        """
        from ..analysis.measure import RiseTimeSpec, measure

        spec = RiseTimeSpec(
            name=name,
            signal=signal,
            target=target,
            reference=reference,
            low_pct=low_pct,
            high_pct=high_pct,
        )
        try:
            rows = measure(self._handle, [spec], return_as="python")
            if rows:
                row = rows[0]
                self._measurements.append(
                    MeasurementResult(
                        name=name,
                        value=float(row.get("value", float("nan"))),
                        units="s",
                        details=dict(row),
                    )
                )
        except Exception as e:
            self._measurements.append(
                MeasurementResult(
                    name=name,
                    value=float("nan"),
                    units="s",
                    details={"error": str(e)},
                )
            )
        return self

    # =========================================================================
    # Result Extraction
    # =========================================================================

    def value(self, name: str) -> float:
        """Get a single measurement value by name.

        Args:
            name: Measurement name

        Returns:
            Measurement value

        Raises:
            KeyError: If measurement not found
        """
        for m in self._measurements:
            if m.name == name:
                return m.value
        available = [m.name for m in self._measurements]
        raise KeyError(f"Measurement '{name}' not found. Available: {available}")

    def values(self) -> dict[str, float]:
        """Get all measurement values as a dictionary.

        Returns:
            Dictionary mapping measurement names to values
        """
        return {m.name: m.value for m in self._measurements}

    def summary(self) -> str:
        """Get a formatted summary of all measurements.

        Returns:
            Multi-line string with measurement results
        """
        if not self._measurements:
            return "No measurements performed"
        lines = ["Measurements:"]
        for m in self._measurements:
            lines.append(f"  {m.name}: {m.value:.4g} {m.units}")
        return "\n".join(lines)

    # =========================================================================
    # Plotting (4.5 - Auto-plot selection)
    # =========================================================================

    def plot(
        self,
        signals: Sequence[str] | str | None = None,
        *,
        title: str | None = None,
        **kwargs: Any,
    ) -> VizFigure:
        """Auto-select and create appropriate plot based on analysis type.

        Automatically chooses:
        - AC analysis → Bode plot
        - Transient analysis → Time series plot
        - Other → Time series plot (fallback)

        Args:
            signals: Signal(s) to plot. If None, plots first available signal.
            title: Plot title
            **kwargs: Additional arguments passed to underlying plot function

        Returns:
            VizFigure for display or further customization

        Example:
            >>> quick_ac(circuit).plot("vout")  # Auto-selects Bode plot
            >>> quick_tran(circuit, "1ms").plot("vout")  # Auto-selects time series
        """
        from ..io.raw_reader import TraceSet
        from ..viz.plots import plot_bode, plot_traces

        ds = self._handle.dataset()
        ts = TraceSet.from_dataset(ds)

        # Normalize signals to list
        if signals is None:
            # Pick first non-coordinate signal
            available = [n for n in ts.names if n.lower() not in ("time", "frequency", "freq")]
            if not available:
                available = list(ts.names)
            signals = available[:1] if available else []
        elif isinstance(signals, str):
            signals = [signals]

        if not signals:
            raise ValueError("No signals available to plot")

        # Auto-select plot type based on analysis
        if self._analysis_type == "ac":
            # Bode plot for AC analysis
            return plot_bode(ts, signals[0], title_mag=title, **kwargs)
        else:
            # Time series for transient/other
            return plot_traces(ts, list(signals), title=title, **kwargs)

    def plot_bode(
        self,
        signal: str = "vout",
        *,
        unwrap_phase: bool = True,
        title: str | None = None,
        **kwargs: Any,
    ) -> VizFigure:
        """Create Bode plot (magnitude and phase vs frequency).

        Args:
            signal: Signal to plot
            unwrap_phase: Whether to unwrap phase
            title: Plot title
            **kwargs: Additional arguments

        Returns:
            VizFigure
        """
        from ..io.raw_reader import TraceSet
        from ..viz.plots import plot_bode

        ds = self._handle.dataset()
        ts = TraceSet.from_dataset(ds)
        return plot_bode(ts, signal, unwrap_phase=unwrap_phase, title_mag=title, **kwargs)

    def plot_traces(
        self,
        signals: Sequence[str] | None = None,
        *,
        title: str | None = None,
        **kwargs: Any,
    ) -> VizFigure:
        """Create time series plot.

        Args:
            signals: Signals to plot
            title: Plot title
            **kwargs: Additional arguments

        Returns:
            VizFigure
        """
        from ..io.raw_reader import TraceSet
        from ..viz.plots import plot_traces

        ds = self._handle.dataset()
        ts = TraceSet.from_dataset(ds)
        return plot_traces(ts, signals, title=title, **kwargs)

    def plot_step(
        self,
        signal: str = "vout",
        *,
        steady_state: float | None = None,
        title: str | None = None,
        **kwargs: Any,
    ) -> VizFigure:
        """Create step response plot with annotations.

        Args:
            signal: Signal to plot
            steady_state: Expected steady-state value
            title: Plot title
            **kwargs: Additional arguments

        Returns:
            VizFigure
        """
        from ..io.raw_reader import TraceSet
        from ..viz.plots import plot_step_response

        ds = self._handle.dataset()
        ts = TraceSet.from_dataset(ds)
        return plot_step_response(ts, signal, steady_state=steady_state, title=title, **kwargs)

    def plot_nyquist(
        self,
        signal: str = "vout",
        *,
        title: str | None = None,
        **kwargs: Any,
    ) -> VizFigure:
        """Create Nyquist plot (imaginary vs real).

        Args:
            signal: Signal to plot
            title: Plot title
            **kwargs: Additional arguments

        Returns:
            VizFigure
        """
        from ..io.raw_reader import TraceSet
        from ..viz.plots import plot_nyquist

        ds = self._handle.dataset()
        ts = TraceSet.from_dataset(ds)
        return plot_nyquist(ts, signal, title=title, **kwargs)

    def __repr__(self) -> str:
        analysis = self._analysis_type or "unknown"
        n_meas = len(self._measurements)
        return f"<ChainableResult analysis={analysis} measurements={n_meas}>"


def wrap_result(handle: ResultHandle) -> ChainableResult:
    """Wrap a ResultHandle for fluent method chaining.

    Args:
        handle: ResultHandle from simulation

    Returns:
        ChainableResult wrapper

    Example:
        >>> from spicelab.shortcuts import quick_ac, wrap_result
        >>> result = wrap_result(quick_ac(circuit))
        >>> result.pm().bw().plot()
    """
    return ChainableResult(handle)
