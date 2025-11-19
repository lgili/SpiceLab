"""Simulation shortcuts for common analysis workflows.

Provides quick_* functions with smart defaults to reduce boilerplate:
- quick_ac: AC frequency sweep with sensible defaults
- quick_tran: Transient analysis with auto timestep calculation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..core.types import ResultHandle

__all__ = ["quick_ac", "quick_tran"]


def quick_ac(
    circuit: Any,
    start: float | str = 1.0,
    stop: float | str = 1e9,
    points_per_decade: int = 20,
    engine: str = "ngspice",
    probes: list[Any] | None = None,
) -> ResultHandle:
    """Run AC frequency sweep with smart defaults.

    Performs a logarithmic (decade) AC sweep from start to stop frequency.
    Automatically configures sweep type and point density.

    Args:
        circuit: Circuit to simulate
        start: Start frequency in Hz (default 1 Hz). Accepts SI suffixes: "1k", "1MHz"
        stop: Stop frequency in Hz (default 1 GHz). Accepts SI suffixes: "10Meg", "1G"
        points_per_decade: Number of points per decade (default 20)
        engine: Simulation engine to use (default "ngspice")
        probes: Optional list of probes to measure

    Returns:
        ResultHandle with AC sweep results

    Example:
        >>> from spicelab.shortcuts import quick_ac
        >>> circuit = rc_lowpass(fc=1000)
        >>> result = quick_ac(circuit, start="10", stop="1Meg")
        >>> ds = result.dataset()
        >>> print(ds.coords["frequency"])
    """
    from ..core.types import AnalysisSpec
    from ..core.units import parse_value_flexible
    from ..engines import run_simulation

    # Parse frequency values (handles both numbers and strings with units)
    if isinstance(start, str):
        start = parse_value_flexible(start)
    if isinstance(stop, str):
        stop = parse_value_flexible(stop)

    # Create AC analysis spec with decade sweep
    analyses = [
        AnalysisSpec(
            "ac",
            {
                "sweep_type": "dec",
                "n": points_per_decade,
                "fstart": float(start),
                "fstop": float(stop),
            },
        )
    ]

    result = run_simulation(circuit, analyses, engine=engine, probes=probes or [])
    return cast("ResultHandle", result)


def quick_tran(
    circuit: Any,
    duration: float | str,
    timestep: float | str | None = None,
    engine: str = "ngspice",
    probes: list[Any] | None = None,
) -> ResultHandle:
    """Run transient analysis with auto timestep calculation.

    Performs transient (time-domain) simulation from t=0 to duration.
    Automatically calculates appropriate timestep if not provided (duration/1000).

    Args:
        circuit: Circuit to simulate
        duration: Total simulation time. Accepts SI suffixes: "1ms", "10u", "1m"
        timestep: Simulation timestep (optional). If None, auto-calculated as duration/1000.
            Accepts SI suffixes: "1ns", "10u"
        engine: Simulation engine to use (default "ngspice")
        probes: Optional list of probes to measure

    Returns:
        ResultHandle with transient results

    Example:
        >>> from spicelab.shortcuts import quick_tran
        >>> circuit = rc_lowpass(fc=1000)
        >>> result = quick_tran(circuit, duration="10ms")
        >>> ds = result.dataset()
        >>> print(ds.coords["time"])

        >>> # Custom timestep
        >>> result = quick_tran(circuit, duration="1ms", timestep="1us")
    """
    from ..core.types import AnalysisSpec
    from ..core.units import parse_value_flexible
    from ..engines import run_simulation

    # Parse duration (handles both numbers and strings with units)
    if isinstance(duration, str):
        duration = parse_value_flexible(duration)

    # Auto-calculate timestep if not provided (duration/1000)
    if timestep is None:
        timestep = float(duration) / 1000.0
    elif isinstance(timestep, str):
        timestep = parse_value_flexible(timestep)

    # Create transient analysis spec
    analyses = [
        AnalysisSpec(
            "tran",
            {
                "tstep": float(timestep),
                "tstop": float(duration),
            },
        )
    ]

    result = run_simulation(circuit, analyses, engine=engine, probes=probes or [])
    return cast("ResultHandle", result)
