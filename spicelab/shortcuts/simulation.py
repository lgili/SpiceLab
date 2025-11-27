"""Simulation shortcuts for common analysis workflows.

Provides quick_* functions with smart defaults to reduce boilerplate:
- quick_op: DC operating point analysis
- quick_ac: AC frequency sweep with sensible defaults
- quick_tran: Transient analysis with auto timestep calculation
- quick_noise: Noise analysis with sensible defaults
- detailed_ac: AC sweep with high point density
- detailed_tran: Transient with fine timestep

Also provides:
- suggest_analysis: Auto-detect suitable analysis from circuit topology
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from ..core.types import ResultHandle

__all__ = [
    "quick_op",
    "quick_ac",
    "quick_tran",
    "quick_noise",
    "detailed_ac",
    "detailed_tran",
    "suggest_analysis",
    "AnalysisSuggestion",
]


# =============================================================================
# DC Operating Point
# =============================================================================


def quick_op(
    circuit: Any,
    engine: str = "ngspice",
    probes: list[Any] | None = None,
) -> ResultHandle:
    """Run DC operating point analysis.

    Calculates the DC bias point of the circuit - all node voltages
    and branch currents with all sources at their DC values.

    Args:
        circuit: Circuit to simulate
        engine: Simulation engine to use (default "ngspice")
        probes: Optional list of probes to measure

    Returns:
        ResultHandle with operating point results

    Example:
        >>> from spicelab.shortcuts import quick_op
        >>> result = quick_op(circuit)
        >>> ds = result.dataset()
        >>> print(ds["V(out)"])
    """
    from ..core.types import AnalysisSpec
    from ..engines import run_simulation

    analyses = [AnalysisSpec("op", {})]
    result = run_simulation(circuit, analyses, engine=engine, probes=probes or [])
    return cast("ResultHandle", result)


# =============================================================================
# AC Analysis Presets
# =============================================================================


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


def detailed_ac(
    circuit: Any,
    start: float | str = 0.1,
    stop: float | str = 10e9,
    points_per_decade: int = 100,
    engine: str = "ngspice",
    probes: list[Any] | None = None,
) -> ResultHandle:
    """Run detailed AC sweep with high point density.

    Same as quick_ac but with 100 points per decade (vs 20) for
    smoother curves and better resolution of peaks/nulls.

    Args:
        circuit: Circuit to simulate
        start: Start frequency in Hz (default 0.1 Hz)
        stop: Stop frequency in Hz (default 10 GHz)
        points_per_decade: Number of points per decade (default 100)
        engine: Simulation engine to use (default "ngspice")
        probes: Optional list of probes to measure

    Returns:
        ResultHandle with AC sweep results

    Example:
        >>> from spicelab.shortcuts import detailed_ac
        >>> result = detailed_ac(circuit, start="1", stop="10Meg")
    """
    return quick_ac(
        circuit,
        start=start,
        stop=stop,
        points_per_decade=points_per_decade,
        engine=engine,
        probes=probes,
    )


# =============================================================================
# Transient Analysis Presets
# =============================================================================


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


def detailed_tran(
    circuit: Any,
    duration: float | str,
    timestep: float | str | None = None,
    engine: str = "ngspice",
    probes: list[Any] | None = None,
) -> ResultHandle:
    """Run detailed transient analysis with fine timestep.

    Same as quick_tran but with finer timestep (duration/10000 vs duration/1000)
    for better time resolution.

    Args:
        circuit: Circuit to simulate
        duration: Total simulation time. Accepts SI suffixes: "1ms", "10u"
        timestep: Simulation timestep (optional). If None, auto-calculated as duration/10000.
        engine: Simulation engine to use (default "ngspice")
        probes: Optional list of probes to measure

    Returns:
        ResultHandle with transient results

    Example:
        >>> from spicelab.shortcuts import detailed_tran
        >>> result = detailed_tran(circuit, duration="1ms")
    """
    from ..core.units import parse_value_flexible

    # Parse duration for auto-timestep calculation
    if isinstance(duration, str):
        parsed_duration = parse_value_flexible(duration)
    else:
        parsed_duration = duration

    # Use finer timestep if not provided
    if timestep is None:
        timestep = float(parsed_duration) / 10000.0

    return quick_tran(
        circuit,
        duration=duration,
        timestep=timestep,
        engine=engine,
        probes=probes,
    )


# =============================================================================
# Noise Analysis
# =============================================================================


def quick_noise(
    circuit: Any,
    output_node: str,
    input_source: str,
    start: float | str = 1.0,
    stop: float | str = 1e6,
    points_per_decade: int = 10,
    engine: str = "ngspice",
    probes: list[Any] | None = None,
) -> ResultHandle:
    """Run noise analysis with sensible defaults.

    Performs noise analysis, computing equivalent input noise and
    output noise across a frequency sweep.

    Args:
        circuit: Circuit to simulate
        output_node: Node where noise is measured (e.g., "vout")
        input_source: Reference input source (e.g., "V1")
        start: Start frequency in Hz (default 1 Hz)
        stop: Stop frequency in Hz (default 1 MHz)
        points_per_decade: Number of points per decade (default 10)
        engine: Simulation engine to use (default "ngspice")
        probes: Optional list of probes to measure

    Returns:
        ResultHandle with noise analysis results

    Example:
        >>> from spicelab.shortcuts import quick_noise
        >>> result = quick_noise(circuit, output_node="vout", input_source="V1")
    """
    from ..core.types import AnalysisSpec
    from ..core.units import parse_value_flexible
    from ..engines import run_simulation

    if isinstance(start, str):
        start = parse_value_flexible(start)
    if isinstance(stop, str):
        stop = parse_value_flexible(stop)

    analyses = [
        AnalysisSpec(
            "noise",
            {
                "output": output_node,
                "src": input_source,
                "sweep_type": "dec",
                "n": points_per_decade,
                "fstart": float(start),
                "fstop": float(stop),
            },
        )
    ]

    result = run_simulation(circuit, analyses, engine=engine, probes=probes or [])
    return cast("ResultHandle", result)


# =============================================================================
# Auto-Detection / Suggestion
# =============================================================================


@dataclass
class AnalysisSuggestion:
    """Suggested analysis based on circuit topology."""

    analysis_type: str  # "op", "ac", "tran", "noise"
    reason: str
    preset_function: str  # Name of suggested function
    suggested_params: dict[str, Any]
    confidence: float  # 0.0 - 1.0


def suggest_analysis(circuit: Any) -> list[AnalysisSuggestion]:
    """Auto-detect suitable analysis types from circuit topology.

    Examines circuit components to suggest appropriate analyses:
    - Circuits with only DC sources → DC operating point
    - Circuits with AC sources or reactive components → AC analysis
    - Circuits with time-varying sources (pulse, sin) → Transient
    - Circuits with resistors and active devices → Noise analysis

    Args:
        circuit: Circuit to analyze

    Returns:
        List of AnalysisSuggestion sorted by confidence (highest first)

    Example:
        >>> from spicelab.shortcuts import suggest_analysis
        >>> suggestions = suggest_analysis(circuit)
        >>> for s in suggestions:
        ...     print(f"{s.analysis_type}: {s.reason} ({s.confidence:.0%})")
    """
    suggestions: list[AnalysisSuggestion] = []

    # Analyze component types
    has_dc_source = False
    has_ac_source = False
    has_time_varying_source = False
    has_reactive = False  # L or C
    has_resistor = False
    has_active_device = False  # transistor, opamp

    # Time constants for transient estimation
    estimated_time_constant: float | None = None
    capacitances: list[float] = []
    resistances: list[float] = []
    inductances: list[float] = []

    for comp in circuit._components:
        comp_type = type(comp).__name__

        # Source detection
        if comp_type in ("Vdc", "Idc"):
            has_dc_source = True
        elif comp_type in ("Vac", "Iac"):
            has_ac_source = True
        elif comp_type in ("Vpulse", "Ipulse", "Vsin", "Isin", "VsinT", "IsinT"):
            has_time_varying_source = True
            has_dc_source = True  # Most also have DC bias

        # Passive components
        elif comp_type == "Resistor":
            has_resistor = True
            if hasattr(comp, "resistance"):
                resistances.append(comp.resistance)
        elif comp_type == "Capacitor":
            has_reactive = True
            if hasattr(comp, "capacitance"):
                capacitances.append(comp.capacitance)
        elif comp_type == "Inductor":
            has_reactive = True
            if hasattr(comp, "inductance"):
                inductances.append(comp.inductance)

        # Active devices
        elif comp_type in (
            "NPN",
            "PNP",
            "NMOS",
            "PMOS",
            "OpAmpIdeal",
            "OpAmp",
            "JFET",
        ):
            has_active_device = True
            has_dc_source = True  # Active circuits need bias

    # Estimate time constant for transient
    if resistances and capacitances:
        # Simple RC time constant estimate
        avg_r = sum(resistances) / len(resistances)
        avg_c = sum(capacitances) / len(capacitances)
        estimated_time_constant = avg_r * avg_c
    elif resistances and inductances:
        # Simple RL time constant
        avg_r = sum(resistances) / len(resistances)
        avg_l = sum(inductances) / len(inductances)
        estimated_time_constant = avg_l / avg_r

    # Estimate cutoff frequency for AC
    estimated_fc: float | None = None
    if estimated_time_constant:
        import math

        estimated_fc = 1.0 / (2 * math.pi * estimated_time_constant)

    # Generate suggestions based on analysis

    # 1. DC Operating Point - always useful for circuits with DC sources
    if has_dc_source or has_active_device:
        suggestions.append(
            AnalysisSuggestion(
                analysis_type="op",
                reason="Circuit has DC sources or active devices requiring bias",
                preset_function="quick_op",
                suggested_params={},
                confidence=0.9 if has_active_device else 0.7,
            )
        )

    # 2. AC Analysis - for circuits with reactive components or AC sources
    if has_reactive or has_ac_source:
        ac_params: dict[str, Any] = {}
        confidence = 0.8

        if estimated_fc:
            # Suggest frequency range centered around estimated fc
            ac_params["start"] = max(1.0, estimated_fc / 1000)
            ac_params["stop"] = estimated_fc * 1000
            confidence = 0.9

        reason = "Circuit has reactive components (L/C)"
        if has_ac_source:
            reason = "Circuit has AC source for frequency response analysis"
            confidence = 0.95

        suggestions.append(
            AnalysisSuggestion(
                analysis_type="ac",
                reason=reason,
                preset_function="quick_ac",
                suggested_params=ac_params,
                confidence=confidence,
            )
        )

    # 3. Transient Analysis - for time-varying sources
    if has_time_varying_source:
        tran_params: dict[str, Any] = {}

        if estimated_time_constant:
            # Simulate for ~5 time constants to reach steady state
            tran_params["duration"] = estimated_time_constant * 5

        suggestions.append(
            AnalysisSuggestion(
                analysis_type="tran",
                reason="Circuit has time-varying source (pulse/sine)",
                preset_function="quick_tran",
                suggested_params=tran_params,
                confidence=0.95,
            )
        )
    elif has_reactive and has_dc_source:
        # Even with DC, reactive circuits have transient response
        tran_params = {}
        if estimated_time_constant:
            tran_params["duration"] = estimated_time_constant * 5

        suggestions.append(
            AnalysisSuggestion(
                analysis_type="tran",
                reason="Reactive circuit has transient response to DC step",
                preset_function="quick_tran",
                suggested_params=tran_params,
                confidence=0.6,
            )
        )

    # 4. Noise Analysis - for circuits with active devices and resistors
    if has_active_device and has_resistor:
        noise_params: dict[str, Any] = {}
        if estimated_fc:
            noise_params["start"] = max(1.0, estimated_fc / 100)
            noise_params["stop"] = estimated_fc * 10

        suggestions.append(
            AnalysisSuggestion(
                analysis_type="noise",
                reason="Active circuit with resistors has thermal/shot noise",
                preset_function="quick_noise",
                suggested_params=noise_params,
                confidence=0.5,  # Lower confidence - needs output/input nodes
            )
        )

    # Sort by confidence (highest first)
    suggestions.sort(key=lambda s: s.confidence, reverse=True)

    return suggestions
