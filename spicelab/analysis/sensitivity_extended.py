"""Extended sensitivity analysis for circuit design (M16).

This module provides additional sensitivity analysis capabilities:
- Temperature sensitivity analysis
- Component tolerance sensitivity
- Sensitivity reports with formatting
- Design margin analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np


# =============================================================================
# Temperature Sensitivity
# =============================================================================


@dataclass
class TemperaturePoint:
    """Result at a specific temperature.

    Attributes:
        temperature: Temperature in Celsius
        value: Output metric value
        normalized: Normalized deviation from nominal
    """

    temperature: float
    value: float
    normalized: float = 0.0


@dataclass
class TemperatureSensitivityResult:
    """Results from temperature sensitivity analysis.

    Attributes:
        metric_name: Name of the measured metric
        nominal_temp: Nominal temperature (usually 25°C)
        nominal_value: Value at nominal temperature
        points: List of temperature points
        tempco: Temperature coefficient (ppm/°C or %/°C)
        tempco_units: Units of tempco
        min_value: Minimum value across temperature range
        max_value: Maximum value across temperature range
        range_pct: Percentage variation across temperature range
    """

    metric_name: str
    nominal_temp: float
    nominal_value: float
    points: list[TemperaturePoint]
    tempco: float
    tempco_units: Literal["ppm/C", "%/C", "unit/C"]
    min_value: float
    max_value: float
    range_pct: float

    @property
    def temperatures(self) -> list[float]:
        """Get list of temperatures."""
        return [p.temperature for p in self.points]

    @property
    def values(self) -> list[float]:
        """Get list of values."""
        return [p.value for p in self.points]

    def value_at(self, temp: float) -> float | None:
        """Get value at specific temperature (if measured)."""
        for p in self.points:
            if abs(p.temperature - temp) < 0.1:
                return p.value
        return None

    def interpolate_at(self, temp: float) -> float:
        """Interpolate value at given temperature."""
        temps = np.array(self.temperatures)
        vals = np.array(self.values)
        return float(np.interp(temp, temps, vals))

    def worst_case_deviation(self) -> tuple[float, float]:
        """Get worst case deviation from nominal.

        Returns:
            Tuple of (temperature, deviation_pct) for worst case
        """
        worst_temp = self.nominal_temp
        worst_dev = 0.0

        for p in self.points:
            dev = abs(p.value - self.nominal_value) / abs(self.nominal_value) * 100
            if dev > worst_dev:
                worst_dev = dev
                worst_temp = p.temperature

        return worst_temp, worst_dev

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "nominal_temp": self.nominal_temp,
            "nominal_value": self.nominal_value,
            "tempco": self.tempco,
            "tempco_units": self.tempco_units,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "range_pct": self.range_pct,
            "points": [
                {"temp": p.temperature, "value": p.value, "normalized": p.normalized}
                for p in self.points
            ],
        }


def temperature_sensitivity(
    objective: Callable[[float], float],
    metric_name: str = "output",
    *,
    temp_range: tuple[float, float] = (-40.0, 85.0),
    nominal_temp: float = 25.0,
    n_points: int = 11,
    tempco_units: Literal["ppm/C", "%/C", "unit/C"] = "%/C",
) -> TemperatureSensitivityResult:
    """Analyze sensitivity to temperature.

    Sweeps temperature and measures how output changes, computing
    temperature coefficient and worst-case deviation.

    Args:
        objective: Function that takes temperature and returns metric value
        metric_name: Name of the output metric
        temp_range: (min, max) temperature in Celsius
        nominal_temp: Nominal/reference temperature
        n_points: Number of temperature points to evaluate
        tempco_units: Units for temperature coefficient

    Returns:
        TemperatureSensitivityResult with analysis data

    Example:
        def run_at_temp(temp):
            # Run simulation at temperature
            result = simulate(circuit, temp=temp)
            return result.gain_db

        sens = temperature_sensitivity(
            run_at_temp,
            metric_name="Gain",
            temp_range=(-40, 125),
        )
        print(f"Tempco: {sens.tempco:.2f} {sens.tempco_units}")
    """
    # Generate temperature points
    temps = np.linspace(temp_range[0], temp_range[1], n_points)

    # Ensure nominal is included
    if nominal_temp not in temps:
        temps = np.sort(np.append(temps, nominal_temp))

    # Evaluate at each temperature
    points: list[TemperaturePoint] = []
    nominal_value = 0.0

    for temp in temps:
        value = objective(temp)
        points.append(TemperaturePoint(temperature=temp, value=value))
        if abs(temp - nominal_temp) < 0.1:
            nominal_value = value

    # Calculate normalized values
    if abs(nominal_value) > 1e-15:
        for p in points:
            p.normalized = (p.value - nominal_value) / nominal_value

    # Calculate temperature coefficient
    values = np.array([p.value for p in points])
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    range_pct = (max_val - min_val) / abs(nominal_value) * 100 if abs(nominal_value) > 1e-15 else 0.0

    # Linear fit for tempco
    temps_arr = np.array([p.temperature for p in points])
    slope, _ = np.polyfit(temps_arr, values, 1)

    # Convert to requested units
    if tempco_units == "ppm/C":
        tempco = slope / nominal_value * 1e6 if abs(nominal_value) > 1e-15 else 0.0
    elif tempco_units == "%/C":
        tempco = slope / nominal_value * 100 if abs(nominal_value) > 1e-15 else 0.0
    else:  # unit/C
        tempco = slope

    return TemperatureSensitivityResult(
        metric_name=metric_name,
        nominal_temp=nominal_temp,
        nominal_value=nominal_value,
        points=points,
        tempco=tempco,
        tempco_units=tempco_units,
        min_value=min_val,
        max_value=max_val,
        range_pct=range_pct,
    )


# =============================================================================
# Tolerance Sensitivity
# =============================================================================


@dataclass
class ComponentTolerance:
    """Definition of a component's tolerance.

    Attributes:
        name: Component reference (e.g., "R1", "C2")
        nominal: Nominal value
        tolerance_pct: Tolerance as percentage (e.g., 5 for 5%)
        value_unit: Unit of the value (e.g., "ohm", "F")
    """

    name: str
    nominal: float
    tolerance_pct: float
    value_unit: str = ""


@dataclass
class ToleranceImpact:
    """Impact of a component's tolerance on output.

    Attributes:
        component: Component name
        nominal: Nominal component value
        tolerance_pct: Component tolerance percentage
        output_sensitivity: dOutput/dComponent (normalized)
        output_contribution: Contribution to total output variation (%)
        min_output: Output at -tolerance
        max_output: Output at +tolerance
    """

    component: str
    nominal: float
    tolerance_pct: float
    output_sensitivity: float
    output_contribution: float
    min_output: float
    max_output: float


@dataclass
class ToleranceSensitivityResult:
    """Results from tolerance sensitivity analysis.

    Attributes:
        metric_name: Name of the output metric
        nominal_output: Output value with all components at nominal
        impacts: List of per-component impacts, sorted by contribution
        total_variation_pct: Total output variation (RSS of contributions)
        worst_case_variation_pct: Worst case output variation (sum of absolutes)
    """

    metric_name: str
    nominal_output: float
    impacts: list[ToleranceImpact]
    total_variation_pct: float
    worst_case_variation_pct: float

    def get_ranking(self) -> list[str]:
        """Get components ranked by impact on output."""
        return [i.component for i in sorted(self.impacts, key=lambda x: x.output_contribution, reverse=True)]

    def get_critical_components(self, threshold_pct: float = 10.0) -> list[str]:
        """Get components contributing more than threshold to output variation."""
        return [i.component for i in self.impacts if i.output_contribution >= threshold_pct]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "nominal_output": self.nominal_output,
            "total_variation_pct": self.total_variation_pct,
            "worst_case_variation_pct": self.worst_case_variation_pct,
            "impacts": [
                {
                    "component": i.component,
                    "nominal": i.nominal,
                    "tolerance_pct": i.tolerance_pct,
                    "sensitivity": i.output_sensitivity,
                    "contribution_pct": i.output_contribution,
                }
                for i in self.impacts
            ],
        }


def tolerance_sensitivity(
    objective: Callable[[dict[str, float]], float],
    components: list[ComponentTolerance],
    metric_name: str = "output",
) -> ToleranceSensitivityResult:
    """Analyze how component tolerances affect output.

    For each component, evaluates how output changes when the component
    is varied by its tolerance, calculating sensitivity and contribution
    to total output variation.

    Args:
        objective: Function that takes dict of {component: value} and returns output
        components: List of components with their tolerances
        metric_name: Name of the output metric

    Returns:
        ToleranceSensitivityResult with per-component impacts

    Example:
        def run_with_values(values):
            circuit.R1.value = values["R1"]
            circuit.C1.value = values["C1"]
            return simulate(circuit).gain

        components = [
            ComponentTolerance("R1", 10000, 5),  # 10k, 5%
            ComponentTolerance("C1", 1e-9, 10),  # 1nF, 10%
        ]

        result = tolerance_sensitivity(run_with_values, components, "Gain")
        print(f"Critical: {result.get_critical_components()}")
    """
    # Get nominal output
    nominal_values = {c.name: c.nominal for c in components}
    nominal_output = objective(nominal_values)

    impacts: list[ToleranceImpact] = []
    contributions_sq: list[float] = []
    worst_case_sum = 0.0

    for comp in components:
        # Evaluate at -tolerance and +tolerance
        delta = comp.nominal * comp.tolerance_pct / 100

        values_low = nominal_values.copy()
        values_low[comp.name] = comp.nominal - delta

        values_high = nominal_values.copy()
        values_high[comp.name] = comp.nominal + delta

        out_low = objective(values_low)
        out_high = objective(values_high)

        # Calculate sensitivity (normalized)
        # dOutput/dComponent * (Component/Output)
        if abs(nominal_output) > 1e-15 and abs(delta) > 1e-15:
            sensitivity = (out_high - out_low) / (2 * delta) * comp.nominal / nominal_output
        else:
            sensitivity = 0.0

        # Output variation due to this component's tolerance
        output_delta = abs(out_high - out_low) / 2
        if abs(nominal_output) > 1e-15:
            contribution = (output_delta / abs(nominal_output) * 100) ** 2
        else:
            contribution = 0.0

        contributions_sq.append(contribution)
        worst_case_sum += abs(output_delta) / abs(nominal_output) * 100 if abs(nominal_output) > 1e-15 else 0.0

        impacts.append(
            ToleranceImpact(
                component=comp.name,
                nominal=comp.nominal,
                tolerance_pct=comp.tolerance_pct,
                output_sensitivity=sensitivity,
                output_contribution=0.0,  # Will be normalized below
                min_output=out_low,
                max_output=out_high,
            )
        )

    # Calculate RSS total and normalize contributions
    total_rss = np.sqrt(sum(contributions_sq))

    for i, impact in enumerate(impacts):
        # Contribution as fraction of total RSS variation
        if total_rss > 0:
            impact.output_contribution = np.sqrt(contributions_sq[i]) / total_rss * 100
        else:
            impact.output_contribution = 0.0

    # Sort by contribution
    impacts.sort(key=lambda x: x.output_contribution, reverse=True)

    return ToleranceSensitivityResult(
        metric_name=metric_name,
        nominal_output=nominal_output,
        impacts=impacts,
        total_variation_pct=total_rss,
        worst_case_variation_pct=worst_case_sum,
    )


# =============================================================================
# Sensitivity Reports
# =============================================================================


@dataclass
class SensitivityReportSection:
    """A section in a sensitivity report."""

    title: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report.

    Combines multiple analyses into a formatted report.
    """

    title: str
    created_at: datetime
    sections: list[SensitivityReportSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_section(self, title: str, content: str, data: dict[str, Any] | None = None) -> None:
        """Add a section to the report."""
        self.sections.append(SensitivityReportSection(title=title, content=content, data=data or {}))

    def to_text(self) -> str:
        """Generate plain text report."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(self.title)
        lines.append("=" * 60)
        lines.append(f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        for section in self.sections:
            lines.append("-" * 40)
            lines.append(section.title)
            lines.append("-" * 40)
            lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Generate HTML report."""
        html_parts: list[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html><head>")
        html_parts.append(f"<title>{self.title}</title>")
        html_parts.append("<style>")
        html_parts.append("""
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; border-bottom: 2px solid #333; }
            h2 { color: #555; margin-top: 30px; }
            table { border-collapse: collapse; margin: 10px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f5f5f5; }
            .critical { color: #d00; font-weight: bold; }
            .warning { color: #fa0; }
            .ok { color: #0a0; }
            pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
            .timestamp { color: #888; font-size: 0.9em; }
        """)
        html_parts.append("</style></head><body>")

        html_parts.append(f"<h1>{self.title}</h1>")
        html_parts.append(f'<p class="timestamp">Generated: {self.created_at.strftime("%Y-%m-%d %H:%M:%S")}</p>')

        for section in self.sections:
            html_parts.append(f"<h2>{section.title}</h2>")
            # Convert content to HTML (preserve newlines)
            content_html = section.content.replace("\n", "<br>")
            html_parts.append(f"<div>{content_html}</div>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    def save(self, path: Path, format: Literal["text", "html"] = "text") -> Path:
        """Save report to file.

        Args:
            path: Output file path
            format: "text" or "html"

        Returns:
            Path to saved file
        """
        path = Path(path)
        if format == "html":
            content = self.to_html()
            if not path.suffix:
                path = path.with_suffix(".html")
        else:
            content = self.to_text()
            if not path.suffix:
                path = path.with_suffix(".txt")

        path.write_text(content)
        return path


def generate_temperature_report_section(result: TemperatureSensitivityResult) -> SensitivityReportSection:
    """Generate report section for temperature sensitivity."""
    lines: list[str] = []
    lines.append(f"Metric: {result.metric_name}")
    lines.append(f"Nominal Temperature: {result.nominal_temp}°C")
    lines.append(f"Nominal Value: {result.nominal_value:.6g}")
    lines.append("")
    lines.append(f"Temperature Range: {result.points[0].temperature:.0f}°C to {result.points[-1].temperature:.0f}°C")
    lines.append(f"Value Range: {result.min_value:.6g} to {result.max_value:.6g}")
    lines.append(f"Variation: {result.range_pct:.2f}%")
    lines.append(f"Temperature Coefficient: {result.tempco:.4f} {result.tempco_units}")
    lines.append("")

    worst_temp, worst_dev = result.worst_case_deviation()
    lines.append(f"Worst Case: {worst_dev:.2f}% deviation at {worst_temp}°C")

    lines.append("")
    lines.append("Temperature Sweep Data:")
    lines.append(f"{'Temp (°C)':>10} {'Value':>15} {'Deviation (%)':>15}")
    lines.append("-" * 42)
    for p in result.points:
        dev_pct = p.normalized * 100
        lines.append(f"{p.temperature:>10.1f} {p.value:>15.6g} {dev_pct:>15.2f}")

    return SensitivityReportSection(
        title="Temperature Sensitivity Analysis",
        content="\n".join(lines),
        data=result.to_dict(),
    )


def generate_tolerance_report_section(result: ToleranceSensitivityResult) -> SensitivityReportSection:
    """Generate report section for tolerance sensitivity."""
    lines: list[str] = []
    lines.append(f"Metric: {result.metric_name}")
    lines.append(f"Nominal Output: {result.nominal_output:.6g}")
    lines.append("")
    lines.append(f"Total Variation (RSS): ±{result.total_variation_pct:.2f}%")
    lines.append(f"Worst Case Variation: ±{result.worst_case_variation_pct:.2f}%")
    lines.append("")

    critical = result.get_critical_components(threshold_pct=10.0)
    if critical:
        lines.append(f"Critical Components (>10% contribution): {', '.join(critical)}")
    else:
        lines.append("No components contribute more than 10% to output variation.")

    lines.append("")
    lines.append("Component Impact Ranking:")
    lines.append(f"{'Component':>12} {'Nominal':>12} {'Tol(%)':>8} {'Sensitivity':>12} {'Contrib(%)':>12}")
    lines.append("-" * 58)

    for impact in result.impacts:
        lines.append(
            f"{impact.component:>12} {impact.nominal:>12.4g} {impact.tolerance_pct:>8.1f} "
            f"{impact.output_sensitivity:>12.4f} {impact.output_contribution:>12.1f}"
        )

    return SensitivityReportSection(
        title="Tolerance Sensitivity Analysis",
        content="\n".join(lines),
        data=result.to_dict(),
    )


def create_sensitivity_report(
    title: str = "Circuit Sensitivity Analysis Report",
    temperature_result: TemperatureSensitivityResult | None = None,
    tolerance_result: ToleranceSensitivityResult | None = None,
    additional_sections: list[SensitivityReportSection] | None = None,
    metadata: dict[str, Any] | None = None,
) -> SensitivityReport:
    """Create a complete sensitivity analysis report.

    Args:
        title: Report title
        temperature_result: Temperature sensitivity analysis result
        tolerance_result: Tolerance sensitivity analysis result
        additional_sections: Extra sections to include
        metadata: Additional metadata

    Returns:
        SensitivityReport object

    Example:
        temp_result = temperature_sensitivity(...)
        tol_result = tolerance_sensitivity(...)

        report = create_sensitivity_report(
            title="Amplifier Sensitivity Analysis",
            temperature_result=temp_result,
            tolerance_result=tol_result,
        )

        report.save(Path("sensitivity_report.html"), format="html")
    """
    report = SensitivityReport(
        title=title,
        created_at=datetime.now(),
        metadata=metadata or {},
    )

    # Add summary section
    summary_lines: list[str] = []
    if temperature_result:
        summary_lines.append(f"• Temperature: {temperature_result.range_pct:.2f}% variation over range")
    if tolerance_result:
        summary_lines.append(f"• Tolerance: {tolerance_result.total_variation_pct:.2f}% RSS variation")
        critical = tolerance_result.get_critical_components()
        if critical:
            summary_lines.append(f"• Critical components: {', '.join(critical)}")

    if summary_lines:
        report.add_section("Summary", "\n".join(summary_lines))

    # Add detailed sections
    if temperature_result:
        section = generate_temperature_report_section(temperature_result)
        report.sections.append(section)

    if tolerance_result:
        section = generate_tolerance_report_section(tolerance_result)
        report.sections.append(section)

    # Add any additional sections
    if additional_sections:
        report.sections.extend(additional_sections)

    return report


# =============================================================================
# Design Margin Analysis
# =============================================================================


@dataclass
class DesignMargin:
    """Design margin for a specification.

    Attributes:
        spec_name: Specification name
        spec_value: Specification limit
        nominal_value: Nominal design value
        margin_pct: Margin as percentage of spec
        margin_sigma: Margin in standard deviations (if variation known)
        variation_pct: Expected variation percentage
        passes: Whether design meets spec with margin
    """

    spec_name: str
    spec_value: float
    spec_type: Literal["min", "max", "target"]
    nominal_value: float
    margin_pct: float
    margin_sigma: float | None
    variation_pct: float
    passes: bool


@dataclass
class DesignMarginResult:
    """Results from design margin analysis.

    Attributes:
        margins: List of margin analyses
        overall_pass: Whether all margins are met
        critical_specs: Specs with lowest margins
    """

    margins: list[DesignMargin]
    overall_pass: bool
    critical_specs: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_pass": self.overall_pass,
            "critical_specs": self.critical_specs,
            "margins": [
                {
                    "spec_name": m.spec_name,
                    "spec_value": m.spec_value,
                    "spec_type": m.spec_type,
                    "nominal_value": m.nominal_value,
                    "margin_pct": m.margin_pct,
                    "margin_sigma": m.margin_sigma,
                    "variation_pct": m.variation_pct,
                    "passes": m.passes,
                }
                for m in self.margins
            ],
        }


def analyze_design_margins(
    specs: list[tuple[str, float, Literal["min", "max", "target"]]],
    nominal_values: dict[str, float],
    variations: dict[str, float],
    required_margin_pct: float = 10.0,
) -> DesignMarginResult:
    """Analyze design margins against specifications.

    Args:
        specs: List of (name, value, type) tuples for specifications
        nominal_values: Dict of nominal values for each spec
        variations: Dict of expected variation (%) for each spec
        required_margin_pct: Minimum required margin percentage

    Returns:
        DesignMarginResult with margin analysis

    Example:
        specs = [
            ("Gain", 40.0, "min"),      # Gain >= 40 dB
            ("Bandwidth", 1e6, "min"),   # BW >= 1 MHz
            ("Noise", 10e-9, "max"),     # Noise <= 10 nV/rtHz
        ]

        result = analyze_design_margins(
            specs,
            nominal_values={"Gain": 45, "Bandwidth": 1.5e6, "Noise": 8e-9},
            variations={"Gain": 5, "Bandwidth": 10, "Noise": 15},
        )
    """
    margins: list[DesignMargin] = []
    all_pass = True
    critical: list[str] = []

    for spec_name, spec_value, spec_type in specs:
        nominal = nominal_values.get(spec_name, 0.0)
        variation = variations.get(spec_name, 0.0)

        # Calculate margin
        if spec_type == "min":
            # nominal should be > spec_value
            margin_abs = nominal - spec_value
            margin_pct = margin_abs / abs(spec_value) * 100 if spec_value != 0 else 0.0
        elif spec_type == "max":
            # nominal should be < spec_value
            margin_abs = spec_value - nominal
            margin_pct = margin_abs / abs(spec_value) * 100 if spec_value != 0 else 0.0
        else:  # target
            margin_abs = abs(spec_value - nominal)
            margin_pct = -margin_abs / abs(spec_value) * 100 if spec_value != 0 else 0.0

        # Calculate sigma if variation known
        margin_sigma = None
        if variation > 0:
            # How many standard deviations of variation fit in the margin
            variation_abs = abs(nominal) * variation / 100
            if variation_abs > 0:
                margin_sigma = margin_abs / variation_abs

        # Check pass/fail
        passes = margin_pct >= required_margin_pct

        if not passes:
            all_pass = False

        # Track critical specs (lowest margins)
        if margin_pct < required_margin_pct * 1.5:
            critical.append(spec_name)

        margins.append(
            DesignMargin(
                spec_name=spec_name,
                spec_value=spec_value,
                spec_type=spec_type,
                nominal_value=nominal,
                margin_pct=margin_pct,
                margin_sigma=margin_sigma,
                variation_pct=variation,
                passes=passes,
            )
        )

    # Sort margins by margin_pct (lowest first for easy identification of problems)
    margins.sort(key=lambda m: m.margin_pct)

    return DesignMarginResult(
        margins=margins,
        overall_pass=all_pass,
        critical_specs=critical,
    )


__all__ = [
    # Temperature sensitivity
    "TemperaturePoint",
    "TemperatureSensitivityResult",
    "temperature_sensitivity",
    # Tolerance sensitivity
    "ComponentTolerance",
    "ToleranceImpact",
    "ToleranceSensitivityResult",
    "tolerance_sensitivity",
    # Reports
    "SensitivityReport",
    "SensitivityReportSection",
    "create_sensitivity_report",
    "generate_temperature_report_section",
    "generate_tolerance_report_section",
    # Design margins
    "DesignMargin",
    "DesignMarginResult",
    "analyze_design_margins",
]
