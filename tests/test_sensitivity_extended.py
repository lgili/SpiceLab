"""Tests for extended sensitivity analysis (Sprint 8 - M16).

Tests for:
- Temperature sensitivity analysis
- Component tolerance sensitivity
- Sensitivity reports
- Design margin analysis
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from spicelab.analysis.sensitivity_extended import (
    ComponentTolerance,
    DesignMargin,
    DesignMarginResult,
    SensitivityReport,
    SensitivityReportSection,
    TemperaturePoint,
    TemperatureSensitivityResult,
    ToleranceImpact,
    ToleranceSensitivityResult,
    analyze_design_margins,
    create_sensitivity_report,
    generate_temperature_report_section,
    generate_tolerance_report_section,
    temperature_sensitivity,
    tolerance_sensitivity,
)


# =============================================================================
# Temperature Sensitivity Tests
# =============================================================================


class TestTemperaturePoint:
    """Tests for TemperaturePoint dataclass."""

    def test_creation(self) -> None:
        """Test creating a temperature point."""
        point = TemperaturePoint(temperature=25.0, value=1.0, normalized=0.0)
        assert point.temperature == 25.0
        assert point.value == 1.0
        assert point.normalized == 0.0


class TestTemperatureSensitivity:
    """Tests for temperature sensitivity analysis."""

    def test_linear_tempco(self) -> None:
        """Test temperature sensitivity with linear response."""
        # Simulate a circuit with 100 ppm/C tempco
        nominal = 1000.0
        tempco_ppm = 100  # ppm/C

        def objective(temp: float) -> float:
            return nominal * (1 + tempco_ppm * 1e-6 * (temp - 25))

        result = temperature_sensitivity(
            objective,
            metric_name="Resistance",
            temp_range=(-40, 85),
            nominal_temp=25.0,
            tempco_units="ppm/C",
        )

        assert result.metric_name == "Resistance"
        assert result.nominal_temp == 25.0
        assert pytest.approx(result.nominal_value, rel=0.01) == 1000.0
        # Tempco should be close to 100 ppm/C
        assert pytest.approx(result.tempco, rel=0.1) == 100

    def test_temperature_range(self) -> None:
        """Test that full temperature range is covered."""

        def objective(temp: float) -> float:
            return 10 - 0.01 * temp  # Linear decrease with temp

        result = temperature_sensitivity(
            objective,
            temp_range=(-40, 125),
            n_points=17,
        )

        # Check range
        assert result.points[0].temperature == pytest.approx(-40, abs=0.1)
        assert result.points[-1].temperature == pytest.approx(125, abs=0.1)
        assert len(result.points) >= 17

    def test_min_max_values(self) -> None:
        """Test min/max value detection."""

        def objective(temp: float) -> float:
            # Parabola with minimum at 50C
            return 100 + (temp - 50) ** 2 / 100

        result = temperature_sensitivity(
            objective,
            temp_range=(0, 100),
            nominal_temp=25.0,
        )

        assert result.min_value < result.nominal_value
        assert result.max_value >= result.nominal_value

    def test_worst_case_deviation(self) -> None:
        """Test worst case deviation calculation."""

        def objective(temp: float) -> float:
            # Sharp increase at high temp
            if temp > 80:
                return 1.0 + 0.05 * (temp - 80)
            return 1.0

        result = temperature_sensitivity(
            objective,
            temp_range=(0, 100),
            nominal_temp=25.0,
        )

        worst_temp, worst_dev = result.worst_case_deviation()
        assert worst_temp > 80  # Worst case should be at high temp
        assert worst_dev > 0

    def test_interpolate_at(self) -> None:
        """Test value interpolation."""

        def objective(temp: float) -> float:
            return temp * 2  # Simple linear

        result = temperature_sensitivity(
            objective,
            temp_range=(0, 100),
            n_points=11,
        )

        # Interpolate at a point between measurements
        interp = result.interpolate_at(55)
        assert pytest.approx(interp, rel=0.1) == 110

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""

        def objective(temp: float) -> float:
            return 100.0

        result = temperature_sensitivity(
            objective,
            temp_range=(0, 50),
            n_points=5,
        )

        d = result.to_dict()
        assert "metric_name" in d
        assert "tempco" in d
        assert "points" in d
        assert len(d["points"]) == 5


# =============================================================================
# Tolerance Sensitivity Tests
# =============================================================================


class TestComponentTolerance:
    """Tests for ComponentTolerance dataclass."""

    def test_creation(self) -> None:
        """Test creating a component tolerance spec."""
        tol = ComponentTolerance("R1", 10000, 5.0, "ohm")
        assert tol.name == "R1"
        assert tol.nominal == 10000
        assert tol.tolerance_pct == 5.0
        assert tol.value_unit == "ohm"


class TestToleranceSensitivity:
    """Tests for tolerance sensitivity analysis."""

    def test_single_component(self) -> None:
        """Test sensitivity with single component."""

        def objective(values: dict[str, float]) -> float:
            # Output = R1 (direct proportional)
            return values["R1"]

        components = [ComponentTolerance("R1", 1000, 5.0)]

        result = tolerance_sensitivity(objective, components, "output")

        assert result.metric_name == "output"
        assert result.nominal_output == 1000
        assert len(result.impacts) == 1
        assert result.impacts[0].component == "R1"
        # 5% tolerance should give ~5% variation
        assert pytest.approx(result.total_variation_pct, rel=0.2) == 5.0

    def test_multiple_components(self) -> None:
        """Test sensitivity with multiple components."""

        def objective(values: dict[str, float]) -> float:
            # Voltage divider: Vout = Vin * R2 / (R1 + R2)
            r1 = values["R1"]
            r2 = values["R2"]
            return 10 * r2 / (r1 + r2)

        components = [
            ComponentTolerance("R1", 10000, 5.0),
            ComponentTolerance("R2", 10000, 5.0),
        ]

        result = tolerance_sensitivity(objective, components)

        assert len(result.impacts) == 2
        # Both resistors affect output
        assert all(i.output_contribution > 0 for i in result.impacts)

    def test_get_ranking(self) -> None:
        """Test component ranking by impact."""

        def objective(values: dict[str, float]) -> float:
            # R1 has 10x more impact than R2
            return values["R1"] * 10 + values["R2"]

        components = [
            ComponentTolerance("R1", 100, 5.0),
            ComponentTolerance("R2", 100, 5.0),
        ]

        result = tolerance_sensitivity(objective, components)

        ranking = result.get_ranking()
        assert ranking[0] == "R1"  # R1 should be first

    def test_get_critical_components(self) -> None:
        """Test critical component identification."""

        def objective(values: dict[str, float]) -> float:
            # Only R1 matters significantly
            return values["R1"] * 100 + values["R2"] * 0.01

        components = [
            ComponentTolerance("R1", 100, 5.0),
            ComponentTolerance("R2", 100, 5.0),
        ]

        result = tolerance_sensitivity(objective, components)

        critical = result.get_critical_components(threshold_pct=50)
        assert "R1" in critical
        # R2 should not be critical
        assert "R2" not in critical

    def test_rss_calculation(self) -> None:
        """Test RSS (Root Sum Square) total variation."""

        def objective(values: dict[str, float]) -> float:
            # Independent contributions
            return values["R1"] + values["R2"]

        components = [
            ComponentTolerance("R1", 100, 5.0),  # 5% on half = 2.5% contribution
            ComponentTolerance("R2", 100, 5.0),  # 5% on half = 2.5% contribution
        ]

        result = tolerance_sensitivity(objective, components)

        # RSS of equal contributions should be sqrt(2) times individual
        expected_rss = 5.0  # Each contributes 2.5%, RSS = sqrt(2.5^2 + 2.5^2) = 3.54, normalized to sum = 5%
        assert result.total_variation_pct > 0
        assert result.total_variation_pct <= result.worst_case_variation_pct

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""

        def objective(values: dict[str, float]) -> float:
            return values["R1"]

        components = [ComponentTolerance("R1", 1000, 5.0)]

        result = tolerance_sensitivity(objective, components)

        d = result.to_dict()
        assert "impacts" in d
        assert len(d["impacts"]) == 1
        assert d["impacts"][0]["component"] == "R1"


# =============================================================================
# Sensitivity Report Tests
# =============================================================================


class TestSensitivityReportSection:
    """Tests for report sections."""

    def test_creation(self) -> None:
        """Test creating a report section."""
        section = SensitivityReportSection(
            title="Test Section",
            content="This is test content",
            data={"key": "value"},
        )
        assert section.title == "Test Section"
        assert section.content == "This is test content"
        assert section.data["key"] == "value"


class TestSensitivityReport:
    """Tests for sensitivity reports."""

    def test_add_section(self) -> None:
        """Test adding sections to report."""
        from datetime import datetime

        report = SensitivityReport(title="Test Report", created_at=datetime.now())
        report.add_section("Section 1", "Content 1")
        report.add_section("Section 2", "Content 2", {"data": 123})

        assert len(report.sections) == 2
        assert report.sections[0].title == "Section 1"
        assert report.sections[1].data["data"] == 123

    def test_to_text(self) -> None:
        """Test text report generation."""
        from datetime import datetime

        report = SensitivityReport(title="Test Report", created_at=datetime.now())
        report.add_section("Analysis", "Results here")

        text = report.to_text()
        assert "Test Report" in text
        assert "Analysis" in text
        assert "Results here" in text

    def test_to_html(self) -> None:
        """Test HTML report generation."""
        from datetime import datetime

        report = SensitivityReport(title="HTML Test", created_at=datetime.now())
        report.add_section("Section", "Content")

        html = report.to_html()
        assert "<html>" in html
        assert "HTML Test" in html
        assert "<h2>Section</h2>" in html

    def test_save_text(self) -> None:
        """Test saving text report."""
        from datetime import datetime

        report = SensitivityReport(title="Save Test", created_at=datetime.now())
        report.add_section("Data", "Values")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report"
            saved_path = report.save(path, format="text")

            assert saved_path.exists()
            assert saved_path.suffix == ".txt"
            content = saved_path.read_text()
            assert "Save Test" in content

    def test_save_html(self) -> None:
        """Test saving HTML report."""
        from datetime import datetime

        report = SensitivityReport(title="HTML Save", created_at=datetime.now())
        report.add_section("Results", "Data here")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report"
            saved_path = report.save(path, format="html")

            assert saved_path.exists()
            assert saved_path.suffix == ".html"
            content = saved_path.read_text()
            assert "<html>" in content


class TestReportGeneration:
    """Tests for report section generation."""

    def test_temperature_report_section(self) -> None:
        """Test generating temperature report section."""

        def objective(temp: float) -> float:
            return 100 + temp * 0.1

        result = temperature_sensitivity(objective, temp_range=(0, 100))

        section = generate_temperature_report_section(result)
        assert section.title == "Temperature Sensitivity Analysis"
        assert "Temperature Coefficient" in section.content
        assert len(section.data) > 0

    def test_tolerance_report_section(self) -> None:
        """Test generating tolerance report section."""

        def objective(values: dict[str, float]) -> float:
            return values["R1"] + values["R2"]

        components = [
            ComponentTolerance("R1", 1000, 5.0),
            ComponentTolerance("R2", 1000, 10.0),
        ]

        result = tolerance_sensitivity(objective, components)

        section = generate_tolerance_report_section(result)
        assert section.title == "Tolerance Sensitivity Analysis"
        assert "Component Impact Ranking" in section.content

    def test_create_combined_report(self) -> None:
        """Test creating combined sensitivity report."""

        def temp_obj(temp: float) -> float:
            return 100 + temp * 0.01

        def tol_obj(values: dict[str, float]) -> float:
            return values["R1"]

        temp_result = temperature_sensitivity(temp_obj, temp_range=(0, 100))
        tol_result = tolerance_sensitivity(tol_obj, [ComponentTolerance("R1", 1000, 5.0)])

        report = create_sensitivity_report(
            title="Combined Analysis",
            temperature_result=temp_result,
            tolerance_result=tol_result,
        )

        assert len(report.sections) == 3  # Summary + 2 analysis sections
        text = report.to_text()
        assert "Temperature" in text
        assert "Tolerance" in text


# =============================================================================
# Design Margin Tests
# =============================================================================


class TestDesignMargin:
    """Tests for DesignMargin dataclass."""

    def test_creation(self) -> None:
        """Test creating a design margin."""
        margin = DesignMargin(
            spec_name="Gain",
            spec_value=40.0,
            spec_type="min",
            nominal_value=45.0,
            margin_pct=12.5,
            margin_sigma=2.5,
            variation_pct=5.0,
            passes=True,
        )
        assert margin.spec_name == "Gain"
        assert margin.passes is True


class TestAnalyzeDesignMargins:
    """Tests for design margin analysis."""

    def test_all_passing(self) -> None:
        """Test when all specs pass with margin."""
        specs = [
            ("Gain", 40.0, "min"),
            ("Noise", 10e-9, "max"),
        ]
        nominal_values = {"Gain": 50.0, "Noise": 5e-9}
        variations = {"Gain": 5.0, "Noise": 10.0}

        result = analyze_design_margins(specs, nominal_values, variations)

        assert result.overall_pass is True
        assert len(result.margins) == 2

    def test_some_failing(self) -> None:
        """Test when some specs fail."""
        specs = [
            ("Gain", 40.0, "min"),
            ("Noise", 5e-9, "max"),  # Tight spec
        ]
        nominal_values = {"Gain": 50.0, "Noise": 5.5e-9}  # Noise exceeds spec
        variations = {"Gain": 5.0, "Noise": 10.0}

        result = analyze_design_margins(specs, nominal_values, variations, required_margin_pct=10.0)

        assert result.overall_pass is False
        # Noise should be in critical list
        assert "Noise" in result.critical_specs

    def test_margin_calculation_min(self) -> None:
        """Test margin calculation for min spec."""
        specs = [("Gain", 40.0, "min")]
        nominal_values = {"Gain": 44.0}  # 10% above spec
        variations = {"Gain": 0.0}

        result = analyze_design_margins(specs, nominal_values, variations, required_margin_pct=5.0)

        margin = result.margins[0]
        assert margin.margin_pct == pytest.approx(10.0, rel=0.01)
        assert margin.passes is True

    def test_margin_calculation_max(self) -> None:
        """Test margin calculation for max spec."""
        specs = [("Noise", 10.0, "max")]
        nominal_values = {"Noise": 8.0}  # 20% below spec
        variations = {"Noise": 0.0}

        result = analyze_design_margins(specs, nominal_values, variations, required_margin_pct=10.0)

        margin = result.margins[0]
        assert margin.margin_pct == pytest.approx(20.0, rel=0.01)
        assert margin.passes is True

    def test_margin_sigma_calculation(self) -> None:
        """Test margin sigma calculation."""
        specs = [("Gain", 40.0, "min")]
        nominal_values = {"Gain": 50.0}  # 10 units margin
        variations = {"Gain": 10.0}  # 10% = 5 units

        result = analyze_design_margins(specs, nominal_values, variations)

        margin = result.margins[0]
        # Margin is 10 units, variation is 5 units, so ~2 sigma
        assert margin.margin_sigma is not None
        assert margin.margin_sigma == pytest.approx(2.0, rel=0.1)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        specs = [("Gain", 40.0, "min")]
        nominal_values = {"Gain": 50.0}
        variations = {"Gain": 5.0}

        result = analyze_design_margins(specs, nominal_values, variations)

        d = result.to_dict()
        assert "overall_pass" in d
        assert "margins" in d
        assert len(d["margins"]) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for extended sensitivity analysis."""

    def test_complete_analysis_workflow(self) -> None:
        """Test complete analysis workflow."""
        # 1. Define circuit behavior
        def temp_response(temp: float) -> float:
            # Gain with -0.1%/C tempco
            return 100 * (1 - 0.001 * (temp - 25))

        def component_response(values: dict[str, float]) -> float:
            # Simple gain: Rf/Rin
            return values["Rf"] / values["Rin"]

        # 2. Run temperature analysis
        temp_result = temperature_sensitivity(
            temp_response,
            metric_name="Gain",
            temp_range=(-40, 85),
            tempco_units="%/C",
        )

        # 3. Run tolerance analysis
        tol_result = tolerance_sensitivity(
            component_response,
            [
                ComponentTolerance("Rf", 100000, 1.0),
                ComponentTolerance("Rin", 10000, 5.0),
            ],
            metric_name="Gain",
        )

        # 4. Analyze design margins (with wider spec range to pass)
        margin_result = analyze_design_margins(
            specs=[("Gain", 8.0, "min"), ("Gain", 12.0, "max")],
            nominal_values={"Gain": 10.0},
            variations={"Gain": tol_result.total_variation_pct},
            required_margin_pct=5.0,  # Lower threshold
        )

        # 5. Generate report
        report = create_sensitivity_report(
            title="Complete Analysis",
            temperature_result=temp_result,
            tolerance_result=tol_result,
        )

        # Verify results
        assert temp_result.tempco < 0  # Negative tempco
        assert tol_result.get_ranking()[0] == "Rin"  # Rin has more impact (5% vs 1%)
        assert margin_result.overall_pass is True
        assert len(report.sections) == 3

    def test_save_and_load_report(self) -> None:
        """Test saving report in different formats."""

        def objective(temp: float) -> float:
            return 100.0

        result = temperature_sensitivity(objective, temp_range=(0, 100))
        report = create_sensitivity_report(temperature_result=result)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as text
            text_path = report.save(Path(tmpdir) / "report", format="text")
            assert text_path.exists()

            # Save as HTML
            html_path = report.save(Path(tmpdir) / "report_html", format="html")
            assert html_path.exists()

            # Verify content
            text_content = text_path.read_text()
            html_content = html_path.read_text()

            assert "Temperature Sensitivity" in text_content
            assert "<html>" in html_content
