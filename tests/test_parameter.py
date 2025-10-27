"""Tests for Phase 2: Parameter System."""

from __future__ import annotations

import numpy as np
import pytest
from spicelab.core.expressions import (
    ExpressionError,
    safe_eval_expression,
    validate_expression_dependencies,
)
from spicelab.core.parameter import (
    LogNormalTolerance,
    NormalTolerance,
    Parameter,
    ParameterRef,
    TriangularTolerance,
    UniformTolerance,
)
from spicelab.core.units import Unit, format_si_value, parse_si_value


# ======================================================================================
# Unit Tests
# ======================================================================================
def test_unit_enum():
    """Unit enum has expected attributes."""
    assert Unit.OHM.suffix == "Ohm"
    assert Unit.FARAD.suffix == "F"
    assert Unit.VOLT.suffix == "V"
    assert Unit.SECOND.suffix == "s"


def test_format_si_value_basic():
    """format_si_value produces SPICE-compatible strings."""
    assert format_si_value(10_000, Unit.OHM) == "10kOhm"
    assert format_si_value(1_500_000, Unit.OHM) == "1.5MegOhm"
    assert format_si_value(100e-9, Unit.FARAD) == "100nF"
    assert format_si_value(1e-6, Unit.FARAD) == "1uF"


def test_format_si_value_edge_cases():
    """format_si_value handles edge cases."""
    assert format_si_value(0, Unit.OHM) == "0Ohm"
    assert format_si_value(-10_000, Unit.VOLT) == "-10kV"
    assert format_si_value(1, Unit.AMPERE) == "1A"


def test_parse_si_value():
    """parse_si_value converts SPICE strings to floats."""
    assert parse_si_value("10k") == pytest.approx(10_000)
    assert parse_si_value("1.5Meg") == pytest.approx(1_500_000)
    assert parse_si_value("100n") == pytest.approx(100e-9)
    assert parse_si_value("1u") == pytest.approx(1e-6)


def test_parse_si_value_with_unit():
    """parse_si_value strips unit suffix."""
    assert parse_si_value("10kOhm", Unit.OHM) == pytest.approx(10_000)
    assert parse_si_value("100nF", Unit.FARAD) == pytest.approx(100e-9)
    assert parse_si_value("1.5MegOhm", Unit.OHM) == pytest.approx(1_500_000)


def test_parse_si_value_invalid():
    """parse_si_value raises on invalid input."""
    with pytest.raises(ValueError):
        parse_si_value("abc")

    with pytest.raises(ValueError):
        parse_si_value("")


# ======================================================================================
# Tolerance Tests
# ======================================================================================
def test_normal_tolerance_sample():
    """NormalTolerance samples from normal distribution."""
    tol = NormalTolerance(sigma_pct=5.0)
    rng = np.random.default_rng(42)

    samples = [tol.sample(10_000, rng) for _ in range(1000)]

    # Check mean and std
    mean = np.mean(samples)
    std = np.std(samples)

    assert 9_800 < mean < 10_200  # Within ±2%
    assert 400 < std < 600  # 5% of 10k = 500


def test_normal_tolerance_bounds():
    """NormalTolerance bounds are ±3σ."""
    tol = NormalTolerance(sigma_pct=5.0)
    lower, upper = tol.bounds(10_000)

    # ±3σ = ±15% for 5% tolerance
    assert lower == pytest.approx(8_500, rel=0.01)
    assert upper == pytest.approx(11_500, rel=0.01)


def test_uniform_tolerance():
    """UniformTolerance samples uniformly."""
    tol = UniformTolerance(plus_minus_pct=10.0)
    rng = np.random.default_rng(42)

    samples = [tol.sample(10_000, rng) for _ in range(1000)]

    # All samples within bounds
    assert all(9_000 <= s <= 11_000 for s in samples)

    # Bounds are exact
    lower, upper = tol.bounds(10_000)
    assert lower == 9_000
    assert upper == 11_000


def test_lognormal_tolerance():
    """LogNormalTolerance produces positive values."""
    tol = LogNormalTolerance(sigma_pct=20.0)
    rng = np.random.default_rng(42)

    samples = [tol.sample(10_000, rng) for _ in range(100)]

    # All positive
    assert all(s > 0 for s in samples)

    # Asymmetric (more spread on high side)
    lower, upper = tol.bounds(10_000)
    assert (10_000 - lower) < (upper - 10_000)


def test_triangular_tolerance():
    """TriangularTolerance peaks at nominal."""
    tol = TriangularTolerance(plus_minus_pct=10.0)
    rng = np.random.default_rng(42)

    samples = [tol.sample(10_000, rng) for _ in range(1000)]

    # All within bounds
    assert all(9_000 <= s <= 11_000 for s in samples)

    # Most samples near center (rough check)
    center_samples = sum(1 for s in samples if 9_500 <= s <= 10_500)
    assert center_samples > 400  # Expect >40% in center 10%


# ======================================================================================
# Parameter Tests
# ======================================================================================
def test_parameter_basic():
    """Parameter can be created with nominal and unit."""
    param = Parameter(name="Rload", nominal=10_000, unit=Unit.OHM)

    assert param.name == "Rload"
    assert param.nominal == 10_000
    assert param.unit == Unit.OHM
    assert param.tolerance is None


def test_parameter_with_tolerance():
    """Parameter with tolerance can sample values."""
    param = Parameter(name="Rload", nominal=10_000, unit=Unit.OHM, tolerance=NormalTolerance(5.0))

    # Sample values
    values = [param.sample_value(seed=i) for i in range(100)]

    # Check variation
    assert any(v != 10_000 for v in values)
    mean = np.mean(values)
    assert 9_700 < mean < 10_300


def test_parameter_with_expression():
    """Parameter with expression evaluates in context."""
    param = Parameter(
        name="R_total",
        nominal=0,  # Nominal ignored when expression present
        unit=Unit.OHM,
        expression="R1 + R2",
    )

    context = {"R1": 1000, "R2": 2000}
    result = param.evaluate(context)

    assert result == 3000


def test_parameter_expression_no_sample():
    """Parameter with expression cannot be sampled."""
    param = Parameter(name="R_total", nominal=0, unit=Unit.OHM, expression="R1 + R2")

    with pytest.raises(ValueError, match="has expression"):
        param.sample_value(seed=42)


def test_parameter_expression_and_tolerance_invalid():
    """Parameter cannot have both expression and tolerance."""
    with pytest.raises(ValueError, match="cannot have both"):
        Parameter(
            name="bad",
            nominal=10_000,
            unit=Unit.OHM,
            expression="R1 + R2",
            tolerance=NormalTolerance(5.0),
        )


def test_parameter_to_spice():
    """Parameter generates .param statement."""
    param = Parameter(name="Rload", nominal=10_000, unit=Unit.OHM)

    spice = param.to_spice()
    assert ".param Rload=10kOhm" in spice


def test_parameter_to_spice_expression():
    """Parameter with expression generates .param with expression."""
    param = Parameter(name="R_total", nominal=0, unit=Unit.OHM, expression="R1 + R2")

    spice = param.to_spice()
    assert ".param R_total=R1 + R2" in spice


def test_parameter_ref():
    """ParameterRef creates reference string."""
    ref = ParameterRef("Rload")

    assert ref.name == "Rload"
    assert str(ref) == "{Rload}"


# ======================================================================================
# Expression Tests
# ======================================================================================
def test_safe_eval_simple():
    """safe_eval_expression handles simple arithmetic."""
    context = {"R1": 1000, "R2": 2000}

    assert safe_eval_expression("R1 + R2", context) == 3000
    assert safe_eval_expression("R1 * 2", context) == 2000
    assert safe_eval_expression("R2 / 2", context) == 1000


def test_safe_eval_math_functions():
    """safe_eval_expression supports math functions."""
    context = {"R1": 1000, "R2": 2000}

    result = safe_eval_expression("sqrt(R1**2 + R2**2)", context)
    assert result == pytest.approx(2236.067977, rel=1e-6)

    result = safe_eval_expression("log10(R1)", context)
    assert result == pytest.approx(3.0, rel=1e-6)


def test_safe_eval_constants():
    """safe_eval_expression supports constants."""
    context = {}

    result = safe_eval_expression("pi", context)
    assert result == pytest.approx(3.14159265, rel=1e-6)

    result = safe_eval_expression("e", context)
    assert result == pytest.approx(2.71828182, rel=1e-6)


def test_safe_eval_complex():
    """safe_eval_expression handles complex expressions."""
    context = {"R": 1000, "C": 1e-6}

    # RC time constant
    result = safe_eval_expression("R * C", context)
    assert result == pytest.approx(0.001, rel=1e-6)

    # Cutoff frequency
    result = safe_eval_expression("1 / (2 * pi * R * C)", context)
    assert result == pytest.approx(159.154943, rel=1e-4)


def test_safe_eval_undefined_parameter():
    """safe_eval_expression raises on undefined parameter."""
    context = {"R1": 1000}

    with pytest.raises(ExpressionError, match="Undefined parameter"):
        safe_eval_expression("R1 + R2", context)


def test_safe_eval_invalid_expression():
    """safe_eval_expression raises on invalid syntax."""
    context = {"R1": 1000}

    with pytest.raises(ExpressionError):
        safe_eval_expression("R1 +", context)


def test_safe_eval_unsafe_function():
    """safe_eval_expression blocks unsafe functions."""
    context = {"R1": 1000}

    with pytest.raises(ExpressionError, match="simple function calls"):
        safe_eval_expression("__import__('os').system('ls')", context)


def test_validate_dependencies_simple():
    """validate_expression_dependencies sorts simple DAG."""
    expressions = {
        "R_total": "R1 + R2",
        "R1": "1000",
        "R2": "2000",
    }

    order = validate_expression_dependencies(expressions)

    # R1 and R2 must come before R_total
    assert order.index("R1") < order.index("R_total")
    assert order.index("R2") < order.index("R_total")


def test_validate_dependencies_complex():
    """validate_expression_dependencies handles multi-level dependencies."""
    expressions = {
        "tau": "R_total * C1",
        "R_total": "R1 + R2",
        "R1": "1000",
        "R2": "2000",
        "C1": "1e-6",
    }

    order = validate_expression_dependencies(expressions)

    # R1, R2, C1 must come before R_total
    assert order.index("R1") < order.index("R_total")
    assert order.index("R2") < order.index("R_total")

    # R_total must come before tau
    assert order.index("R_total") < order.index("tau")


def test_validate_dependencies_circular():
    """validate_expression_dependencies detects cycles."""
    expressions = {
        "A": "B + 1",
        "B": "C + 1",
        "C": "A + 1",  # Circular!
    }

    with pytest.raises(ExpressionError, match="Circular dependency"):
        validate_expression_dependencies(expressions)


# ======================================================================================
# Integration Tests
# ======================================================================================
def test_parameter_workflow():
    """Complete workflow: define, sample, generate netlist."""
    # Define parameters
    r_load = Parameter(
        name="Rload",
        nominal=10_000,
        unit=Unit.OHM,
        tolerance=NormalTolerance(5.0),
        description="Load resistance",
    )

    c_filter = Parameter(
        name="Cfilt",
        nominal=1e-6,
        unit=Unit.FARAD,
        tolerance=UniformTolerance(10.0),
        description="Filter capacitor",
    )

    tau = Parameter(
        name="tau",
        nominal=0,
        unit=Unit.SECOND,
        expression="Rload * Cfilt",
        description="Time constant",
    )

    # Sample values
    rng = np.random.default_rng(42)
    r_sample = r_load.sample_value(rng=rng)
    c_sample = c_filter.sample_value(rng=rng)

    assert r_sample != 10_000  # Varied
    assert c_sample != 1e-6  # Varied

    # Evaluate expression
    context = {"Rload": r_sample, "Cfilt": c_sample}
    tau_value = tau.evaluate(context)

    assert tau_value == pytest.approx(r_sample * c_sample)

    # Generate SPICE
    assert "Rload=" in r_load.to_spice()
    assert "Cfilt=" in c_filter.to_spice()
    assert "tau=Rload * Cfilt" in tau.to_spice()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
