"""Tests for SPICE syntax compatibility module.

Tests conversion of LTspice-specific syntax to ngspice equivalents.
"""

import pytest
from spicelab.utils.spice_compat import (
    convert_directives_for_engine,
    convert_expression_for_engine,
    convert_if_to_ternary,
    convert_param_directive,
)


class TestIfToTernaryConversion:
    """Test IF() to ternary operator conversion."""

    def test_simple_if(self):
        """Simple IF(cond, true, false) conversion."""
        result = convert_if_to_ternary("IF(T1<0,1,0)")
        assert result == "((T1<0) ? (1) : (0))"

    def test_if_with_spaces(self):
        """IF with spaces in arguments."""
        result = convert_if_to_ternary("IF( T1 < 0, 1, 0 )")
        assert "?" in result
        assert ":" in result

    def test_if_in_expression(self):
        """IF embedded in a larger expression."""
        result = convert_if_to_ternary("R0*(1+A*T1+IF(T1<0,C*T1,0))")
        assert "R0*(1+A*T1+(" in result
        assert "?" in result

    def test_nested_if(self):
        """Nested IF functions."""
        result = convert_if_to_ternary("IF(x>0, IF(x>10, 2, 1), 0)")
        # Should have two ternary operators
        assert result.count("?") == 2
        assert result.count(":") == 2

    def test_no_if(self):
        """Expression without IF - should remain unchanged."""
        expr = "R0 * (1 + A*T1 + B*T1**2)"
        result = convert_if_to_ternary(expr)
        assert result == expr

    def test_case_insensitive(self):
        """IF should be matched case-insensitively."""
        result1 = convert_if_to_ternary("IF(x,1,0)")
        result2 = convert_if_to_ternary("if(x,1,0)")
        result3 = convert_if_to_ternary("If(x,1,0)")
        # All should produce the same result
        assert "?" in result1
        assert "?" in result2
        assert "?" in result3

    def test_pt1000_unit_parameter(self):
        """Real PT1000 UNIT parameter from LTspice."""
        result = convert_if_to_ternary("(IF(T1<0,1,0))")
        assert "(((T1<0) ? (1) : (0)))" == result

    def test_complex_condition(self):
        """IF with complex condition."""
        result = convert_if_to_ternary("IF(x>0 && y<10, a+b, c*d)")
        assert "?" in result
        assert "x>0 && y<10" in result or "x>0&&y<10" in result


class TestParamDirectiveConversion:
    """Test .param directive conversion for different engines."""

    def test_param_with_if_ngspice(self):
        """Convert .param with IF() for ngspice."""
        directive = ".param UNIT=(IF(T1<0,1,0))"
        result = convert_param_directive(directive, "ngspice")
        assert ".param" in result
        assert "?" in result
        assert "IF(" not in result

    def test_param_without_if(self):
        """Param without IF should remain mostly unchanged."""
        directive = ".param R0=1000"
        result = convert_param_directive(directive, "ngspice")
        assert ".param R0=1000" == result

    def test_param_with_spaces_ngspice(self):
        """Spaces should be removed for ngspice."""
        directive = ".param Rrtd=R0 * (1 + A*T1)"
        result = convert_param_directive(directive, "ngspice")
        # ngspice doesn't allow spaces in param expressions
        assert " * " not in result
        assert " + " not in result
        assert "R0*(1+A*T1)" in result

    def test_param_ltspice_unchanged(self):
        """LTspice target should keep IF() and spaces."""
        directive = ".param UNIT=(IF(T1<0,1,0))"
        result = convert_param_directive(directive, "ltspice")
        assert "IF(" in result

    def test_non_param_unchanged(self):
        """Non-.param directives should be unchanged."""
        directive = ".tran 1u 10m"
        result = convert_param_directive(directive, "ngspice")
        assert result == directive


class TestExpressionConversion:
    """Test general expression conversion."""

    def test_ngspice_converts_if(self):
        """ngspice target should convert IF()."""
        expr = "IF(x,1,0)"
        result = convert_expression_for_engine(expr, "ngspice")
        assert "?" in result

    def test_ltspice_keeps_if(self):
        """ltspice target should keep IF()."""
        expr = "IF(x,1,0)"
        result = convert_expression_for_engine(expr, "ltspice")
        assert "IF(" in result


class TestDirectivesConversion:
    """Test batch directive conversion."""

    def test_multiple_directives(self):
        """Convert multiple directives at once."""
        directives = [
            ".param A=1",
            ".param B=(IF(x<0,1,0))",
            ".tran 1u 10m",
            ".param C=A * B",
        ]
        result = convert_directives_for_engine(directives, "ngspice")

        assert len(result) == 4
        assert ".param A=1" == result[0]
        assert "?" in result[1]  # IF converted
        assert ".tran 1u 10m" == result[2]  # unchanged
        assert "*" in result[3]  # spaces removed


class TestPT1000Integration:
    """Integration tests with real PT1000 parameters."""

    def test_full_rrtd_expression(self):
        """Test full Rrtd expression from PT1000 circuit."""
        # This is the actual expression from the ASC file
        expr = "R0 * (1 + A*T1 + B*T1**2 + C*(T1-100)*T1**3*IF(T1<0,1,0))"
        result = convert_if_to_ternary(expr)

        # IF should be converted to ternary
        assert "IF(" not in result
        assert "?" in result
        assert ":" in result

        # Main expression structure should be preserved
        assert "R0" in result
        assert "A*T1" in result
        assert "B*T1**2" in result

    def test_pt1000_param_set(self):
        """Test the complete set of PT1000 parameters."""
        params = [
            ".param A=3.9083m",
            ".param B=-577.5n",
            ".param C=-4.183p",
            ".param R0=1000",
            ".param UNIT=(IF(T1<0,1,0))",
            ".param Rrtd=R0 * (1 + A*T1 + B*T1**2 + C*(T1-100)*T1**3*UNIT)",
            ".param T1=0",
        ]

        result = convert_directives_for_engine(params, "ngspice")

        # Check UNIT conversion
        assert "?" in result[4]
        assert "IF(" not in result[4]

        # Check Rrtd has no spaces
        assert " * " not in result[5]
        assert " + " not in result[5]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
