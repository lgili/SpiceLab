"""Tests for Model Library (Sprint 4 - M9).

Tests for:
- Op-amp models
- Transistor models (BJT)
- MOSFET models
- Diode models
- Model parameter validation
"""

import pytest

from spicelab.library import (
    ValidationResult,
    get_parameter_info,
    list_components,
    list_parameters,
    parse_model_card,
    parse_spice_value,
    search_components,
    validate_component_params,
    validate_model_card,
)


# ============================================================================
# SPICE Value Parsing Tests
# ============================================================================


class TestParseSpiceValue:
    """Tests for SPICE value parsing."""

    def test_basic_numeric(self):
        """Test basic numeric values."""
        assert parse_spice_value("100") == 100.0
        assert parse_spice_value("3.14159") == 3.14159
        assert parse_spice_value("0") == 0.0

    def test_scientific_notation(self):
        """Test scientific notation."""
        assert parse_spice_value("1e-9") == 1e-9
        assert parse_spice_value("2.5E-12") == 2.5e-12
        assert parse_spice_value("1E6") == 1e6

    def test_spice_suffixes(self):
        """Test SPICE engineering suffixes."""
        # Large values
        assert parse_spice_value("10k") == pytest.approx(10e3)
        assert parse_spice_value("10K") == pytest.approx(10e3)
        assert parse_spice_value("1MEG") == pytest.approx(1e6)
        assert parse_spice_value("1meg") == pytest.approx(1e6)
        assert parse_spice_value("1G") == pytest.approx(1e9)
        assert parse_spice_value("1T") == pytest.approx(1e12)

        # Small values
        assert parse_spice_value("100m") == pytest.approx(100e-3)
        assert parse_spice_value("100u") == pytest.approx(100e-6)
        assert parse_spice_value("100n") == pytest.approx(100e-9)
        assert parse_spice_value("100p") == pytest.approx(100e-12)
        assert parse_spice_value("100f") == pytest.approx(100e-15)

    def test_combined_values(self):
        """Test values with decimal and suffix."""
        assert parse_spice_value("4.7k") == pytest.approx(4700.0)
        assert parse_spice_value("2.2u") == pytest.approx(2.2e-6)
        assert parse_spice_value("0.1n") == pytest.approx(0.1e-9)


# ============================================================================
# Model Card Parsing Tests
# ============================================================================


class TestParseModelCard:
    """Tests for model card parsing."""

    def test_diode_model(self):
        """Test parsing diode model card."""
        card = ".model D1N4148 D(Is=2.52e-9 Rs=0.568 N=1.906)"
        name, mtype, params = parse_model_card(card)

        assert name == "D1N4148"
        assert mtype == "D"
        assert params["Is"] == pytest.approx(2.52e-9)
        assert params["Rs"] == pytest.approx(0.568)
        assert params["N"] == pytest.approx(1.906)

    def test_bjt_model(self):
        """Test parsing BJT model card."""
        card = ".model Q2N2222 NPN(Is=14.34E-15 Bf=256 Vaf=74)"
        name, mtype, params = parse_model_card(card)

        assert name == "Q2N2222"
        assert mtype == "NPN"
        assert params["Is"] == pytest.approx(14.34e-15)
        assert params["Bf"] == pytest.approx(256)
        assert params["Vaf"] == pytest.approx(74)

    def test_mosfet_model(self):
        """Test parsing MOSFET model card."""
        card = ".model M2N7000 NMOS(Vto=2.1 KP=0.1 Lambda=0.04)"
        name, mtype, params = parse_model_card(card)

        assert name == "M2N7000"
        assert mtype == "NMOS"
        assert params["Vto"] == pytest.approx(2.1)
        assert params["KP"] == pytest.approx(0.1)  # Case preserved from input
        assert params["Lambda"] == pytest.approx(0.04)

    def test_with_spice_suffixes(self):
        """Test parsing with SPICE suffixes."""
        card = ".model D1N4148 D(Is=2.5n Cjo=4p)"
        name, mtype, params = parse_model_card(card)

        assert params["Is"] == pytest.approx(2.5e-9)
        assert params["Cjo"] == pytest.approx(4e-12)

    def test_case_insensitive(self):
        """Test case insensitivity."""
        card = ".MODEL d1n4148 d(is=2.5e-9 rs=0.5)"
        name, mtype, params = parse_model_card(card)

        assert name == "d1n4148"
        assert mtype == "D"
        assert "Is" in params or "is" in params

    def test_invalid_card(self):
        """Test invalid model card."""
        name, mtype, params = parse_model_card("INVALID")

        assert name is None
        assert mtype is None
        assert params == {}


# ============================================================================
# Model Validation Tests
# ============================================================================


class TestValidateModelCard:
    """Tests for model card validation."""

    def test_valid_diode(self):
        """Test valid diode model."""
        card = ".model D1N4148 D(Is=2.52e-9 Rs=0.568 N=1.906)"
        result = validate_model_card(card)

        assert result.valid
        assert result.model_type == "D"
        assert result.model_name == "D1N4148"
        assert not result.has_errors

    def test_valid_bjt(self):
        """Test valid BJT model."""
        card = ".model Q2N3904 NPN(Is=6.7e-15 Bf=416 Vaf=74)"
        result = validate_model_card(card)

        assert result.valid
        assert result.model_type == "NPN"

    def test_parameter_out_of_range(self):
        """Test parameter out of typical range."""
        # N (emission coefficient) typically 0.5-4.0
        card = ".model DBAD D(Is=1e-9 N=10)"  # N too high
        result = validate_model_card(card)

        assert result.has_warnings
        assert any("N" in issue.message for issue in result.issues)

    def test_unknown_parameter(self):
        """Test unknown parameter detection."""
        card = ".model DTEST D(Is=1e-9 XYZ=123)"
        result = validate_model_card(card)

        # Unknown params generate info messages
        assert any("XYZ" in issue.message for issue in result.issues)

    def test_invalid_syntax(self):
        """Test invalid syntax detection."""
        result = validate_model_card("NOT A MODEL CARD")

        assert not result.valid
        assert result.has_errors


class TestValidateComponentParams:
    """Tests for component parameter validation."""

    def test_valid_diode_params(self):
        """Test valid diode parameters."""
        params = {"Is": 1e-14, "Rs": 0.5, "N": 1.0}
        result = validate_component_params("D", params)

        assert result.valid
        assert not result.has_warnings

    def test_param_below_min(self):
        """Test parameter below minimum."""
        params = {"Bf": 1}  # Bf typically 10-1000
        result = validate_component_params("NPN", params)

        assert result.has_warnings
        assert any("Bf" in issue.message for issue in result.issues)

    def test_unknown_model_type(self):
        """Test unknown model type."""
        result = validate_component_params("UNKNOWN", {"x": 1})

        # Should succeed but with warning
        assert result.valid
        assert result.has_warnings


# ============================================================================
# Parameter Info Tests
# ============================================================================


class TestParameterInfo:
    """Tests for parameter information retrieval."""

    def test_get_diode_param(self):
        """Test getting diode parameter info."""
        info = get_parameter_info("D", "Is")

        assert info is not None
        assert info.name == "Is"
        assert "saturation" in info.description.lower()
        assert info.unit == "A"

    def test_get_bjt_param(self):
        """Test getting BJT parameter info."""
        info = get_parameter_info("NPN", "Bf")

        assert info is not None
        assert "beta" in info.description.lower()

    def test_unknown_param(self):
        """Test unknown parameter."""
        info = get_parameter_info("D", "UNKNOWN")
        assert info is None

    def test_list_diode_params(self):
        """Test listing diode parameters."""
        params = list_parameters("D")

        assert len(params) > 0
        names = [p.name for p in params]
        assert "Is" in names
        assert "N" in names
        assert "Rs" in names


# ============================================================================
# Component Library Tests
# ============================================================================


class TestOpAmpLibrary:
    """Tests for op-amp component library."""

    def test_list_opamps(self):
        """Test listing op-amp components."""
        components = list_components(category="opamp")
        names = [c.name for c in components]

        assert "opamp.ideal" in names
        assert "opamp.lm741" in names
        assert "opamp.tl072" in names
        assert "opamp.lm358" in names

    def test_search_audio_opamps(self):
        """Test searching for audio op-amps."""
        results = search_components(name_contains="ne5532")
        names = [r.name for r in results]

        assert "opamp.ne5532" in names

    def test_opamp_count(self):
        """Test we have expected number of op-amps."""
        components = list_components(category="opamp")
        # Should have at least 10 op-amps now
        assert len(list(components)) >= 10


class TestTransistorLibrary:
    """Tests for transistor component library."""

    def test_list_bjts(self):
        """Test listing BJT components."""
        components = list_components(category="bjt")
        names = [c.name for c in components]

        assert "bjt.2n2222" in names
        assert "bjt.2n3904" in names
        assert "bjt.2n3906" in names

    def test_list_mosfets(self):
        """Test listing MOSFET components."""
        components = list_components(category="mosfet")
        names = [c.name for c in components]

        assert "mosfet.2n7000" in names
        assert "mosfet.irf540n" in names

    def test_bjt_count(self):
        """Test we have expected number of BJTs."""
        components = list_components(category="bjt")
        # Should have multiple NPN and PNP transistors
        assert len(list(components)) >= 10


class TestDiodeLibrary:
    """Tests for diode component library."""

    def test_list_diodes(self):
        """Test listing diode components."""
        components = list_components(category="diode")
        names = [c.name for c in components]

        assert "diode.1n4148" in names
        assert "diode.1n4001" in names

    def test_search_zener(self):
        """Test searching for Zener diodes."""
        # Search by name containing "z" for zener
        results = search_components(name_contains="z")

        assert len(results) > 0

    def test_search_schottky(self):
        """Test searching for Schottky diodes."""
        # Schottky diodes like 1N5817
        results = search_components(name_contains="1n581")

        assert len(results) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for model library."""

    def test_all_bjt_models_valid(self):
        """Test all BJT models have valid model cards."""
        from spicelab.library.transistors import _BJTS

        for bjt in _BJTS:
            result = validate_model_card(bjt.model_card)
            assert result.valid, f"Invalid model: {bjt.slug}: {result}"

    def test_all_mosfet_models_valid(self):
        """Test all MOSFET models have valid model cards."""
        from spicelab.library.transistors import _MOSFETS

        for mosfet in _MOSFETS:
            result = validate_model_card(mosfet.model_card)
            assert result.valid, f"Invalid model: {mosfet.slug}: {result}"

    def test_all_diode_models_valid(self):
        """Test all diode models have valid model cards."""
        from spicelab.library.diodes import _COMMON_DIODES

        for diode in _COMMON_DIODES:
            if diode.model_card:
                result = validate_model_card(diode.model_card)
                assert result.valid, f"Invalid model: {diode.part_number}: {result}"

    def test_validation_result_str(self):
        """Test ValidationResult string representation."""
        card = ".model D1N4148 D(Is=2.52e-9 N=1.906)"
        result = validate_model_card(card)

        result_str = str(result)
        assert "VALID" in result_str
        assert "D1N4148" in result_str
