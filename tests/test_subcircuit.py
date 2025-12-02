"""Tests for hierarchical subcircuit support (Sprint 7 - M8).

Tests for:
- SubcircuitDefinition creation and SPICE generation
- SubcircuitParameter validation
- SubcircuitLibrary management
- File loading and parsing
- Dependency tracking
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spicelab.core.subcircuit import (
    SubcircuitDefinition,
    SubcircuitLibrary,
    SubcircuitParameter,
    SubcircuitPort,
    get_subcircuit_library,
    register_subcircuit,
)


class TestSubcircuitPort:
    """Tests for SubcircuitPort."""

    def test_port_creation(self) -> None:
        """Test creating a port."""
        port = SubcircuitPort("in", "Input signal")
        assert port.name == "in"
        assert port.description == "Input signal"

    def test_port_default_description(self) -> None:
        """Test port with default empty description."""
        port = SubcircuitPort("out")
        assert port.name == "out"
        assert port.description == ""


class TestSubcircuitParameter:
    """Tests for SubcircuitParameter."""

    def test_parameter_creation(self) -> None:
        """Test creating a parameter."""
        param = SubcircuitParameter("R", 1000, "Resistance", min_value=1, max_value=1e6)
        assert param.name == "R"
        assert param.default == 1000
        assert param.min_value == 1
        assert param.max_value == 1e6

    def test_parameter_validate_in_range(self) -> None:
        """Test parameter validation within range."""
        param = SubcircuitParameter("R", 1000, min_value=100, max_value=10000)
        assert param.validate(500) is True
        assert param.validate(100) is True
        assert param.validate(10000) is True

    def test_parameter_validate_out_of_range(self) -> None:
        """Test parameter validation out of range."""
        param = SubcircuitParameter("R", 1000, min_value=100, max_value=10000)
        assert param.validate(50) is False
        assert param.validate(20000) is False

    def test_parameter_validate_string(self) -> None:
        """Test that string values always pass validation."""
        param = SubcircuitParameter("R", 1000, min_value=100)
        assert param.validate("1k") is True
        assert param.validate("{R_value}") is True

    def test_parameter_validate_no_limits(self) -> None:
        """Test validation with no limits."""
        param = SubcircuitParameter("R", 1000)
        assert param.validate(0.001) is True
        assert param.validate(1e12) is True


class TestSubcircuitDefinition:
    """Tests for SubcircuitDefinition."""

    def test_basic_definition(self) -> None:
        """Test creating a basic subcircuit definition."""
        defn = SubcircuitDefinition(
            name="RC_FILTER",
            ports=[SubcircuitPort("in"), SubcircuitPort("out"), SubcircuitPort("gnd")],
            body="R1 in out 1k\nC1 out gnd 1n",
        )
        assert defn.name == "RC_FILTER"
        assert len(defn.ports) == 3
        assert defn.port_names == ["in", "out", "gnd"]

    def test_definition_with_parameters(self) -> None:
        """Test definition with parameters."""
        defn = SubcircuitDefinition(
            name="RC_FILTER",
            ports=[SubcircuitPort("in"), SubcircuitPort("out")],
            body="R1 in out {R}\nC1 out 0 {C}",
            parameters=[
                SubcircuitParameter("R", 1000),
                SubcircuitParameter("C", 1e-9),
            ],
        )
        assert defn.parameter_names == ["R", "C"]
        assert defn.default_params == {"R": 1000, "C": 1e-9}

    def test_to_spice_basic(self) -> None:
        """Test generating SPICE output."""
        defn = SubcircuitDefinition(
            name="SIMPLE",
            ports=[SubcircuitPort("a"), SubcircuitPort("b")],
            body="R1 a b 1k",
        )
        spice = defn.to_spice()
        assert ".SUBCKT SIMPLE a b" in spice
        assert "R1 a b 1k" in spice
        assert ".ENDS SIMPLE" in spice

    def test_to_spice_with_params(self) -> None:
        """Test SPICE output with parameters."""
        defn = SubcircuitDefinition(
            name="PARAM_TEST",
            ports=[SubcircuitPort("in"), SubcircuitPort("out")],
            body="R1 in out {R}",
            parameters=[SubcircuitParameter("R", 1000)],
        )
        spice = defn.to_spice()
        assert "params: R=1000" in spice

    def test_instantiate(self) -> None:
        """Test generating X-element instance."""
        defn = SubcircuitDefinition(
            name="BUFFER",
            ports=[SubcircuitPort("in"), SubcircuitPort("out"), SubcircuitPort("vdd")],
            body="* buffer body",
        )
        inst = defn.instantiate("1", ["input", "output", "vcc"])
        assert inst == "X1 input output vcc BUFFER"

    def test_instantiate_with_params(self) -> None:
        """Test instantiation with parameter overrides."""
        defn = SubcircuitDefinition(
            name="RC",
            ports=[SubcircuitPort("in"), SubcircuitPort("out")],
            body="R1 in out {R}",
            parameters=[SubcircuitParameter("R", 1000)],
        )
        inst = defn.instantiate("2", ["a", "b"], params={"R": "2k"})
        assert inst == "X2 a b RC R=2k"

    def test_instantiate_wrong_node_count(self) -> None:
        """Test that wrong node count raises error."""
        defn = SubcircuitDefinition(
            name="TWO_PORT",
            ports=[SubcircuitPort("a"), SubcircuitPort("b")],
            body="R1 a b 1k",
        )
        with pytest.raises(ValueError, match="has 2 ports"):
            defn.instantiate("1", ["only_one"])

    def test_validate_params_unknown(self) -> None:
        """Test validation catches unknown parameters."""
        defn = SubcircuitDefinition(
            name="TEST",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 {R}",
            parameters=[SubcircuitParameter("R", 1000)],
        )
        errors = defn.validate_params({"R": 500, "UNKNOWN": 1})
        assert len(errors) == 1
        assert "Unknown parameter 'UNKNOWN'" in errors[0]

    def test_validate_params_out_of_range(self) -> None:
        """Test validation catches out-of-range values."""
        defn = SubcircuitDefinition(
            name="TEST",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 {R}",
            parameters=[SubcircuitParameter("R", 1000, min_value=100, max_value=10000)],
        )
        errors = defn.validate_params({"R": 50})
        assert len(errors) == 1
        assert "out of range" in errors[0]

    def test_from_spice_basic(self) -> None:
        """Test parsing from SPICE text."""
        spice = """
.SUBCKT MYRESISTOR a b
R1 a b 1k
.ENDS MYRESISTOR
"""
        defn = SubcircuitDefinition.from_spice(spice)
        assert defn.name == "MYRESISTOR"
        assert defn.port_names == ["a", "b"]
        assert "R1 a b 1k" in defn.body

    def test_from_spice_with_params(self) -> None:
        """Test parsing with parameters."""
        spice = """
.SUBCKT PARAMRES in out params: R=1000 C=1e-9
R1 in out {R}
C1 out 0 {C}
.ENDS PARAMRES
"""
        defn = SubcircuitDefinition.from_spice(spice)
        assert defn.name == "PARAMRES"
        assert len(defn.parameters) == 2
        assert defn.parameters[0].name == "R"
        assert defn.parameters[0].default == 1000
        assert defn.parameters[1].name == "C"

    def test_from_spice_no_subckt(self) -> None:
        """Test error when no .SUBCKT found."""
        with pytest.raises(ValueError, match="No .SUBCKT"):
            SubcircuitDefinition.from_spice("R1 a b 1k")

    def test_from_spice_no_ends(self) -> None:
        """Test error when no .ENDS found."""
        with pytest.raises(ValueError, match="No .ENDS"):
            SubcircuitDefinition.from_spice(".SUBCKT TEST a b\nR1 a b 1k")


class TestSubcircuitLibrary:
    """Tests for SubcircuitLibrary."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving definitions."""
        lib = SubcircuitLibrary()
        defn = SubcircuitDefinition(
            name="TEST_SUB",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 1k",
        )
        lib.register(defn)

        retrieved = lib.get("TEST_SUB")
        assert retrieved is not None
        assert retrieved.name == "TEST_SUB"

    def test_register_duplicate_raises(self) -> None:
        """Test that duplicate registration raises error."""
        lib = SubcircuitLibrary()
        defn = SubcircuitDefinition(
            name="DUP",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 1k",
        )
        lib.register(defn)

        with pytest.raises(ValueError, match="already registered"):
            lib.register(defn)

    def test_register_or_replace(self) -> None:
        """Test register_or_replace overwrites."""
        lib = SubcircuitLibrary()
        defn1 = SubcircuitDefinition(
            name="REPLACEABLE",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 1k",
            description="Original",
        )
        lib.register(defn1)

        defn2 = SubcircuitDefinition(
            name="REPLACEABLE",
            ports=[SubcircuitPort("a"), SubcircuitPort("b")],
            body="R1 a b 2k",
            description="Replaced",
        )
        lib.register_or_replace(defn2)

        retrieved = lib.get("REPLACEABLE")
        assert retrieved is not None
        assert retrieved.description == "Replaced"
        assert len(retrieved.ports) == 2

    def test_getitem(self) -> None:
        """Test dictionary-style access."""
        lib = SubcircuitLibrary()
        defn = SubcircuitDefinition(
            name="INDEXED",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 1k",
        )
        lib.register(defn)

        assert lib["INDEXED"].name == "INDEXED"

    def test_getitem_missing(self) -> None:
        """Test KeyError for missing subcircuit."""
        lib = SubcircuitLibrary()
        with pytest.raises(KeyError, match="not found"):
            _ = lib["NONEXISTENT"]

    def test_contains(self) -> None:
        """Test 'in' operator."""
        lib = SubcircuitLibrary()
        defn = SubcircuitDefinition(
            name="CONTAINS_TEST",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 1k",
        )
        lib.register(defn)

        assert "CONTAINS_TEST" in lib
        assert "MISSING" not in lib

    def test_names_property(self) -> None:
        """Test names property."""
        lib = SubcircuitLibrary()
        lib.register(SubcircuitDefinition("A", [SubcircuitPort("x")], "R1 x 0 1k"))
        lib.register(SubcircuitDefinition("B", [SubcircuitPort("x")], "R1 x 0 1k"))

        names = lib.names
        assert "A" in names
        assert "B" in names

    def test_categories(self) -> None:
        """Test category organization."""
        lib = SubcircuitLibrary()
        lib.register(
            SubcircuitDefinition("FILTER1", [SubcircuitPort("a")], "R1 a 0 1k", category="filters")
        )
        lib.register(
            SubcircuitDefinition("FILTER2", [SubcircuitPort("a")], "R1 a 0 1k", category="filters")
        )
        lib.register(
            SubcircuitDefinition("AMP1", [SubcircuitPort("a")], "R1 a 0 1k", category="amplifiers")
        )

        assert "filters" in lib.categories
        assert "amplifiers" in lib.categories
        assert lib.list_by_category("filters") == ["FILTER1", "FILTER2"]
        assert lib.list_by_category("amplifiers") == ["AMP1"]

    def test_load_file(self) -> None:
        """Test loading from file."""
        content = """
.SUBCKT RESISTOR_PAIR a b c
R1 a b 1k
R2 b c 1k
.ENDS RESISTOR_PAIR

.SUBCKT CAP_PAIR a b c
C1 a b 1n
C2 b c 1n
.ENDS CAP_PAIR
"""
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False, mode="w") as f:
            f.write(content)
            path = Path(f.name)

        try:
            lib = SubcircuitLibrary()
            loaded = lib.load_file(path)

            assert "RESISTOR_PAIR" in loaded
            assert "CAP_PAIR" in loaded
            assert lib.get("RESISTOR_PAIR") is not None
            assert lib.get("CAP_PAIR") is not None
        finally:
            path.unlink()

    def test_load_file_not_found(self) -> None:
        """Test error for missing file."""
        lib = SubcircuitLibrary()
        with pytest.raises(FileNotFoundError):
            lib.load_file(Path("/nonexistent/file.sub"))

    def test_load_file_caching(self) -> None:
        """Test that files are only loaded once."""
        content = ".SUBCKT ONCE a b\nR1 a b 1k\n.ENDS ONCE"
        with tempfile.NamedTemporaryFile(suffix=".sub", delete=False, mode="w") as f:
            f.write(content)
            path = Path(f.name)

        try:
            lib = SubcircuitLibrary()
            loaded1 = lib.load_file(path)
            loaded2 = lib.load_file(path)

            assert loaded1 == loaded2
            assert len(lib.names) == 1
        finally:
            path.unlink()

    def test_load_directory(self) -> None:
        """Test loading all files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "file1.sub").write_text(
                ".SUBCKT SUB1 a b\nR1 a b 1k\n.ENDS SUB1"
            )
            (Path(tmpdir) / "file2.sub").write_text(
                ".SUBCKT SUB2 a b\nC1 a b 1n\n.ENDS SUB2"
            )

            lib = SubcircuitLibrary()
            results = lib.load_directory(Path(tmpdir))

            assert len(results) == 2
            assert "SUB1" in lib
            assert "SUB2" in lib

    def test_to_spice(self) -> None:
        """Test generating combined SPICE output."""
        lib = SubcircuitLibrary()
        lib.register(SubcircuitDefinition("A", [SubcircuitPort("x")], "R1 x 0 1k"))
        lib.register(SubcircuitDefinition("B", [SubcircuitPort("y")], "C1 y 0 1n"))

        spice = lib.to_spice()
        assert ".SUBCKT A" in spice
        assert ".SUBCKT B" in spice

    def test_to_spice_subset(self) -> None:
        """Test generating SPICE for subset."""
        lib = SubcircuitLibrary()
        lib.register(SubcircuitDefinition("A", [SubcircuitPort("x")], "R1 x 0 1k"))
        lib.register(SubcircuitDefinition("B", [SubcircuitPort("y")], "C1 y 0 1n"))

        spice = lib.to_spice(["A"])
        assert ".SUBCKT A" in spice
        assert ".SUBCKT B" not in spice

    def test_get_dependencies(self) -> None:
        """Test dependency detection."""
        lib = SubcircuitLibrary()
        lib.register(SubcircuitDefinition("INNER", [SubcircuitPort("a")], "R1 a 0 1k"))
        lib.register(
            SubcircuitDefinition(
                "OUTER",
                [SubcircuitPort("in"), SubcircuitPort("out")],
                "X1 in mid INNER\nX2 mid out INNER",
            )
        )

        deps = lib.get_dependencies("OUTER")
        assert "INNER" in deps

    def test_get_all_dependencies_transitive(self) -> None:
        """Test transitive dependency detection."""
        lib = SubcircuitLibrary()
        lib.register(SubcircuitDefinition("LEVEL0", [SubcircuitPort("a")], "R1 a 0 1k"))
        lib.register(
            SubcircuitDefinition("LEVEL1", [SubcircuitPort("a")], "X1 a 0 LEVEL0")
        )
        lib.register(
            SubcircuitDefinition("LEVEL2", [SubcircuitPort("a")], "X1 a 0 LEVEL1")
        )

        deps = lib.get_all_dependencies("LEVEL2")
        assert "LEVEL1" in deps
        assert "LEVEL0" in deps

    def test_clear(self) -> None:
        """Test clearing library."""
        lib = SubcircuitLibrary()
        lib.register(SubcircuitDefinition("TO_CLEAR", [SubcircuitPort("a")], "R1 a 0 1k"))

        lib.clear()

        assert len(lib.names) == 0
        assert "TO_CLEAR" not in lib


class TestGlobalLibrary:
    """Tests for global library functions."""

    def test_get_library_singleton(self) -> None:
        """Test that get_subcircuit_library returns same instance."""
        lib1 = get_subcircuit_library()
        lib2 = get_subcircuit_library()
        assert lib1 is lib2

    def test_register_and_get_global(self) -> None:
        """Test global register and get functions."""
        # Clear any existing
        get_subcircuit_library().clear()

        defn = SubcircuitDefinition(
            name="GLOBAL_TEST",
            ports=[SubcircuitPort("a")],
            body="R1 a 0 1k",
        )
        register_subcircuit(defn)

        from spicelab.core.subcircuit import get_subcircuit

        retrieved = get_subcircuit("GLOBAL_TEST")
        assert retrieved is not None
        assert retrieved.name == "GLOBAL_TEST"

        # Cleanup
        get_subcircuit_library().clear()


class TestIntegration:
    """Integration tests for subcircuit system."""

    def test_create_hierarchical_circuit(self) -> None:
        """Test creating a hierarchical circuit structure."""
        # Define an RC filter subcircuit
        rc_filter = SubcircuitDefinition(
            name="RC_LOWPASS",
            ports=[
                SubcircuitPort("in", "Input"),
                SubcircuitPort("out", "Output"),
                SubcircuitPort("gnd", "Ground"),
            ],
            body="""R1 in out {R}
C1 out gnd {C}""",
            parameters=[
                SubcircuitParameter("R", 1000, "Filter resistance"),
                SubcircuitParameter("C", 1e-9, "Filter capacitance"),
            ],
            category="filters",
        )

        # Register in library
        lib = SubcircuitLibrary()
        lib.register(rc_filter)

        # Generate SPICE definition
        spice = rc_filter.to_spice()
        assert ".SUBCKT RC_LOWPASS in out gnd params: R=1000 C=1e-09" in spice
        assert ".ENDS RC_LOWPASS" in spice

        # Generate instances with different parameters
        inst1 = rc_filter.instantiate("1", ["input", "mid", "0"], params={"R": "1k", "C": "10n"})
        inst2 = rc_filter.instantiate("2", ["mid", "output", "0"], params={"R": "10k", "C": "1n"})

        assert inst1 == "X1 input mid 0 RC_LOWPASS R=1k C=10n"
        assert inst2 == "X2 mid output 0 RC_LOWPASS R=10k C=1n"

    def test_nested_subcircuits(self) -> None:
        """Test nested subcircuit dependencies."""
        lib = SubcircuitLibrary()

        # Level 0: Basic resistor
        lib.register(
            SubcircuitDefinition(
                name="VAR_RES",
                ports=[SubcircuitPort("a"), SubcircuitPort("b")],
                body="R1 a b {R}",
                parameters=[SubcircuitParameter("R", 1000)],
            )
        )

        # Level 1: Voltage divider using VAR_RES
        lib.register(
            SubcircuitDefinition(
                name="VDIV",
                ports=[SubcircuitPort("in"), SubcircuitPort("out"), SubcircuitPort("gnd")],
                body="X1 in out VAR_RES R={R1}\nX2 out gnd VAR_RES R={R2}",
                parameters=[
                    SubcircuitParameter("R1", 1000),
                    SubcircuitParameter("R2", 1000),
                ],
            )
        )

        # Level 2: Buffer using VDIV
        lib.register(
            SubcircuitDefinition(
                name="BUFFERED_VDIV",
                ports=[SubcircuitPort("in"), SubcircuitPort("out"), SubcircuitPort("gnd")],
                body="X1 in mid gnd VDIV R1=10k R2=10k\nE1 out gnd mid gnd 1",
            )
        )

        # Check dependency chain
        deps = lib.get_all_dependencies("BUFFERED_VDIV")
        assert "VDIV" in deps
        assert "VAR_RES" in deps

        # Generate all required definitions
        all_defs = lib.to_spice(["VAR_RES", "VDIV", "BUFFERED_VDIV"])
        assert all_defs.count(".SUBCKT") == 3
        assert all_defs.count(".ENDS") == 3
