"""Tests for Component with ParameterRef integration (Phase 2.3/2.4)."""

from __future__ import annotations

import pytest
from spicelab.core.components import Capacitor, Inductor, Resistor
from spicelab.core.parameter import ParameterRef


# Helper function for net mapping
def simple_net_of(port):  # noqa: D103
    return "n1" if port.name == "a" else "n2"


# ======================================================================================
# Backward Compatibility Tests (legacy value field)
# ======================================================================================
def test_resistor_legacy_value():
    """Resistor with legacy value field still works."""
    r = Resistor(ref="1", value="10k")

    assert r.ref == "1"
    assert r.value == "10k"
    assert r.resistance is None

    # SPICE card uses value field
    card = r.spice_card(simple_net_of)
    assert card == "R1 n1 n2 10k"


def test_capacitor_legacy_value():
    """Capacitor with legacy value field still works."""
    c = Capacitor(ref="1", value="100n")

    assert c.ref == "1"
    assert c.value == "100n"
    assert c.capacitance is None

    card = c.spice_card(simple_net_of)
    assert card == "C1 n1 n2 100n"


def test_inductor_legacy_value():
    """Inductor with legacy value field still works."""
    ind = Inductor(ref="1", value="1u")

    assert ind.ref == "1"
    assert ind.value == "1u"
    assert ind.inductance is None

    card = ind.spice_card(simple_net_of)
    assert card == "L1 n1 n2 1u"


# ======================================================================================
# Typed Field Tests (new resistance/capacitance/inductance fields)
# ======================================================================================
def test_resistor_typed_float():
    """Resistor with typed resistance field (float)."""
    r = Resistor(ref="1", resistance=10_000)

    assert r.ref == "1"
    assert r.resistance == 10_000
    assert r.value == ""  # Legacy field empty

    card = r.spice_card(simple_net_of)
    assert card == "R1 n1 n2 10000"


def test_capacitor_typed_float():
    """Capacitor with typed capacitance field (float)."""
    c = Capacitor(ref="1", capacitance=100e-9)

    assert c.ref == "1"
    assert c.capacitance == 100e-9

    card = c.spice_card(simple_net_of)
    assert card == "C1 n1 n2 1e-07"


def test_inductor_typed_float():
    """Inductor with typed inductance field (float)."""
    ind = Inductor(ref="1", inductance=1e-6)

    assert ind.ref == "1"
    assert ind.inductance == 1e-6

    card = ind.spice_card(simple_net_of)
    assert card == "L1 n1 n2 1e-06"


# ======================================================================================
# ParameterRef Integration Tests
# ======================================================================================
def test_resistor_parameter_ref():
    """Resistor with ParameterRef references parameter by name."""
    r = Resistor(ref="1", resistance=ParameterRef("Rload"))

    assert r.ref == "1"
    assert isinstance(r.resistance, ParameterRef)
    assert r.resistance.name == "Rload"

    card = r.spice_card(simple_net_of)

    # ParameterRef.__str__() returns {name}
    assert card == "R1 n1 n2 {Rload}"


def test_capacitor_parameter_ref():
    """Capacitor with ParameterRef."""
    c = Capacitor(ref="1", capacitance=ParameterRef("Cfilt"))

    assert isinstance(c.capacitance, ParameterRef)
    assert c.capacitance.name == "Cfilt"

    card = c.spice_card(simple_net_of)
    assert card == "C1 n1 n2 {Cfilt}"


def test_inductor_parameter_ref():
    """Inductor with ParameterRef."""
    ind = Inductor(ref="1", inductance=ParameterRef("Lcoil"))

    assert isinstance(ind.inductance, ParameterRef)
    assert ind.inductance.name == "Lcoil"

    card = ind.spice_card(simple_net_of)
    assert card == "L1 n1 n2 {Lcoil}"


# ======================================================================================
# Mixed Usage Tests
# ======================================================================================
def test_resistor_both_value_and_resistance():
    """When both value and resistance are set, resistance takes precedence."""
    r = Resistor(ref="1", value="1k", resistance=10_000)

    # Typed field wins
    card = r.spice_card(simple_net_of)
    assert card == "R1 n1 n2 10000"


def test_component_ports_still_work():
    """Ports are created correctly regardless of value type."""
    r = Resistor(ref="1", resistance=ParameterRef("R1"))

    assert len(r.ports) == 2
    assert r.ports[0].name == "a"
    assert r.ports[1].name == "b"


# ======================================================================================
# Integration with Circuit (preview - full integration in P2.6)
# ======================================================================================
def test_multiple_components_with_params():
    """Multiple components can reference different parameters."""
    r1 = Resistor(ref="1", resistance=ParameterRef("Rload"))
    r2 = Resistor(ref="2", resistance=ParameterRef("Rsense"))
    c1 = Capacitor(ref="1", capacitance=ParameterRef("Cfilt"))

    assert "Rload" in r1.spice_card(simple_net_of)
    assert "Rsense" in r2.spice_card(simple_net_of)
    assert "Cfilt" in c1.spice_card(simple_net_of)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
