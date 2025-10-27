"""Tests for Circuit.with_param() integration (Phase 2.6)."""

from __future__ import annotations

import pytest
from spicelab.core.circuit_v2 import ImmutableCircuit
from spicelab.core.components import Capacitor, Resistor
from spicelab.core.net import GND, Net
from spicelab.core.parameter import NormalTolerance, Parameter, ParameterRef, UniformTolerance
from spicelab.core.units import Unit


# ======================================================================================
# Circuit.with_param() Tests
# ======================================================================================
def test_circuit_with_param_basic():
    """Circuit can store parameters."""
    r_param = Parameter("Rload", 10_000, Unit.OHM)

    circuit = ImmutableCircuit("test").with_param("Rload", r_param)

    assert "Rload" in circuit.params
    assert circuit.params["Rload"] is r_param


def test_circuit_with_param_multiple():
    """Circuit can store multiple parameters."""
    r_param = Parameter("Rload", 10_000, Unit.OHM)
    c_param = Parameter("Cfilt", 1e-6, Unit.FARAD)

    circuit = ImmutableCircuit("test").with_param("Rload", r_param).with_param("Cfilt", c_param)

    assert len(circuit.params) == 2
    assert "Rload" in circuit.params
    assert "Cfilt" in circuit.params


def test_circuit_with_param_immutable():
    """with_param() returns new circuit (immutability)."""
    r_param = Parameter("Rload", 10_000, Unit.OHM)

    circuit1 = ImmutableCircuit("test")
    circuit2 = circuit1.with_param("Rload", r_param)

    # circuit1 unchanged
    assert len(circuit1.params) == 0
    assert "Rload" not in circuit1.params

    # circuit2 has parameter
    assert len(circuit2.params) == 1
    assert "Rload" in circuit2.params


def test_circuit_with_param_override():
    """Later with_param() overrides earlier value."""
    r1 = Parameter("Rload", 10_000, Unit.OHM)
    r2 = Parameter("Rload", 20_000, Unit.OHM)

    circuit = (
        ImmutableCircuit("test").with_param("Rload", r1).with_param("Rload", r2)  # Override
    )

    assert circuit.params["Rload"].nominal == 20_000


# ======================================================================================
# Netlist Generation with Parameters
# ======================================================================================
def test_netlist_with_param_declarations():
    """build_netlist() generates .param statements."""
    r_param = Parameter("Rload", 10_000, Unit.OHM)
    c_param = Parameter("Cfilt", 1e-6, Unit.FARAD)

    circuit = (
        ImmutableCircuit("rc_filter").with_param("Rload", r_param).with_param("Cfilt", c_param)
    )

    netlist = circuit.build_netlist()

    # Should contain .param statements
    assert ".param Rload=10kOhm" in netlist
    assert ".param Cfilt=1uF" in netlist

    # Check ordering (params before components)
    param_idx = netlist.index(".param Rload")
    end_idx = netlist.index(".end")
    assert param_idx < end_idx


def test_netlist_with_param_and_components():
    """Components using ParameterRef generate correct netlist."""
    # Define parameters
    r_param = Parameter("Rload", 10_000, Unit.OHM)
    c_param = Parameter("Cfilt", 100e-9, Unit.FARAD)

    # Create components with ParameterRef
    r1 = Resistor(ref="1", resistance=ParameterRef("Rload"))
    c1 = Capacitor(ref="1", capacitance=ParameterRef("Cfilt"))

    # Build circuit
    vin = Net("in")
    vout = Net("out")

    circuit = (
        ImmutableCircuit("rc_filter")
        .with_param("Rload", r_param)
        .with_param("Cfilt", c_param)
        .add(r1)
        .add(c1)
        .connect(r1.ports[0], vin)
        .connect(r1.ports[1], vout)
        .connect(c1.ports[0], vout)
        .connect(c1.ports[1], GND)
    )

    netlist = circuit.build_netlist()

    # Check .param declarations
    assert ".param Rload=10kOhm" in netlist
    assert ".param Cfilt=100nF" in netlist

    # Check component cards use {param_name}
    assert "R1" in netlist
    assert "{Rload}" in netlist
    assert "C1" in netlist
    assert "{Cfilt}" in netlist


def test_netlist_param_with_tolerance():
    """Parameters with tolerance generate correct .param."""
    r_param = Parameter(
        "Rload",
        10_000,
        Unit.OHM,
        tolerance=NormalTolerance(5.0),
        description="Load resistance with 5% tolerance",
    )

    circuit = ImmutableCircuit("test").with_param("Rload", r_param)
    netlist = circuit.build_netlist()

    # Tolerance not in .param (used for Monte Carlo)
    assert ".param Rload=10kOhm" in netlist
    # Description might be in comment
    if "Load resistance" in netlist:
        assert "* Load resistance" in netlist or "Load resistance" in netlist


def test_netlist_param_with_expression():
    """Parameters with expressions generate correct .param."""
    r1_param = Parameter("R1", 1_000, Unit.OHM)
    r2_param = Parameter("R2", 2_000, Unit.OHM)
    rtotal_param = Parameter("Rtotal", 0, Unit.OHM, expression="R1 + R2")

    circuit = (
        ImmutableCircuit("test")
        .with_param("R1", r1_param)
        .with_param("R2", r2_param)
        .with_param("Rtotal", rtotal_param)
    )

    netlist = circuit.build_netlist()

    assert ".param R1=1kOhm" in netlist
    assert ".param R2=2kOhm" in netlist
    assert ".param Rtotal=R1 + R2" in netlist


def test_netlist_no_params():
    """Circuit without parameters generates netlist without .param section."""
    r1 = Resistor(ref="1", value="10k")  # Legacy value

    circuit = (
        ImmutableCircuit("test").add(r1).connect(r1.ports[0], Net("in")).connect(r1.ports[1], GND)
    )

    netlist = circuit.build_netlist()

    # No .param section
    assert ".param" not in netlist

    # But component is there
    assert "R1" in netlist
    assert "10k" in netlist


# ======================================================================================
# Integration Tests
# ======================================================================================
def test_complete_workflow():
    """Complete workflow: define params, add components, generate netlist."""
    # Define parameters with tolerances
    r_load = Parameter(
        "Rload", 10_000, Unit.OHM, tolerance=UniformTolerance(10.0), description="Load resistance"
    )

    c_filt = Parameter(
        "Cfilt", 1e-6, Unit.FARAD, tolerance=NormalTolerance(5.0), description="Filter capacitor"
    )

    tau = Parameter(
        "tau", 0, Unit.SECOND, expression="Rload * Cfilt", description="RC time constant"
    )

    # Create components using ParameterRef
    r1 = Resistor(ref="load", resistance=ParameterRef("Rload"))
    c1 = Capacitor(ref="filt", capacitance=ParameterRef("Cfilt"))

    # Build circuit
    vin = Net("in")
    vout = Net("out")

    circuit = (
        ImmutableCircuit("rc_lowpass")
        .with_param("Rload", r_load)
        .with_param("Cfilt", c_filt)
        .with_param("tau", tau)
        .add(r1)
        .add(c1)
        .connect(r1.ports[0], vin)
        .connect(r1.ports[1], vout)
        .connect(c1.ports[0], vout)
        .connect(c1.ports[1], GND)
        .add_directive(".ac dec 10 1 100k")
    )

    # Generate netlist
    netlist = circuit.build_netlist()

    # Validate structure
    assert "* rc_lowpass" in netlist
    assert "* Parameters" in netlist
    assert ".param Rload=10kOhm" in netlist
    assert ".param Cfilt=1uF" in netlist
    assert ".param tau=Rload * Cfilt" in netlist
    assert "Rload" in netlist and "{Rload}" in netlist  # Component with ParameterRef
    assert "Cfilt" in netlist and "{Cfilt}" in netlist
    assert ".ac dec 10 1 100k" in netlist
    assert ".end" in netlist

    # Check ordering: params before components
    param_idx = netlist.index(".param Rload")
    comp_idx = netlist.index("{Rload}")  # Find ParameterRef usage
    assert param_idx < comp_idx


def test_mixed_legacy_and_typed():
    """Circuit can mix legacy value and typed ParameterRef components."""
    r1 = Resistor(ref="1", value="1k")  # Legacy
    r2 = Resistor(ref="2", resistance=ParameterRef("Rload"))  # Typed

    r_param = Parameter("Rload", 10_000, Unit.OHM)

    circuit = (
        ImmutableCircuit("mixed")
        .with_param("Rload", r_param)
        .add(r1)
        .add(r2)
        .connect(r1.ports[0], Net("in"))
        .connect(r1.ports[1], Net("mid"))
        .connect(r2.ports[0], Net("mid"))
        .connect(r2.ports[1], GND)
    )

    netlist = circuit.build_netlist()

    # Both styles work
    assert "R1" in netlist
    assert "1k" in netlist
    assert "R2" in netlist
    assert "{Rload}" in netlist
    assert ".param Rload=10kOhm" in netlist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
