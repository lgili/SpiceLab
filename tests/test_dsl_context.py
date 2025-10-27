from __future__ import annotations

import pytest
from spicelab.core.net import GND
from spicelab.dsl import (
    IC,
    TEMP,
    Circuit,
    Directive,
    DSLContextError,
    Net,
    Option,
    Param,
    R,
    V,
)


def test_circuit_context_param_option_temp_ic_directive() -> None:
    with Circuit("rc_example") as ctx:
        vin = Net("vin")
        vout = Net("vout")
        gnd = Net("0")

        Param("Rval", "10k")
        Option(reltol=1e-3, abstol=1e-6)
        TEMP(27, 85)
        IC(vout="0")
        Directive(".save V(vout)")

        V("VIN", vin, gnd, 5.0)
        R("R1", vin, vout, "Rval")

        circuit = ctx.circuit

    netlist = circuit.build_netlist().strip().splitlines()
    assert netlist[0] == "* rc_example"
    assert ".param Rval=10k" in netlist
    assert any(line.startswith(".option ") for line in netlist)
    assert ".temp 27 85" in netlist
    assert ".ic V(vout)=0" in netlist
    assert ".save V(vout)" in netlist
    assert "VVIN vin 0 5" in netlist
    assert "RR1 vin vout Rval" in netlist


def test_param_outside_context_raises() -> None:
    with pytest.raises(DSLContextError):
        Param("R", 10)


def test_directive_safe_guard() -> None:
    with Circuit("safe_directive"):
        with pytest.raises(ValueError):
            Directive("save V(out)")
        Directive(".include models.lib")


def test_net_alias_and_ground() -> None:
    with Circuit("aliases") as ctx:
        a = Net("VIN")
        b = Net("vin")
        g = Net("GND")
        assert a is b
        assert g is GND
        V("VS", a, g, 1.0)
        circuit = ctx.circuit

    lines = circuit.build_netlist().splitlines()
    assert any(line.startswith("VVS ") for line in lines)
