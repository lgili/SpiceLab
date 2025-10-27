from __future__ import annotations

import importlib

import pytest

dsl_spec = importlib.import_module("spicelab.dsl.spec")

BuildResult = dsl_spec.BuildResult
CircuitSpec = dsl_spec.CircuitSpec
ComponentSpec = dsl_spec.ComponentSpec
NetSpec = dsl_spec.NetSpec
build_circuit_from_spec = dsl_spec.build_circuit_from_spec
load_circuit_spec = dsl_spec.load_circuit_spec


def test_build_circuit_from_spec_basic_rc() -> None:
    spec = CircuitSpec(
        name="rc_lowpass",
        nets=[
            NetSpec(name="vin", aliases=["in"]),
            NetSpec(name="vout"),
        ],
        components=[
            ComponentSpec(kind="vdc", nodes=["vin", "0"], value="5"),
            ComponentSpec(kind="resistor", nodes=["vin", "vout"], value="1k"),
            ComponentSpec(kind="capacitor", nodes=["vout", "0"], value="100n"),
        ],
        directives=[".tran 1us 1ms"],
    )

    result = build_circuit_from_spec(spec)
    circuit = result.circuit

    netlist = circuit.build_netlist().strip().splitlines()
    assert netlist[0] == "* rc_lowpass"
    assert "VV1 in 0 5" in netlist
    assert "RR1 in vout 1k" in netlist
    assert "CC1 vout 0 100n" in netlist
    assert ".tran 1us 1ms" in netlist
    assert netlist[-1] == ".end"

    assert {ref for ref in result.components} == {"V1", "R1", "C1"}


def test_load_circuit_spec_from_dict_aliases() -> None:
    data = {
        "name": "alias_test",
        "nets": [
            {"name": "out", "aliases": ["vout", "out"]},
        ],
        "components": [
            {"kind": "inductor", "nodes": ["out", "gnd"], "value": "10u"},
        ],
    }

    result = load_circuit_spec(data)
    circuit = result.circuit
    lines = circuit.build_netlist().splitlines()
    assert any(line.startswith("LL1 out 0 10u") for line in lines)


def test_component_spec_node_validation() -> None:
    spec = CircuitSpec(
        name="invalid",
        components=[ComponentSpec(kind="resistor", nodes=["a"], value="1k")],
    )
    with pytest.raises(ValueError):
        build_circuit_from_spec(spec)
