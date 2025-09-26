from __future__ import annotations

import math

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.net import GND, Net
from spicelab.utils.synth import RCDesign, design_rc_lowpass
from spicelab.utils.topologies import opamp_buffer, opamp_inverting


def test_design_rc_lowpass_prefers_resistor() -> None:
    design = design_rc_lowpass(1_000.0, prefer_R=True)
    assert isinstance(design, RCDesign)
    assert math.isclose(design.fc, 1.0 / (2 * math.pi * design.R * design.C), rel_tol=1e-6)


def test_design_rc_lowpass_prefers_capacitor() -> None:
    design = design_rc_lowpass(10_000.0, prefer_R=False)
    assert design.tau == pytest.approx(design.R * design.C, rel=1e-6)


def test_opamp_topologies_wire_components() -> None:
    circuit = Circuit("amp")
    inp = Net("inp")
    out = Net("out")
    ref = GND

    buffer = opamp_buffer(circuit, inp, out)
    assert buffer in circuit._components

    amp, rin, rf = opamp_inverting(circuit, inp, out, ref, Rin="1k", Rf="10k")
    assert amp in circuit._components
    assert rin.value == "1k"
    assert rf.value == "10k"
