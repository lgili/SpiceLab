import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.opamp_closed_loop import build_opamp_closed_loop


def test_build_opamp_closed_loop_netlist() -> None:
    c = build_opamp_closed_loop()
    net = c.build_netlist()
    assert "opamp_cl" in net
    # Expect an opamp instance (OpAmpIdeal emits a VCVS 'E' card)
    assert any(
        line.startswith("E1 ") or line.startswith("O1 ") or "OpAmp" in line or line.startswith("X")
        for line in net.splitlines()
    )
