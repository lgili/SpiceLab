import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.rc_highpass import build_rc_highpass


def test_build_rc_highpass_netlist_contains_components() -> None:
    c = build_rc_highpass()
    net = c.build_netlist()
    assert "rc_highpass" in net
    assert any(line.startswith("V1 ") or line.startswith("V") for line in net.splitlines())
    assert any(line.startswith("C1 ") or line.startswith("C") for line in net.splitlines())
    assert any(line.startswith("R1 ") or line.startswith("R") for line in net.splitlines())
