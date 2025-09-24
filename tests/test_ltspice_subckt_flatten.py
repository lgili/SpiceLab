import os
import tempfile

from spicelab.io.ltspice_parser import from_ltspice_file


def _write(p: str, text: str) -> str:
    with open(p, "w") as f:
        f.write(text)
    return p


def test_flatten_single_level_subckt_rc() -> None:
    text = """* top with subckt
.param R=1k C=100n
.subckt RC in out
R1 in out {R}
C1 out 0 {C}
.ends
XU1 vin vout RC
V1 vin 0 1
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        lines = net.splitlines()
        # After flatten, we expect R and C with instance-prefixed refs (e.g., RU1_1)
        assert any(line.startswith("RU1_") for line in lines)
    assert any(line.startswith("CU1_") for line in lines)


def test_flatten_nested_one_level_and_params() -> None:
    text = """* nested subckt
.subckt AMP in out
E1 out 0 in 0 10
.ends
.subckt TOP a b
X1 a b AMP
.ends
XU a b TOP PARAMX=ignored
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x2.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        lines = net.splitlines()
    # Expect expanded VCVS with ref prefixed by instance chain (e.g., EU_1_1 or EU1_1)
    assert any(line.split()[0].startswith("EU") for line in lines)
