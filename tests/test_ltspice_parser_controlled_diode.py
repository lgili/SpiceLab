import os
import tempfile

from spicelab.io.ltspice_parser import from_ltspice_file


def _write(p: str, text: str) -> str:
    with open(p, "w") as f:
        f.write(text)
    return p


def test_parse_vcvs_vccs() -> None:
    text = """* vcvs vccs
E1 np nn ncp ncn 10
G1 np2 nn2 ncp2 ncn2 1e-3
R1 np nn 1k
R2 np2 nn2 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert any(line.startswith("E1 ") for line in net.splitlines())
        assert any(line.startswith("G1 ") for line in net.splitlines())


def test_parse_cccs_ccvs() -> None:
    text = """* cccs ccvs
V1 ctrl 0 1
F2 np nn V1 2.0
H3 np2 nn2 V1 1000
R1 np nn 1k
R2 np2 nn2 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert any(line.startswith("F2 ") and " V1 " in line for line in net.splitlines())
        assert any(line.startswith("H3 ") and " V1 " in line for line in net.splitlines())


def test_parse_diode_minimal() -> None:
    text = """* diode
D1 a c Dmod
R1 a 0 1k
R2 c 0 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert any(
            line.startswith("D1 ") and line.strip().endswith(" Dmod") for line in net.splitlines()
        )
