import os
import tempfile

from spicelab.io.ltspice_parser import from_ltspice_file


def _write(p: str, text: str) -> str:
    with open(p, "w") as f:
        f.write(text)
    return p


def test_parse_v_sin_basic() -> None:
    text = """* vsin
V1 n1 0 SIN(0 1 1k)
R1 n1 0 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert "SIN(" in net and "V1" in net


def test_parse_v_pwl_basic() -> None:
    text = """* vpwl
V1 n1 0 PWL(0 0 1m 5 2m 0)
R1 n1 0 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert "PWL(" in net and "V1" in net


def test_parse_i_sin_and_pwl_basic() -> None:
    text = """* isin/ipwl
I1 a 0 SIN(0 1m 1k)
I2 b 0 PWL(0 0 1m 1m)
R1 a 0 1k
R2 b 0 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert net.count("SIN(") >= 1 and net.count("PWL(") >= 1
