import os
import tempfile

from spicelab.io.ltspice_parser import from_ltspice_file


def _write(p: str, text: str) -> str:
    with open(p, "w") as f:
        f.write(text)
    return p


def test_parse_v_pulse_basic() -> None:
    text = """* rc pulse
V1 n1 0 PULSE(0 5 1u 10n 10n 1u 2u)
R1 n1 n2 1k
C1 n2 0 100n
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert "PULSE(" in net and "V1" in net


def test_parse_include_and_param_substitution() -> None:
    main = """* main
.include vals.inc
V1 in 0 5
R1 in out {RVAL}
C1 out 0 {CVAL}
.end
"""
    vals = ".param RVAL=2k CVAL=220n\n"
    with tempfile.TemporaryDirectory() as td:
        p_main = _write(os.path.join(td, "main.cir"), main)
        _ = _write(os.path.join(td, "vals.inc"), vals)
        c = from_ltspice_file(p_main)
        net = c.build_netlist()
        assert " 2k" in net and " 220n" in net


def test_parse_i_pulse_basic() -> None:
    text = """* i pulse
I1 n1 0 PULSE(0 1m 0 1n 1n 1u 2u)
R1 n1 0 1k
.end
"""
    with tempfile.TemporaryDirectory() as td:
        p = _write(os.path.join(td, "x.cir"), text)
        c = from_ltspice_file(p)
        net = c.build_netlist()
        assert "PULSE(" in net and "I1" in net
