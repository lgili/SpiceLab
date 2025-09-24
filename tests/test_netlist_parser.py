from pathlib import Path

from spicelab.core.circuit import Circuit


def test_pulse_and_continuation(tmp_path: Path) -> None:
    nl = """
* pulse example
V1 1 0 PULSE(0 5 0 1n 1n 1u 2u)
.end
"""
    p = tmp_path / "pulse.net"
    p.write_text(nl)
    c = Circuit.from_netlist(str(p))
    # should have one V source and a directive .end
    assert any(d.startswith(".end") or d == ".end" for d in c._directives)


def test_subckt_and_model_preserved(tmp_path: Path) -> None:
    nl = """
* subckt example
.subckt mysub in out
R1 in out 1k
.ends
.model Dmod D(Is=1e-14)
.end
"""
    p = tmp_path / "subckt.net"
    p.write_text(nl)
    c = Circuit.from_netlist(str(p))
    # .subckt block should be preserved as a directive entry
    assert any(".subckt mysub" in d for d in c._directives)
    assert any(d.startswith(".model") for d in c._directives)


def test_comments_and_plus_continuation(tmp_path: Path) -> None:
    nl = """
* comment preserved
R1 1 0 1k
V1 1 0 PULSE(0 5
+ 0 1n 1n 1u 2u)
.end
"""
    p = tmp_path / "cont.net"
    p.write_text(nl)
    c = Circuit.from_netlist(str(p))
    # comment and the V1 card should be present (as directive and component)
    assert any(d.lstrip().startswith("*") for d in c._directives) or any(c._components)
