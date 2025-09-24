from pathlib import Path

from spicelab.io.spice_parser import preprocess_netlist, tokenize_card


def test_preprocess_continuation_and_subckt(tmp_path: Path) -> None:
    nl = """
* example
.subckt mysub in out
R1 in out 1k
.ends
V1 1 0 PULSE(0 5
+ 0 1n 1n 1u 2u)
.end
"""
    lines = preprocess_netlist(nl)
    # subckt should be one entry
    assert any(line.lstrip().lower().startswith(".subckt") for line in lines)
    # PULSE should remain as single logical line
    assert any("PULSE(" in line for line in lines)


def test_tokenize_parenthesis_group() -> None:
    line = "V1 1 0 PULSE(0 5 0 1n 1n 1u 2u)"
    toks = tokenize_card(line)
    assert toks[0] == "V1"
    assert toks[-1].startswith("PULSE(")
