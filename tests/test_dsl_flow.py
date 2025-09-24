from spicelab.core.components import Resistor
from spicelab.dsl.flow import Parallel, S, chain


def test_chain_and_parallel_to_circuit() -> None:
    # Build two branches in parallel between same nets
    br1 = chain(Resistor("1", "1k"))
    br2 = S(Resistor("2", "2k")) >> Resistor("3", "3k")
    from spicelab.dsl.flow import Chain

    par = Parallel([br1, Chain(br2.items)])
    c, a, b = par.to_circuit("p")
    # Netlist should contain all resistors; ports connected will be validated by build
    net = c.build_netlist()
    assert "R1" in net and "R2" in net and "R3" in net
