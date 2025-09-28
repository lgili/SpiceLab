from __future__ import annotations

from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.viz.circuit import components_for_net, connectivity_table, summary_table


def test_connectivity_and_summary_tables_smoke() -> None:
    c = Circuit("smoke")
    v1 = Vdc(ref="V1", value="5V")
    r1 = Resistor(ref="R1", value="1k")
    c.add(v1)
    c.add(r1)
    n1 = Net("n1")
    c.connect(v1.ports[0], n1)
    c.connect(v1.ports[1], GND)
    c.connect(r1.ports[0], n1)
    c.connect(r1.ports[1], GND)

    df = connectivity_table(c)
    assert not df.empty

    summary = summary_table(c)
    assert "V1" in summary and "R1" in summary

    comps = components_for_net(c, "n1")
    assert comps == ["R1", "V1"]
