from pathlib import Path

from spicelab.core.circuit import Circuit
from spicelab.core.components import OpAmpIdeal, Resistor, Vdc
from spicelab.core.net import GND
from spicelab.io.ltspice_asc import (
    circuit_from_asc,
    circuit_to_asc_text,
    parse_asc,
    schematic_to_circuit,
)


def _type_counts(circuit: Circuit) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comp in circuit._components:
        counts[type(comp).__name__] = counts.get(type(comp).__name__, 0) + 1
    return counts


def test_roundtrip_simple_circuit() -> None:
    # build simple circuit programmatically
    c = Circuit("roundtrip_test")
    v = Vdc("V1", "3.3")
    r = Resistor("R1", "10k")
    oa = OpAmpIdeal("OA1")
    c.add(v, r, oa)
    # connect: V+ -> R1 node -> OA input+ ; R1 other -> GND ; V- -> GND ; OA out -> GND (simple)
    c.connect(v.ports[0], r.ports[0])
    c.connect(r.ports[1], GND)
    c.connect(v.ports[1], GND)
    # connect OA inp to R node, inn to GND, out to a new net (connect to GND here to keep simple)
    c.connect(oa.ports[0], r.ports[0])
    c.connect(oa.ports[1], GND)
    c.connect(oa.ports[2], GND)

    asc_text = circuit_to_asc_text(c, include_wires=True)
    schematic = parse_asc(asc_text)
    c2 = schematic_to_circuit(schematic)

    # compare type counts
    t1 = _type_counts(c)
    t2 = _type_counts(c2)
    assert t1 == t2, f"Round-trip changed component type counts: {t1} != {t2}"


def test_roundtrip_save_and_load(tmp_path: Path) -> None:
    # same as above but write to file and load via circuit_from_asc
    c = Circuit("roundtrip_file")
    v = Vdc("V1", "5")
    r = Resistor("R1", "1k")
    c.add(v, r)
    c.connect(v.ports[0], r.ports[0])
    c.connect(r.ports[1], GND)
    c.connect(v.ports[1], GND)

    asc_text = circuit_to_asc_text(c)
    p = tmp_path / "rt.asc"
    p.write_text(asc_text, encoding="utf-8")

    c_loaded = circuit_from_asc(p)
    assert _type_counts(c) == _type_counts(c_loaded)
