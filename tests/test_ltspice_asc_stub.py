from spicelab.core.circuit import Circuit
from spicelab.core.components import VCVS, Capacitor, Diode, Resistor, Vdc
from spicelab.core.net import GND, Net
from spicelab.io.ltspice_asc import (
    circuit_to_asc_text,
    circuit_to_schematic,
    parse_asc,
    schematic_to_circuit,
)


def _build_simple_circuit() -> Circuit:
    c = Circuit("asc_roundtrip")
    v1 = Vdc("1", 5.0)
    r1 = Resistor("1", "1k")
    c1 = Capacitor("1", "10n")

    vin = Net("vin")
    vout = Net("vout")

    c.add(v1, r1, c1)
    c.connect(v1.ports[0], vin)
    c.connect(r1.ports[0], vin)
    c.connect(r1.ports[1], vout)
    c.connect(c1.ports[0], vout)
    c.connect(c1.ports[1], GND)
    c.connect(v1.ports[1], GND)
    return c


def test_parse_generated_schematic() -> None:
    circuit = _build_simple_circuit()
    text = circuit_to_asc_text(circuit)

    parsed = parse_asc(text)
    assert len(parsed.components) == 3
    assert any(comp.symbol == "voltage" for comp in parsed.components)

    reconstructed = schematic_to_circuit(parsed)
    netlist = reconstructed.build_netlist()
    assert "R1" in netlist and "C1" in netlist
    assert "vin" in netlist and "vout" in netlist
    cap = next(comp for comp in parse_asc(text).components if comp.symbol == "cap")
    assert cap.orientation in {"R0", "R90"}


def test_roundtrip_via_text() -> None:
    circuit = _build_simple_circuit()
    asc_text = circuit_to_asc_text(circuit)
    restored = schematic_to_circuit(parse_asc(asc_text))
    assert "R1" in restored.build_netlist()


def test_geometry_fallback_without_spice_line() -> None:
    circuit = _build_simple_circuit()
    schem = circuit_to_schematic(circuit)
    for comp in schem.components:
        comp.attributes.pop("SpiceLine", None)
    restored = schematic_to_circuit(schem)
    netlist = restored.build_netlist()
    assert "vin" in netlist and "vout" in netlist


def test_diode_roundtrip() -> None:
    circuit = Circuit("dio_test")
    d = Diode("1", "DMOD")
    n1 = Net("n1")
    circuit.add(d)
    circuit.connect(d.ports[0], n1)
    circuit.connect(d.ports[1], GND)

    asc_text = circuit_to_asc_text(circuit)
    restored = schematic_to_circuit(parse_asc(asc_text))
    assert "D1 n1 0 DMOD" in restored.build_netlist()


def test_export_without_wires() -> None:
    circuit = _build_simple_circuit()
    schem = circuit_to_schematic(circuit, include_wires=False)
    assert not schem.wires
    asc_text = circuit_to_asc_text(circuit, include_wires=False)
    assert "SpiceLine" in asc_text


def test_vcvs_roundtrip_with_spiceline() -> None:
    circuit = Circuit("vcvs")
    e1 = VCVS("1", "100")
    circuit.add(e1)
    nets = [Net(name) for name in ("out", "ref", "cp", "cn")]
    for port, net in zip(e1.ports, nets, strict=False):
        circuit.connect(port, net)

    asc_text = circuit_to_asc_text(circuit)
    assert "SpiceLine" in asc_text
    restored = schematic_to_circuit(parse_asc(asc_text))
    netlist = restored.build_netlist().replace("\n", " ")
    assert "E1 out ref cp cn 100" in netlist


def test_include_wires_true_emits_geometry() -> None:
    circuit = _build_simple_circuit()
    schem = circuit_to_schematic(circuit, include_wires=True)
    assert schem.wires
    for comp in schem.components:
        comp.attributes.pop("SpiceLine", None)
    restored = schematic_to_circuit(schem)
    netlist = restored.build_netlist()
    assert "vin" in netlist and "vout" in netlist
