from pathlib import Path

from cat.core.circuit import Circuit
from cat.core.components import Capacitor, Resistor, Vdc
from cat.core.net import GND, Net
from cat.io.ltspice_asc import (
    circuit_to_asc_text,
    parse_asc,
    save_circuit_as_asc,
    schematic_to_circuit,
)

vin = Net("vin")
vout = Net("vout")

circuit = Circuit("rc")
V1 = Vdc("1", 5.0)
R1 = Resistor("1", "1k")
C1 = Capacitor("1", "10n")

circuit.add(V1, R1, C1)
circuit.connect(V1.ports[0], vin)
circuit.connect(R1.ports[0], vin)
circuit.connect(R1.ports[1], vout)
circuit.connect(C1.ports[0], vout)
circuit.connect(C1.ports[1], GND)
circuit.connect(V1.ports[1], GND)

asc_text = circuit_to_asc_text(circuit)
print("Generated .asc:\n", asc_text)

out_path = Path("rc.asc")
save_circuit_as_asc(circuit, out_path)
print("Saved schematic to", out_path.resolve())

reloaded = schematic_to_circuit(parse_asc(asc_text))
print("Round-tripped netlist:\n", reloaded.build_netlist())
