"""Enable-ports demo for AnalogMux8.

Run:
  PYTHONPATH=src python -m examples.mux_enable_ports_demo
"""

from __future__ import annotations

from cat.analysis import OP
from cat.core.circuit import Circuit
from cat.core.components import AnalogMux8, Resistor, Vdc
from cat.core.net import GND, Net


def main() -> None:
    c = Circuit("mux_en")
    vin = Net("vin")
    vout = Net("vout")

    V1 = Vdc("1", 5.0)
    M = AnalogMux8(ref="MU1", r_series=100.0, enable_ports=True, emit_model=True)
    RL = Resistor("L", 1000.0)

    c.add(V1, M, RL)
    c.connect(V1.ports[0], vin)
    c.connect(V1.ports[1], GND)
    c.connect(M.ports[0], vin)
    # Route out2 to vout
    c.connect(M.ports[1 + 2], vout)
    c.connect(RL.ports[0], vout)
    c.connect(RL.ports[1], GND)

    # Drive en2 high, others low
    for i in range(8):
        V_en = Vdc(f"EN{i}", 1.0 if i == 2 else 0.0)
        c.add(V_en)
        c.connect(V_en.ports[0], M.ports[1 + 8 + i])
        c.connect(V_en.ports[1], GND)

    r = OP().run(c)
    vout_final = float(r.traces["v(vout)"].values[-1])
    print("V(vout)=", vout_final)


if __name__ == "__main__":
    main()
